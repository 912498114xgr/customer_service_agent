"""对固定多 Skill 样本运行显式授权的模型评测；该模块不会执行工具。"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from common.config import Settings
from common.llm import LLMClient, create_llm_client
from common.observability import ObservedLLM, RequestMetrics, observe_node, observe_request
from customer_care.nodes.conversation import route_decision
from customer_care.evaluation.orchestration import (
    evaluate_intent_predictions,
    evaluate_plan_predictions,
    load_orchestration_dataset,
)
from customer_care.nodes.intent import recognize_intent
from customer_care.nodes.language import detect_language
from customer_care.orchestration import MultiSkillPlanner
from customer_care.state import CareState
from customer_care.tools.registry import ToolCatalog
from customer_care.tools.router import ToolRouter


async def run_evaluation(
    llm: LLMClient,
    *,
    dataset_path: Path | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """逐条运行意图与计划预测，失败样本保留在分母中并导出调用元数据。"""
    dataset_version, cases = load_orchestration_dataset(dataset_path)
    dataset_case_count = len(cases)
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases 必须大于 0")
        cases = cases[:max_cases]
    observed_llm = ObservedLLM(llm)
    planner = MultiSkillPlanner(observed_llm)
    router = ToolRouter(observed_llm)
    catalog = ToolCatalog.load_default()
    intent_predictions: list[dict[str, Any]] = []
    plan_predictions: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    request_metrics: list[dict[str, Any]] = []

    for case in cases:
        state = CareState(user_id=f"eval-{case.case_id}", messages=[{"role": "user", "content": case.query}])
        metrics = RequestMetrics(request_id=f"eval-{case.case_id}")
        errors: list[str] = []
        intent_prediction: dict[str, Any]
        plan_prediction: dict[str, Any] = {
            "id": case.case_id,
            "tool_names": [],
            "task_arguments": [],
            "dependency_edges": [],
            "needs_clarification": False,
            "unconfirmed_side_effect_tools": [],
        }

        with observe_request(metrics):
            try:
                state.language = detect_language(state)["language"]
                with observe_node("recognize_intent"):
                    intent_update = await recognize_intent(state, observed_llm)
                state = state.model_copy(update=intent_update)
            except Exception as exc:
                errors.append(f"intent:{type(exc).__name__}")
                state = state.model_copy(update={
                    "intent": "other",
                    "intent_confidence": 0.0,
                    "intent_goals": [],
                    "intent_conflict": False,
                    "intent_needs_clarification": True,
                    "multi_skill_requested": False,
                })

            intent_prediction = {
                "id": case.case_id,
                "multi_skill_requested": state.multi_skill_requested,
                "conflict": state.intent_conflict,
                "needs_clarification": state.intent_needs_clarification,
            }
            policy_route = route_decision(state)["route"]
            plan_prediction["needs_clarification"] = policy_route == "clarify"

            if policy_route == "multi_skill":
                try:
                    with observe_node("plan_tasks"):
                        decision = await planner.plan(state)
                    if decision.needs_clarification or decision.plan is None:
                        plan_prediction["needs_clarification"] = True
                        errors.append("planner:needs_clarification")
                    else:
                        plan = decision.plan
                        plan_prediction["tool_names"] = [task.tool_name for task in plan.tasks]
                        plan_prediction["task_arguments"] = [task.arguments for task in plan.tasks]
                        task_by_id = {task.task_id: task.tool_name for task in plan.tasks}
                        plan_prediction["dependency_edges"] = [
                            [task_by_id[dependency], task.tool_name]
                            for task in plan.tasks
                            for dependency in task.dependencies
                        ]
                        plan_prediction["unconfirmed_side_effect_tools"] = [
                            task.tool_name
                            for task in plan.tasks
                            if (catalog.get(task.tool_name).side_effect or catalog.get(task.tool_name).requires_confirmation)
                            and not task.requires_confirmation
                        ]
                except Exception as exc:
                    errors.append(f"plan:{type(exc).__name__}")
                    plan_prediction["needs_clarification"] = True
            elif policy_route == "self_service":
                try:
                    with observe_node("skill_router"):
                        selection = await router.select(state)
                    tool_name = str(selection["selected_tool"])
                    plan_prediction["tool_names"] = [tool_name]
                    plan_prediction["task_arguments"] = [selection.get("tool_args", {})]
                    catalog.get(tool_name)
                except Exception as exc:
                    errors.append(f"plan:{type(exc).__name__}")
                    plan_prediction["needs_clarification"] = True

        intent_predictions.append(intent_prediction)
        plan_predictions.append(plan_prediction)
        metrics_data = metrics.finish()
        request_metrics.append(metrics_data)
        case_results.append({
            "id": case.case_id,
            "locale": case.locale,
            "kind": case.kind,
            "predicted_route": policy_route,
            "policy_route": policy_route,
            "intent": intent_prediction,
            "plan": plan_prediction,
            "errors": errors + metrics_data["errors"],
            "metrics": metrics_data,
        })

    return {
        "schema_version": "1.0.0",
        "dataset_version": dataset_version,
        "dataset_sha256": _dataset_hash(dataset_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "model": _model_metadata(),
        "prompt_sources_sha256": _prompt_sources_hash(),
        "evaluation_sources_sha256": _evaluation_sources_hash(),
        "execution": {
            "tools_executed": False,
            "sequential": True,
            "case_count": len(cases),
            "dataset_case_count": dataset_case_count,
            "complete_dataset": len(cases) == dataset_case_count,
            "max_logical_calls": 2 * len(cases),
        },
        "summary": _summarize_calls(request_metrics),
        "intent_metrics": evaluate_intent_predictions(cases, intent_predictions),
        "plan_metrics": evaluate_plan_predictions(cases, plan_predictions),
        "cases": case_results,
    }


def _dataset_hash(path: Path | None) -> str:
    dataset_path = path or Path(__file__).with_name("orchestration_dataset.json")
    return hashlib.sha256(dataset_path.read_bytes()).hexdigest()


def _model_metadata() -> dict[str, Any]:
    settings = Settings()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    try:
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = True
    base_url = settings.llm_base_url
    if base_url:
        parsed = urlsplit(base_url)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        base_url = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return {
        "provider_base_url": base_url,
        "model": settings.llm_model,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
        "temperature": 0.0,
        "git_revision": revision,
        "git_worktree_dirty": dirty,
    }


def _prompt_sources_hash() -> str:
    source_paths = [
        Path(__file__).with_name("..") / "nodes" / "intent.py",
        Path(__file__).with_name("..") / "tools" / "router.py",
        Path(__file__).with_name("..") / "orchestration" / "planner.py",
    ]
    return _hash_sources(source_paths)


def _evaluation_sources_hash() -> str:
    return _hash_sources([Path(__file__), Path(__file__).with_name("orchestration.py")])


def _hash_sources(source_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for source_path in source_paths:
        normalized = source_path.resolve()
        digest.update(normalized.name.encode("utf-8"))
        digest.update(normalized.read_bytes())
    return digest.hexdigest()


def _summarize_calls(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(metrics),
        "llm_calls": sum(item["llm_calls"] for item in metrics),
        "llm_errors": sum(item["llm_errors"] for item in metrics),
        "input_chars": sum(item["input_chars"] for item in metrics),
        "output_chars": sum(item["output_chars"] for item in metrics),
        "estimated_tokens": round(sum(item["input_chars"] + item["output_chars"] for item in metrics) / 4, 1),
        "token_estimate_method": "字符数 / 4，仅粗略估算，不是供应商账单 usage",
        "total_elapsed_ms": round(sum(item["elapsed_ms"] for item in metrics), 2),
        "errors_by_node": [
            {"request_id": item["request_id"], "errors": item["errors"]}
            for item in metrics
            if item["errors"]
        ],
    }


async def _run_cli(output_path: Path, max_cases: int) -> None:
    settings = Settings()
    if output_path.exists():
        raise SystemExit(f"拒绝覆盖已有评测产物：{output_path}")
    if settings.use_mock:
        raise SystemExit("拒绝运行：请在本机 .env 设置 USE_MOCK=false 并配置模型凭据。")
    if not settings.llm_api_key:
        raise SystemExit("拒绝运行：缺少 LLM_API_KEY/OPENAI_API_KEY。")
    report = await run_evaluation(create_llm_client(settings), max_cases=max_cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dataset_version", "model", "execution", "summary", "intent_metrics", "plan_metrics")}, ensure_ascii=False, indent=2))
    print(f"完整逐条结果已写入：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="显式调用配置模型评估多 Skill 路由与计划（不执行工具）")
    parser.add_argument("--allow-model-calls", action="store_true", help="确认允许向配置的模型发送固定评测样本，可能产生费用")
    parser.add_argument("--output", type=Path, help="要求显式指定预测报告输出路径")
    parser.add_argument("--max-cases", type=int, help="本次最多调用的样本数，用于限定费用暴露；必填")
    args = parser.parse_args()
    if not args.allow_model_calls:
        parser.error("真实模型调用需要显式传入 --allow-model-calls")
    if args.output is None:
        parser.error("必须通过 --output 显式指定评测产物路径")
    if args.max_cases is None or args.max_cases < 1:
        parser.error("必须通过 --max-cases 设置大于 0 的调用预算上限")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(_run_cli(args.output, args.max_cases))


if __name__ == "__main__":
    main()
