"""客服路由的可复现离线评测实现。"""

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from common.mock_llm import MockLLM
from customer_care.nodes.conversation import route_decision
from customer_care.state import CareState
from customer_care.tools.registry import ToolCatalog
from customer_care.tools.router import ToolRouter


@dataclass(frozen=True)
class EvaluationCase:
    """单条评测样本及其人工标签。"""

    case_id: str
    query: str
    intent: str
    expected_tool: str


@dataclass(frozen=True)
class CaseResult:
    """单条样本的候选结果。"""

    case_id: str
    query: str
    expected_tool: str
    candidates: tuple[str, ...]

    @property
    def top1(self) -> str | None:
        return self.candidates[0] if self.candidates else None

    def hit_at(self, k: int) -> bool:
        return self.expected_tool in self.candidates[:k]


@dataclass(frozen=True)
class EvaluationReport:
    """评测汇总和可供报告展示的逐条结果。"""

    total: int
    top1_accuracy: float
    recall_at_3: float
    recall_at_10: float
    by_category: dict[str, dict[str, float | int]]
    bad_cases: tuple[CaseResult, ...]
    cases: tuple[CaseResult, ...]

    def as_dict(self) -> dict[str, Any]:
        """转成稳定的 JSON 结构，便于后续保存到 CI 或实验记录。"""
        return {
            "total": self.total,
            "top1_accuracy": self.top1_accuracy,
            "recall_at_3": self.recall_at_3,
            "recall_at_10": self.recall_at_10,
            "by_category": self.by_category,
            "bad_cases": [
                {
                    "id": item.case_id,
                    "query": item.query,
                    "expected_tool": item.expected_tool,
                    "candidates": list(item.candidates),
                }
                for item in self.bad_cases
            ],
        }


def load_dataset(path: Path | None = None) -> list[EvaluationCase]:
    """加载 JSON 标签集，并在重复 ID 或缺字段时尽早失败。"""
    path = path or Path(__file__).with_name("dataset.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        EvaluationCase(
            case_id=item["id"],
            query=item["query"],
            intent=item["intent"],
            expected_tool=item["expected_tool"],
        )
        for item in raw
    ]
    ids = [item.case_id for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("评测集中的 id 必须唯一")
    if not cases:
        raise ValueError("评测集不能为空")
    return cases


def evaluate_dataset(cases: list[EvaluationCase] | None = None) -> EvaluationReport:
    """运行词袋粗筛基线，不发起网络请求，也不依赖真实模型。"""
    cases = cases or load_dataset()
    router = ToolRouter(MockLLM())
    results: list[CaseResult] = []
    for case in cases:
        candidates = tuple(tool.name for tool in router.coarse_candidates(case.query, case.intent))
        results.append(
            CaseResult(
                case_id=case.case_id,
                query=case.query,
                expected_tool=case.expected_tool,
                candidates=candidates,
            )
        )

    total = len(results)
    top1_accuracy = sum(item.hit_at(1) for item in results) / total
    recall_at_3 = sum(item.hit_at(3) for item in results) / total
    recall_at_10 = sum(item.hit_at(10) for item in results) / total
    by_category = _category_metrics(cases, results)
    return EvaluationReport(
        total=total,
        top1_accuracy=top1_accuracy,
        recall_at_3=recall_at_3,
        recall_at_10=recall_at_10,
        by_category=by_category,
        bad_cases=tuple(item for item in results if not item.hit_at(3)),
        cases=tuple(results),
    )


def evaluate_route_policy() -> dict[str, Any]:
    """评估低置信度、负面情绪和歧义的确定性路由策略。"""
    cases = [
        ("clear", CareState(user_id="u", intent="refund", intent_confidence=0.9), "self_service"),
        ("low_confidence", CareState(user_id="u", intent="refund", intent_confidence=0.4), "human"),
        ("negative_emotion", CareState(user_id="u", intent="refund", intent_confidence=0.9, emotion="angry"), "human"),
        ("ambiguous", CareState(user_id="u", intent="refund", intent_confidence=0.9, rewrite_ambiguous=True), "clarify"),
        ("other_intent", CareState(user_id="u", intent="other", intent_confidence=0.9), "clarify"),
    ]
    results = []
    for case_id, state, expected in cases:
        actual = route_decision(state)["route"]
        results.append({"id": case_id, "expected": expected, "actual": actual, "passed": actual == expected})
    return {"total": len(results), "accuracy": sum(item["passed"] for item in results) / len(results), "cases": results}


def _category_metrics(cases: list[EvaluationCase], results: list[CaseResult]) -> dict[str, dict[str, float | int]]:
    """按工具类别分组，帮助定位是某一类工具还是整体召回有问题。"""
    catalog = ToolCatalog.load_default()
    groups: dict[str, list[tuple[EvaluationCase, CaseResult]]] = {}
    for case, result in zip(cases, results):
        category = catalog.get(case.expected_tool).category
        groups.setdefault(category, []).append((case, result))
    metrics: dict[str, dict[str, float | int]] = {}
    for category, items in groups.items():
        metrics[category] = {
            "total": len(items),
            "top1_accuracy": sum(result.hit_at(1) for _, result in items) / len(items),
            "recall_at_3": sum(result.hit_at(3) for _, result in items) / len(items),
            "recall_at_10": sum(result.hit_at(10) for _, result in items) / len(items),
        }
    return metrics


def format_report(report: EvaluationReport, route_report: dict[str, Any]) -> str:
    """生成适合终端和提交记录粘贴的中文报告。"""
    lines = [
        "客服路由离线基线",
        f"样本数：{report.total}",
        f"Top-1：{report.top1_accuracy:.1%}",
        f"Recall@3：{report.recall_at_3:.1%}",
        f"Recall@10：{report.recall_at_10:.1%}",
        "",
        "按工具类别：",
    ]
    for category, metrics in report.by_category.items():
        lines.append(
            f"- {category}: n={metrics['total']}, Top-1={metrics['top1_accuracy']:.1%}, "
            f"R@3={metrics['recall_at_3']:.1%}, R@10={metrics['recall_at_10']:.1%}"
        )
    lines.extend(["", f"确定性路由策略准确率：{route_report['accuracy']:.1%}", "", "Top-3 未召回的 bad case："])
    if report.bad_cases:
        for case in report.bad_cases:
            lines.append(f"- {case.case_id}: 期望 {case.expected_tool}；候选 {', '.join(case.candidates)}；问题：{case.query}")
    else:
        lines.append("- 无")
    return "\n".join(lines)


def main() -> None:
    """命令行入口：输出人类可读报告，并保留机器可读 JSON。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = evaluate_dataset()
    route_report = evaluate_route_policy()
    print(format_report(report, route_report))
    print("\n机器可读结果：")
    print(json.dumps({"routing": report.as_dict(), "route_policy": route_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
