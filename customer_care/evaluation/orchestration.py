"""多 Skill 入口与任务计划的固定样本评测。"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from customer_care.orchestration.detection import is_multi_task_request
from customer_care.tools.registry import ToolCatalog


@dataclass(frozen=True)
class OrchestrationCase:
    case_id: str
    locale: str
    kind: str
    query: str
    expected_multi_task: bool
    expected_tools: tuple[str, ...]
    expected_task_arguments: tuple[dict[str, Any], ...]
    expected_dependencies: tuple[tuple[str, str], ...]
    expected_clarification: bool
    conflict: bool


def load_orchestration_dataset(path: Path | None = None) -> tuple[str, list[OrchestrationCase]]:
    """加载固定回归样本，并拒绝重复 ID、重复 query 和非法标签。"""
    path = path or Path(__file__).with_name("orchestration_dataset.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("version"), str):
        raise ValueError("多 Skill 评测集必须声明版本")
    raw_cases = raw.get("cases", [])
    argument_map = raw.get("expected_arguments", {})
    raw_ids = {item.get("id") for item in raw_cases}
    if set(argument_map) - raw_ids:
        raise ValueError("expected_arguments 包含未知 case id")
    cases = [
        OrchestrationCase(
            case_id=item["id"],
            locale=item["locale"],
            kind=item["kind"],
            query=item["query"],
            expected_multi_task=item["expected_multi_task"],
            expected_tools=tuple(item["expected_tools"]),
            expected_task_arguments=tuple(argument_map.get(item["id"], [])),
            expected_dependencies=tuple(tuple(edge) for edge in item["expected_dependencies"]),
            expected_clarification=item["expected_clarification"],
            conflict=item["conflict"],
        )
        for item in raw.get("cases", [])
    ]
    ids = [case.case_id for case in cases]
    queries = [_normalize_query(case.query) for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("多 Skill 评测集不能为空，且 case id 必须唯一")
    if len(queries) != len(set(queries)):
        raise ValueError("多 Skill 评测集存在重复 query")
    if any(not case.query.strip() or not case.locale or not case.kind for case in cases):
        raise ValueError("多 Skill 评测样本缺少 query、locale 或 kind")
    if any(type(case.expected_multi_task) is not bool or type(case.expected_clarification) is not bool for case in cases):
        raise ValueError("多 Skill 评测标签必须是布尔值")
    if any(case.conflict and not case.expected_clarification for case in cases):
        raise ValueError("冲突样本必须要求澄清")
    if any(case.expected_clarification and case.expected_tools for case in cases):
        raise ValueError("需要澄清的样本不能带可执行工具标签")
    if any(len(case.expected_tools) != len(case.expected_task_arguments) for case in cases):
        raise ValueError("每个期望 Skill 必须有一个对应的参数标签")
    if any(not isinstance(arguments, dict) for case in cases for arguments in case.expected_task_arguments):
        raise ValueError("expected_arguments 必须是与工具列表等长的对象数组")
    if any(not isinstance(tool, str) for case in cases for tool in case.expected_tools):
        raise ValueError("expected_tools 必须只包含字符串")
    if any(len(edge) != 2 or any(not isinstance(node, str) for node in edge) for case in cases for edge in case.expected_dependencies):
        raise ValueError("expected_dependencies 必须是二元字符串数组")
    catalog = ToolCatalog.load_default()
    for case in cases:
        for tool_name in case.expected_tools:
            catalog.get(tool_name)
    return raw["version"], cases


def evaluate_detector(cases: list[OrchestrationCase] | None = None) -> dict[str, Any]:
    """用固定多语言样本测量当前连接词启发式的入口质量。"""
    cases = cases or load_orchestration_dataset()[1]
    results = [
        {
            "id": case.case_id,
            "expected": case.expected_multi_task,
            "actual": is_multi_task_request(case.query),
            "locale": case.locale,
            "kind": case.kind,
        }
        for case in cases
    ]
    true_positive = sum(item["expected"] and item["actual"] for item in results)
    false_positive = sum(not item["expected"] and item["actual"] for item in results)
    false_negative = sum(item["expected"] and not item["actual"] for item in results)
    total = len(results)
    by_locale: dict[str, dict[str, float | int | None]] = {}
    by_kind: dict[str, dict[str, float | int | None]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kind_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[item["locale"]].append(item)
        kind_groups[item["kind"]].append(item)
    for locale, items in groups.items():
        by_locale[locale] = _binary_metrics(items)
    for kind, items in kind_groups.items():
        by_kind[kind] = _binary_metrics(items)
    return {
        "total": total,
        "accuracy": sum(item["expected"] == item["actual"] for item in results) / total,
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": _f1(true_positive, false_positive, false_negative),
        "by_locale": by_locale,
        "by_kind": by_kind,
        "false_positives": [item["id"] for item in results if not item["expected"] and item["actual"]],
        "false_negatives": [item["id"] for item in results if item["expected"] and not item["actual"]],
        "cases": results,
    }


def evaluate_plan_predictions(
    cases: list[OrchestrationCase], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """用统一接口评分模型导出的任务、依赖、澄清和冲突决策。"""
    case_by_id = {case.case_id: case for case in cases}
    prediction_by_id: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or not isinstance(prediction.get("id"), str):
            raise ValueError("每条预测必须是包含字符串 id 的对象")
        case_id = prediction.get("id")
        if case_id not in case_by_id:
            raise ValueError(f"预测中包含未知 case id：{case_id}")
        if case_id in prediction_by_id:
            raise ValueError(f"预测中 case id 重复：{case_id}")
        if not isinstance(prediction.get("tool_names", []), list):
            raise ValueError(f"预测 {case_id} 的 tool_names 必须是数组")
        if not isinstance(prediction.get("dependency_edges", []), list):
            raise ValueError(f"预测 {case_id} 的 dependency_edges 必须是数组")
        if any(not isinstance(name, str) for name in prediction.get("tool_names", [])):
            raise ValueError(f"预测 {case_id} 的 tool_names 必须只包含字符串")
        if not isinstance(prediction.get("task_arguments", []), list) or len(prediction.get("task_arguments", [])) != len(prediction.get("tool_names", [])):
            raise ValueError(f"预测 {case_id} 必须为每个 tool_name 提供对应 task_arguments")
        if any(not isinstance(arguments, dict) for arguments in prediction.get("task_arguments", [])):
            raise ValueError(f"预测 {case_id} 的 task_arguments 必须是对象数组")
        if any(not isinstance(edge, list) or len(edge) != 2 or any(not isinstance(node, str) for node in edge) for edge in prediction.get("dependency_edges", [])):
            raise ValueError(f"预测 {case_id} 的 dependency_edges 必须是二元字符串数组")
        if type(prediction.get("needs_clarification", False)) is not bool:
            raise ValueError(f"预测 {case_id} 的 needs_clarification 必须是布尔值")
        if not isinstance(prediction.get("unconfirmed_side_effect_tools", []), list):
            raise ValueError(f"预测 {case_id} 的 unconfirmed_side_effect_tools 必须是数组")
        if any(not isinstance(name, str) for name in prediction.get("unconfirmed_side_effect_tools", [])):
            raise ValueError(f"预测 {case_id} 的 unconfirmed_side_effect_tools 必须只包含字符串")
        prediction_by_id[case_id] = prediction

    true_positive = false_positive = false_negative = 0
    task_tp = task_fp = task_fn = expected_task_total = 0
    dependency_tp = dependency_fp = dependency_fn = 0
    exact = clarification_correct = conflict_tp = unsafe_conflict = side_effect_violations = 0
    clarification_tp = clarification_fp = clarification_fn = 0
    bad_cases: list[dict[str, Any]] = []
    for case in cases:
        prediction = prediction_by_id.get(case.case_id, {})
        predicted_tools = Counter(prediction.get("tool_names", []))
        expected_tools = Counter(case.expected_tools)
        predicted_tasks = _task_counter(prediction.get("tool_names", []), prediction.get("task_arguments", []))
        expected_tasks = _task_counter(list(case.expected_tools), list(case.expected_task_arguments))
        true_positive += sum((predicted_tools & expected_tools).values())
        false_positive += sum((predicted_tools - expected_tools).values())
        false_negative += sum((expected_tools - predicted_tools).values())
        task_tp += sum((predicted_tasks & expected_tasks).values())
        task_fp += sum((predicted_tasks - expected_tasks).values())
        task_fn += sum((expected_tasks - predicted_tasks).values())
        expected_task_total += sum(expected_tasks.values())
        predicted_edges = {tuple(edge) for edge in prediction.get("dependency_edges", []) if isinstance(edge, list) and len(edge) == 2}
        expected_edges = set(case.expected_dependencies)
        dependency_tp += len(predicted_edges & expected_edges)
        dependency_fp += len(predicted_edges - expected_edges)
        dependency_fn += len(expected_edges - predicted_edges)
        needs_clarification = prediction.get("needs_clarification", False)
        clarification_correct += needs_clarification == case.expected_clarification
        clarification_tp += needs_clarification and case.expected_clarification
        clarification_fp += needs_clarification and not case.expected_clarification
        clarification_fn += not needs_clarification and case.expected_clarification
        conflict_tp += case.conflict and needs_clarification
        unsafe_conflict += case.conflict and bool(predicted_tools)
        side_effect_violations += len(prediction.get("unconfirmed_side_effect_tools", []))
        is_exact = (
            predicted_tasks == expected_tasks
            and predicted_edges == expected_edges
            and needs_clarification == case.expected_clarification
            and not prediction.get("unconfirmed_side_effect_tools")
        )
        exact += is_exact
        if not is_exact or prediction.get("unconfirmed_side_effect_tools"):
            bad_cases.append({
                "id": case.case_id,
                "expected_tools": list(case.expected_tools),
                "actual_tools": list(predicted_tools.elements()),
                "expected_task_arguments": list(case.expected_task_arguments),
                "actual_task_arguments": prediction.get("task_arguments", []),
                "expected_clarification": case.expected_clarification,
                "actual_clarification": needs_clarification,
                "expected_dependencies": [list(edge) for edge in case.expected_dependencies],
                "actual_dependencies": [list(edge) for edge in sorted(predicted_edges)],
                "unconfirmed_side_effect_tools": prediction.get("unconfirmed_side_effect_tools", []),
            })
    conflict_total = sum(case.conflict for case in cases)
    clarification_total = len(cases)
    return {
        "total": len(cases),
        "plan_tool_precision": _ratio(true_positive, true_positive + false_positive),
        "plan_tool_recall": _ratio(true_positive, true_positive + false_negative),
        "plan_tool_f1": _f1(true_positive, false_positive, false_negative),
        "plan_task_precision": _ratio(task_tp, task_tp + task_fp),
        "plan_task_recall": _ratio(task_tp, task_tp + task_fn),
        "plan_task_f1": _f1(task_tp, task_fp, task_fn),
        "argument_exact_accuracy": _ratio(task_tp, expected_task_total),
        "dependency_f1": _f1(dependency_tp, dependency_fp, dependency_fn),
        "exact_plan_accuracy": exact / len(cases),
        "clarification_accuracy": clarification_correct / clarification_total,
        "clarification_precision": _ratio(clarification_tp, clarification_tp + clarification_fp),
        "clarification_recall": _ratio(clarification_tp, clarification_tp + clarification_fn),
        "conflict_recall": _ratio(conflict_tp, conflict_total),
        "unsafe_conflict_execution_count": unsafe_conflict,
        "unconfirmed_side_effect_count": side_effect_violations,
        "bad_cases": bad_cases,
    }


def format_detector_report(version: str, report: dict[str, Any]) -> str:
    """生成启发式入口基线摘要。"""
    return "\n".join([
        f"多 Skill 入口基线 v{version}（连接词启发式，不是 LLM 计划准确率）",
        f"样本数：{report['total']}",
        f"Accuracy：{report['accuracy']:.1%}；Precision：{report['precision']:.1%}；Recall：{report['recall']:.1%}；F1：{report['f1']:.1%}",
        f"误报：{', '.join(report['false_positives']) or '无'}",
        f"漏报：{', '.join(report['false_negatives']) or '无'}",
        "按语言：",
        *[
            f"- {locale}: n={metrics['total']}, P={_display_metric(metrics['precision'])}, R={_display_metric(metrics['recall'])}, F1={_display_metric(metrics['f1'])}, Acc={metrics['accuracy']:.1%}"
            for locale, metrics in sorted(report["by_locale"].items())
        ],
        "按样本类型：",
        *[
            f"- {kind}: n={metrics['total']}, P={_display_metric(metrics['precision'])}, R={_display_metric(metrics['recall'])}, F1={_display_metric(metrics['f1'])}, Acc={metrics['accuracy']:.1%}"
            for kind, metrics in sorted(report["by_kind"].items())
        ],
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="评估客服多 Skill 入口与计划")
    parser.add_argument("--predictions", type=Path, help="模型计划预测 JSON 文件")
    args = parser.parse_args()
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    version, cases = load_orchestration_dataset()
    detector_report = evaluate_detector(cases)
    print(format_detector_report(version, detector_report))
    if args.predictions:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        plan_report = evaluate_plan_predictions(cases, predictions)
        print("\n多 Skill 计划评分")
        for key, value in plan_report.items():
            if key != "bad_cases":
                print(f"{key}: {value}")
        print("bad_cases:")
        print(json.dumps(plan_report["bad_cases"], ensure_ascii=False, indent=2))


def _binary_metrics(items: list[dict[str, Any]]) -> dict[str, float | int | None]:
    tp = sum(item["expected"] and item["actual"] for item in items)
    fp = sum(not item["expected"] and item["actual"] for item in items)
    fn = sum(item["expected"] and not item["actual"] for item in items)
    return {
        "total": len(items),
        "accuracy": sum(item["expected"] == item["actual"] for item in items) / len(items),
        "precision": _optional_ratio(tp, tp + fp),
        "recall": _optional_ratio(tp, tp + fn),
        "f1": _optional_f1(tp, fp, fn),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _optional_f1(tp: int, fp: int, fn: int) -> float | None:
    precision = _optional_ratio(tp, tp + fp)
    recall = _optional_ratio(tp, tp + fn)
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _display_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _task_counter(tool_names: list[str], task_arguments: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter(
        (tool_name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        for tool_name, arguments in zip(tool_names, task_arguments)
    )


def _normalize_query(query: str) -> str:
    return re.sub(r"[^\w]+", "", query.lower(), flags=re.UNICODE)


if __name__ == "__main__":
    main()
