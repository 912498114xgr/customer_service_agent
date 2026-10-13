import json
from pathlib import Path

import pytest

from customer_care.evaluation import orchestration as orchestration_evaluation
from customer_care.evaluation.orchestration import (
    evaluate_detector,
    evaluate_plan_predictions,
    load_orchestration_dataset,
)


def test_multi_skill_dataset_is_versioned_and_exposes_heuristic_failures() -> None:
    version, cases = load_orchestration_dataset()

    report = evaluate_detector(cases)

    assert version == "1.0.0"
    assert report["total"] == 25
    assert report["accuracy"] < 0.7
    assert "multi-spanish-y" in report["false_negatives"]
    assert "single-angry-refund-and" in report["false_positives"]
    assert {"en", "zh", "es"} == set(report["by_locale"])


def test_plan_scorer_reports_exact_plan_clarification_and_conflict_metrics() -> None:
    _, cases = load_orchestration_dataset()
    predictions = [
        {
            "id": case.case_id,
            "tool_names": list(case.expected_tools),
            "task_arguments": list(case.expected_task_arguments),
            "dependency_edges": [list(edge) for edge in case.expected_dependencies],
            "needs_clarification": case.expected_clarification,
        }
        for case in cases
    ]

    report = evaluate_plan_predictions(cases, predictions)

    assert report["plan_tool_f1"] == 1.0
    assert report["dependency_f1"] == 1.0
    assert report["exact_plan_accuracy"] == 1.0
    assert report["clarification_accuracy"] == 1.0
    assert report["conflict_recall"] == 1.0
    assert report["unsafe_conflict_execution_count"] == 0


def test_plan_scorer_flags_conflict_execution_and_missing_skill() -> None:
    _, cases = load_orchestration_dataset()
    predictions = []
    for case in cases:
        prediction = {
            "id": case.case_id,
            "tool_names": list(case.expected_tools),
            "task_arguments": list(case.expected_task_arguments),
            "dependency_edges": [list(edge) for edge in case.expected_dependencies],
            "needs_clarification": case.expected_clarification,
        }
        if case.case_id == "conflict-cancel-and-keep":
            prediction["tool_names"] = ["check_cancel_eligibility"]
            prediction["task_arguments"] = [{"order_id": "A"}]
            prediction["needs_clarification"] = False
        if case.case_id == "multi-order-status-receipt":
            prediction["tool_names"] = ["get_order_status"]
            prediction["task_arguments"] = [{"order_id": "A"}]
        if case.case_id == "multi-plus-coupon":
            prediction["task_arguments"] = [{"coupon_id": "WRONG"}, {"order_id": "A"}]
        if case.case_id == "single-plus-negation":
            prediction["tool_names"] = ["update_contact_preference"]
            prediction["task_arguments"] = [{"preference": "weekly"}]
            prediction["needs_clarification"] = False
            prediction["unconfirmed_side_effect_tools"] = ["update_contact_preference"]
        predictions.append(prediction)

    report = evaluate_plan_predictions(cases, predictions)

    assert report["exact_plan_accuracy"] < 1.0
    assert report["unsafe_conflict_execution_count"] == 1
    assert report["conflict_recall"] == 0.5
    assert report["plan_tool_recall"] < 1.0
    assert report["plan_task_recall"] < report["plan_tool_recall"]
    assert report["unconfirmed_side_effect_count"] == 1
    assert {"conflict-cancel-and-keep", "multi-order-status-receipt", "multi-plus-coupon", "single-plus-negation"} <= {item["id"] for item in report["bad_cases"]}


def test_plan_scorer_rejects_unknown_or_duplicate_prediction_ids() -> None:
    _, cases = load_orchestration_dataset()

    with pytest.raises(ValueError, match="未知 case id"):
        evaluate_plan_predictions(cases, [{"id": "not-in-dataset"}])
    with pytest.raises(ValueError, match="重复"):
        evaluate_plan_predictions(cases, [{"id": cases[0].case_id}, {"id": cases[0].case_id}])


def test_dataset_loader_rejects_duplicate_cases_and_misaligned_argument_labels(tmp_path: Path) -> None:
    dataset_path = Path(orchestration_evaluation.__file__).with_name("orchestration_dataset.json")
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    duplicate_path = tmp_path / "duplicate.json"
    raw["cases"].append(raw["cases"][0].copy())
    duplicate_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="case id 必须唯一"):
        load_orchestration_dataset(duplicate_path)

    raw["cases"].pop()
    raw["expected_arguments"]["multi-order-status-receipt"].pop()
    misaligned_path = tmp_path / "misaligned.json"
    misaligned_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="每个期望 Skill"):
        load_orchestration_dataset(misaligned_path)
