import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

import pytest

from common.llm import ChatMessage, LLMClient
from customer_care.evaluation.live import _model_metadata, main, run_evaluation


class _EvaluationLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> str:
        self.calls += 1
        prompt = str(messages[-1]["content"])
        if "识别客服消息" in prompt:
            current_query = prompt.rsplit("当前消息：", 1)[-1]
            if "show my receipt" in current_query:
                return json.dumps({
                    "intent": "order",
                    "entities": {"order_id": "A"},
                    "emotion": "neutral",
                    "confidence": 0.99,
                    "goals": [
                        {"intent": "order", "objective": "check order status"},
                        {"intent": "order", "objective": "show receipt"},
                    ],
                    "conflict": False,
                    "needs_clarification": False,
                })
            return json.dumps({
                "intent": "order",
                "entities": {"order_id": "A"},
                "emotion": "angry" if "angry" in current_query.lower() else "neutral",
                "confidence": 0.99,
                "goals": [{
                    "intent": "refund" if "refund" in current_query.lower() else "order",
                    "objective": "check refund status" if "refund" in current_query.lower() else "check order status",
                }],
                "conflict": False,
                "needs_clarification": False,
            })
        if "最少的客服 Skill 任务" in prompt:
            return json.dumps({
                "tasks": [
                    {"task_id": "status", "intent": "get_order_status", "tool_name": "get_order_status", "capability": "get_order_status", "arguments": {"order_id": "A"}, "dependencies": []},
                    {"task_id": "receipt", "intent": "get_order_receipt", "tool_name": "get_order_receipt", "capability": "get_order_receipt", "arguments": {"order_id": "A"}, "dependencies": []},
                ],
                "rationale": "two independent read requests",
                "confidence": 0.99,
            })
        return json.dumps({"tool_name": "get_order_status", "arguments": {"order_id": "A"}})


def test_live_evaluation_runs_actual_nodes_without_executing_tools(tmp_path: Path) -> None:
    dataset_path = Path("customer_care/evaluation/orchestration_dataset.json")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected_ids = {"multi-order-status-receipt", "single-order-status", "single-angry-refund-and"}
    dataset["cases"] = [case for case in dataset["cases"] if case["id"] in selected_ids]
    dataset["expected_arguments"] = {
        case_id: arguments
        for case_id, arguments in dataset["expected_arguments"].items()
        if case_id in selected_ids
    }
    selected_path = tmp_path / "dataset.json"
    selected_path.write_text(json.dumps(dataset), encoding="utf-8")
    llm = _EvaluationLLM()

    report = asyncio.run(run_evaluation(llm, dataset_path=selected_path))

    assert report["execution"]["tools_executed"] is False
    assert report["execution"]["case_count"] == 3
    assert report["summary"]["llm_calls"] == llm.calls == 5
    assert report["intent_metrics"]["metrics"]["multi_skill_requested"]["f1"] == 1.0
    assert report["plan_metrics"]["exact_plan_accuracy"] < 1.0
    assert len(report["prompt_sources_sha256"]) == 64
    assert len(report["evaluation_sources_sha256"]) == 64
    assert report["cases"][0]["policy_route"] == "multi_skill"
    angry = next(case for case in report["cases"] if case["id"] == "single-angry-refund-and")
    assert angry["policy_route"] == "human"
    assert angry["plan"]["tool_names"] == []
    assert angry["metrics"]["llm_calls"] == 1
    assert all(not result["errors"] for result in report["cases"])


def test_live_evaluation_cli_requires_explicit_model_call_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["live", "--output", str(output), "--max-cases", "1"])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert not output.exists()


def test_live_evaluation_refuses_mock_mode_even_after_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    monkeypatch.setenv("USE_MOCK", "true")
    monkeypatch.setattr(sys, "argv", [
        "live", "--allow-model-calls", "--output", str(output), "--max-cases", "1"
    ])

    with pytest.raises(SystemExit, match="拒绝运行"):
        main()

    assert not output.exists()


def test_model_metadata_does_not_export_base_url_credentials_or_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://user:password@example.test/v1?token=secret")

    metadata = _model_metadata()

    assert metadata["provider_base_url"] == "https://example.test/v1"
    assert "password" not in json.dumps(metadata)
    assert "secret" not in json.dumps(metadata)


def test_live_evaluation_does_not_overwrite_existing_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("keep this", encoding="utf-8")
    monkeypatch.setenv("USE_MOCK", "true")
    monkeypatch.setattr(sys, "argv", [
        "live", "--allow-model-calls", "--output", str(output), "--max-cases", "1"
    ])

    with pytest.raises(SystemExit, match="拒绝覆盖"):
        main()

    assert output.read_text(encoding="utf-8") == "keep this"
