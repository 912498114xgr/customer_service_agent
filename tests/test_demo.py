import asyncio
import time

import pytest

from customer_care.demo import create_scenario_llm, demo_scenarios
from customer_care.service import ConfirmationError, CustomerCareService


def test_demo_scenarios_are_discoverable_and_disclosed_as_scripted() -> None:
    scenarios = demo_scenarios()

    assert {scenario["id"] for scenario in scenarios} == {"multi-skill", "conflict", "invalid-intent", "side-effect", "multi-confirmation"}
    assert all(scenario["query"] and scenario["description"] for scenario in scenarios)


def test_multi_skill_demo_runs_main_graph_and_local_tools() -> None:
    service = CustomerCareService(create_scenario_llm("multi-skill"))

    state, trace = asyncio.run(service.chat("demo", "Check order A-17 status and show my receipt"))

    assert state.route == "multi_skill"
    assert state.execution_status == "succeeded"
    assert {result["tool_name"] for result in state.task_results} == {"get_order_status", "get_order_receipt"}
    assert trace["intent_goals"]
    assert trace["metrics"]["llm_calls"] == 3
    assert {span["node"] for span in trace["metrics"]["llm_spans"]} == {"recognize_intent", "plan_tasks", "reply"}


def test_conflict_demo_stops_before_planner_or_tools() -> None:
    service = CustomerCareService(create_scenario_llm("conflict"))

    state, trace = asyncio.run(service.chat("demo", "Cancel order A-17, but keep order A-17 active"))

    assert state.route == "clarify"
    assert state.intent_conflict is True
    assert state.task_plan is None
    assert state.task_results == []
    assert state.selected_tool is None
    assert trace["metrics"]["llm_calls"] == 2
    assert [span["node"] for span in trace["metrics"]["llm_spans"]] == ["recognize_intent", "reply"]


def test_invalid_intent_demo_fails_closed() -> None:
    service = CustomerCareService(create_scenario_llm("invalid-intent"))

    state, _ = asyncio.run(service.chat("demo", "Where is order A-17?"))

    assert state.route == "clarify"
    assert state.selected_tool is None
    assert state.intent_confidence == 0.0


def test_single_skill_side_effect_waits_for_confirmation_and_resumes_once() -> None:
    service = CustomerCareService(create_scenario_llm("side-effect"))

    state, trace = asyncio.run(service.chat("demo", "Change my notification preference to weekly"))

    assert state.route == "awaiting_confirmation"
    assert state.awaiting_confirmation is True
    assert state.execution_status == "awaiting_confirmation"
    assert state.task_results == []
    assert state.confirmation_status == "pending"
    assert state.confirmation_token_hash != trace["confirmation"]["token"]
    token = trace["confirmation"]["token"]

    completed, confirmation_trace = asyncio.run(service.confirm_action("demo", token))

    assert completed.confirmation_status == "completed"
    assert completed.awaiting_confirmation is False
    assert completed.execution_status == "succeeded"
    assert completed.task_results[0]["tool_name"] == "update_contact_preference"
    assert confirmation_trace["confirmation_status"] == "completed"
    with pytest.raises(ConfirmationError, match="没有可确认"):
        asyncio.run(service.confirm_action("demo", token))


def test_wrong_confirmation_token_does_not_consume_pending_action() -> None:
    service = CustomerCareService(create_scenario_llm("side-effect"))
    state, _ = asyncio.run(service.chat("demo", "Change my notification preference to weekly"))

    with pytest.raises(ConfirmationError, match="令牌无效"):
        asyncio.run(service.confirm_action("demo", "x" * 48))

    persisted = asyncio.run(service._memory.load("demo"))
    assert persisted.confirmation_status == "pending"
    assert persisted.confirmation_token_hash == state.confirmation_token_hash


def test_expired_confirmation_cannot_execute_and_is_cleared() -> None:
    service = CustomerCareService(create_scenario_llm("side-effect"))
    state, trace = asyncio.run(service.chat("demo", "Change my notification preference to weekly"))
    state.confirmation_expires_at = time.time() - 1
    asyncio.run(service._memory.save("demo", state))

    with pytest.raises(ConfirmationError, match="已过期") as error:
        asyncio.run(service.confirm_action("demo", trace["confirmation"]["token"]))

    assert error.value.status_code == 410
    expired = asyncio.run(service._memory.load("demo"))
    assert expired.confirmation_status == "expired"
    assert expired.confirmation_token_hash is None
    assert expired.task_results == []


def test_declining_confirmation_consumes_token_without_running_tool() -> None:
    service = CustomerCareService(create_scenario_llm("multi-confirmation"))
    pending, trace = asyncio.run(service.chat("demo", "Request a review for refund R-17 and explain the refund policy"))
    token = trace["confirmation"]["token"]

    declined, decline_trace = asyncio.run(service.cancel_action("demo", token))

    assert declined.confirmation_status == "declined"
    assert declined.execution_status == "cancelled"
    assert declined.awaiting_confirmation is False
    assert {result["status"] for result in declined.task_results} == {"cancelled"}
    assert all("用户拒绝执行" in result["error"] for result in declined.task_results)
    assert decline_trace["confirmation_status"] == "declined"
    assert service._executor.cache_size == 0
    with pytest.raises(ConfirmationError, match="没有可拒绝"):
        asyncio.run(service.cancel_action("demo", token))


def test_concurrent_confirmation_attempts_execute_at_most_once_in_one_process() -> None:
    service = CustomerCareService(create_scenario_llm("side-effect"))
    _, trace = asyncio.run(service.chat("demo", "Change my notification preference to weekly"))
    token = trace["confirmation"]["token"]

    async def confirm_twice():
        return await asyncio.gather(
            service.confirm_action("demo", token),
            service.confirm_action("demo", token),
            return_exceptions=True,
        )

    results = asyncio.run(confirm_twice())

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, ConfirmationError) for result in results) == 1
    persisted = asyncio.run(service._memory.load("demo"))
    assert persisted.confirmation_status == "completed"
    assert len(persisted.task_results) == 1


def test_uncertain_execution_requires_reconciliation_and_consumes_token(monkeypatch: pytest.MonkeyPatch) -> None:
    service = CustomerCareService(create_scenario_llm("side-effect"))
    _, trace = asyncio.run(service.chat("demo", "Change my notification preference to weekly"))
    token = trace["confirmation"]["token"]

    async def fail_after_confirmation(*args, **kwargs):
        raise TimeoutError("下游响应超时，无法确定副作用是否已发生")

    monkeypatch.setattr(service._executor, "execute", fail_after_confirmation)
    state, result_trace = asyncio.run(service.confirm_action("demo", token))

    assert state.confirmation_status == "reconciliation_required"
    assert state.execution_status == "failed"
    assert state.route == "human"
    assert result_trace["confirmation_status"] == "reconciliation_required"
    assert "not be retried" in state.answer
    with pytest.raises(ConfirmationError, match="没有可确认"):
        asyncio.run(service.confirm_action("demo", token))


def test_new_turn_does_not_replay_previous_confirmation_result() -> None:
    service = CustomerCareService(create_scenario_llm("side-effect"))
    _, pending_trace = asyncio.run(service.chat("demo", "Change my notification preference to weekly"))
    asyncio.run(service.confirm_action("demo", pending_trace["confirmation"]["token"]))

    state, trace = asyncio.run(service.chat("demo", "Show my account details"))

    assert trace["task_results"] == []
    assert trace["execution_status"] is None
    assert trace["route"] == "clarify"
    assert state.confirmation_status == "completed"


def test_chat_cannot_replace_an_unconfirmed_action() -> None:
    service = CustomerCareService(create_scenario_llm("side-effect"))
    asyncio.run(service.chat("demo", "Change my notification preference to weekly"))

    with pytest.raises(ConfirmationError, match="待确认操作"):
        asyncio.run(service.chat("demo", "Show my account details"))


def test_multi_skill_plan_pauses_as_a_whole_then_executes_original_plan() -> None:
    service = CustomerCareService(create_scenario_llm("multi-confirmation"))

    pending, trace = asyncio.run(service.chat("demo", "Request a review for refund R-17 and explain the refund policy"))

    assert pending.route == "awaiting_confirmation"
    assert pending.task_results
    assert all(result["status"] == "awaiting_confirmation" for result in pending.task_results)
    token = trace["confirmation"]["token"]

    completed, _ = asyncio.run(service.confirm_action("demo", token))

    assert completed.execution_status == "succeeded"
    assert {result["task_id"] for result in completed.task_results} == {"review", "policy"}
    assert completed.confirmation_status == "completed"

