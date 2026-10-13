import asyncio

from customer_care.demo import create_scenario_llm, demo_scenarios
from customer_care.service import CustomerCareService


def test_demo_scenarios_are_discoverable_and_disclosed_as_scripted() -> None:
    scenarios = demo_scenarios()

    assert {scenario["id"] for scenario in scenarios} == {"multi-skill", "conflict", "invalid-intent"}
    assert all(scenario["query"] and scenario["description"] for scenario in scenarios)


def test_multi_skill_demo_runs_main_graph_and_local_tools() -> None:
    service = CustomerCareService(create_scenario_llm("multi-skill"))

    state, trace = asyncio.run(service.chat("demo", "Check order A-17 status and show my receipt"))

    assert state.route == "multi_skill"
    assert state.execution_status == "succeeded"
    assert {result["tool_name"] for result in state.task_results} == {"get_order_status", "get_order_receipt"}
    assert trace["intent_goals"]


def test_conflict_demo_stops_before_planner_or_tools() -> None:
    service = CustomerCareService(create_scenario_llm("conflict"))

    state, _ = asyncio.run(service.chat("demo", "Cancel order A-17, but keep it active"))

    assert state.route == "clarify"
    assert state.intent_conflict is True
    assert state.task_plan is None
    assert state.task_results == []
    assert state.selected_tool is None


def test_invalid_intent_demo_fails_closed() -> None:
    service = CustomerCareService(create_scenario_llm("invalid-intent"))

    state, _ = asyncio.run(service.chat("demo", "Where is order A-17?"))

    assert state.route == "clarify"
    assert state.selected_tool is None
    assert state.intent_confidence == 0.0

