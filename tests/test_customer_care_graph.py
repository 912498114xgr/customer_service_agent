import asyncio

from common.mock_llm import MockLLM
from customer_care.graph import build_customer_care_graph
from customer_care.state import CareState


def test_refund_status_happy_path_runs_through_self_service() -> None:
    llm = MockLLM(
        [
            '{"intent":"refund","entities":{"order_id":"R-17"},"emotion":"neutral","confidence":0.96,"goals":[{"intent":"refund","objective":"查询退款状态"}],"conflict":false,"needs_clarification":false}',
            '{"tool_name":"get_refund_status","arguments":{"order_id":"R-17"}}',
            "Your refund for order R-17 is processing and should arrive in about 3 days.",
        ]
    )
    graph = build_customer_care_graph(llm)
    initial = CareState(
        user_id="u1",
        messages=[{"role": "user", "content": "Where is my refund for order R-17?"}],
    )

    result = asyncio.run(graph.ainvoke(initial.model_dump()))
    state = CareState.model_validate(result)

    assert state.language == "en"
    assert state.route == "self_service"
    assert state.selected_tool == "get_refund_status"
    assert state.tool_result["status"] == "processing"
    assert "3 days" in (state.answer or "")
    assert state.knowledge_sources
    assert state.knowledge_sources[0]["id"] in {"refund-policy", "refund-timeline"}


def test_angry_user_is_handed_to_human_without_running_tools() -> None:
    llm = MockLLM(
        [
            '{"intent":"refund","entities":{"order_id":"R-17"},"emotion":"angry","confidence":0.96,"goals":[{"intent":"refund","objective":"查询退款状态"}],"conflict":false,"needs_clarification":false}',
            "I’m connecting you with a support specialist.",
        ]
    )
    graph = build_customer_care_graph(llm)
    initial = CareState(
        user_id="u1",
        messages=[{"role": "user", "content": "I am angry about my refund for order R-17."}],
    )

    result = CareState.model_validate(asyncio.run(graph.ainvoke(initial.model_dump())))

    assert result.route == "human"
    assert result.handoff_summary is not None
    assert result.selected_tool is None
    assert "support specialist" in (result.answer or "")


def test_ambiguous_reference_routes_to_clarification() -> None:
    llm = MockLLM(
        [
            '{"intent":"other","entities":{},"emotion":"neutral","confidence":0.30,"goals":[{"intent":"other","objective":"澄清用户请求"}],"conflict":false,"needs_clarification":true}',
            '{"rewritten_query":"What about that?","ambiguous":true,"explanation":"No known referent."}',
            "Which order are you asking about? Please share the order ID.",
        ]
    )
    graph = build_customer_care_graph(llm)
    initial = CareState(user_id="u1", messages=[{"role": "user", "content": "What about that?"}])

    result = CareState.model_validate(asyncio.run(graph.ainvoke(initial.model_dump())))

    assert result.route == "clarify"
    assert result.rewrite_ambiguous is True
    assert result.selected_tool is None
    assert "order ID" in (result.answer or "")


def test_conflicting_request_is_clarified_without_planning_or_tool_call() -> None:
    llm = MockLLM([
        '{"intent":"account","entities":{},"emotion":"neutral","confidence":0.96,"goals":[{"intent":"account","objective":"turn on weekly notifications"},{"intent":"account","objective":"leave current notification setting"}],"conflict":true,"needs_clarification":true}',
        "I found conflicting instructions. Should I proceed with the change, or leave the current setting as it is?",
    ])
    state = CareState(user_id="u1", messages=[{"role": "user", "content": "Set notifications to weekly, but leave current notification setting"}])

    result = CareState.model_validate(asyncio.run(build_customer_care_graph(llm).ainvoke(state.model_dump())))

    assert result.route == "clarify"
    assert result.intent_conflict is True
    assert result.multi_skill_requested is False
    assert result.task_plan is None
    assert result.selected_tool is None
    assert len(llm.calls) == 2
    assert "conflicting instructions" in (result.answer or "")

