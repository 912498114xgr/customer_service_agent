import asyncio

from common.mock_llm import MockLLM
from customer_care.graph import build_customer_care_graph
from customer_care.state import CareState


def test_refund_status_happy_path_runs_through_self_service() -> None:
    llm = MockLLM(
        [
            '{"intent":"refund","entities":{"order_id":"R-17"},"emotion":"neutral","confidence":0.96}',
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


def test_angry_user_is_handed_to_human_without_running_tools() -> None:
    llm = MockLLM(
        [
            '{"intent":"refund","entities":{"order_id":"R-17"},"emotion":"angry","confidence":0.96}',
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
            '{"intent":"other","entities":{},"emotion":"neutral","confidence":0.30}',
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

