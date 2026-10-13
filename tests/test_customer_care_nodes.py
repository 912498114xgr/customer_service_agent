import asyncio

from common.mock_llm import MockLLM
from customer_care.nodes.intent import recognize_intent
from customer_care.nodes.language import detect_language
from customer_care.nodes.rewrite import rewrite_query
from customer_care.state import CareState
from customer_care.tools.registry import ToolCatalog
from customer_care.tools.router import ToolRouter


def test_intent_node_validates_multilingual_structured_result() -> None:
    llm = MockLLM(
        ['{"intent":"refund","entities":{"order_id":"R-17"},"emotion":"neutral","confidence":0.94}']
    )
    state = CareState(user_id="u1", messages=[{"role": "user", "content": "¿Dónde está mi reembolso del pedido R-17?"}])

    update = asyncio.run(recognize_intent(state, llm))

    assert update["intent"] == "refund"
    assert update["entities"] == {"order_id": "R-17"}
    assert update["intent_confidence"] == 0.94


def test_invalid_intent_output_falls_back_to_clarification_safe_values() -> None:
    llm = MockLLM(["I think this is a refund request."])
    state = CareState(user_id="u1", messages=[{"role": "user", "content": "Help me"}])

    update = asyncio.run(recognize_intent(state, llm))

    assert update["intent"] == "other"
    assert update["intent_confidence"] == 0.0


def test_language_detection_distinguishes_spanish_and_portuguese_order_phrases() -> None:
    spanish = CareState(user_id="u1", messages=[{"role": "user", "content": "¿Dónde está mi pedido?"}])
    portuguese = CareState(user_id="u2", messages=[{"role": "user", "content": "Onde está meu pedido?"}])

    assert detect_language(spanish)["language"] == "es"
    assert detect_language(portuguese)["language"] == "pt"


def test_query_rewrite_resolves_only_with_an_explicit_entity() -> None:
    llm = MockLLM(
        ['{"rewritten_query":"Can I get a refund for order A123?","ambiguous":false,"explanation":"it refers to the known order."}']
    )
    state = CareState(
        user_id="u1",
        messages=[{"role": "user", "content": "Can I get a refund for it?"}],
        entities={"order_id": "A123"},
    )

    update = asyncio.run(rewrite_query(state, llm))

    assert update["rewritten_query"] == "Can I get a refund for order A123?"
    assert update["rewrite_ambiguous"] is False


def test_query_rewrite_preserves_ambiguous_question() -> None:
    llm = MockLLM(
        ['{"rewritten_query":"What about that?","ambiguous":true,"explanation":"No known referent."}']
    )
    query = "What about that?"
    state = CareState(user_id="u1", messages=[{"role": "user", "content": query}])

    update = asyncio.run(rewrite_query(state, llm))

    assert update["rewritten_query"] == query
    assert update["rewrite_ambiguous"] is True


def test_tool_catalog_has_thirty_standardized_tools() -> None:
    tools = ToolCatalog.load_default().tools

    assert len(tools) == 30
    assert {tool.category for tool in tools} == {"orders", "delivery", "refunds", "coupons", "account"}
    assert all(tool.trigger_keywords and tool.pos_examples and tool.neg_examples for tool in tools)


def test_two_level_router_bounds_candidates_and_rejects_non_candidate_choice() -> None:
    llm = MockLLM(['{"tool_name":"not_a_candidate","arguments":{"order_id":"R-17"}}'])
    router = ToolRouter(llm, max_candidates=10)
    state = CareState(
        user_id="u1",
        intent="refund",
        rewritten_query="My refund for order R-17 is still pending",
        entities={"order_id": "R-17"},
    )
    candidates = router._coarse_filter(state.rewritten_query or "", state.intent)

    update = asyncio.run(router.select(state))

    assert len(candidates) <= 10
    assert "get_refund_status" in {tool.name for tool in candidates}
    assert update["selected_tool"] in {tool.name for tool in candidates}
    assert update["selected_tool"] != "not_a_candidate"
    assert update["tool_args"] == {"order_id": "R-17"}

