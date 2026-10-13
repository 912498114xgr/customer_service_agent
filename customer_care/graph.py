"""客服 Agent 的 LangGraph 状态图装配。"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from common.llm import LLMClient, create_llm_client
from customer_care.nodes.conversation import clarify, human_handoff, reply, route_decision, self_service
from customer_care.nodes.intent import recognize_intent
from customer_care.nodes.language import detect_language
from customer_care.nodes.rewrite import rewrite_query
from customer_care.knowledge import KnowledgeRetriever
from customer_care.state import CareState
from customer_care.tools.registry import ToolRegistry
from customer_care.tools.router import ToolRouter


def build_customer_care_graph(
    llm: LLMClient | None = None,
    *,
    registry: ToolRegistry | None = None,
    router: ToolRouter | None = None,
    knowledge: KnowledgeRetriever | None = None,
) -> Any:
    """构建包含澄清、自助和人工分支的客服状态图。"""
    llm = llm or create_llm_client()
    registry = registry or ToolRegistry()
    router = router or ToolRouter(llm)
    knowledge = knowledge or KnowledgeRetriever.load_default()
    graph = StateGraph(CareState)

    graph.add_node("detect_language", detect_language)

    async def intent_node(raw_state: dict[str, Any]) -> dict[str, object]:
        return await recognize_intent(CareState.model_validate(raw_state), llm)

    async def rewrite_node(raw_state: dict[str, Any]) -> dict[str, object]:
        return await rewrite_query(CareState.model_validate(raw_state), llm)

    def routing_node(raw_state: dict[str, Any]) -> dict[str, str]:
        return route_decision(CareState.model_validate(raw_state))

    def clarify_node(raw_state: dict[str, Any]) -> dict[str, str]:
        return clarify(CareState.model_validate(raw_state))

    def handoff_node(raw_state: dict[str, Any]) -> dict[str, str]:
        return human_handoff(CareState.model_validate(raw_state))

    async def tool_router_node(raw_state: dict[str, Any]) -> dict[str, object]:
        return await router.select(CareState.model_validate(raw_state))

    async def self_service_node(raw_state: dict[str, Any]) -> dict[str, object]:
        return await self_service(CareState.model_validate(raw_state), registry)

    async def reply_node(raw_state: dict[str, Any]) -> dict[str, str]:
        return await reply(CareState.model_validate(raw_state), llm)

    def knowledge_node(raw_state: dict[str, Any]) -> dict[str, object]:
        state = CareState.model_validate(raw_state)
        query = state.rewritten_query or " ".join(
            str(message.get("content", "")) for message in state.messages[-2:] if message.get("role") == "user"
        )
        hits = knowledge.search(query, language=state.language, topic=state.intent, top_k=3)
        return {
            "knowledge_context": knowledge.context(hits),
            "knowledge_sources": [hit.as_dict() for hit in hits],
        }

    graph.add_node("recognize_intent", intent_node)
    graph.add_node("rewrite_query", rewrite_node)
    graph.add_node("route_decision", routing_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("human_handoff", handoff_node)
    graph.add_node("skill_router", tool_router_node)
    graph.add_node("self_service", self_service_node)
    graph.add_node("reply", reply_node)
    graph.add_node("retrieve_knowledge", knowledge_node)

    graph.add_edge(START, "detect_language")
    graph.add_edge("detect_language", "recognize_intent")
    graph.add_edge("recognize_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "route_decision")
    graph.add_conditional_edges(
        "route_decision",
        lambda state: state.route,
        {"clarify": "clarify", "human": "human_handoff", "self_service": "skill_router"},
    )
    graph.add_edge("clarify", "retrieve_knowledge")
    graph.add_edge("human_handoff", "retrieve_knowledge")
    graph.add_edge("skill_router", "self_service")
    graph.add_edge("self_service", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "reply")
    graph.add_edge("reply", END)
    return graph.compile()

