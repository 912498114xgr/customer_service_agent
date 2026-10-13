"""客服 Agent 的 LangGraph 状态图装配。"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from common.llm import LLMClient, create_llm_client
from customer_care.nodes.conversation import clarify, human_handoff, reply, route_decision, self_service
from customer_care.nodes.intent import recognize_intent
from customer_care.nodes.language import detect_language
from customer_care.nodes.rewrite import rewrite_query
from customer_care.knowledge import KnowledgeRetriever
from customer_care.orchestration import MultiSkillExecutor, MultiSkillPlanner
from customer_care.orchestration.detection import is_multi_task_request
from customer_care.state import CareState
from customer_care.tools.registry import ToolRegistry
from customer_care.tools.router import ToolRouter


def build_customer_care_graph(
    llm: LLMClient | None = None,
    *,
    registry: ToolRegistry | None = None,
    router: ToolRouter | None = None,
    knowledge: KnowledgeRetriever | None = None,
    planner: MultiSkillPlanner | None = None,
    executor: MultiSkillExecutor | None = None,
) -> Any:
    """构建包含澄清、自助和人工分支的客服状态图。"""
    llm = llm or create_llm_client()
    registry = registry or ToolRegistry()
    router = router or ToolRouter(llm)
    knowledge = knowledge or KnowledgeRetriever.load_default()
    planner = planner or MultiSkillPlanner(llm)
    executor = executor or MultiSkillExecutor(registry)
    graph = StateGraph(CareState)

    graph.add_node("detect_language", detect_language)

    async def intent_node(raw_state: dict[str, Any]) -> dict[str, object]:
        return await recognize_intent(CareState.model_validate(raw_state), llm)

    async def rewrite_node(raw_state: dict[str, Any]) -> dict[str, object]:
        return await rewrite_query(CareState.model_validate(raw_state), llm)

    def routing_node(raw_state: dict[str, Any]) -> dict[str, str]:
        return route_decision(CareState.model_validate(raw_state))

    def multi_task_detector(raw_state: dict[str, Any]) -> dict[str, bool]:
        state = CareState.model_validate(raw_state)
        query = (state.rewritten_query or "").lower()
        if not query:
            query = " ".join(str(message.get("content", "")) for message in state.messages[-1:]).lower()
        return {"multi_skill_requested": is_multi_task_request(query)}

    async def plan_tasks_node(raw_state: dict[str, Any]) -> dict[str, object]:
        state = CareState.model_validate(raw_state)
        decision = await planner.plan(state)
        if decision.plan is None or decision.needs_clarification:
            return {
                "route": "clarify",
                "answer": decision.reason or "I need a little more detail before I can handle these requests safely.",
            }
        return {
            "task_plan": decision.plan.model_dump(mode="json"),
            "route": "execute",
        }

    async def execute_plan_node(raw_state: dict[str, Any]) -> dict[str, object]:
        state = CareState.model_validate(raw_state)
        if not state.task_plan:
            return {"execution_status": "failed", "route": "human"}

        async def replan_task(task: Any, error: str):
            decision = await planner.replan(state, task, error)
            if decision.needs_clarification:
                return None
            return decision.plan

        report = await executor.execute(
            planner_task_plan(state.task_plan),
            request_id=f"graph-{state.user_id}-{len(state.messages)}",
            confirmed=False,
            max_replans=1,
            replan=replan_task,
        )
        update: dict[str, object] = {
            "task_results": [item.model_dump(mode="json") for item in report.results],
            "execution_status": report.status,
            "awaiting_confirmation": report.status == "awaiting_confirmation",
            "replan_count": report.replan_count,
        }
        if report.status in {"succeeded", "partial", "cached"}:
            update["route"] = "multi_skill"
        if report.status == "awaiting_confirmation":
            update["route"] = "human"
            update["handoff_summary"] = "多 Skill 计划包含需要用户确认的副作用操作。"
        elif report.status in {"failed", "partial"}:
            update["route"] = "human"
            update["handoff_summary"] = "多 Skill 执行未能完整满足请求，需人工核对已完成与失败的任务。"
        return update

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
    graph.add_node("detect_multi_task", multi_task_detector)
    graph.add_node("plan_tasks", plan_tasks_node)
    graph.add_node("execute_plan", execute_plan_node)

    graph.add_edge(START, "detect_language")
    graph.add_edge("detect_language", "recognize_intent")
    graph.add_edge("recognize_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "detect_multi_task")
    graph.add_edge("detect_multi_task", "route_decision")
    graph.add_conditional_edges(
        "route_decision",
        lambda state: state.route,
        {"clarify": "clarify", "human": "human_handoff", "self_service": "skill_router", "multi_skill": "plan_tasks"},
    )
    graph.add_conditional_edges(
        "plan_tasks",
        lambda state: state.route or "execute",
        {"clarify": "clarify", "execute": "execute_plan"},
    )
    graph.add_edge("execute_plan", "retrieve_knowledge")
    graph.add_edge("clarify", "retrieve_knowledge")
    graph.add_edge("human_handoff", "retrieve_knowledge")
    graph.add_edge("skill_router", "self_service")
    graph.add_edge("self_service", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "reply")
    graph.add_edge("reply", END)
    return graph.compile()


def planner_task_plan(raw: dict[str, Any]):
    """将状态中的 JSON 计划恢复为受校验的领域对象。"""
    from customer_care.orchestration.models import TaskPlan

    return TaskPlan.model_validate(raw)

