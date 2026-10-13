"""把状态图、会话记忆和用户输入组合成可调用的客服服务。"""

from typing import Any
from uuid import uuid4

from common.llm import LLMClient
from customer_care.graph import build_customer_care_graph
from customer_care.memory.base import MemoryStore
from customer_care.memory.in_memory import InMemoryMemoryStore
from customer_care.memory.policy import MemoryPolicy, compact_state
from customer_care.state import CareState
from common.observability import RequestMetrics, log_event


class CustomerCareService:
    """客服会话编排服务；每次请求都从记忆加载并保存完整状态。"""

    def __init__(self, llm: LLMClient, memory: MemoryStore | None = None, *, memory_policy: MemoryPolicy | None = None) -> None:
        self._memory_policy = memory_policy or MemoryPolicy()
        self._memory = memory or InMemoryMemoryStore(ttl_seconds=self._memory_policy.ttl_seconds)
        self._graph = build_customer_care_graph(llm)

    async def chat(self, user_id: str, content: str) -> tuple[CareState, dict[str, Any]]:
        """处理一条用户消息，返回最终状态和便于学习的决策轨迹。"""
        content = content.strip()
        if not content:
            raise ValueError("用户消息不能为空")

        metrics = RequestMetrics(request_id=uuid4().hex)
        metrics.node("load_memory")
        state = compact_state(await self._memory.load(user_id), self._memory_policy)
        state.messages.append({"role": "user", "content": content})
        metrics.node("customer_care_graph")
        result = CareState.model_validate(await self._graph.ainvoke(state.model_dump()))
        result.messages.append({"role": "assistant", "content": result.answer or ""})
        result = compact_state(result, self._memory_policy)
        metrics.node("save_memory")
        version = await self._memory.save(user_id, result)
        metrics.output_chars = len(result.answer or "")
        metrics_data = metrics.finish()
        log_event("customer_care_request", request_id=metrics_data["request_id"], user_id=user_id, route=result.route, **{key: value for key, value in metrics_data.items() if key != "request_id"})
        return result, {
            "request_id": metrics_data["request_id"],
            "metrics": metrics_data,
            "version": version,
            "language": result.language,
            "intent": result.intent,
            "intent_confidence": result.intent_confidence,
            "entities": result.entities,
            "rewritten_query": result.rewritten_query,
            "rewrite_ambiguous": result.rewrite_ambiguous,
            "route": result.route,
            "selected_tool": result.selected_tool,
            "tool_args": result.tool_args,
            "tool_result": result.tool_result,
            "handoff_summary": result.handoff_summary,
            "context_summary": result.context_summary,
            "knowledge_sources": result.knowledge_sources,
        }

    async def reset(self, user_id: str) -> None:
        """当前内存后端没有删除接口，重置通过替换会话状态实现。"""
        await self._memory.save(user_id, CareState(user_id=user_id))

