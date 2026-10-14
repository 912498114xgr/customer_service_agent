"""把状态图、会话记忆和用户输入组合成可调用的客服服务。"""

import asyncio
import hashlib
import secrets
import time
from typing import Any
from uuid import uuid4

from common.llm import LLMClient
from common.observability import ObservedLLM, RequestMetrics, log_event, observe_request
from customer_care.graph import build_customer_care_graph
from customer_care.memory.base import MemoryStore
from customer_care.memory.in_memory import InMemoryMemoryStore
from customer_care.memory.policy import MemoryPolicy, compact_state
from customer_care.orchestration import MultiSkillExecutor
from customer_care.orchestration.models import TaskPlan
from customer_care.state import CareState
from customer_care.tools.registry import ToolRegistry

_CONFIRMATION_TTL_SECONDS = 300


class ConfirmationError(ValueError):
    """待确认操作不存在、失效或不能安全恢复。"""

    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


class CustomerCareService:
    """客服会话编排服务；每次请求都从记忆加载并保存完整状态。"""

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryStore | None = None,
        *,
        memory_policy: MemoryPolicy | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._memory_policy = memory_policy or MemoryPolicy()
        self._memory = memory or InMemoryMemoryStore(ttl_seconds=self._memory_policy.ttl_seconds)
        registry = registry or ToolRegistry()
        self._executor = MultiSkillExecutor(registry)
        self._confirmation_locks: dict[str, asyncio.Lock] = {}
        self._graph = build_customer_care_graph(ObservedLLM(llm), registry=registry, executor=self._executor)

    async def chat(self, user_id: str, content: str) -> tuple[CareState, dict[str, Any]]:
        """处理一条用户消息，返回最终状态和便于学习的决策轨迹。"""
        content = content.strip()
        if not content:
            raise ValueError("用户消息不能为空")

        metrics = RequestMetrics(request_id=uuid4().hex)
        metrics.node("load_memory")
        state = compact_state(await self._memory.load(user_id), self._memory_policy)
        if state.confirmation_status == "pending" and (
            state.confirmation_expires_at is None or state.confirmation_expires_at <= time.time()
        ):
            state.confirmation_status = "expired"
            state.confirmation_token_hash = None
            state.confirmation_expires_at = None
            state.awaiting_confirmation = False
            state.task_plan = None
            await self._memory.save(user_id, state)
        if state.confirmation_status in {"pending", "executing", "reconciliation_required"}:
            raise ConfirmationError("当前会话有待确认操作，请先确认或等待处理完成。")
        state = state.model_copy(update={
            "intent": None,
            "intent_confidence": 0.0,
            "intent_goals": [],
            "intent_conflict": False,
            "intent_needs_clarification": False,
            "route": None,
            "selected_tool": None,
            "tool_args": {},
            "tool_result": {},
            "rewritten_query": None,
            "rewrite_ambiguous": False,
            "rewrite_explanation": None,
            "task_plan": None,
            "task_results": [],
            "execution_status": None,
            "awaiting_confirmation": False,
            "replan_count": 0,
            "multi_skill_requested": False,
            "handoff_summary": None,
            "knowledge_context": "",
            "knowledge_sources": [],
            "answer": None,
            "emotion": None,
        })
        state.messages.append({"role": "user", "content": content})
        metrics.node("customer_care_graph")
        with observe_request(metrics):
            result = CareState.model_validate(await self._graph.ainvoke(state.model_dump()))
        result.messages.append({"role": "assistant", "content": result.answer or ""})
        result = compact_state(result, self._memory_policy)
        confirmation: dict[str, Any] | None = None
        if result.awaiting_confirmation:
            confirmation = self._issue_confirmation(result)
        metrics.node("save_memory")
        version = await self._memory.save(user_id, result)
        metrics.response_chars = len(result.answer or "")
        metrics_data = metrics.finish()
        log_event("customer_care_request", request_id=metrics_data["request_id"], user_id=user_id, route=result.route, **{key: value for key, value in metrics_data.items() if key != "request_id"})
        return result, {
            "request_id": metrics_data["request_id"],
            "metrics": metrics_data,
            "version": version,
            "language": result.language,
            "intent": result.intent,
            "intent_confidence": result.intent_confidence,
            "intent_goals": result.intent_goals,
            "intent_conflict": result.intent_conflict,
            "intent_needs_clarification": result.intent_needs_clarification,
            "entities": result.entities,
            "rewritten_query": result.rewritten_query,
            "rewrite_ambiguous": result.rewrite_ambiguous,
            "route": result.route,
            "selected_tool": result.selected_tool,
            "tool_args": result.tool_args,
            "tool_result": result.tool_result,
            "multi_skill_requested": result.multi_skill_requested,
            "task_plan": result.task_plan,
            "task_results": result.task_results,
            "execution_status": result.execution_status,
            "awaiting_confirmation": result.awaiting_confirmation,
            "replan_count": result.replan_count,
            "handoff_summary": result.handoff_summary,
            "context_summary": result.context_summary,
            "knowledge_sources": result.knowledge_sources,
            "confirmation": confirmation,
        }

    async def confirm_action(self, user_id: str, token: str) -> tuple[CareState, dict[str, Any]]:
        """校验一次性确认令牌，先持久化消费状态，再执行原计划。"""
        lock = self._confirmation_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            metrics = RequestMetrics(request_id=uuid4().hex)
            metrics.node("load_confirmation")
            state = compact_state(await self._memory.load(user_id), self._memory_policy)
            if state.confirmation_status != "pending" or not state.confirmation_token_hash:
                raise ConfirmationError("没有可确认的待处理操作。")
            supplied_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            if not state.task_plan:
                raise ConfirmationError("待确认计划缺失，已拒绝执行。")

            confirmation_id = state.confirmation_id
            if not confirmation_id:
                raise ConfirmationError("确认操作标识缺失，已拒绝执行。")
            claim_result, state = await self._memory.claim_confirmation(
                user_id, confirmation_id, supplied_hash, time.time(), "executing"
            )
            self._raise_claim_error(claim_result)
            plan = TaskPlan.model_validate(state.task_plan)

            metrics.node("execute_confirmed_plan")
            try:
                report = await self._executor.execute(plan, request_id=confirmation_id, confirmed=True)
            except Exception as exc:
                state.confirmation_status = "reconciliation_required"
                state.execution_status = "failed"
                state.awaiting_confirmation = False
                state.route = "human"
                state.answer = "The confirmed action could not be verified. It will not be retried automatically; support must reconcile its status."
                await self._memory.save(user_id, state)
                metrics_data = metrics.finish()
                log_event(
                    "customer_care_confirmation",
                    request_id=metrics_data["request_id"],
                    user_id=user_id,
                    status=state.confirmation_status,
                    execution_status=state.execution_status,
                    error_type=type(exc).__name__,
                )
                return state, {
                    "request_id": metrics_data["request_id"],
                    "metrics": metrics_data,
                    "route": state.route,
                    "confirmation_status": state.confirmation_status,
                    "execution_status": state.execution_status,
                    "task_results": state.task_results,
                }

            state.task_results = [item.model_dump(mode="json") for item in report.results]
            state.execution_status = report.status
            state.awaiting_confirmation = False
            state.confirmation_status = "completed" if report.status in {"succeeded", "cached"} else "reconciliation_required"
            state.route = "self_service" if state.confirmation_status == "completed" else "human"
            state.answer = (
                "Your confirmed request was completed by the local demo tool."
                if state.confirmation_status == "completed"
                else "The confirmed request could not be completed. A support specialist must review it."
            )
            version = await self._memory.save(user_id, state)
            metrics_data = metrics.finish()
            log_event("customer_care_confirmation", request_id=metrics_data["request_id"], user_id=user_id, status=state.confirmation_status, execution_status=report.status)
            return state, {
                "request_id": metrics_data["request_id"],
                "metrics": metrics_data,
                "version": version,
                "route": state.route,
                "confirmation_status": state.confirmation_status,
                "execution_status": state.execution_status,
                "task_results": state.task_results,
            }

    async def cancel_action(self, user_id: str, token: str) -> tuple[CareState, dict[str, Any]]:
        """消费确认令牌并拒绝执行，不调用任何工具。"""
        lock = self._confirmation_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            state = compact_state(await self._memory.load(user_id), self._memory_policy)
            if state.confirmation_status != "pending" or not state.confirmation_token_hash:
                raise ConfirmationError("没有可拒绝的待确认操作。")
            supplied_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            confirmation_id = state.confirmation_id
            if not confirmation_id:
                raise ConfirmationError("确认操作标识缺失，已拒绝操作。")
            claim_result, state = await self._memory.claim_confirmation(
                user_id, confirmation_id, supplied_hash, time.time(), "declined"
            )
            self._raise_claim_error(claim_result)
            version = await self._memory.save(user_id, state)
            return state, {
                "request_id": uuid4().hex,
                "version": version,
                "route": state.route,
                "confirmation_status": state.confirmation_status,
                "execution_status": state.execution_status,
                "task_results": state.task_results,
            }

    async def reset(self, user_id: str) -> None:
        """当前内存后端没有删除接口，重置通过替换会话状态实现。"""
        await self._memory.save(user_id, CareState(user_id=user_id))

    @staticmethod
    def _issue_confirmation(state: CareState) -> dict[str, Any]:
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + _CONFIRMATION_TTL_SECONDS
        state.confirmation_id = uuid4().hex
        state.confirmation_token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        state.confirmation_expires_at = expires_at
        state.confirmation_status = "pending"
        return {
            "token": token,
            "expires_at": expires_at,
            "ttl_seconds": _CONFIRMATION_TTL_SECONDS,
            "actions": [
                {"tool_name": task.get("tool_name"), "arguments": task.get("arguments", {})}
                for task in (state.task_plan or {}).get("tasks", [])
                if task.get("requires_confirmation")
            ],
        }

    @staticmethod
    def _raise_claim_error(result: str) -> None:
        if result == "expired":
            raise ConfirmationError("确认已过期，请重新提交请求。", 410)
        if result == "invalid_token":
            raise ConfirmationError("确认令牌无效。", 403)
        if result == "missing_plan":
            raise ConfirmationError("待确认计划缺失，已拒绝执行。")
        if result in {"not_pending", "invalid_confirmation", "no_state"}:
            raise ConfirmationError("没有可处理的待确认操作。")
        if result != "claimed":
            raise ConfirmationError("确认状态无法安全迁移。")

