"""多 Skill 任务规划、治理校验和有限重规划。"""

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.llm import LLMClient
from common.structured_output import parse_model_json
from customer_care.state import CareState
from customer_care.tools.registry import ToolCatalog, ToolDefinition
from customer_care.orchestration.models import SkillTask, TaskPlan

_LOGGER = logging.getLogger(__name__)

_INTENT_CATEGORY = {
    "refund": "refunds",
    "track_order": "delivery",
    "order": "orders",
    "coupon": "coupons",
    "account": "account",
}
_SIDE_EFFECT_TOOLS = {"create_refund_review_ticket", "update_contact_preference"}
_HIGH_RISK_TOOLS = {"create_refund_review_ticket", "update_contact_preference", "check_cancel_eligibility"}


class PlanningDecision(BaseModel):
    """规划结果；失败时明确要求澄清而不是猜测执行。"""

    plan: TaskPlan | None = None
    needs_clarification: bool = False
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)


class _RawTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    intent: str
    tool_name: str
    capability: str
    arguments: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)


class _RawPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[_RawTask] = Field(min_length=1, max_length=8)
    rationale: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)


class MultiSkillPlanner:
    """在模型提议之前执行 Skill 治理，禁止候选外工具和错误类别直接执行。"""

    def __init__(self, llm: LLMClient, catalog: ToolCatalog | None = None) -> None:
        self._llm = llm
        self._catalog = catalog or ToolCatalog.load_default()

    async def plan(self, state: CareState) -> PlanningDecision:
        query = state.rewritten_query or _latest_query(state)
        prompt = (
            "把用户请求拆成最少的客服 Skill 任务，只返回 JSON。\n"
            '{"tasks":[{"task_id":"task-1","intent":"get_refund_status","tool_name":"get_refund_status","capability":"get_refund_status",'
            '"arguments":{"order_id":"R-17"},"dependencies":[]}],"rationale":"...","confidence":0.0}\n'
            "独立只读查询可以并行；只有存在事实依赖时才填写 dependencies。"
            "不能猜测缺失实体。只选择给定工具目录中的工具。capability 必须逐字使用该工具目录声明的 capability；"
            "intent 必须是工具目录给出的 task_intents 之一，表示这个子任务的具体目标，不要只填宽泛领域。\n"
            f"重规划约束：{state.context_summary or '无'}\n"
            f"用户问题：{query}\n当前意图：{state.intent}\n已知实体：{state.entities}\n工具目录：{self._tool_descriptions()}"
        )
        try:
            raw = parse_model_json(await self._llm.complete([{"role": "user", "content": prompt}]), _RawPlan)
            plan = self._build_plan(raw, state)
        except Exception as exc:
            _LOGGER.exception("多 Skill 规划失败，禁止直接执行")
            return PlanningDecision(needs_clarification=True, reason=f"计划无法验证：{exc}")
        if raw.confidence < 0.55:
            return PlanningDecision(plan=plan, needs_clarification=True, reason="规划置信度低", confidence=raw.confidence)
        return PlanningDecision(plan=plan, confidence=raw.confidence)

    async def replan(self, state: CareState, failed_task: SkillTask, error: str) -> PlanningDecision:
        """为失败任务请求替代计划，并显式提供失败工具和原因。"""
        replan_state = state.model_copy(update={
            "context_summary": (
                f"只修复 task_id={failed_task.task_id}，目标能力={failed_task.capability}；"
                f"失败工具={failed_task.tool_name}，失败类型={error}。"
                "保留同一用户目标，不要重复已成功的其他任务；若没有安全替代工具则返回低置信度并要求澄清。"
            )
        })
        return await self.plan(replan_state)

    def _build_plan(self, raw: _RawPlan, state: CareState) -> TaskPlan:
        tasks: list[SkillTask] = []
        for raw_task in raw.tasks:
            tool = self._catalog.get(raw_task.tool_name)
            if raw_task.intent not in tool.task_intents:
                raise ValueError(f"任务 {raw_task.task_id} 的目标意图不受工具支持")
            if raw_task.capability != tool.capability:
                raise ValueError(f"任务 {raw_task.task_id} 声明的能力与工具目录不一致")
            expected_category = _INTENT_CATEGORY.get(raw_task.intent)
            if expected_category and tool.category != expected_category:
                raise ValueError(f"任务 {raw_task.task_id} 的意图与工具类别不匹配")
            arguments = dict(raw_task.arguments)
            for key in tool.argument_names:
                if key not in arguments and key in state.entities:
                    arguments[key] = state.entities[key]
            requires_confirmation = tool.requires_confirmation or tool.name in _SIDE_EFFECT_TOOLS
            risk = "high" if tool.name in _HIGH_RISK_TOOLS else ("medium" if requires_confirmation else "low")
            tasks.append(
                SkillTask(
                    task_id=raw_task.task_id,
                    intent=raw_task.intent,
                    tool_name=tool.name,
                    capability=raw_task.capability,
                    arguments={key: value for key, value in arguments.items() if key in tool.argument_names},
                    dependencies=raw_task.dependencies,
                    risk_level=risk,
                    requires_confirmation=requires_confirmation,
                    max_attempts=1 if requires_confirmation else 2,
                )
            )
        return TaskPlan(tasks=tasks, requires_confirmation=any(task.requires_confirmation for task in tasks), rationale=raw.rationale)

    def _tool_descriptions(self) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "capability": tool.capability, "task_intents": tool.task_intents, "category": tool.category, "description": tool.description, "arguments": tool.argument_names}
            for tool in self._catalog.tools
        ]


def _latest_query(state: CareState) -> str:
    for message in reversed(state.messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""
