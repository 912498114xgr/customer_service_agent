"""多语言意图、实体与情绪识别节点。"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import LLMClient
from common.structured_output import parse_model_json
from customer_care.nodes.language import latest_user_message
from customer_care.state import CareState

_LOGGER = logging.getLogger(__name__)


class IntentGoal(BaseModel):
    """当前请求中的一个独立目标；具体工具仍由后续 Planner 选择。"""

    intent: Literal["refund", "track_order", "coupon", "account", "order", "other"]
    objective: str = Field(min_length=1, max_length=240)


class IntentResult(BaseModel):
    intent: Literal["refund", "track_order", "coupon", "account", "order", "other"]
    entities: dict[str, str | float | int | bool | None] = Field(default_factory=dict)
    emotion: Literal["neutral", "frustrated", "angry", "upset"] = "neutral"
    confidence: float = Field(ge=0, le=1)
    goals: list[IntentGoal] = Field(min_length=1, max_length=8)
    conflict: bool
    needs_clarification: bool


async def recognize_intent(state: CareState, llm: LLMClient) -> dict[str, object]:
    """调用模型识别意图，并在结构化输出无效时走安全澄清路径。"""
    query = latest_user_message(state)
    if not query:
        return {"intent": "other", "intent_confidence": 0.0, "emotion": "neutral", "intent_needs_clarification": True}

    history = state.messages[-8:]
    prompt = (
        "识别客服消息的意图、实体、情绪和独立目标，只返回 JSON。\n"
        '格式：{"intent":"refund|track_order|coupon|account|order|other",'
        '"entities":{"order_id":null,"coupon_id":null,"amount":null},'
        '"emotion":"neutral|frustrated|angry|upset","confidence":0.0,'
        '"goals":[{"intent":"refund","objective":"查询退款状态"}],'
        '"conflict":false,"needs_clarification":false}\n'
        "goals 按用户明确提出的独立业务目标拆分，不按句子数量或连接词拆分；一个目标只能有一项。"
        "同一领域中的两个不同目标也要分别列出。单一目标只输出一个 goal。"
        "若用户提出互相矛盾、无法同时满足的要求，将 conflict 和 needs_clarification 设为 true；"
        "若关键含义不确定或信息不足以判断目标，也设 needs_clarification=true。不要自行替用户选边。\n"
        "只抽取文本中明确出现的实体，不猜订单号或金额。\n"
        "示例：\n"
        '用户：Quiero saber el estado de mi reembolso. -> {"intent":"refund","entities":{},"emotion":"neutral","confidence":0.96,"goals":[{"intent":"refund","objective":"consultar estado del reembolso"}],"conflict":false,"needs_clarification":false}\n'
        '用户：Check order A123 and show my receipt. -> {"intent":"order","entities":{"order_id":"A123"},"emotion":"neutral","confidence":0.98,"goals":[{"intent":"order","objective":"consultar estado del pedido"},{"intent":"order","objective":"consultar recibo"}],"conflict":false,"needs_clarification":false}\n'
        f"语言：{state.language}\n历史摘要：{state.context_summary or '无'}\n上下文：{history}\n当前消息：{query}"
    )
    try:
        result = parse_model_json(
            await llm.complete([{"role": "user", "content": prompt}]),
            IntentResult,
        )
    except Exception:
        _LOGGER.exception("意图识别调用或结构化输出校验失败")
        return {
            "intent": "other",
            "intent_confidence": 0.0,
            "entities": {},
            "emotion": "neutral",
            "intent_goals": [],
            "intent_conflict": False,
            "intent_needs_clarification": True,
            "multi_skill_requested": False,
        }

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "entities": result.entities,
        "emotion": result.emotion,
        "intent_goals": [goal.model_dump(mode="json") for goal in result.goals],
        "intent_conflict": result.conflict,
        "intent_needs_clarification": result.needs_clarification,
        "multi_skill_requested": len(result.goals) > 1 and not result.conflict and not result.needs_clarification,
    }

