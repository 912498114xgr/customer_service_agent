"""多语言意图、实体与情绪识别节点。"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import LLMClient
from common.structured_output import parse_model_json
from customer_care.nodes.language import latest_user_message
from customer_care.state import CareState

_LOGGER = logging.getLogger(__name__)


class IntentResult(BaseModel):
    intent: Literal["refund", "track_order", "coupon", "account", "order", "other"]
    entities: dict[str, str | float | int | bool | None] = Field(default_factory=dict)
    emotion: Literal["neutral", "frustrated", "angry", "upset"] = "neutral"
    confidence: float = Field(ge=0, le=1)


async def recognize_intent(state: CareState, llm: LLMClient) -> dict[str, object]:
    """调用模型识别意图，并在结构化输出无效时走安全澄清路径。"""
    query = latest_user_message(state)
    if not query:
        return {"intent": "other", "intent_confidence": 0.0, "emotion": "neutral"}

    history = state.messages[-8:]
    prompt = (
        "识别客服消息的意图、实体和情绪，只返回 JSON。\n"
        '格式：{"intent":"refund|track_order|coupon|account|order|other",'
        '"entities":{"order_id":null,"coupon_id":null,"amount":null},'
        '"emotion":"neutral|frustrated|angry|upset","confidence":0.0}\n'
        "只抽取文本中明确出现的实体，不猜订单号或金额。\n"
        "示例：\n"
        '用户：Quiero saber el estado de mi reembolso. -> {"intent":"refund","entities":{},"emotion":"neutral","confidence":0.96}\n'
        '用户：Meu pedido ainda não chegou. -> {"intent":"track_order","entities":{},"emotion":"frustrated","confidence":0.91}\n'
        '用户：Where is order A123? -> {"intent":"track_order","entities":{"order_id":"A123"},"emotion":"neutral","confidence":0.98}\n'
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
        }

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "entities": result.entities,
        "emotion": result.emotion,
    }

