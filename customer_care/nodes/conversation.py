"""客服路由、自助执行、澄清、转人工与最终回复节点。"""

import logging
from typing import Any

from common.llm import LLMClient
from customer_care.nodes.language import latest_user_message
from customer_care.state import CareState
from customer_care.tools.registry import ToolRegistry

_LOGGER = logging.getLogger(__name__)

_CLARIFY_TEXT = {
    "en": "Which order are you asking about? Please share the order ID.",
    "es": "¿Sobre qué pedido nos consulta? Comparta el número de pedido, por favor.",
    "pt": "Sobre qual pedido você está perguntando? Informe o número do pedido, por favor.",
}
_HANDOFF_TEXT = {
    "en": "I’m connecting you with a support specialist. I’ll include the details you shared.",
    "es": "Le conectaré con un agente de soporte y compartiré los detalles necesarios.",
    "pt": "Vou encaminhar você para um atendente e compartilhar os detalhes necessários.",
}


def route_decision(state: CareState) -> dict[str, str]:
    """按歧义、识别置信度和情绪选择澄清、自助或人工。"""
    if state.rewrite_ambiguous:
        return {"route": "clarify"}
    if state.intent is None or state.intent == "other":
        return {"route": "clarify"}
    if state.intent_confidence < 0.55 or state.emotion in {"frustrated", "angry", "upset"}:
        return {"route": "human"}
    return {"route": "self_service"}


def clarify(state: CareState) -> dict[str, str]:
    """以用户语言提出最小必要追问。"""
    return {"answer": _CLARIFY_TEXT.get(state.language, _CLARIFY_TEXT["en"])}


def human_handoff(state: CareState) -> dict[str, str]:
    """生成最小化人工摘要，不自动创建外部工单或执行资金操作。"""
    summary = (
        f"intent={state.intent or 'unknown'}; "
        f"query={latest_user_message(state)[:300]}; "
        f"entities={state.entities}"
    )
    return {
        "handoff_summary": summary,
        "answer": _HANDOFF_TEXT.get(state.language, _HANDOFF_TEXT["en"]),
    }


async def self_service(state: CareState, registry: ToolRegistry) -> dict[str, Any]:
    """只调用本地工具注册表，并把结果写回状态。"""
    if not state.selected_tool:
        return {"tool_result": {"status": "not_executed", "reason": "missing_tool"}}
    try:
        result = registry.execute(state.selected_tool, state.tool_args)
    except ValueError as exc:
        return {"tool_result": {"status": "not_executed", "reason": str(exc)}}
    return {"tool_result": result}


async def reply(state: CareState, llm: LLMClient) -> dict[str, str]:
    """根据工具结果或人工摘要生成回复；模型异常时使用本地模板兜底。"""
    context: dict[str, Any] = {
        "tool_result": state.tool_result,
        "clarification": state.answer if state.route == "clarify" else None,
    }
    prompt = (
        "你是多语言外卖客服，只根据给定事实回复用户。不要承诺已经执行退款、赔付或账户变更；"
        "需要人工处理时要明确说明，不要泄露内部摘要。只输出面向用户的简短回复。\n"
        f"回复语言代码：{state.language}\n用户问题：{latest_user_message(state)}\n事实数据：{context}"
    )
    try:
        answer = (await llm.complete([{"role": "user", "content": prompt}])).strip()
    except Exception:
        _LOGGER.exception("最终回复生成失败，改用本地模板")
        answer = ""
    if not answer or answer == "mock response":
        answer = _fallback_reply(state)
    return {"answer": answer}


def _fallback_reply(state: CareState) -> str:
    if state.route == "clarify":
        return _CLARIFY_TEXT.get(state.language, _CLARIFY_TEXT["en"])
    if state.route == "human":
        return _HANDOFF_TEXT.get(state.language, _HANDOFF_TEXT["en"])
    if state.tool_result.get("status") == "processing":
        order_id = state.tool_result.get("order_id", "your order")
        days = state.tool_result.get("estimated_days", 3)
        return f"The refund for {order_id} is being processed. The current estimate is {days} days."
    return "I found the relevant order information. Would you like help with anything else?"

