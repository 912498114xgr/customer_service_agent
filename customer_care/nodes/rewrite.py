"""基于对话上下文进行多语言指代消解。"""

import logging
import re

from pydantic import BaseModel, Field

from common.llm import LLMClient
from common.structured_output import parse_model_json
from customer_care.nodes.language import latest_user_message
from customer_care.state import CareState

_LOGGER = logging.getLogger(__name__)

_REFERENCE = re.compile(
    r"\b(it|this|that|they|them|one|he|she|esto|esta|este|eso|esa|ese|lo|la|"
    r"isso|isto|esse|essa|ele|ela|aquilo)\b",
    re.I,
)


class RewriteResult(BaseModel):
    rewritten_query: str
    ambiguous: bool
    explanation: str = Field(default="")


async def rewrite_query(state: CareState, llm: LLMClient) -> dict[str, object]:
    """有明确指代时调用模型；上下文不足或输出异常时标记歧义。"""
    query = latest_user_message(state)
    if not _REFERENCE.search(query):
        return {
            "rewritten_query": query,
            "rewrite_ambiguous": False,
            "rewrite_explanation": "未发现需要消解的指代词。",
        }

    prompt = (
        "你负责客服对话中的指代消解，只返回 JSON。\n"
        '格式：{"rewritten_query":"...","ambiguous":false,"explanation":"..."}\n'
        "若上下文没有唯一且明确的指代对象，必须令 ambiguous=true，并保持原问题，不得猜测。\n"
        "仅用西班牙语、葡萄牙语或英语重写，保持原语言，不添加事实。\n"
        "示例：\n"
        '上下文实体：order_id=A123；问题：Can I get a refund for it? -> {"rewritten_query":"Can I get a refund for order A123?","ambiguous":false,"explanation":"it 指代已识别的订单。"}\n'
        '上下文实体：无；问题：What about that? -> {"rewritten_query":"What about that?","ambiguous":true,"explanation":"没有可确定的指代对象。"}\n'
        f"语言：{state.language}\n历史摘要：{state.context_summary or '无'}\n历史：{state.messages[-8:]}\n已识别实体：{state.entities}\n当前问题：{query}"
    )
    try:
        result = parse_model_json(
            await llm.complete([{"role": "user", "content": prompt}]),
            RewriteResult,
        )
    except Exception:
        _LOGGER.exception("查询改写调用或结构化输出校验失败")
        return {
            "rewritten_query": query,
            "rewrite_ambiguous": True,
            "rewrite_explanation": "模型输出无法验证，转入澄清。",
        }

    if result.ambiguous:
        return {
            "rewritten_query": query,
            "rewrite_ambiguous": True,
            "rewrite_explanation": result.explanation or "上下文不足以确定指代对象。",
        }
    if not result.rewritten_query.strip():
        return {
            "rewritten_query": query,
            "rewrite_ambiguous": True,
            "rewrite_explanation": "模型没有提供有效改写，转入澄清。",
        }
    return {
        "rewritten_query": result.rewritten_query.strip(),
        "rewrite_ambiguous": False,
        "rewrite_explanation": result.explanation,
    }

