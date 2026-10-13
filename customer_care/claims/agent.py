"""只生成理赔建议和人工工单，不提供资金执行能力。"""

import logging
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from common.llm import LLMClient
from common.structured_output import parse_model_json

_LOGGER = logging.getLogger(__name__)

ClaimAction = Literal["refund", "compensation", "waiver", "none"]


class ClaimRequest(BaseModel):
    user_id: str
    order_id: str
    user_message: str
    requested_amount: Decimal = Field(ge=0)
    verified_order_total: Decimal = Field(ge=0)
    evidence_verified: bool = False
    risk_flags: list[str] = Field(default_factory=list)


class ClaimSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ClaimAction
    rationale: str
    confidence: float = Field(ge=0, le=1)


class ClaimDecision(BaseModel):
    action: ClaimAction
    rationale: str
    confidence: float
    requested_amount: Decimal
    rule_ceiling: Decimal
    suggested_amount: Decimal
    requires_human_review: Literal[True] = True
    auto_execute: Literal[False] = False
    review_reasons: list[str]
    ticket: dict[str, str]


async def evaluate_claim(
    request: ClaimRequest,
    llm: LLMClient,
    *,
    compensation_cap: Decimal = Decimal("10.00"),
    confidence_threshold: float = 0.75,
) -> ClaimDecision:
    """模型提出方案；金额上限只由确定性规则计算，最终始终转人工。"""
    prompt = (
        "分析售后诉求，只建议动作和理由，不要计算、修改或输出任何金额。只返回 JSON。\n"
        '{"action":"refund|compensation|waiver|none","rationale":"...","confidence":0.0}\n'
        "退款、赔付或豁免都只是建议，不能声称已执行。\n"
        f"用户诉求：{request.user_message}\n订单编号：{request.order_id}\n"
        f"证据已核验：{request.evidence_verified}\n风险标记：{request.risk_flags}"
    )
    try:
        suggestion = parse_model_json(
            await llm.complete([{"role": "user", "content": prompt}]),
            ClaimSuggestion,
        )
    except Exception:
        _LOGGER.exception("理赔建议调用或输出校验失败，转人工判断")
        suggestion = ClaimSuggestion(
            action="none",
            rationale="模型建议不可验证，需要人工判断。",
            confidence=0.0,
        )

    if suggestion.action == "refund":
        rule_ceiling = request.verified_order_total
    elif suggestion.action == "compensation":
        rule_ceiling = min(request.verified_order_total, compensation_cap)
    else:
        rule_ceiling = Decimal("0")
    suggested_amount = min(request.requested_amount, rule_ceiling)

    review_reasons = ["退款、赔付和豁免均需人工二次确认。"]
    if not request.evidence_verified:
        review_reasons.append("订单证据尚未核验。")
    if request.risk_flags:
        review_reasons.append("存在高风险标记：" + ", ".join(request.risk_flags))
    if suggestion.confidence < confidence_threshold:
        review_reasons.append("模型置信度低于阈值。")
    if request.requested_amount > rule_ceiling:
        review_reasons.append("诉求金额超过规则上限，建议金额已按上限截断。")
    if suggestion.action == "none":
        review_reasons.append("模型未能提出可验证的理赔方案。")

    return ClaimDecision(
        action=suggestion.action,
        rationale=suggestion.rationale,
        confidence=suggestion.confidence,
        requested_amount=request.requested_amount,
        rule_ceiling=rule_ceiling,
        suggested_amount=suggested_amount,
        review_reasons=review_reasons,
        ticket={
            "user_id": request.user_id,
            "order_id": request.order_id,
            "status": "pending_human_review",
        },
    )

