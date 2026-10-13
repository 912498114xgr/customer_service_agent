"""结合确定性红线和模型维度评分输出质检结果。"""

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.llm import LLMClient
from common.structured_output import parse_model_json
from customer_care.quality.rules import RuleHit, check_hard_rules

_LOGGER = logging.getLogger(__name__)


class QualityDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: int = Field(ge=0, le=100)
    politeness: int = Field(ge=0, le=100)
    resolution: int = Field(ge=0, le=100)
    compliance: int = Field(ge=0, le=100)


class ModelScore(QualityDimensions):
    rationale: str


class QualityResult(BaseModel):
    score: int
    hit_rules: list[RuleHit]
    dimensions: QualityDimensions
    rationale: str
    passed: bool


async def score_quality(
    messages: list[dict[str, Any]],
    llm: LLMClient,
    *,
    pass_threshold: int = 80,
) -> QualityResult:
    """红线命中直接判零；否则按准确、礼貌、解决、规范四维加权。"""
    hits = check_hard_rules(messages)
    prompt = (
        "为客服对话的软性服务质量评分，只返回 JSON。\n"
        '{"accuracy":0,"politeness":0,"resolution":0,"compliance":0,"rationale":"..."}\n'
        "每个分数必须是 0 到 100 的整数。不要依据未出现的事实推断。\n"
        f"对话：{messages}"
    )
    try:
        model_score = parse_model_json(
            await llm.complete([{"role": "user", "content": prompt}]),
            ModelScore,
        )
        rationale = model_score.rationale
    except Exception:
        _LOGGER.exception("模型质检评分调用或输出校验失败，按零分处理")
        model_score = ModelScore(
            accuracy=0,
            politeness=0,
            resolution=0,
            compliance=0,
            rationale="模型评分输出不可验证，按零分处理。",
        )
        rationale = model_score.rationale

    weighted_score = round(
        model_score.accuracy * 0.35
        + model_score.politeness * 0.25
        + model_score.resolution * 0.25
        + model_score.compliance * 0.15
    )
    final_score = 0 if hits else weighted_score
    if hits:
        rationale = f"命中硬性规则，一票否决。模型评分仅作参考：{rationale}"
    return QualityResult(
        score=final_score,
        hit_rules=hits,
        dimensions=QualityDimensions(
            accuracy=model_score.accuracy,
            politeness=model_score.politeness,
            resolution=model_score.resolution,
            compliance=model_score.compliance,
        ),
        rationale=rationale,
        passed=final_score >= pass_threshold,
    )

