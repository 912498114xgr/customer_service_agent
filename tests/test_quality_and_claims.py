import asyncio
from decimal import Decimal

from common.mock_llm import MockLLM
from customer_care.claims import ClaimRequest, evaluate_claim
from customer_care.quality.rules import check_hard_rules
from customer_care.quality.scoring import score_quality


def test_hard_rules_detect_insult_privacy_promise_and_timeout() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "You are an idiot. We will refund you at user@example.com.",
            "response_delay_seconds": 90,
        }
    ]

    hits = check_hard_rules(messages)

    assert {hit.rule_id for hit in hits} == {
        "insult",
        "privacy_leak",
        "unfulfilled_promise",
        "response_timeout",
    }


def test_quality_redline_is_veto_even_when_model_score_is_high() -> None:
    llm = MockLLM(
        ['{"accuracy":100,"politeness":100,"resolution":100,"compliance":100,"rationale":"good"}']
    )

    result = asyncio.run(
        score_quality(
            [{"role": "assistant", "content": "You are an idiot."}],
            llm,
        )
    )

    assert result.score == 0
    assert result.passed is False
    assert result.hit_rules[0].rule_id == "insult"


def test_quality_uses_weighted_score_without_hard_rule_hits() -> None:
    llm = MockLLM(
        ['{"accuracy":90,"politeness":80,"resolution":70,"compliance":100,"rationale":"clear answer"}']
    )

    result = asyncio.run(
        score_quality(
            [{"role": "assistant", "content": "Your refund is processing."}],
            llm,
        )
    )

    assert result.score == 84
    assert result.passed is True
    assert result.hit_rules == []


def test_claim_amount_is_capped_by_rules_and_never_auto_executed() -> None:
    llm = MockLLM(
        ['{"action":"compensation","rationale":"Delivery was delayed.","confidence":0.92}']
    )
    request = ClaimRequest(
        user_id="u1",
        order_id="R-17",
        user_message="Please compensate me for the delay.",
        requested_amount=Decimal("25.00"),
        verified_order_total=Decimal("40.00"),
        evidence_verified=True,
    )

    result = asyncio.run(evaluate_claim(request, llm))

    assert result.rule_ceiling == Decimal("10.00")
    assert result.suggested_amount == Decimal("10.00")
    assert result.requires_human_review is True
    assert result.auto_execute is False
    assert result.ticket["status"] == "pending_human_review"


def test_claim_high_risk_and_low_confidence_are_escalated() -> None:
    llm = MockLLM(
        ['{"action":"refund","rationale":"Possible duplicate charge.","confidence":0.51}']
    )
    request = ClaimRequest(
        user_id="u2",
        order_id="R-18",
        user_message="I was charged twice.",
        requested_amount=Decimal("8.00"),
        verified_order_total=Decimal("18.00"),
        risk_flags=["repeated_dispute"],
    )

    result = asyncio.run(evaluate_claim(request, llm))

    assert result.auto_execute is False
    assert any("高风险" in reason for reason in result.review_reasons)
    assert any("置信度" in reason for reason in result.review_reasons)

