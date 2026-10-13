"""离线验证退款客服、自助工具、质检和理赔安全边界。"""

import asyncio
from decimal import Decimal

from common.mock_llm import MockLLM
from customer_care.claims import ClaimRequest, evaluate_claim
from customer_care.graph import build_customer_care_graph
from customer_care.quality.scoring import score_quality
from customer_care.state import CareState


async def run_self_test() -> None:
    """执行不访问外部 API、不连接 Redis 的端到端检查。"""
    graph_llm = MockLLM(
        [
            '{"intent":"refund","entities":{"order_id":"R-17"},"emotion":"neutral","confidence":0.96,"goals":[{"intent":"refund","objective":"查询退款状态"}],"conflict":false,"needs_clarification":false}',
            '{"tool_name":"get_refund_status","arguments":{"order_id":"R-17"}}',
            "Your refund for order R-17 is processing and should arrive in about 3 days.",
        ]
    )
    graph = build_customer_care_graph(graph_llm)
    initial_state = CareState(
        user_id="demo-user",
        messages=[{"role": "user", "content": "Where is my refund for order R-17?"}],
    )
    conversation = CareState.model_validate(await graph.ainvoke(initial_state.model_dump()))
    assert conversation.route == "self_service"
    assert conversation.tool_result.get("status") == "processing"
    assert conversation.answer
    print("客服退款链路：通过")

    quality_llm = MockLLM(
        ['{"accuracy":90,"politeness":90,"resolution":85,"compliance":95,"rationale":"clear and safe"}']
    )
    quality = await score_quality(
        [
            {"role": "user", "content": "Where is my refund?"},
            {"role": "assistant", "content": conversation.answer},
        ],
        quality_llm,
    )
    assert quality.passed
    print(f"会话质检：通过（{quality.score} 分）")

    claim_llm = MockLLM(
        ['{"action":"compensation","rationale":"Delivery was delayed.","confidence":0.9}']
    )
    claim = await evaluate_claim(
        ClaimRequest(
            user_id="demo-user",
            order_id="R-17",
            user_message="Please compensate me for the delay.",
            requested_amount=Decimal("25.00"),
            verified_order_total=Decimal("40.00"),
            evidence_verified=True,
        ),
        claim_llm,
    )
    assert claim.suggested_amount == Decimal("10.00")
    assert claim.auto_execute is False
    print("理赔金额规则与人工兜底：通过")


def main() -> None:
    asyncio.run(run_self_test())


if __name__ == "__main__":
    main()

