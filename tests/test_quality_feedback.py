from customer_care.quality.feedback import HumanFeedback, feedback_to_eval_record, handoff_recommendation
from customer_care.state import CareState


def test_handoff_recommendation_contains_actionable_reasons() -> None:
    state = CareState(user_id="u1", route="human", emotion="angry", handoff_summary="refund case")

    result = handoff_recommendation(state)

    assert result["requires_human"] is True
    assert result["priority"] == "high"
    assert result["reasons"]


def test_human_feedback_can_become_eval_record() -> None:
    state = CareState(user_id="u1", route="self_service", intent="refund", selected_tool="get_refund_status", messages=[{"role":"user","content":"Where is my refund?"}])
    feedback = HumanFeedback.create("u1", False, corrected_answer="Please wait three days.", reason="回答缺少时间范围")

    record = feedback_to_eval_record(state, feedback)

    assert record["human_accepted"] is False
    assert record["query"] == "Where is my refund?"
