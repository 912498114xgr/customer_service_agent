"""确认令牌的存储层原子迁移规则。"""

import hmac

from customer_care.state import CareState


def claim_state(
    state: CareState,
    confirmation_id: str,
    token_hash: str,
    now: float,
    target_status: str,
) -> tuple[str, CareState]:
    """根据当前快照校验并生成确认状态迁移。"""
    if state.confirmation_status != "pending" or not state.confirmation_token_hash:
        return "not_pending", state
    if state.confirmation_id != confirmation_id:
        return "invalid_confirmation", state
    if not hmac.compare_digest(state.confirmation_token_hash, token_hash):
        return "invalid_token", state
    if state.confirmation_expires_at is None or state.confirmation_expires_at <= now:
        expired = state.model_copy(update={
            "confirmation_status": "expired",
            "confirmation_token_hash": None,
            "confirmation_expires_at": None,
            "awaiting_confirmation": False,
            "task_plan": None,
        })
        return "expired", expired
    if target_status == "executing":
        if not state.task_plan:
            return "missing_plan", state
        claimed = state.model_copy(update={
            "confirmation_status": "executing",
            "confirmation_token_hash": None,
            "confirmation_expires_at": None,
            "messages": [*state.messages, {"role": "user", "content": "[用户确认待处理操作]"}],
        })
        return "claimed", claimed
    if target_status == "declined":
        declined = state.model_copy(update={
            "confirmation_status": "declined",
            "confirmation_token_hash": None,
            "confirmation_expires_at": None,
            "awaiting_confirmation": False,
            "execution_status": "cancelled",
            "tool_result": {"status": "cancelled", "reason": "user_declined"},
            "task_results": [
                {**result, "status": "cancelled", "error": "用户拒绝执行，工具未调用"}
                for result in state.task_results
            ],
            "route": "clarify",
            "answer": "The operation was cancelled. No tool was run.",
            "messages": [*state.messages, {"role": "user", "content": "[用户拒绝待执行操作]"}],
        })
        return "claimed", declined
    raise ValueError("target_status must be executing or declined")
