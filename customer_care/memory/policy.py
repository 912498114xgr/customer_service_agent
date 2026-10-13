"""会话上下文裁剪与摘要策略。"""

from dataclasses import dataclass

from customer_care.state import CareState


@dataclass(frozen=True)
class MemoryPolicy:
    """限制单个会话的消息数、字符数和存活时间。"""

    max_messages: int = 16
    max_chars: int = 6000
    ttl_seconds: int = 86400

    def __post_init__(self) -> None:
        if self.max_messages < 2 or self.max_chars < 200 or self.ttl_seconds <= 0:
            raise ValueError("会话记忆限制必须为正数，且至少保留两条消息")


def compact_state(state: CareState, policy: MemoryPolicy) -> CareState:
    """保留最新上下文，并将被裁剪的旧消息压缩成可解释摘要。"""
    messages = list(state.messages)
    serialized_size = sum(len(str(message.get("content", ""))) for message in messages)
    if len(messages) <= policy.max_messages and serialized_size <= policy.max_chars:
        return state

    retained: list[dict[str, object]] = []
    size = 0
    for message in reversed(messages):
        content_size = len(str(message.get("content", "")))
        if retained and (len(retained) >= policy.max_messages or size + content_size > policy.max_chars):
            break
        retained.append(message)
        size += content_size
    retained.reverse()
    removed = messages[: len(messages) - len(retained)]
    summary_parts = [
        f"{message.get('role', 'unknown')}: {str(message.get('content', '')).strip()[:160]}"
        for message in removed
        if str(message.get("content", "")).strip()
    ]
    old_summary = state.context_summary or ""
    new_summary = "；".join(part for part in [old_summary, *summary_parts] if part)
    return state.model_copy(update={"messages": retained, "context_summary": new_summary[-2000:]})
