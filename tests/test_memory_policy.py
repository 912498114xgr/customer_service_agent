import asyncio

from customer_care.memory import InMemoryMemoryStore, MemoryPolicy, compact_state
from customer_care.state import CareState


def test_compact_state_keeps_recent_messages_and_creates_summary() -> None:
    state = CareState(
        user_id="u1",
        messages=[{"role": "user", "content": f"message-{index}"} for index in range(8)],
    )

    compacted = compact_state(state, MemoryPolicy(max_messages=4, max_chars=1000, ttl_seconds=60))

    assert len(compacted.messages) == 4
    assert compacted.messages[0]["content"] == "message-4"
    assert "message-0" in (compacted.context_summary or "")


def test_in_memory_store_expires_state_after_ttl() -> None:
    async def scenario() -> CareState:
        store = InMemoryMemoryStore(ttl_seconds=1)
        await store.save("u1", CareState(user_id="u1", intent="refund"))
        store._expires_at["u1"] = 0
        return await store.load("u1")

    loaded = asyncio.run(scenario())

    assert loaded.user_id == "u1"
    assert loaded.intent is None
