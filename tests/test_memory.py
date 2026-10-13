import asyncio

import pytest

from customer_care.memory import InMemoryMemoryStore, VersionNotFoundError
from customer_care.state import CareState


def test_load_returns_empty_state_for_new_user() -> None:
    store = InMemoryMemoryStore()

    state = asyncio.run(store.load("user-1"))

    assert state.user_id == "user-1"
    assert state.messages == []
    assert state.language == "en"


def test_save_load_and_history_round_trip() -> None:
    store = InMemoryMemoryStore()
    state = CareState(
        user_id="user-1",
        messages=[{"role": "user", "content": "Where is my refund?"}],
        intent="refund",
    )

    version = asyncio.run(store.save("user-1", state))
    loaded = asyncio.run(store.load("user-1"))

    assert version.startswith("00000001-")
    assert loaded == state
    assert asyncio.run(store.history("user-1")) == state.messages


def test_rollback_restores_snapshot_without_deleting_later_versions() -> None:
    store = InMemoryMemoryStore()
    first = CareState(user_id="user-1", messages=[{"role": "user", "content": "first"}])
    second = CareState(user_id="user-1", messages=[{"role": "user", "content": "second"}])

    first_version = asyncio.run(store.save("user-1", first))
    second_version = asyncio.run(store.save("user-1", second))
    restored = asyncio.run(store.rollback("user-1", first_version))

    assert restored == first
    assert asyncio.run(store.load("user-1")) == first
    assert asyncio.run(store.rollback("user-1", second_version)) == second


def test_rollback_unknown_version_raises() -> None:
    store = InMemoryMemoryStore()

    with pytest.raises(VersionNotFoundError):
        asyncio.run(store.rollback("user-1", "missing-version"))


def test_save_rejects_state_for_a_different_user() -> None:
    store = InMemoryMemoryStore()
    state = CareState(user_id="user-2")

    with pytest.raises(ValueError, match="state.user_id"):
        asyncio.run(store.save("user-1", state))

