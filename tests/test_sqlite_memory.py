import asyncio

import pytest

from customer_care.memory import SQLiteMemoryStore, VersionNotFoundError
from customer_care.state import CareState


def test_sqlite_store_persists_snapshots_and_rollback(tmp_path) -> None:
    database = tmp_path / "memory.db"
    first_store = SQLiteMemoryStore(database)
    first = CareState(user_id="u-1", messages=[{"role": "user", "content": "first"}])
    second = CareState(user_id="u-1", messages=[{"role": "user", "content": "second"}])
    first_version = asyncio.run(first_store.save("u-1", first))
    asyncio.run(first_store.save("u-1", second))

    reopened = SQLiteMemoryStore(database)
    assert asyncio.run(reopened.load("u-1")) == second
    assert asyncio.run(reopened.rollback("u-1", first_version)) == first
    assert asyncio.run(reopened.load("u-1")) == first


def test_sqlite_rollback_rejects_unknown_version(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")

    with pytest.raises(VersionNotFoundError):
        asyncio.run(store.rollback("u-1", "missing"))


def test_sqlite_store_requires_a_file_for_durable_connections() -> None:
    with pytest.raises(ValueError, match="file path"):
        SQLiteMemoryStore(":memory:")
