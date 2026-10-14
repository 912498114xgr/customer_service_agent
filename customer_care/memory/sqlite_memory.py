"""基于 SQLite 事务的本地持久化会话快照存储。"""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from customer_care.memory.base import MemoryStore, VersionNotFoundError
from customer_care.memory.confirmation import claim_state
from customer_care.memory.versioning import make_version, parse_state, serialize_state
from customer_care.state import CareState


class SQLiteMemoryStore(MemoryStore):
    """面向单机学习/演练的持久化存储，确认认领使用 BEGIN IMMEDIATE 串行化。"""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path == ":memory:":
            raise ValueError("SQLiteMemoryStore requires a file path for cross-connection persistence")
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    user_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (user_id, version)
                );
                CREATE TABLE IF NOT EXISTS current_states (
                    user_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    async def load(self, user_id: str) -> CareState:
        return await asyncio.to_thread(self._load, user_id)

    def _load(self, user_id: str) -> CareState:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM current_states WHERE user_id = ?", (user_id,)
            ).fetchone()
        return parse_state(row["payload"]) if row else CareState(user_id=user_id)

    async def save(self, user_id: str, state: CareState) -> str:
        if state.user_id != user_id:
            raise ValueError("state.user_id must match user_id")
        return await asyncio.to_thread(self._save, user_id, state)

    def _save(self, user_id: str, state: CareState) -> str:
        payload = serialize_state(state)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute(
                "SELECT COUNT(*) FROM snapshots WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            version = make_version(count + 1, payload)
            connection.execute(
                "INSERT INTO snapshots(user_id, version, payload) VALUES (?, ?, ?)",
                (user_id, version, payload),
            )
            connection.execute(
                "INSERT INTO current_states(user_id, version, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET version=excluded.version, payload=excluded.payload",
                (user_id, version, payload),
            )
            connection.commit()
        return version

    async def history(self, user_id: str) -> list[dict[str, object]]:
        return (await self.load(user_id)).messages

    async def rollback(self, user_id: str, version: str) -> CareState:
        return await asyncio.to_thread(self._rollback, user_id, version)

    def _rollback(self, user_id: str, version: str) -> CareState:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM snapshots WHERE user_id = ? AND version = ?",
                (user_id, version),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise VersionNotFoundError(f"unknown memory version: {version}")
            connection.execute(
                "INSERT INTO current_states(user_id, version, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET version=excluded.version, payload=excluded.payload",
                (user_id, version, row["payload"]),
            )
            connection.commit()
        return parse_state(row["payload"])

    async def claim_confirmation(
        self,
        user_id: str,
        confirmation_id: str,
        token_hash: str,
        now: float,
        target_status: str,
    ) -> tuple[str, CareState]:
        return await asyncio.to_thread(
            self._claim_confirmation,
            user_id,
            confirmation_id,
            token_hash,
            now,
            target_status,
        )

    def _claim_confirmation(
        self,
        user_id: str,
        confirmation_id: str,
        token_hash: str,
        now: float,
        target_status: str,
    ) -> tuple[str, CareState]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM current_states WHERE user_id = ?", (user_id,)
            ).fetchone()
            state = parse_state(row["payload"]) if row else CareState(user_id=user_id)
            result, updated = claim_state(state, confirmation_id, token_hash, now, target_status)
            if result in {"claimed", "expired"}:
                payload = serialize_state(updated)
                count = connection.execute(
                    "SELECT COUNT(*) FROM snapshots WHERE user_id = ?", (user_id,)
                ).fetchone()[0]
                version = make_version(count + 1, payload)
                connection.execute(
                    "INSERT INTO snapshots(user_id, version, payload) VALUES (?, ?, ?)",
                    (user_id, version, payload),
                )
                connection.execute(
                    "UPDATE current_states SET version = ?, payload = ? WHERE user_id = ?",
                    (version, payload, user_id),
                )
            connection.commit()
        return result, updated
