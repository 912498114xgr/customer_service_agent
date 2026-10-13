"""带不可变状态快照的进程内记忆后端。"""

import asyncio

from customer_care.memory.base import MemoryStore, VersionNotFoundError
from customer_care.memory.versioning import make_version, parse_state, serialize_state
from customer_care.state import CareState


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._snapshots: dict[str, dict[str, str]] = {}
        self._current_versions: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    async def load(self, user_id: str) -> CareState:
        async with self._lock_for(user_id):
            version = self._current_versions.get(user_id)
            if version is None:
                return CareState(user_id=user_id)
            return parse_state(self._snapshots[user_id][version])

    async def save(self, user_id: str, state: CareState) -> str:
        if state.user_id != user_id:
            raise ValueError("state.user_id must match user_id")

        async with self._lock_for(user_id):
            snapshots = self._snapshots.setdefault(user_id, {})
            payload = serialize_state(state)
            version = make_version(len(snapshots) + 1, payload)
            snapshots[version] = payload
            self._current_versions[user_id] = version
            return version

    async def history(self, user_id: str) -> list[dict[str, object]]:
        state = await self.load(user_id)
        return state.messages

    async def rollback(self, user_id: str, version: str) -> CareState:
        async with self._lock_for(user_id):
            payload = self._snapshots.get(user_id, {}).get(version)
            if payload is None:
                raise VersionNotFoundError(f"unknown memory version: {version}")
            self._current_versions[user_id] = version
            return parse_state(payload)

