"""带不可变状态快照的进程内记忆后端。"""

import asyncio
import time

from customer_care.memory.base import MemoryStore, VersionNotFoundError
from customer_care.memory.confirmation import claim_state
from customer_care.memory.versioning import make_version, parse_state, serialize_state
from customer_care.state import CareState


class InMemoryMemoryStore(MemoryStore):
    def __init__(self, *, ttl_seconds: int | None = None) -> None:
        self._snapshots: dict[str, dict[str, str]] = {}
        self._current_versions: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._expires_at: dict[str, float] = {}
        self._ttl_seconds = ttl_seconds

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    async def load(self, user_id: str) -> CareState:
        async with self._lock_for(user_id):
            self._purge_if_expired(user_id)
            version = self._current_versions.get(user_id)
            if version is None:
                return CareState(user_id=user_id)
            return parse_state(self._snapshots[user_id][version])

    async def save(self, user_id: str, state: CareState) -> str:
        if state.user_id != user_id:
            raise ValueError("state.user_id must match user_id")

        async with self._lock_for(user_id):
            self._purge_if_expired(user_id)
            snapshots = self._snapshots.setdefault(user_id, {})
            payload = serialize_state(state)
            version = make_version(len(snapshots) + 1, payload)
            snapshots[version] = payload
            self._current_versions[user_id] = version
            if self._ttl_seconds is not None:
                self._expires_at[user_id] = time.monotonic() + self._ttl_seconds
            return version

    async def history(self, user_id: str) -> list[dict[str, object]]:
        state = await self.load(user_id)
        return state.messages

    async def rollback(self, user_id: str, version: str) -> CareState:
        async with self._lock_for(user_id):
            self._purge_if_expired(user_id)
            payload = self._snapshots.get(user_id, {}).get(version)
            if payload is None:
                raise VersionNotFoundError(f"unknown memory version: {version}")
            self._current_versions[user_id] = version
            if self._ttl_seconds is not None:
                self._expires_at[user_id] = time.monotonic() + self._ttl_seconds
            return parse_state(payload)

    async def claim_confirmation(
        self,
        user_id: str,
        confirmation_id: str,
        token_hash: str,
        now: float,
        target_status: str,
    ) -> tuple[str, CareState]:
        async with self._lock_for(user_id):
            self._purge_if_expired(user_id)
            current = self._current_versions.get(user_id)
            state = (
                parse_state(self._snapshots[user_id][current])
                if current is not None
                else CareState(user_id=user_id)
            )
            result, updated = claim_state(state, confirmation_id, token_hash, now, target_status)
            if result in {"claimed", "expired"}:
                snapshots = self._snapshots.setdefault(user_id, {})
                payload = serialize_state(updated)
                version = make_version(len(snapshots) + 1, payload)
                snapshots[version] = payload
                self._current_versions[user_id] = version
                if self._ttl_seconds is not None:
                    self._expires_at[user_id] = time.monotonic() + self._ttl_seconds
            return result, updated

    def _purge_if_expired(self, user_id: str) -> None:
        expires_at = self._expires_at.get(user_id)
        if expires_at is None or expires_at > time.monotonic():
            return
        self._snapshots.pop(user_id, None)
        self._current_versions.pop(user_id, None)
        self._expires_at.pop(user_id, None)

