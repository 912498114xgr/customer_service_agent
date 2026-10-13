"""基于 Redis 的对话状态与版本快照存储。"""

from urllib.parse import quote

from redis.asyncio import Redis

from customer_care.memory.base import MemoryStore, VersionNotFoundError
from customer_care.memory.versioning import make_version, parse_state, serialize_state
from customer_care.state import CareState

_SAVE_SCRIPT = """
local sequence = redis.call('INCR', KEYS[1])
local version = string.format('%08d', sequence) .. '-' .. ARGV[2]
redis.call('SET', KEYS[2], ARGV[1])
redis.call('HSET', KEYS[3], version, ARGV[1])
if tonumber(ARGV[3]) and tonumber(ARGV[3]) > 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[3])
    redis.call('EXPIRE', KEYS[2], ARGV[3])
    redis.call('EXPIRE', KEYS[3], ARGV[3])
end
return version
"""

_ROLLBACK_SCRIPT = """
local payload = redis.call('HGET', KEYS[1], ARGV[1])
if not payload then
    return false
end
redis.call('SET', KEYS[2], payload)
return payload
"""


class RedisMemoryStore(MemoryStore):
    def __init__(self, client: Redis, *, key_prefix: str = "customer-care", ttl_seconds: int | None = None) -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    def _keys(self, user_id: str) -> tuple[str, str, str]:
        safe_user_id = quote(user_id, safe="")
        root = f"{self._key_prefix}:{safe_user_id}"
        return f"{root}:sequence", f"{root}:current", f"{root}:versions"

    async def load(self, user_id: str) -> CareState:
        _, current_key, _ = self._keys(user_id)
        payload = await self._client.get(current_key)
        if payload is None:
            return CareState(user_id=user_id)
        return parse_state(payload)

    async def save(self, user_id: str, state: CareState) -> str:
        if state.user_id != user_id:
            raise ValueError("state.user_id must match user_id")

        sequence_key, current_key, versions_key = self._keys(user_id)
        payload = serialize_state(state)
        digest = make_version(0, payload).split("-", maxsplit=1)[1]
        result = await self._client.eval(
            _SAVE_SCRIPT,
            3,
            sequence_key,
            current_key,
            versions_key,
            payload,
            digest,
            str(self._ttl_seconds or 0),
        )
        return result.decode("utf-8") if isinstance(result, bytes) else str(result)

    async def history(self, user_id: str) -> list[dict[str, object]]:
        state = await self.load(user_id)
        return state.messages

    async def rollback(self, user_id: str, version: str) -> CareState:
        _, current_key, versions_key = self._keys(user_id)
        payload = await self._client.eval(
            _ROLLBACK_SCRIPT,
            2,
            versions_key,
            current_key,
            version,
        )
        if payload is None or payload is False:
            raise VersionNotFoundError(f"unknown memory version: {version}")
        return parse_state(payload)

