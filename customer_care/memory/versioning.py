"""记忆后端共用的序列化逻辑和版本标识生成。"""

import hashlib
import json

from customer_care.state import CareState


def serialize_state(state: CareState) -> str:
    return json.dumps(
        state.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_state(payload: str | bytes) -> CareState:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return CareState.model_validate_json(payload)


def make_version(sequence: int, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{sequence:08d}-{digest}"

