"""本地可验证的请求观测和成本估算。"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger("customer_care")


@dataclass
class RequestMetrics:
    """一轮请求的节点、耗时、错误和成本估算。"""

    request_id: str
    started_at: float = field(default_factory=time.perf_counter)
    nodes: list[str] = field(default_factory=list)
    llm_calls: int = 0
    input_chars: int = 0
    output_chars: int = 0
    errors: list[str] = field(default_factory=list)

    def node(self, name: str) -> None:
        self.nodes.append(name)

    def llm(self, input_chars: int, output_chars: int) -> None:
        self.llm_calls += 1
        self.input_chars += input_chars
        self.output_chars += output_chars

    def error(self, message: str) -> None:
        self.errors.append(message[:200])

    def finish(self) -> dict[str, Any]:
        elapsed_ms = round((time.perf_counter() - self.started_at) * 1000, 2)
        estimated_tokens = (self.input_chars + self.output_chars) / 4
        return {
            "request_id": self.request_id,
            "elapsed_ms": elapsed_ms,
            "nodes": self.nodes,
            "llm_calls": self.llm_calls,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "estimated_tokens": round(estimated_tokens, 1),
            "errors": self.errors,
        }


def log_event(event: str, **fields: Any) -> None:
    """输出不包含消息正文和密钥的结构化日志。"""
    _LOGGER.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))
