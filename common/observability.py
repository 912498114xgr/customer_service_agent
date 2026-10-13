"""本地可验证的请求观测和成本估算。"""

import json
import logging
import time
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from common.llm import ChatMessage, LLMClient

_LOGGER = logging.getLogger("customer_care")
_ACTIVE_REQUEST: ContextVar["RequestMetrics | None"] = ContextVar("active_request_metrics", default=None)
_ACTIVE_NODE: ContextVar[str] = ContextVar("active_observation_node", default="unknown")


@dataclass
class RequestMetrics:
    """一轮请求的节点、耗时、错误和成本估算。"""

    request_id: str
    started_at: float = field(default_factory=time.perf_counter)
    nodes: list[str] = field(default_factory=list)
    llm_calls: int = 0
    input_chars: int = 0
    output_chars: int = 0
    response_chars: int = 0
    llm_elapsed_ms: float = 0.0
    llm_errors: int = 0
    llm_spans: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def node(self, name: str) -> None:
        self.nodes.append(name)

    def llm(
        self,
        input_chars: int,
        output_chars: int,
        elapsed_ms: float,
        node: str,
        error_type: str | None = None,
    ) -> None:
        self.llm_calls += 1
        self.input_chars += input_chars
        self.output_chars += output_chars
        self.llm_elapsed_ms = round(self.llm_elapsed_ms + elapsed_ms, 2)
        span = {
            "node": node,
            "elapsed_ms": elapsed_ms,
            "input_chars": input_chars,
            "output_chars": output_chars,
            "error_type": error_type,
        }
        self.llm_spans.append(span)
        if error_type:
            self.llm_errors += 1
            self.errors.append(f"{node}: {error_type}")

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
            "response_chars": self.response_chars,
            "llm_elapsed_ms": round(self.llm_elapsed_ms, 2),
            "llm_errors": self.llm_errors,
            "llm_spans": self.llm_spans,
            "estimated_tokens": round(estimated_tokens, 1),
            "token_estimate_method": "字符数 / 4，仅为粗略估算",
            "errors": self.errors,
        }


@contextmanager
def observe_request(metrics: RequestMetrics):
    """将指标绑定到当前异步任务，避免并发请求互相污染。"""
    token: Token[RequestMetrics | None] = _ACTIVE_REQUEST.set(metrics)
    try:
        yield
    finally:
        _ACTIVE_REQUEST.reset(token)


@contextmanager
def observe_node(name: str):
    """标记当前模型调用所属图节点。"""
    token: Token[str] = _ACTIVE_NODE.set(name)
    try:
        yield
    finally:
        _ACTIVE_NODE.reset(token)


class ObservedLLM:
    """记录逻辑模型调用耗时和字符量，不保存提示词或回复正文。"""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    async def complete(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> str:
        metrics = _ACTIVE_REQUEST.get()
        if metrics is None:
            return await self._inner.complete(messages, temperature=temperature)

        input_chars = sum(len(str(message.get("content", ""))) for message in messages)
        started_at = time.perf_counter()
        try:
            response = await self._inner.complete(messages, temperature=temperature)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            metrics.llm(input_chars, 0, elapsed_ms, _ACTIVE_NODE.get(), type(exc).__name__)
            raise
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        metrics.llm(input_chars, len(response), elapsed_ms, _ACTIVE_NODE.get())
        return response


def log_event(event: str, **fields: Any) -> None:
    """输出不包含消息正文和密钥的结构化日志。"""
    _LOGGER.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))
