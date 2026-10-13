"""用于本地开发和测试的确定性离线模型实现。"""

from collections import deque
from collections.abc import Mapping, Sequence

from common.llm import ChatMessage


class MockLLM:
    def __init__(self, responses: Sequence[str] | None = None, *, default_response: str = "mock response") -> None:
        self._responses = deque(responses or [])
        self._default_response = default_response
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(list(messages))
        if self._responses:
            return self._responses.popleft()
        return self._default_response

