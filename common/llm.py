"""LLM 客户端接口及 OpenAI 兼容实现。"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from openai import AsyncOpenAI

from common.config import Settings

ChatMessage = Mapping[str, str]


class LLMClient(Protocol):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, *, api_key: str, base_url: str | None, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),  # type: ignore[arg-type]  # SDK 的消息类型定义比公共接口更严格。
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """按配置创建客户端；Mock 模式不需要凭据或网络。"""
    settings = settings or Settings()
    if settings.use_mock:
        from common.mock_llm import MockLLM

        return MockLLM()

    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY is required when USE_MOCK=false")
    return OpenAICompatibleLLM(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )

