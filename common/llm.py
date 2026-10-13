"""LLM 客户端接口及 OpenAI 兼容实现。"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from common.config import Settings

ChatMessage = Mapping[str, str]


class LLMUnavailableError(RuntimeError):
    """模型请求不可用；上层节点应进入安全降级路径。"""


class LLMClient(Protocol):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str: ...


class OpenAICompatibleLLM:
    """OpenAI Chat Completions 兼容客户端，带超时、重试和脱敏异常。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("LLM_API_KEY cannot be empty when USE_MOCK=false")
        if not model.strip():
            raise ValueError("LLM_MODEL cannot be empty when USE_MOCK=false")
        if timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than 0")
        if max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES cannot be negative")

        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._model = model

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str:
        if not messages:
            raise ValueError("模型消息不能为空")
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),  # type: ignore[arg-type]  # SDK 的消息类型定义比公共接口更严格。
                temperature=temperature,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as exc:
            raise LLMUnavailableError("模型请求暂时不可用，已交由上层安全降级") from exc
        except Exception as exc:
            raise LLMUnavailableError("模型请求失败，请检查模型配置、网络和服务状态") from exc

        if not response.choices:
            raise LLMUnavailableError("模型返回为空，已交由上层安全降级")
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
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

