import asyncio

import pytest

from common.config import Settings
from common.llm import LLMUnavailableError, OpenAICompatibleLLM, create_llm_client
from common.mock_llm import MockLLM


def test_mock_llm_runs_without_credentials_or_network() -> None:
    settings = Settings(use_mock=True, llm_api_key=None)
    llm = create_llm_client(settings)

    result = asyncio.run(llm.complete([{"role": "user", "content": "refund status"}]))

    assert isinstance(llm, MockLLM)
    assert result == "mock response"
    assert len(llm.calls) == 1


def test_real_llm_requires_api_key() -> None:
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        create_llm_client(Settings(use_mock=False, llm_api_key=None))


def test_settings_accepts_openai_compatible_environment_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "GLM-5")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "18")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "1")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key == "test-key"
    assert settings.llm_base_url == "https://example.test/v1"
    assert settings.llm_model == "GLM-5"
    assert settings.llm_timeout_seconds == 18
    assert settings.llm_max_retries == 1


def test_llm_environment_names_take_precedence_over_openai_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_MODEL", "llm-model")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key == "llm-key"
    assert settings.llm_model == "llm-model"


def test_real_llm_passes_timeout_and_retry_settings_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("common.llm.AsyncOpenAI", FakeAsyncOpenAI)

    client = OpenAICompatibleLLM(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        timeout_seconds=12,
        max_retries=3,
    )

    assert client is not None
    assert captured == {
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "timeout": 12,
        "max_retries": 3,
    }


def test_real_llm_converts_provider_failure_to_sanitized_error() -> None:
    class FailingCompletions:
        async def create(self, **_: object) -> object:
            raise RuntimeError("secret-api-key must not leak")

    class FakeClient:
        class Chat:
            completions = FailingCompletions()

        chat = Chat()

    llm = OpenAICompatibleLLM(
        api_key="test-key",
        base_url=None,
        model="test-model",
        client=FakeClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(LLMUnavailableError, match="模型请求失败") as error:
        asyncio.run(llm.complete([{"role": "user", "content": "hello"}]))
    assert "secret-api-key" not in str(error.value)

