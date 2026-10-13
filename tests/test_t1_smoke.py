import asyncio

import pytest

from common.config import Settings
from common.llm import create_llm_client
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

