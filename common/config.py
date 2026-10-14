"""通过环境变量加载应用配置。"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    use_mock: bool = True
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    llm_model: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL"),
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS"),
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias=AliasChoices("LLM_MAX_RETRIES", "OPENAI_MAX_RETRIES"),
    )
    memory_ttl_seconds: int = Field(default=86400, gt=0, le=2592000)
    memory_max_messages: int = Field(default=16, ge=2, le=100)
    memory_max_chars: int = Field(default=6000, ge=200, le=100000)
    redis_url: str = "redis://localhost:6379/0"
    sqlite_memory_path: str | None = Field(
        default=None,
        validation_alias="CUSTOMER_CARE_SQLITE_PATH",
    )
    k8s_context: str | None = None

