from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEARMATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = (
        "postgresql+asyncpg://gearmate:replace-with-local-password@localhost:5432/gearmate"
    )
    rentflow_base_url: str = "http://localhost:8080"
    rentflow_connect_timeout_seconds: float = 5.0
    rentflow_read_timeout_seconds: float = 20.0
    tool_timeout_seconds: float = 30.0
    catalog_equipment_roles: str = (
        "action_camera,camera,capture_card,drone,laptop,lens,lighting,microphone,tripod"
    )
    jwt_public_key_path: Path | None = None
    jwt_issuer: str = "rentflow-server"
    jwt_audience: str = "rentflow-platform"
    cors_allowed_origins: str = "http://localhost:5173"

    model_provider: str = "openai-compatible"
    model_base_url: str | None = None
    model_id: str | None = None
    model_api_key: SecretStr | None = None
    model_connect_timeout_seconds: float = 5.0
    model_first_token_timeout_seconds: float = 30.0
    model_request_timeout_seconds: float = 120.0
    model_max_output_tokens: int = 4096
    run_timeout_seconds: float = 180.0
    max_model_rounds: int = 6
    max_tool_calls: int = 10
    max_tool_concurrency: int = 4
    max_tool_result_items: int = 20
    event_poll_interval_seconds: float = 0.5
    sse_heartbeat_seconds: float = 15.0
    context_history_token_budget: int = 12000
    context_summary_trigger_tokens: int = 8000
    context_summary_max_output_tokens: int = 1024
    context_recent_messages: int = 8
    context_source_message_limit: int = 100
    action_resolution_max_output_tokens: int = 256
    rental_period_extraction_max_output_tokens: int = 512
    requirements_extraction_max_output_tokens: int = 512
    rental_period_max_advance_days: int = 90

    @field_validator(
        "jwt_public_key_path",
        "model_base_url",
        "model_id",
        "model_api_key",
        mode="before",
    )
    @classmethod
    def blank_as_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "model_connect_timeout_seconds",
        "model_first_token_timeout_seconds",
        "model_request_timeout_seconds",
        "run_timeout_seconds",
        "rentflow_connect_timeout_seconds",
        "rentflow_read_timeout_seconds",
        "tool_timeout_seconds",
        "event_poll_interval_seconds",
        "sse_heartbeat_seconds",
    )
    @classmethod
    def positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout values must be positive")
        return value

    @field_validator("model_max_output_tokens")
    @classmethod
    def positive_output_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("model_max_output_tokens must be positive")
        return value

    @field_validator(
        "max_model_rounds",
        "max_tool_calls",
        "max_tool_concurrency",
        "max_tool_result_items",
        "context_history_token_budget",
        "context_summary_trigger_tokens",
        "context_summary_max_output_tokens",
        "context_recent_messages",
        "context_source_message_limit",
        "action_resolution_max_output_tokens",
        "rental_period_extraction_max_output_tokens",
        "requirements_extraction_max_output_tokens",
        "rental_period_max_advance_days",
    )
    @classmethod
    def positive_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("agent limits must be positive")
        return value

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        return tuple(
            origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()
        )

    @property
    def equipment_roles(self) -> tuple[str, ...]:
        return tuple(
            role.strip() for role in self.catalog_equipment_roles.split(",") if role.strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
