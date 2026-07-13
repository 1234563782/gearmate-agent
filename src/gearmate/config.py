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

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        return tuple(
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
