from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
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
        "action_camera,camera,capture_card,drone,laptop,lens,lighting,microphone,projector,"
        "smartphone,tripod"
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
    chat_model_max_concurrency: int = 6
    action_model_max_concurrency: int = 4
    main_model_max_concurrency: int = 3
    background_model_max_concurrency: int = 1
    chat_queue_capacity: int = 40
    chat_queue_timeout_seconds: float = 8.0
    chat_requests_per_minute: int = 200
    chat_tokens_per_minute: int = 500000
    chat_max_retries: int = 2
    chat_retry_base_delay_seconds: float = 0.5
    chat_circuit_breaker_threshold: int = 5
    chat_circuit_breaker_cooldown_seconds: float = 30.0
    semantic_search_enabled: bool = False
    embedding_base_url: str | None = None
    embedding_model_id: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 32
    embedding_max_concurrency: int = 2
    embedding_online_max_concurrency: int = 2
    embedding_refresh_max_concurrency: int = 1
    embedding_queue_capacity: int = 100
    embedding_queue_timeout_seconds: float = 15.0
    embedding_requests_per_minute: int = 300
    embedding_tokens_per_minute: int = 1000000
    embedding_max_retries: int = 3
    embedding_retry_base_delay_seconds: float = 0.5
    embedding_circuit_breaker_threshold: int = 5
    embedding_circuit_breaker_cooldown_seconds: float = 60.0
    semantic_search_top_k: int = 20
    semantic_search_min_score: float = 0.65
    semantic_vector_weight: float = 0.85
    semantic_lexical_weight: float = 0.15
    catalog_sync_on_startup: bool = False
    catalog_sync_interval_seconds: float = 900.0
    catalog_sync_retry_seconds: float = 30.0
    conversation_retention_hours: float = 24.0
    conversation_cleanup_interval_seconds: float = 3600.0
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
    user_memory_enabled: bool = False
    user_memory_mode: Literal["off", "shadow", "active"] = "off"
    user_memory_retrieval_limit: int = 10
    user_memory_max_items: int = 100
    user_memory_min_confidence: float = 0.85
    user_memory_retention_days: int = 180
    user_memory_extraction_max_output_tokens: int = 256
    action_resolution_max_output_tokens: int = 256
    rental_period_extraction_max_output_tokens: int = 512
    requirements_extraction_max_output_tokens: int = 512
    rental_period_max_advance_days: int = 90
    intent_pre_router_mode: Literal["off", "shadow", "enforce"] = "off"
    intent_pre_router_pure_social_enabled: bool = True
    intent_pre_router_pending_confirmation_enabled: bool = True
    intent_pre_router_pending_date_enabled: bool = True

    @field_validator(
        "jwt_public_key_path",
        "model_base_url",
        "model_id",
        "model_api_key",
        "embedding_base_url",
        "embedding_model_id",
        "embedding_api_key",
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
        "catalog_sync_interval_seconds",
        "catalog_sync_retry_seconds",
        "conversation_retention_hours",
        "conversation_cleanup_interval_seconds",
        "chat_queue_timeout_seconds",
        "chat_retry_base_delay_seconds",
        "chat_circuit_breaker_cooldown_seconds",
        "embedding_queue_timeout_seconds",
        "embedding_retry_base_delay_seconds",
        "embedding_circuit_breaker_cooldown_seconds",
    )
    @classmethod
    def positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout values must be positive")
        return value

    @field_validator(
        "semantic_search_min_score",
        "semantic_vector_weight",
        "semantic_lexical_weight",
        "user_memory_min_confidence",
    )
    @classmethod
    def unit_interval(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("semantic search scores and weights must be between 0 and 1")
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
        "user_memory_retrieval_limit",
        "user_memory_max_items",
        "user_memory_retention_days",
        "user_memory_extraction_max_output_tokens",
        "action_resolution_max_output_tokens",
        "rental_period_extraction_max_output_tokens",
        "requirements_extraction_max_output_tokens",
        "rental_period_max_advance_days",
        "embedding_dimensions",
        "embedding_batch_size",
        "semantic_search_top_k",
        "chat_model_max_concurrency",
        "action_model_max_concurrency",
        "main_model_max_concurrency",
        "background_model_max_concurrency",
        "chat_requests_per_minute",
        "chat_tokens_per_minute",
        "chat_circuit_breaker_threshold",
        "embedding_max_concurrency",
        "embedding_online_max_concurrency",
        "embedding_refresh_max_concurrency",
        "embedding_requests_per_minute",
        "embedding_tokens_per_minute",
        "embedding_circuit_breaker_threshold",
    )
    @classmethod
    def positive_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("agent limits must be positive")
        return value

    @field_validator(
        "chat_queue_capacity",
        "chat_max_retries",
        "embedding_queue_capacity",
        "embedding_max_retries",
    )
    @classmethod
    def non_negative_limit(cls, value: int) -> int:
        if value < 0:
            raise ValueError("queue capacities and retry counts must not be negative")
        return value

    @model_validator(mode="after")
    def workload_concurrency_does_not_exceed_total(self) -> "Settings":
        if max(
            self.action_model_max_concurrency,
            self.main_model_max_concurrency,
            self.background_model_max_concurrency,
        ) > self.chat_model_max_concurrency:
            raise ValueError("chat workload concurrency must not exceed total concurrency")
        if max(
            self.embedding_online_max_concurrency,
            self.embedding_refresh_max_concurrency,
        ) > self.embedding_max_concurrency:
            raise ValueError("embedding workload concurrency must not exceed total concurrency")
        return self

    @field_validator("embedding_dimensions")
    @classmethod
    def supported_embedding_dimensions(cls, value: int) -> int:
        if value != 1024:
            raise ValueError("embedding_dimensions must match the vector(1024) database column")
        return value

    @model_validator(mode="after")
    def semantic_weights_sum_to_one(self) -> "Settings":
        total = self.semantic_vector_weight + self.semantic_lexical_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("semantic vector and lexical weights must sum to 1")
        return self

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
