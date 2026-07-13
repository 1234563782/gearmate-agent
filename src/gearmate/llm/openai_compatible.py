from dataclasses import dataclass

from pydantic import SecretStr

from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse


class ModelConfigurationError(ValueError):
    pass


class ModelInvocationDisabledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    base_url: str
    model_id: str
    api_key: SecretStr
    connect_timeout_seconds: float
    first_token_timeout_seconds: float
    request_timeout_seconds: float
    max_output_tokens: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleConfig":
        if settings.model_base_url is None:
            raise ModelConfigurationError("GEARMATE_MODEL_BASE_URL is required")
        if settings.model_id is None:
            raise ModelConfigurationError("GEARMATE_MODEL_ID is required")
        if settings.model_api_key is None:
            raise ModelConfigurationError("GEARMATE_MODEL_API_KEY is required")
        return cls(
            base_url=settings.model_base_url,
            model_id=settings.model_id,
            api_key=settings.model_api_key,
            connect_timeout_seconds=settings.model_connect_timeout_seconds,
            first_token_timeout_seconds=settings.model_first_token_timeout_seconds,
            request_timeout_seconds=settings.model_request_timeout_seconds,
            max_output_tokens=settings.model_max_output_tokens,
        )


class OpenAICompatibleChatModel:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self._config = config

    @property
    def config(self) -> OpenAICompatibleConfig:
        return self._config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise ModelInvocationDisabledError(
            "Model invocation is disabled until the model ADR and graph workflow are approved"
        )
