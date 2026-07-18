from gearmate.config import Settings
from gearmate.llm.governed import GovernedChatModel
from gearmate.llm.openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleConfig,
)
from gearmate.llm.port import ChatModelPort
from gearmate.resilience import AsyncModelGovernor, GovernorConfig


def build_chat_model(settings: Settings) -> ChatModelPort:
    if settings.model_provider == "openai-compatible":
        inner = OpenAICompatibleChatModel(OpenAICompatibleConfig.from_settings(settings))
        governor = AsyncModelGovernor(
            GovernorConfig(
                name=f"chat:{settings.model_id or 'unknown'}",
                max_concurrency=settings.chat_model_max_concurrency,
                lane_limits={
                    "action": settings.action_model_max_concurrency,
                    "main": settings.main_model_max_concurrency,
                    "background": settings.background_model_max_concurrency,
                },
                queue_capacity=settings.chat_queue_capacity,
                queue_timeout_seconds=settings.chat_queue_timeout_seconds,
                requests_per_minute=settings.chat_requests_per_minute,
                tokens_per_minute=settings.chat_tokens_per_minute,
                max_retries=settings.chat_max_retries,
                retry_base_delay_seconds=settings.chat_retry_base_delay_seconds,
                circuit_breaker_threshold=settings.chat_circuit_breaker_threshold,
                circuit_breaker_cooldown_seconds=(
                    settings.chat_circuit_breaker_cooldown_seconds
                ),
            )
        )
        return GovernedChatModel(inner, governor)
    raise ValueError(f"Unsupported model provider: {settings.model_provider}")
