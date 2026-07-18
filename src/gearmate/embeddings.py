from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from openai import AsyncOpenAI
from pydantic import SecretStr

from gearmate.config import Settings
from gearmate.resilience import AsyncModelGovernor, GovernorConfig, GovernorSnapshot

EmbeddingWorkload = Literal["online", "refresh"]


class EmbeddingPort(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(
        self,
        texts: tuple[str, ...],
        *,
        workload: EmbeddingWorkload = "online",
    ) -> tuple[tuple[float, ...], ...]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEmbeddingConfig:
    base_url: str
    model_id: str
    api_key: SecretStr
    dimensions: int
    connect_timeout_seconds: float
    request_timeout_seconds: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleEmbeddingConfig":
        base_url = settings.embedding_base_url or settings.model_base_url
        api_key = settings.embedding_api_key or settings.model_api_key
        if base_url is None:
            raise ValueError("GEARMATE_EMBEDDING_BASE_URL is required")
        if settings.embedding_model_id is None:
            raise ValueError("GEARMATE_EMBEDDING_MODEL_ID is required")
        if api_key is None:
            raise ValueError("GEARMATE_EMBEDDING_API_KEY is required")
        return cls(
            base_url=base_url,
            model_id=settings.embedding_model_id,
            api_key=api_key,
            dimensions=settings.embedding_dimensions,
            connect_timeout_seconds=settings.model_connect_timeout_seconds,
            request_timeout_seconds=settings.model_request_timeout_seconds,
        )


class OpenAICompatibleEmbeddingModel:
    def __init__(self, config: OpenAICompatibleEmbeddingConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value(),
            timeout=httpx.Timeout(
                timeout=config.request_timeout_seconds,
                connect=config.connect_timeout_seconds,
            ),
            max_retries=0,
        )

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    async def embed(
        self,
        texts: tuple[str, ...],
        *,
        workload: EmbeddingWorkload = "online",
    ) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        response = await self._client.embeddings.create(
            model=self._config.model_id,
            input=list(texts),
            dimensions=self._config.dimensions,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings = tuple(tuple(float(value) for value in item.embedding) for item in ordered)
        if len(embeddings) != len(texts):
            raise ValueError("Embedding provider returned an unexpected result count")
        if any(len(item) != self._config.dimensions for item in embeddings):
            raise ValueError("Embedding provider returned an unexpected vector dimension")
        return embeddings

    async def close(self) -> None:
        await self._client.close()


class GovernedEmbeddingModel:
    def __init__(self, inner: EmbeddingPort, governor: AsyncModelGovernor) -> None:
        self._inner = inner
        self._governor = governor

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def governor_snapshot(self) -> GovernorSnapshot:
        return self._governor.snapshot

    async def embed(
        self,
        texts: tuple[str, ...],
        *,
        workload: EmbeddingWorkload = "online",
    ) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        return await self._governor.run(
            lambda: self._inner.embed(texts, workload=workload),
            lane=workload,
            estimated_tokens=estimate_embedding_tokens(texts),
        )

    async def close(self) -> None:
        await self._inner.close()


def estimate_embedding_tokens(texts: tuple[str, ...]) -> int:
    return max(1, (sum(len(text) for text in texts) + 3) // 4)


def build_embedding_model(settings: Settings) -> EmbeddingPort | None:
    if not settings.semantic_search_enabled:
        return None
    inner = OpenAICompatibleEmbeddingModel(
        OpenAICompatibleEmbeddingConfig.from_settings(settings)
    )
    governor = AsyncModelGovernor(
        GovernorConfig(
            name=f"embedding:{settings.embedding_model_id or 'unknown'}",
            max_concurrency=settings.embedding_max_concurrency,
            lane_limits={
                "online": settings.embedding_online_max_concurrency,
                "refresh": settings.embedding_refresh_max_concurrency,
            },
            queue_capacity=settings.embedding_queue_capacity,
            queue_timeout_seconds=settings.embedding_queue_timeout_seconds,
            requests_per_minute=settings.embedding_requests_per_minute,
            tokens_per_minute=settings.embedding_tokens_per_minute,
            max_retries=settings.embedding_max_retries,
            retry_base_delay_seconds=settings.embedding_retry_base_delay_seconds,
            circuit_breaker_threshold=settings.embedding_circuit_breaker_threshold,
            circuit_breaker_cooldown_seconds=(
                settings.embedding_circuit_breaker_cooldown_seconds
            ),
        )
    )
    return GovernedEmbeddingModel(inner, governor)
