import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from gearmate.config import Settings
from gearmate.embeddings import GovernedEmbeddingModel
from gearmate.llm.governed import GovernedChatModel
from gearmate.llm.types import ModelMessage, ModelRequest, ModelResponse
from gearmate.resilience import (
    AsyncModelGovernor,
    GovernorConfig,
    ModelCircuitOpenError,
    ModelQueueFullError,
    ModelQueueTimeoutError,
)
from gearmate.resilience.governor import retry_delay_seconds


def governor_config(
    *,
    max_concurrency: int = 2,
    lane_limits: dict[str, int] | None = None,
    queue_capacity: int = 4,
    queue_timeout_seconds: float = 0.2,
    max_retries: int = 0,
    circuit_breaker_threshold: int = 5,
    requests_per_minute: int = 1_000_000,
) -> GovernorConfig:
    return GovernorConfig(
        name="test",
        max_concurrency=max_concurrency,
        lane_limits=lane_limits or {"action": max_concurrency},
        queue_capacity=queue_capacity,
        queue_timeout_seconds=queue_timeout_seconds,
        requests_per_minute=requests_per_minute,
        tokens_per_minute=1_000_000,
        max_retries=max_retries,
        retry_base_delay_seconds=0.01,
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_cooldown_seconds=1,
    )


async def wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(1):
        while not predicate():  # noqa: ASYNC110 - observes scheduler-controlled test state
            await asyncio.sleep(0)


async def test_governor_enforces_total_concurrency() -> None:
    governor = AsyncModelGovernor(governor_config())
    release = asyncio.Event()
    active = 0
    peak = 0

    async def operation() -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await release.wait()
            return "ok"
        finally:
            active -= 1

    tasks = [
        asyncio.create_task(governor.run(operation, lane="action", estimated_tokens=1))
        for _ in range(3)
    ]
    await wait_until(lambda: active == 2)

    assert peak == 2
    assert governor.snapshot.peak_active == 2

    release.set()
    assert await asyncio.gather(*tasks) == ["ok", "ok", "ok"]
    assert governor.snapshot.active == 0


async def test_governor_isolates_workload_lane_concurrency() -> None:
    governor = AsyncModelGovernor(
        governor_config(
            max_concurrency=3,
            lane_limits={"action": 1, "main": 2},
        )
    )
    release = asyncio.Event()
    active = {"action": 0, "main": 0}
    peak = {"action": 0, "main": 0}

    def operation(lane: str) -> Callable[[], Awaitable[str]]:
        async def run() -> str:
            active[lane] += 1
            peak[lane] = max(peak[lane], active[lane])
            try:
                await release.wait()
                return lane
            finally:
                active[lane] -= 1

        return run

    tasks = [
        asyncio.create_task(governor.run(operation("action"), lane="action", estimated_tokens=1)),
        asyncio.create_task(governor.run(operation("action"), lane="action", estimated_tokens=1)),
        asyncio.create_task(governor.run(operation("main"), lane="main", estimated_tokens=1)),
        asyncio.create_task(governor.run(operation("main"), lane="main", estimated_tokens=1)),
    ]
    await wait_until(lambda: active == {"action": 1, "main": 2})

    assert peak == {"action": 1, "main": 2}

    release.set()
    await asyncio.gather(*tasks)


async def test_governor_rejects_when_admission_queue_is_full() -> None:
    governor = AsyncModelGovernor(
        governor_config(max_concurrency=1, queue_capacity=1, queue_timeout_seconds=1)
    )
    release = asyncio.Event()

    async def blocked() -> str:
        await release.wait()
        return "ok"

    first = asyncio.create_task(governor.run(blocked, lane="action", estimated_tokens=1))
    await wait_until(lambda: governor.snapshot.active == 1)
    second = asyncio.create_task(governor.run(blocked, lane="action", estimated_tokens=1))
    await wait_until(lambda: governor.snapshot.admitted == 2)

    with pytest.raises(ModelQueueFullError):
        await governor.run(blocked, lane="action", estimated_tokens=1)

    assert governor.snapshot.queue_rejected == 1
    release.set()
    await asyncio.gather(first, second)


async def test_governor_times_out_waiting_for_capacity() -> None:
    governor = AsyncModelGovernor(
        governor_config(max_concurrency=1, queue_capacity=1, queue_timeout_seconds=0.02)
    )
    release = asyncio.Event()

    async def blocked() -> str:
        await release.wait()
        return "ok"

    first = asyncio.create_task(governor.run(blocked, lane="action", estimated_tokens=1))
    await wait_until(lambda: governor.snapshot.active == 1)

    with pytest.raises(ModelQueueTimeoutError):
        await governor.run(blocked, lane="action", estimated_tokens=1)

    assert governor.snapshot.queue_timeouts == 1
    release.set()
    await first


async def test_governor_bounds_rate_limit_wait_by_queue_timeout() -> None:
    governor = AsyncModelGovernor(
        governor_config(
            max_concurrency=1,
            queue_timeout_seconds=0.02,
            requests_per_minute=1,
        )
    )

    async def operation() -> str:
        return "ok"

    assert await governor.run(operation, lane="action", estimated_tokens=1) == "ok"
    with pytest.raises(ModelQueueTimeoutError):
        await governor.run(operation, lane="action", estimated_tokens=1)


def test_retry_delay_honors_provider_retry_after() -> None:
    error = RuntimeError("limited")
    error.response = SimpleNamespace(headers={"retry-after": "3.5"})  # type: ignore[attr-defined]

    assert retry_delay_seconds(error, attempt=3, base_delay=1) == 3.5


async def test_governor_retries_retryable_failure() -> None:
    class RetryableError(RuntimeError):
        pass

    governor = AsyncModelGovernor(
        governor_config(max_retries=1),
        retryable=lambda error: isinstance(error, RetryableError),
    )
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RetryableError
        return "ok"

    assert await governor.run(flaky, lane="action", estimated_tokens=1) == "ok"
    assert attempts == 2
    assert governor.snapshot.retries == 1
    assert governor.snapshot.completed == 1


async def test_governor_does_not_retry_non_retryable_failure() -> None:
    governor = AsyncModelGovernor(governor_config(max_retries=2), retryable=lambda _: False)
    attempts = 0

    async def rejected() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        await governor.run(rejected, lane="action", estimated_tokens=1)

    assert attempts == 1
    assert governor.snapshot.retries == 0
    assert governor.snapshot.failed == 1


async def test_governor_releases_semaphore_during_retry_backoff() -> None:
    class RetryableError(RuntimeError):
        pass

    governor = AsyncModelGovernor(
        governor_config(max_concurrency=1, max_retries=1),
        retryable=lambda error: isinstance(error, RetryableError),
    )
    first_failed = asyncio.Event()
    sequence: list[str] = []
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            sequence.append("failed")
            first_failed.set()
            raise RetryableError
        sequence.append("retried")
        return "first"

    async def other() -> str:
        sequence.append("other")
        return "second"

    first = asyncio.create_task(governor.run(flaky, lane="action", estimated_tokens=1))
    await first_failed.wait()
    second = asyncio.create_task(governor.run(other, lane="action", estimated_tokens=1))

    assert await second == "second"
    assert await first == "first"
    assert sequence == ["failed", "other", "retried"]


async def test_governor_opens_circuit_after_provider_failures() -> None:
    class RetryableError(RuntimeError):
        pass

    governor = AsyncModelGovernor(
        governor_config(circuit_breaker_threshold=1),
        retryable=lambda error: isinstance(error, RetryableError),
    )

    async def rejected() -> str:
        raise RetryableError

    with pytest.raises(RetryableError):
        await governor.run(rejected, lane="action", estimated_tokens=1)
    with pytest.raises(ModelCircuitOpenError):
        await governor.run(rejected, lane="action", estimated_tokens=1)

    assert governor.snapshot.failed == 1
    assert governor.snapshot.circuit_rejected == 1


class RecordingGovernor:
    def __init__(self) -> None:
        self.lanes: list[str] = []
        self.estimated_tokens: list[int] = []

    async def run(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        lane: str,
        estimated_tokens: int,
    ) -> Any:
        self.lanes.append(lane)
        self.estimated_tokens.append(estimated_tokens)
        return await operation()


class FakeChatModel:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(text=request.workload, finish_reason="stop")

    async def close(self) -> None:
        return None


async def test_chat_wrapper_routes_request_workload() -> None:
    governor = RecordingGovernor()
    model = GovernedChatModel(FakeChatModel(), governor)  # type: ignore[arg-type]
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="search"),),
        max_output_tokens=32,
        workload="action",
    )

    response = await model.complete(request)

    assert response.text == "action"
    assert governor.lanes == ["action"]
    assert governor.estimated_tokens[0] >= 33


class FakeEmbeddingModel:
    model_id = "fake"
    dimensions = 2

    def __init__(self) -> None:
        self.workloads: list[str] = []

    async def embed(
        self,
        texts: tuple[str, ...],
        *,
        workload: str = "online",
    ) -> tuple[tuple[float, ...], ...]:
        self.workloads.append(workload)
        return tuple((1.0, 0.0) for _ in texts)

    async def close(self) -> None:
        return None


async def test_embedding_wrapper_routes_online_and_refresh_workloads() -> None:
    governor = RecordingGovernor()
    inner = FakeEmbeddingModel()
    model = GovernedEmbeddingModel(inner, governor)  # type: ignore[arg-type]

    await model.embed(("query",))
    await model.embed(("catalog",), workload="refresh")

    assert governor.lanes == ["online", "refresh"]
    assert inner.workloads == ["online", "refresh"]


def test_settings_rejects_lane_concurrency_above_total() -> None:
    with pytest.raises(ValidationError, match="chat workload concurrency"):
        Settings(
            _env_file=None,
            chat_model_max_concurrency=1,
            action_model_max_concurrency=2,
        )


def test_governor_config_rejects_invalid_rate_limit() -> None:
    with pytest.raises(ValueError, match="rate limits"):
        GovernorConfig(
            name="invalid",
            max_concurrency=1,
            lane_limits={"main": 1},
            queue_capacity=0,
            queue_timeout_seconds=1,
            requests_per_minute=0,
            tokens_per_minute=1,
            max_retries=0,
            retry_base_delay_seconds=1,
            circuit_breaker_threshold=1,
            circuit_breaker_cooldown_seconds=1,
        )
