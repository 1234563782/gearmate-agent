import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import TypeVar

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

logger = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


class ModelGovernorError(RuntimeError):
    code = "MODEL_GOVERNOR_ERROR"


class ModelQueueFullError(ModelGovernorError):
    code = "MODEL_QUEUE_FULL"


class ModelQueueTimeoutError(ModelGovernorError):
    code = "MODEL_QUEUE_TIMEOUT"


class ModelCircuitOpenError(ModelGovernorError):
    code = "MODEL_CIRCUIT_OPEN"


@dataclass(frozen=True, slots=True)
class GovernorConfig:
    name: str
    max_concurrency: int
    lane_limits: Mapping[str, int]
    queue_capacity: int
    queue_timeout_seconds: float
    requests_per_minute: int
    tokens_per_minute: int
    max_retries: int
    retry_base_delay_seconds: float
    circuit_breaker_threshold: int
    circuit_breaker_cooldown_seconds: float

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if not self.lane_limits:
            raise ValueError("at least one workload lane is required")
        if any(limit < 1 for limit in self.lane_limits.values()):
            raise ValueError("workload lane concurrency must be positive")
        if any(limit > self.max_concurrency for limit in self.lane_limits.values()):
            raise ValueError("workload lane concurrency must not exceed total concurrency")
        if self.queue_capacity < 0:
            raise ValueError("queue_capacity must not be negative")
        if self.queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be positive")
        if self.requests_per_minute < 1 or self.tokens_per_minute < 1:
            raise ValueError("model rate limits must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.retry_base_delay_seconds <= 0:
            raise ValueError("retry_base_delay_seconds must be positive")
        if self.circuit_breaker_threshold < 1:
            raise ValueError("circuit_breaker_threshold must be positive")
        if self.circuit_breaker_cooldown_seconds <= 0:
            raise ValueError("circuit_breaker_cooldown_seconds must be positive")


@dataclass(frozen=True, slots=True)
class GovernorSnapshot:
    admitted: int
    active: int
    peak_active: int
    completed: int
    failed: int
    retries: int
    queue_rejected: int
    queue_timeouts: int
    circuit_rejected: int


@dataclass(slots=True)
class _GovernorMetrics:
    admitted: int = 0
    active: int = 0
    peak_active: int = 0
    completed: int = 0
    failed: int = 0
    retries: int = 0
    queue_rejected: int = 0
    queue_timeouts: int = 0
    circuit_rejected: int = 0

    def snapshot(self) -> GovernorSnapshot:
        return GovernorSnapshot(
            admitted=self.admitted,
            active=self.active,
            peak_active=self.peak_active,
            completed=self.completed,
            failed=self.failed,
            retries=self.retries,
            queue_rejected=self.queue_rejected,
            queue_timeouts=self.queue_timeouts,
            circuit_rejected=self.circuit_rejected,
        )


class _TokenBucket:
    def __init__(self, tokens_per_minute: int) -> None:
        self._capacity = float(tokens_per_minute)
        self._refill_per_second = self._capacity / 60.0
        self._tokens = self._capacity
        self._updated_at = monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: int) -> None:
        requested = min(self._capacity, float(max(1, amount)))
        while True:
            async with self._lock:
                now = monotonic()
                elapsed = max(0.0, now - self._updated_at)
                self._tokens = min(
                    self._capacity,
                    self._tokens + elapsed * self._refill_per_second,
                )
                self._updated_at = now
                if self._tokens >= requested:
                    self._tokens -= requested
                    return
                delay = (requested - self._tokens) / self._refill_per_second
            await asyncio.sleep(delay)


class _CircuitBreaker:
    def __init__(self, threshold: int, cooldown_seconds: float) -> None:
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._open_until = 0.0
        self._lock = asyncio.Lock()

    async def check(self) -> None:
        async with self._lock:
            now = monotonic()
            if self._open_until > now:
                raise ModelCircuitOpenError("Model circuit breaker is open")
            if self._open_until:
                self._open_until = 0.0
                self._failures = 0

    async def success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._open_until = 0.0

    async def failure(self) -> bool:
        async with self._lock:
            self._failures += 1
            if self._failures < self._threshold:
                return False
            self._open_until = monotonic() + self._cooldown_seconds
            return True


class AsyncModelGovernor:
    def __init__(
        self,
        config: GovernorConfig,
        *,
        retryable: Callable[[Exception], bool] | None = None,
    ) -> None:
        self._config = config
        self._total = asyncio.Semaphore(config.max_concurrency)
        self._lanes = {
            name: asyncio.Semaphore(limit) for name, limit in config.lane_limits.items()
        }
        self._requests = _TokenBucket(config.requests_per_minute)
        self._tokens = _TokenBucket(config.tokens_per_minute)
        self._circuit = _CircuitBreaker(
            config.circuit_breaker_threshold,
            config.circuit_breaker_cooldown_seconds,
        )
        self._retryable = retryable or is_retryable_provider_error
        self._admitted = 0
        self._admission_lock = asyncio.Lock()
        self._metrics = _GovernorMetrics()

    @property
    def snapshot(self) -> GovernorSnapshot:
        return self._metrics.snapshot()

    async def run(
        self,
        operation: Callable[[], Awaitable[ResultT]],
        *,
        lane: str,
        estimated_tokens: int,
    ) -> ResultT:
        if lane not in self._lanes:
            raise ValueError(f"Unknown model workload lane: {lane}")
        await self._admit()
        try:
            return await self._run_admitted(operation, lane, estimated_tokens)
        finally:
            async with self._admission_lock:
                self._admitted -= 1

    async def _run_admitted(
        self,
        operation: Callable[[], Awaitable[ResultT]],
        lane: str,
        estimated_tokens: int,
    ) -> ResultT:
        attempt = 0
        while True:
            try:
                await self._circuit.check()
            except ModelCircuitOpenError:
                self._metrics.circuit_rejected += 1
                raise
            acquired = False
            try:
                async with asyncio.timeout(self._config.queue_timeout_seconds):
                    await self._requests.acquire(1)
                    await self._tokens.acquire(estimated_tokens)
                    await self._acquire_lane(lane)
                    acquired = True
            except TimeoutError as error:
                self._metrics.queue_timeouts += 1
                raise ModelQueueTimeoutError("Timed out waiting for model capacity") from error

            try:
                result = await operation()
            except Exception as error:
                retryable = self._retryable(error)
                if retryable:
                    opened = await self._circuit.failure()
                    if opened:
                        logger.warning("Model circuit opened (governor=%s)", self._config.name)
                if not retryable or attempt >= self._config.max_retries:
                    self._metrics.failed += 1
                    raise
                delay = retry_delay_seconds(
                    error,
                    attempt=attempt,
                    base_delay=self._config.retry_base_delay_seconds,
                )
            else:
                await self._circuit.success()
                self._metrics.completed += 1
                return result
            finally:
                if acquired:
                    self._release_lane(lane)

            self._metrics.retries += 1
            logger.warning(
                "Retrying model request (governor=%s, attempt=%d, delay=%.3f)",
                self._config.name,
                attempt + 1,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1

    async def _admit(self) -> None:
        async with self._admission_lock:
            maximum = self._config.max_concurrency + self._config.queue_capacity
            if self._admitted >= maximum:
                self._metrics.queue_rejected += 1
                raise ModelQueueFullError("Model request queue is full")
            self._admitted += 1
            self._metrics.admitted += 1

    async def _acquire_lane(self, lane: str) -> None:
        lane_semaphore = self._lanes[lane]
        await lane_semaphore.acquire()
        try:
            await self._total.acquire()
        except BaseException:
            lane_semaphore.release()
            raise
        self._metrics.active += 1
        self._metrics.peak_active = max(self._metrics.peak_active, self._metrics.active)

    def _release_lane(self, lane: str) -> None:
        self._metrics.active -= 1
        self._total.release()
        self._lanes[lane].release()


def is_retryable_provider_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    ):
        return True
    status_code = (
        error.status_code
        if isinstance(error, APIStatusError)
        else getattr(error, "status_code", None)
    )
    return status_code == 429 or (isinstance(status_code, int) and status_code >= 500)


def retry_delay_seconds(error: Exception, *, attempt: int, base_delay: float) -> float:
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return max(0.0, retry_after)
    return float(base_delay * (2**attempt) + random.uniform(0.0, base_delay * 0.5))


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        return max(0.0, (target - datetime.now(UTC)).total_seconds())
