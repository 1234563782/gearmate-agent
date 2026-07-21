from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from gearmate.agent.service import RunCoordinator
from gearmate.config import Settings
from gearmate.llm.types import ModelMessage, ModelRequest, ModelResponse
from gearmate.memory import ConversationContext
from gearmate.prompts.loader import RenderedPrompt
from gearmate.resilience import (
    ModelCircuitOpenError,
    ModelQueueFullError,
    ModelQueueTimeoutError,
)


class FailingModel:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise self._error

    async def close(self) -> None:
        return None


class FakeMemory:
    async def build_context(self, conversation_id: str) -> ConversationContext:
        now = datetime(2026, 7, 18, 12, tzinfo=UTC)
        return ConversationContext(
            messages=(ModelMessage(role="user", content="search for a camera"),),
            pending_product_search=None,
            recent_product_search=None,
            timezone="Asia/Shanghai",
            now_utc=now,
            now_local=now.astimezone(ZoneInfo("Asia/Shanghai")),
        )


class FakeRepository:
    def __init__(self) -> None:
        self.finalized: dict[str, object] | None = None

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        return None

    async def finalize_run(self, run_id: str, **kwargs: object) -> None:
        self.finalized = kwargs


@pytest.mark.parametrize(
    ("error", "expected_stop_reason", "expected_error_code"),
    [
        (ModelQueueFullError("full"), "MODEL_BUSY", "MODEL_QUEUE_FULL"),
        (ModelQueueTimeoutError("timeout"), "MODEL_BUSY", "MODEL_QUEUE_TIMEOUT"),
        (ModelCircuitOpenError("open"), "MODEL_UNAVAILABLE", "MODEL_CIRCUIT_OPEN"),
    ],
)
async def test_run_persists_specific_model_capacity_error(
    error: Exception,
    expected_stop_reason: str,
    expected_error_code: str,
) -> None:
    repository = FakeRepository()
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            Settings(_env_file=None),
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._memory = FakeMemory()  # type: ignore[assignment]
        coordinator._model = FailingModel(error)

        await coordinator._execute(
            run_id="run-1",
            conversation_id="conversation-1",
            user_id="user-1",
            access_token="token",
            message="search for a camera",
        )

    assert repository.finalized is not None
    assert repository.finalized["status"] == "FAILED"
    assert repository.finalized["stop_reason"] == expected_stop_reason
    assert repository.finalized["error_code"] == expected_error_code
