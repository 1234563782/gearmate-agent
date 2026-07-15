from datetime import UTC, datetime

import httpx

from gearmate.agent.service import RunCoordinator
from gearmate.config import Settings
from gearmate.llm.types import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
)
from gearmate.memory import ConversationMessageMemory
from gearmate.prompts.loader import RenderedPrompt
from gearmate.tools.contracts import RentalPeriodInput


class FakeRepository:
    def __init__(self, message: str) -> None:
        self.message = message
        self.remembered: RentalPeriodInput | None = None
        self.events: list[tuple[str, dict[str, object]]] = []
        self.finalized: dict[str, object] | None = None

    async def conversation_timezone(self, conversation_id: str) -> str:
        return "Asia/Shanghai"

    async def conversation_state(self, conversation_id: str) -> None:
        return None

    async def latest_conversation_summary(self, conversation_id: str) -> None:
        return None

    async def recent_conversation_messages(
        self,
        conversation_id: str,
        limit: int,
        after_event_id: str | None = None,
    ) -> list[ConversationMessageMemory]:
        return [
            ConversationMessageMemory(
                event_id="01J00000000000000000000001",
                role="user",
                content=self.message,
                created_at=datetime.now(UTC),
            )
        ]

    async def conversation_messages_after(
        self, conversation_id: str, after_event_id: str | None, limit: int
    ) -> list[ConversationMessageMemory]:
        return []

    async def upsert_conversation_rental_period(
        self, conversation_id: str, rental_period: RentalPeriodInput
    ) -> None:
        self.remembered = rental_period

    async def save_conversation_summary(self, **kwargs: object) -> None:
        raise AssertionError("summary should not run for one message")

    async def append_event(
        self, run_id: str, event_type: str, payload: dict[str, object]
    ) -> None:
        self.events.append((event_type, payload))

    async def finalize_run(self, run_id: str, **kwargs: object) -> None:
        self.finalized = kwargs


class SequenceModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def close(self) -> None:
        return None


async def test_run_resolves_and_remembers_natural_language_period() -> None:
    message = "2035 年 7 月 20 日 9 点到 7 月 22 日 18 点租一台相机"
    repository = FakeRepository(message)
    period_arguments = {
        "startAt": "2035-07-20T09:00:00+08:00",
        "endAt": "2035-07-22T18:00:00+08:00",
    }
    model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=50, output_tokens=10),
                tool_calls=(
                    ModelToolCall(
                        id="extract-1",
                        name="set_rental_period",
                        arguments=period_arguments,
                    ),
                ),
            ),
            ModelResponse(
                text="租期已经确认。",
                finish_reason="stop",
                usage=ModelUsage(input_tokens=30, output_tokens=7),
            ),
        ]
    )
    settings = Settings(_env_file=None, context_summary_trigger_tokens=8000)
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = model

        await coordinator._execute(
            run_id="run-1",
            conversation_id="conversation-1",
            access_token="token",
            message=message,
            rental_period=None,
        )

    assert repository.remembered == RentalPeriodInput.model_validate(period_arguments)
    assert repository.finalized is not None
    assert repository.finalized["status"] == "COMPLETED"
    assert repository.finalized["input_tokens"] == 80
    assert repository.finalized["output_tokens"] == 17
    assert repository.finalized["model_rounds"] == 2
    assert any(event_type == "assistant.completed" for event_type, _ in repository.events)
    assert len(model.requests) == 2
