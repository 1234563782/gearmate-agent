from datetime import UTC, datetime

import httpx

from gearmate.agent.service import RunCoordinator
from gearmate.config import Settings
from gearmate.llm.types import ModelMessage, ModelRequest, ModelResponse, ModelToolCall
from gearmate.memory import ConversationContext
from gearmate.prompts.loader import RenderedPrompt
from gearmate.user_memory import UserMemoryContext


class FakeRepository:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.finalized: dict[str, object] | None = None

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self.events.append((event_type, payload))

    async def finalize_run(self, run_id: str, **kwargs: object) -> None:
        self.finalized = kwargs


class FakeConversationMemory:
    def __init__(self, message: str) -> None:
        self.message = message
        self.summarized = False

    async def build_context(self, conversation_id: str) -> ConversationContext:
        now = datetime(2026, 7, 20, tzinfo=UTC)
        return ConversationContext(
            messages=(ModelMessage(role="user", content=self.message),),
            rental_period=None,
            rental_requirements=None,
            pending_product_search=None,
            pending_rental_action=None,
            recent_product_search=None,
            timezone="Asia/Shanghai",
            now_utc=now,
            now_local=now,
        )

    async def maybe_summarize(self, conversation_id: str, model: object) -> None:
        self.summarized = True


class FakeUserMemory:
    def __init__(self) -> None:
        self.built_for: str | None = None
        self.extracted: dict[str, object] | None = None

    async def build_context(self, user_id: str) -> UserMemoryContext:
        self.built_for = user_id

        class SoftContext(UserMemoryContext):
            def prompt_context(self) -> str:
                return "User long-term preferences: preferred_brand: Sony"

        return SoftContext()

    async def extract_and_store(self, **kwargs: object) -> None:
        self.extracted = kwargs


class SequenceModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=(
                    ModelToolCall(
                        id="action-1",
                        name="resolve_agent_action",
                        arguments={"action": "chat"},
                    ),
                ),
            )
        return ModelResponse(text="Hello", finish_reason="stop")

    async def close(self) -> None:
        return None


async def test_run_reads_and_extracts_user_memory_with_authenticated_user_id() -> None:
    message = "Hello"
    repository = FakeRepository()
    user_memory = FakeUserMemory()
    model = SequenceModel()
    settings = Settings(_env_file=None, user_memory_enabled=True, user_memory_mode="active")
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
            user_memory=user_memory,  # type: ignore[arg-type]
        )
        coordinator._memory = FakeConversationMemory(message)  # type: ignore[assignment]
        coordinator._model = model

        await coordinator._execute(
            run_id="run-1",
            conversation_id="conversation-1",
            user_id="user-1",
            access_token="token",
            message=message,
            rental_period=None,
        )

    assert user_memory.built_for == "user-1"
    assert user_memory.extracted is not None
    assert user_memory.extracted["user_id"] == "user-1"
    assert repository.finalized is not None
    assert any(
        "preferred_brand: Sony" in item.content
        for request in model.requests
        for item in request.messages
        if item.role == "system"
    )
