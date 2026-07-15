from datetime import UTC, datetime, timedelta

from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelUsage
from gearmate.memory import (
    SUMMARY_SYSTEM_PROMPT,
    ConversationMemoryService,
    ConversationMessageMemory,
    ConversationStateMemory,
    ConversationSummaryMemory,
)
from gearmate.tools.contracts import RentalPeriodInput


class FakeRepository:
    def __init__(self) -> None:
        self.state: ConversationStateMemory | None = None
        self.summary: ConversationSummaryMemory | None = None
        self.recent: list[ConversationMessageMemory] = []
        self.after: list[ConversationMessageMemory] = []
        self.remembered: tuple[str, RentalPeriodInput] | None = None
        self.saved_summary: dict[str, object] | None = None
        self.recent_after_event_id: str | None = None

    async def conversation_timezone(self, conversation_id: str) -> str:
        return "Asia/Shanghai"

    async def upsert_conversation_rental_period(
        self, conversation_id: str, rental_period: RentalPeriodInput
    ) -> None:
        self.remembered = (conversation_id, rental_period)

    async def conversation_state(
        self, conversation_id: str
    ) -> ConversationStateMemory | None:
        return self.state

    async def latest_conversation_summary(
        self, conversation_id: str
    ) -> ConversationSummaryMemory | None:
        return self.summary

    async def recent_conversation_messages(
        self,
        conversation_id: str,
        limit: int,
        after_event_id: str | None = None,
    ) -> list[ConversationMessageMemory]:
        self.recent_after_event_id = after_event_id
        return self.recent[-limit:]

    async def conversation_messages_after(
        self, conversation_id: str, after_event_id: str | None, limit: int
    ) -> list[ConversationMessageMemory]:
        return self.after[:limit]

    async def save_conversation_summary(
        self,
        conversation_id: str,
        content: str,
        through_event_id: str,
        source_message_count: int,
        estimated_tokens: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.saved_summary = {
            "conversation_id": conversation_id,
            "content": content,
            "through_event_id": through_event_id,
            "source_message_count": source_message_count,
            "estimated_tokens": estimated_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


class FakeModel:
    def __init__(self, text: str = "用户准备公司直播, 租期尚未确认。") -> None:
        self.text = text
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text=self.text,
            finish_reason="stop",
            usage=ModelUsage(input_tokens=120, output_tokens=18),
        )

    async def close(self) -> None:
        return None


def settings(**overrides: int) -> Settings:
    return Settings(
        _env_file=None,
        context_history_token_budget=overrides.get("history_budget", 12000),
        context_summary_trigger_tokens=overrides.get("summary_trigger", 8000),
        context_summary_max_output_tokens=overrides.get("summary_output", 256),
        context_recent_messages=overrides.get("recent_messages", 2),
        context_source_message_limit=overrides.get("source_limit", 100),
    )


def message(index: int, role: str, content: str) -> ConversationMessageMemory:
    return ConversationMessageMemory(
        event_id=f"01J000000000000000000000{index:02d}",
        role="user" if role == "user" else "assistant",
        content=content,
        created_at=datetime(2026, 7, 15, tzinfo=UTC) + timedelta(seconds=index),
    )


async def test_build_context_recalls_period_and_keeps_newest_message() -> None:
    repository = FakeRepository()
    start = datetime(2026, 7, 20, 9, tzinfo=UTC)
    end = datetime(2026, 7, 22, 18, tzinfo=UTC)
    repository.state = ConversationStateMemory(start, end)
    repository.summary = ConversationSummaryMemory(
        content="用户计划公司直播。",
        through_event_id="01J00000000000000000000000",
        source_message_count=4,
        estimated_tokens=12,
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )
    repository.recent = [
        message(1, "user", "较早消息" * 80),
        message(2, "assistant", "较早回答" * 80),
        message(3, "user", "最新消息"),
    ]

    service = ConversationMemoryService(
        repository, settings(history_budget=55)
    )
    now = datetime(2026, 7, 15, 2, 30, tzinfo=UTC)
    context = await service.build_context("conversation-1", now_utc=now)

    assert context.rental_period == RentalPeriodInput(start_at=start, end_at=end)
    assert context.timezone == "Asia/Shanghai"
    assert context.now_utc == now
    assert context.now_local.isoformat() == "2026-07-15T10:30:00+08:00"
    assert context.messages[0].role == "system"
    assert "用户当地时间: 2026-07-15T10:30:00+08:00" in context.messages[0].content
    assert any(
        "库存、价格和报价都可能已经过期" in item.content
        for item in context.messages
    )
    assert context.messages[-1].content == "最新消息"
    assert all(item.content != "较早消息" * 80 for item in context.messages)
    assert repository.recent_after_event_id == repository.summary.through_event_id


async def test_remember_rental_period_delegates_to_repository() -> None:
    repository = FakeRepository()
    period = RentalPeriodInput(
        start_at=datetime(2026, 8, 1, tzinfo=UTC),
        end_at=datetime(2026, 8, 3, tzinfo=UTC),
    )
    service = ConversationMemoryService(repository, settings())

    await service.remember_rental_period("conversation-2", period)

    assert repository.remembered == ("conversation-2", period)


async def test_summary_compacts_old_messages_and_preserves_recent_tail() -> None:
    repository = FakeRepository()
    repository.after = [
        message(index, "user" if index % 2 else "assistant", f"消息 {index}" * 20)
        for index in range(1, 7)
    ]
    model = FakeModel()
    service = ConversationMemoryService(
        repository,
        settings(summary_trigger=1, recent_messages=2),
    )

    usage = await service.maybe_summarize("conversation-3", model)

    assert usage == ModelUsage(input_tokens=120, output_tokens=18)
    assert repository.saved_summary is not None
    assert repository.saved_summary["through_event_id"] == repository.after[3].event_id
    assert repository.saved_summary["source_message_count"] == 4
    assert len(model.requests) == 1
    assert model.requests[0].messages[0].content == SUMMARY_SYSTEM_PROMPT
    assert "必须在新一轮中重新查询" in model.requests[0].messages[0].content
    assert "消息 5" not in model.requests[0].messages[1].content


async def test_summary_does_not_call_model_below_threshold() -> None:
    repository = FakeRepository()
    repository.after = [message(1, "user", "短消息")]
    model = FakeModel()
    service = ConversationMemoryService(repository, settings())

    usage = await service.maybe_summarize("conversation-4", model)

    assert usage is None
    assert model.requests == []
    assert repository.saved_summary is None
