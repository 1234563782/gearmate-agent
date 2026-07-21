from datetime import UTC, datetime, timedelta

from gearmate.actions import PendingProductSearch
from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelUsage
from gearmate.memory import (
    SUMMARY_SYSTEM_PROMPT,
    ConversationMemoryService,
    ConversationMessageMemory,
    ConversationStateMemory,
    ConversationSummaryMemory,
)
from gearmate.search import RecentProductReference, RecentProductSearch


class FakeRepository:
    def __init__(self) -> None:
        self.state: ConversationStateMemory | None = None
        self.summary: ConversationSummaryMemory | None = None
        self.recent: list[ConversationMessageMemory] = []
        self.after: list[ConversationMessageMemory] = []
        self.pending_search: PendingProductSearch | None = None
        self.pending_cleared = False
        self.recent_search: RecentProductSearch | None = None
        self.saved_summary: dict[str, object] | None = None
        self.recent_after_event_id: str | None = None

    async def conversation_timezone(self, conversation_id: str) -> str:
        return "Asia/Shanghai"

    async def upsert_pending_product_search(
        self,
        conversation_id: str,
        pending_search: PendingProductSearch,
    ) -> None:
        self.pending_search = pending_search

    async def clear_pending_product_search(self, conversation_id: str) -> None:
        self.pending_cleared = True
        self.pending_search = None

    async def upsert_recent_product_search(
        self,
        conversation_id: str,
        recent_search: RecentProductSearch,
    ) -> None:
        self.recent_search = recent_search

    async def conversation_state(self, conversation_id: str) -> ConversationStateMemory | None:
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
    def __init__(self, text: str = "用户准备购买用于剪辑的电脑。") -> None:
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


async def test_build_context_keeps_summary_and_newest_message_within_budget() -> None:
    repository = FakeRepository()
    repository.summary = ConversationSummaryMemory(
        content="用户计划购买办公电脑。",
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

    service = ConversationMemoryService(repository, settings(history_budget=55))
    now = datetime(2026, 7, 15, 2, 30, tzinfo=UTC)
    context = await service.build_context("conversation-1", now_utc=now)

    assert context.timezone == "Asia/Shanghai"
    assert context.now_local.isoformat() == "2026-07-15T10:30:00+08:00"
    assert context.messages[-1].content == "最新消息"
    assert all(item.content != "较早消息" * 80 for item in context.messages)
    assert repository.recent_after_event_id == repository.summary.through_event_id


async def test_pending_product_search_is_remembered_and_cleared() -> None:
    repository = FakeRepository()
    service = ConversationMemoryService(repository, settings())
    pending = PendingProductSearch(equipment_role="laptop", max_price="8000")

    await service.remember_pending_product_search("conversation-search", pending)
    await service.clear_pending_product_search("conversation-search")

    assert repository.pending_search is None
    assert repository.pending_cleared is True


async def test_build_context_restores_pending_product_search() -> None:
    pending = PendingProductSearch(equipment_role="laptop", target_price="7000")
    repository = FakeRepository()
    repository.state = ConversationStateMemory(pending_product_search=pending)

    context = await ConversationMemoryService(repository, settings()).build_context(
        "conversation-search",
        now_utc=datetime(2026, 7, 15, tzinfo=UTC),
    )

    assert context.pending_product_search == pending


async def test_recent_product_search_is_remembered_and_restored() -> None:
    recent = RecentProductSearch(
        items=(
            RecentProductReference(
                position=1,
                product_id="01J00000000000000000000105",
                name="MacBook Pro 14",
                brand="Apple",
                model="MacBook Pro 14",
                equipment_role="laptop",
            ),
        )
    )
    repository = FakeRepository()
    service = ConversationMemoryService(repository, settings())

    await service.remember_recent_product_search("conversation-1", recent)
    repository.state = ConversationStateMemory(recent_product_search=recent)
    context = await service.build_context(
        "conversation-1",
        now_utc=datetime(2026, 7, 15, 6, tzinfo=UTC),
    )

    assert repository.recent_search == recent
    assert context.recent_product_search == recent


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
    assert model.requests[0].messages[0].content == SUMMARY_SYSTEM_PROMPT


async def test_summary_does_not_call_model_below_threshold() -> None:
    repository = FakeRepository()
    repository.after = [message(1, "user", "短消息")]
    model = FakeModel()
    service = ConversationMemoryService(repository, settings())

    usage = await service.maybe_summarize("conversation-4", model)

    assert usage is None
    assert model.requests == []
