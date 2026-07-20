from datetime import UTC, datetime, timedelta

from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelToolCall, ModelUsage
from gearmate.tools.contracts import ProductSearchInput, ProductSearchResult, ProductSummary
from gearmate.tools.registry import ToolRegistry
from gearmate.user_memory import (
    MemoryKey,
    UserMemoryContext,
    UserMemoryRecord,
    UserMemoryService,
    UserMemoryWrite,
)


def record(
    memory_id: str,
    key: MemoryKey,
    value: str,
    *,
    user_id: str = "user-1",
) -> UserMemoryRecord:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    return UserMemoryRecord(
        id=memory_id,
        user_id=user_id,
        memory_type="CONSTRAINT" if key == "excluded_brand" else "PREFERENCE",
        memory_key=key,
        value=value,
        summary=f"{key}: {value}",
        value_identity_hash="b" * 64,
        capture_mode="MODEL_EXTRACTED",
        confidence=0.98,
        status="ACTIVE",
        source_conversation_id="conversation-1",
        source_run_id="run-1",
        source_event_id="event-1",
        source_message_hash="a" * 64,
        source_created_at=now,
        valid_from=now,
        last_confirmed_at=now,
        valid_to=None,
        expires_at=now + timedelta(days=180),
        created_at=now,
        updated_at=now,
    )


class FakeRepository:
    def __init__(self) -> None:
        self.items: list[UserMemoryRecord] = []
        self.writes: list[UserMemoryWrite] = []
        self.forgotten: list[tuple[str, MemoryKey, str]] = []

    async def active_user_memories(
        self,
        user_id: str,
        *,
        now_utc: datetime,
        limit: int,
    ) -> list[UserMemoryRecord]:
        return [item for item in self.items if item.user_id == user_id][:limit]

    async def user_memory(
        self,
        user_id: str,
        memory_id: str,
        *,
        now_utc: datetime,
    ) -> UserMemoryRecord:
        for item in self.items:
            if (
                item.user_id == user_id
                and item.id == memory_id
                and (item.expires_at is None or item.expires_at > now_utc)
            ):
                return item
        raise LookupError("User memory not found")

    async def canonical_user_memory_identity(
        self,
        memory_key: MemoryKey,
        value: str,
    ) -> str:
        if memory_key in ("preferred_brand", "excluded_brand") and value.casefold() in {
            "sony",
            "索尼",
        }:
            return "Sony"
        return value

    async def upsert_user_memory(self, memory: UserMemoryWrite) -> UserMemoryRecord:
        self.writes.append(memory)
        stored = record(
            f"memory-{len(self.writes)}",
            memory.memory_key,
            memory.value,
            user_id=memory.user_id,
        )
        self.items.append(stored)
        return stored

    async def forget_user_memory(
        self,
        user_id: str,
        memory_key: MemoryKey,
        value_identity_hash: str,
        *,
        forgotten_at: datetime,
    ) -> int:
        self.forgotten.append((user_id, memory_key, value_identity_hash))
        return 1

    async def replace_user_memory(
        self,
        user_id: str,
        memory_id: str,
        replacement: UserMemoryWrite,
    ) -> UserMemoryRecord:
        self.writes.append(replacement)
        return record(memory_id, replacement.memory_key, replacement.value, user_id=user_id)

    async def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        return any(item.user_id == user_id and item.id == memory_id for item in self.items)

    async def delete_user_memories(self, user_id: str) -> int:
        return sum(item.user_id == user_id for item in self.items)

    async def user_message_event_id(self, run_id: str) -> str | None:
        return "event-1"


class FakeModel:
    def __init__(self, arguments: dict[str, object]) -> None:
        self.arguments = arguments
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text="",
            finish_reason="tool_calls",
            usage=ModelUsage(input_tokens=40, output_tokens=8),
            tool_calls=(
                ModelToolCall(
                    id="memory-call-1",
                    name="extract_user_memories",
                    arguments=self.arguments,
                ),
            ),
        )

    async def close(self) -> None:
        return None


def settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        user_memory_enabled=True,
        user_memory_mode=overrides.get("mode", "active"),
        user_memory_min_confidence=overrides.get("min_confidence", 0.85),
    )


async def test_build_context_is_scoped_by_user_and_renders_soft_preferences() -> None:
    repository = FakeRepository()
    repository.items = [
        record("memory-1", "preferred_brand", "Sony"),
        record("memory-2", "excluded_brand", "Canon"),
        record("memory-3", "preferred_brand", "Apple", user_id="user-2"),
    ]

    context = await UserMemoryService(repository, settings()).build_context("user-1")

    assert context.preferred_brands == ("Sony",)
    assert context.excluded_brands == ("Canon",)
    assert context.prompt_context() is not None
    assert "current user message always wins" in context.prompt_context()
    assert "Apple" not in context.prompt_context()


async def test_extracts_only_explicit_allowed_durable_memory() -> None:
    repository = FakeRepository()
    model = FakeModel(
        {
            "candidates": [
                {
                    "operation": "REMEMBER",
                    "memory_type": "PREFERENCE",
                    "memory_key": "preferred_brand",
                    "value": "Sony",
                    "confidence": 0.98,
                    "explicit": True,
                },
                {
                    "operation": "REMEMBER",
                    "memory_type": "PREFERENCE",
                    "memory_key": "preferred_use_case",
                    "value": "budget 500",
                    "confidence": 0.99,
                    "explicit": True,
                },
                {
                    "operation": "REMEMBER",
                    "memory_type": "PREFERENCE",
                    "memory_key": "preferred_brand",
                    "value": "Canon",
                    "confidence": 0.99,
                    "explicit": False,
                },
            ]
        }
    )
    service = UserMemoryService(repository, settings())

    outcome = await service.extract_and_store(
        user_id="user-1",
        conversation_id="conversation-1",
        run_id="run-1",
        message="I prefer Sony. My budget today is 500.",
        model=model,
        now_utc=datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert outcome is not None
    assert outcome.stored_count == 1
    assert len(outcome.candidates) == 1
    assert repository.writes[0].memory_key == "preferred_brand"
    assert repository.writes[0].value == "Sony"
    assert repository.writes[0].source_event_id == "event-1"
    assert model.requests[0].workload == "background"


async def test_shadow_mode_extracts_without_writing() -> None:
    repository = FakeRepository()
    model = FakeModel(
        {
            "candidates": [
                {
                    "operation": "REMEMBER",
                    "memory_type": "CONSTRAINT",
                    "memory_key": "excluded_brand",
                    "value": "Canon",
                    "confidence": 0.95,
                    "explicit": True,
                }
            ]
        }
    )

    outcome = await UserMemoryService(repository, settings(mode="shadow")).extract_and_store(
        user_id="user-1",
        conversation_id="conversation-1",
        run_id="run-1",
        message="Do not recommend Canon in the future.",
        model=model,
    )

    assert outcome is not None
    assert outcome.stored_count == 0
    assert len(outcome.candidates) == 1
    assert repository.writes == []


async def test_forget_uses_catalog_identity_for_brand_alias() -> None:
    repository = FakeRepository()
    model = FakeModel(
        {
            "candidates": [
                {
                    "operation": "FORGET",
                    "memory_type": "CONSTRAINT",
                    "memory_key": "excluded_brand",
                    "value": "索尼",
                    "confidence": 0.99,
                    "explicit": True,
                }
            ]
        }
    )

    outcome = await UserMemoryService(repository, settings()).extract_and_store(
        user_id="user-1",
        conversation_id="conversation-1",
        run_id="run-1",
        message="我不排斥索尼了",
        model=model,
    )

    assert outcome is not None
    assert outcome.stored_count == 1
    assert repository.forgotten == [
        (
            "user-1",
            "excluded_brand",
            UserMemoryService._value_identity_hash("Sony"),
        )
    ]


async def test_cross_key_brand_identity_uses_unicode_casefold() -> None:
    repository = FakeRepository()
    model = FakeModel(
        {
            "candidates": [
                {
                    "operation": "REMEMBER",
                    "memory_type": "PREFERENCE",
                    "memory_key": "preferred_brand",
                    "value": "Straße",
                    "confidence": 0.99,
                    "explicit": True,
                },
                {
                    "operation": "REMEMBER",
                    "memory_type": "CONSTRAINT",
                    "memory_key": "excluded_brand",
                    "value": "STRASSE",
                    "confidence": 0.99,
                    "explicit": True,
                },
            ]
        }
    )

    await UserMemoryService(repository, settings()).extract_and_store(
        user_id="user-1",
        conversation_id="conversation-1",
        run_id="run-1",
        message="I used to prefer Straße, but now I exclude STRASSE.",
        model=model,
    )

    assert len(repository.writes) == 2
    assert repository.writes[0].value_identity_hash == repository.writes[1].value_identity_hash
    assert repository.writes[0].normalized_hash != repository.writes[1].normalized_hash


async def test_shadow_mode_does_not_inject_existing_memories() -> None:
    repository = FakeRepository()
    repository.items = [record("memory-1", "preferred_brand", "Sony")]

    context = await UserMemoryService(repository, settings(mode="shadow")).build_context("user-1")

    assert context.items == ()
    assert context.prompt_context() is None


def product(product_id: str, brand: str) -> ProductSummary:
    return ProductSummary(
        product_id=product_id,
        category_id="01J00000000000000000000999",
        equipment_role="camera",
        name=f"{brand} Camera",
        brand=brand,
        model=f"{brand}-1",
        daily_rate="100.00",
        fixed_deposit="500.00",
    )


def test_tool_registry_applies_brand_memory_as_soft_ranking() -> None:
    registry = object.__new__(ToolRegistry)
    registry._preferred_brands = frozenset({"sony"})
    registry._excluded_brands = frozenset({"canon"})
    result = ProductSearchResult(
        items=(
            product("01J00000000000000000000001", "Nikon"),
            product("01J00000000000000000000002", "Canon"),
            product("01J00000000000000000000003", "Sony"),
        ),
        page=0,
        size=20,
        total_elements=3,
        total_pages=1,
    )

    ranked = registry._apply_user_preferences(result, ProductSearchInput())

    assert [item.brand for item in ranked.items] == ["Sony", "Nikon"]
    explicit = registry._apply_user_preferences(
        result,
        ProductSearchInput(brand="Canon"),
    )
    assert explicit == result


def test_empty_context_has_no_prompt() -> None:
    assert UserMemoryContext().prompt_context() is None
