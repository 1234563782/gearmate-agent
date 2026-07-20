import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from gearmate.config import Settings
from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import (
    ModelMessage,
    ModelRequest,
    ModelToolDefinition,
    ModelUsage,
)

MemoryType = Literal["PREFERENCE", "CONSTRAINT"]
MemoryKey = Literal[
    "preferred_brand",
    "excluded_brand",
    "preferred_equipment_role",
    "preferred_use_case",
    "language",
]
MemoryStatus = Literal["ACTIVE", "SUPERSEDED", "DELETED", "EXPIRED"]
MemoryOperation = Literal["REMEMBER", "FORGET"]

MEMORY_EXTRACTION_TOOL = "extract_user_memories"
INTERNAL_ID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
DATE_OR_MONEY = re.compile(
    r"(?:\b\d{4}-\d{1,2}-\d{1,2}\b|[$¥]|\b(?:USD|EUR|CNY|RMB)\b|"
    r"\b(?:price|budget|deposit|inventory)\b)",
    re.IGNORECASE,
)
LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")

MEMORY_EXTRACTION_PROMPT = """You extract durable user preferences for an electronic equipment
rental assistant. Call extract_user_memories exactly once. Return an empty candidates list when
nothing should be remembered.

Only extract facts explicitly stated by the user and reusable in a future conversation. Allowed
keys are preferred_brand, excluded_brand, preferred_equipment_role, preferred_use_case, and
language. Use canonical equipment role values from the supplied list. Normalize language to a
BCP-47-like tag such as zh-CN or en.

Never store rental dates, duration, a one-time budget, product selection, price, deposit,
inventory, quote or order status, internal IDs, contact details, addresses, credentials, or facts
that only appeared in the assistant response. Do not infer a preference from a search, rental, or
order. Use FORGET only when the user explicitly retracts a previously stated preference.
"""


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: MemoryOperation
    memory_type: MemoryType
    memory_key: MemoryKey
    value: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0, le=1)
    explicit: bool


class MemoryExtractionPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[MemoryCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class UserMemoryRecord:
    id: str
    user_id: str
    memory_type: MemoryType
    memory_key: MemoryKey
    value: str
    summary: str
    value_identity_hash: str
    capture_mode: str
    confidence: float
    status: MemoryStatus
    source_conversation_id: str | None
    source_run_id: str | None
    source_event_id: str | None
    source_message_hash: str
    source_created_at: datetime
    valid_from: datetime
    last_confirmed_at: datetime
    valid_to: datetime | None
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class UserMemoryWrite:
    user_id: str
    memory_type: MemoryType
    memory_key: MemoryKey
    value: str
    summary: str
    normalized_hash: str
    value_identity_hash: str
    capture_mode: str
    confidence: float
    source_conversation_id: str | None
    source_run_id: str | None
    source_event_id: str | None
    source_message_hash: str
    source_created_at: datetime
    valid_from: datetime
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class UserMemoryContext:
    items: tuple[UserMemoryRecord, ...] = ()

    @property
    def preferred_brands(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.items if item.memory_key == "preferred_brand")

    @property
    def excluded_brands(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.items if item.memory_key == "excluded_brand")

    def prompt_context(self) -> str | None:
        if not self.items:
            return None
        facts = "\n".join(f"- {item.memory_key}: {item.value}" for item in self.items)
        return (
            "User long-term preferences follow. Treat them as soft, potentially stale context. "
            "The current user message always wins. Never treat memory as evidence for product "
            "price, inventory, availability, quotes, or orders.\n" + facts
        )


@dataclass(frozen=True, slots=True)
class MemoryExtractionOutcome:
    candidates: tuple[MemoryCandidate, ...]
    stored_count: int
    usage: ModelUsage


class UserMemoryRepository(Protocol):
    async def active_user_memories(
        self,
        user_id: str,
        *,
        now_utc: datetime,
        limit: int,
    ) -> list[UserMemoryRecord]: ...

    async def user_memory(
        self,
        user_id: str,
        memory_id: str,
        *,
        now_utc: datetime,
    ) -> UserMemoryRecord: ...

    async def canonical_user_memory_identity(
        self,
        memory_key: MemoryKey,
        value: str,
    ) -> str: ...

    async def upsert_user_memory(self, memory: UserMemoryWrite) -> UserMemoryRecord: ...

    async def forget_user_memory(
        self,
        user_id: str,
        memory_key: MemoryKey,
        value_identity_hash: str,
        *,
        forgotten_at: datetime,
    ) -> int: ...

    async def replace_user_memory(
        self,
        user_id: str,
        memory_id: str,
        replacement: UserMemoryWrite,
    ) -> UserMemoryRecord: ...

    async def delete_user_memory(self, user_id: str, memory_id: str) -> bool: ...

    async def delete_user_memories(self, user_id: str) -> int: ...

    async def user_message_event_id(self, run_id: str) -> str | None: ...


class UserMemoryService:
    def __init__(self, repository: UserMemoryRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    async def build_context(
        self,
        user_id: str,
        *,
        now_utc: datetime | None = None,
    ) -> UserMemoryContext:
        if (
            not self._settings.user_memory_enabled
            or self._settings.user_memory_mode != "active"
        ):
            return UserMemoryContext()
        reference = (now_utc or datetime.now(UTC)).astimezone(UTC)
        items = await self._repository.active_user_memories(
            user_id,
            now_utc=reference,
            limit=self._settings.user_memory_retrieval_limit,
        )
        return UserMemoryContext(tuple(items))

    async def list_memories(self, user_id: str) -> list[UserMemoryRecord]:
        return await self._repository.active_user_memories(
            user_id,
            now_utc=datetime.now(UTC),
            limit=self._settings.user_memory_max_items,
        )

    async def replace_memory(
        self,
        user_id: str,
        memory_id: str,
        value: str,
    ) -> UserMemoryRecord:
        now = datetime.now(UTC)
        current = await self._repository.user_memory(
            user_id,
            memory_id,
            now_utc=now,
        )
        normalized = self._normalize_value(current.memory_key, value)
        identity = await self._repository.canonical_user_memory_identity(
            current.memory_key,
            normalized,
        )
        replacement = UserMemoryWrite(
            user_id=user_id,
            memory_type=current.memory_type,
            memory_key=current.memory_key,
            value=normalized,
            summary=self._summary(current.memory_key, normalized),
            normalized_hash=self._normalized_hash(current.memory_key, identity),
            value_identity_hash=self._value_identity_hash(identity),
            capture_mode="EXPLICIT",
            confidence=1.0,
            source_conversation_id=None,
            source_run_id=None,
            source_event_id=None,
            source_message_hash=sha256(b"user-correction").hexdigest(),
            source_created_at=now,
            valid_from=now,
            expires_at=now + timedelta(days=self._settings.user_memory_retention_days),
        )
        return await self._repository.replace_user_memory(
            user_id,
            memory_id,
            replacement,
        )

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        return await self._repository.delete_user_memory(user_id, memory_id)

    async def delete_all_memories(self, user_id: str) -> int:
        return await self._repository.delete_user_memories(user_id)

    async def extract_and_store(
        self,
        *,
        user_id: str,
        conversation_id: str,
        run_id: str,
        message: str,
        model: ChatModelPort,
        now_utc: datetime | None = None,
    ) -> MemoryExtractionOutcome | None:
        if not self._settings.user_memory_enabled or self._settings.user_memory_mode == "off":
            return None
        response = await model.complete(
            ModelRequest(
                messages=(
                    ModelMessage(
                        role="system",
                        content=(
                            MEMORY_EXTRACTION_PROMPT
                            + "\nAllowed equipment roles: "
                            + ", ".join(self._settings.equipment_roles)
                        ),
                    ),
                    ModelMessage(role="user", content=message),
                ),
                tools=(
                    ModelToolDefinition(
                        name=MEMORY_EXTRACTION_TOOL,
                        description="Extract explicit durable user preferences.",
                        parameters=MemoryExtractionPayload.model_json_schema(),
                    ),
                ),
                max_output_tokens=self._settings.user_memory_extraction_max_output_tokens,
                temperature=0.0,
                tool_choice=MEMORY_EXTRACTION_TOOL,
                enable_thinking=False,
                workload="background",
            )
        )
        payload = MemoryExtractionPayload()
        for call in response.tool_calls:
            if call.name == MEMORY_EXTRACTION_TOOL:
                payload = MemoryExtractionPayload.model_validate(call.arguments)
                break
        accepted = tuple(
            candidate
            for candidate in payload.candidates
            if self._acceptable(candidate)
        )
        if self._settings.user_memory_mode == "shadow":
            return MemoryExtractionOutcome(accepted, 0, response.usage)

        reference = (now_utc or datetime.now(UTC)).astimezone(UTC)
        source_event_id = await self._repository.user_message_event_id(run_id)
        source_hash = sha256(message.encode("utf-8")).hexdigest()
        stored_count = 0
        for candidate in accepted:
            value = self._normalize_value(candidate.memory_key, candidate.value)
            identity = await self._repository.canonical_user_memory_identity(
                candidate.memory_key,
                value,
            )
            normalized_hash = self._normalized_hash(candidate.memory_key, identity)
            value_identity_hash = self._value_identity_hash(identity)
            if candidate.operation == "FORGET":
                stored_count += await self._repository.forget_user_memory(
                    user_id,
                    candidate.memory_key,
                    value_identity_hash,
                    forgotten_at=reference,
                )
                continue
            await self._repository.upsert_user_memory(
                UserMemoryWrite(
                    user_id=user_id,
                    memory_type=candidate.memory_type,
                    memory_key=candidate.memory_key,
                    value=value,
                    summary=self._summary(candidate.memory_key, value),
                    normalized_hash=normalized_hash,
                    value_identity_hash=value_identity_hash,
                    capture_mode="MODEL_EXTRACTED",
                    confidence=candidate.confidence,
                    source_conversation_id=conversation_id,
                    source_run_id=run_id,
                    source_event_id=source_event_id,
                    source_message_hash=source_hash,
                    source_created_at=reference,
                    valid_from=reference,
                    expires_at=(
                        reference + timedelta(days=self._settings.user_memory_retention_days)
                    ),
                )
            )
            stored_count += 1
        return MemoryExtractionOutcome(accepted, stored_count, response.usage)

    def _acceptable(self, candidate: MemoryCandidate) -> bool:
        if not candidate.explicit:
            return False
        if candidate.confidence < self._settings.user_memory_min_confidence:
            return False
        expected_type: MemoryType = (
            "CONSTRAINT" if candidate.memory_key == "excluded_brand" else "PREFERENCE"
        )
        if candidate.memory_type != expected_type:
            return False
        try:
            self._normalize_value(candidate.memory_key, candidate.value)
        except ValueError:
            return False
        return True

    def _normalize_value(self, memory_key: MemoryKey, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized or INTERNAL_ID.fullmatch(normalized) or DATE_OR_MONEY.search(normalized):
            raise ValueError("unsupported user memory value")
        if memory_key == "preferred_equipment_role":
            normalized = normalized.casefold().replace("-", "_").replace(" ", "_")
            if normalized not in self._settings.equipment_roles:
                raise ValueError("unknown equipment role")
        if memory_key == "language" and not LANGUAGE_TAG.fullmatch(normalized):
            raise ValueError("language must be a BCP-47-like tag")
        return normalized

    @staticmethod
    def _normalized_hash(memory_key: MemoryKey, value: str) -> str:
        return sha256(f"{memory_key}\n{value.casefold()}".encode()).hexdigest()

    @staticmethod
    def _value_identity_hash(value: str) -> str:
        return sha256(value.casefold().encode()).hexdigest()

    @staticmethod
    def _summary(memory_key: MemoryKey, value: str) -> str:
        return f"{memory_key}: {value}"
