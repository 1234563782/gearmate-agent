from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import delete, exists, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gearmate.actions import PendingProductSearch, PendingRentalAction
from gearmate.ids import new_ulid
from gearmate.memory import (
    ConversationMessageMemory,
    ConversationStateMemory,
    ConversationSummaryMemory,
)
from gearmate.persistence.models import (
    ACTIVE_RUN_STATUSES,
    AgentRun,
    CatalogAlias,
    Conversation,
    ConversationState,
    ConversationSummary,
    RunEvent,
    UserMemory,
)
from gearmate.requirements import RentalRequirements
from gearmate.search import RecentProductSearch
from gearmate.tools.contracts import RentalPeriodInput
from gearmate.user_memory import (
    MemoryKey,
    MemoryStatus,
    MemoryType,
    UserMemoryRecord,
    UserMemoryWrite,
)


class ActiveRunConflict(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    conversation_id: str
    status: str
    stop_reason: str | None
    state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EventRecord:
    id: str
    sequence_no: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


class AgentRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def create_conversation(
        self,
        user_id: str,
        timezone: str,
        title: str | None,
    ) -> Conversation:
        conversation = Conversation(
            id=new_ulid(),
            user_id=user_id,
            timezone=timezone,
            title=title,
        )
        async with self._sessions.begin() as session:
            session.add(conversation)
        return conversation

    async def list_conversations(self, user_id: str) -> list[Conversation]:
        async with self._sessions() as session:
            result = await session.scalars(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
            )
            return list(result)

    async def require_conversation(self, conversation_id: str, user_id: str) -> Conversation:
        async with self._sessions() as session:
            conversation = await session.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            if conversation is None:
                raise LookupError("Conversation not found")
            return conversation

    async def create_run(
        self,
        conversation_id: str,
        *,
        model_provider: str,
        model_id: str | None,
        prompt_version: str,
        prompt_hash: str,
        initial_state: dict[str, Any],
        user_message: str,
    ) -> AgentRun:
        run = AgentRun(
            id=new_ulid(),
            conversation_id=conversation_id,
            status="RUNNING",
            model_provider=model_provider,
            model_id=model_id,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            state=initial_state,
        )
        try:
            async with self._sessions.begin() as session:
                await session.execute(
                    update(Conversation)
                    .where(Conversation.id == conversation_id)
                    .values(updated_at=datetime.now(UTC))
                )
                session.add(run)
                session.add_all(
                    (
                        RunEvent(
                            id=new_ulid(),
                            run_id=run.id,
                            sequence_no=1,
                            event_type="run.started",
                            payload={
                                "runId": run.id,
                                "conversationId": conversation_id,
                                "promptVersion": prompt_version,
                                "promptHash": prompt_hash,
                            },
                        ),
                        RunEvent(
                            id=new_ulid(),
                            run_id=run.id,
                            sequence_no=2,
                            event_type="user.message",
                            payload={"content": user_message},
                        ),
                    )
                )
        except IntegrityError as error:
            raise ActiveRunConflict("Conversation already has an active run") from error
        return run

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> EventRecord:
        async with self._sessions.begin() as session:
            run = await session.scalar(
                select(AgentRun).where(AgentRun.id == run_id).with_for_update()
            )
            if run is None:
                raise LookupError("Run not found")
            current = await session.scalar(
                select(func.max(RunEvent.sequence_no)).where(RunEvent.run_id == run_id)
            )
            sequence_no = int(current or 0) + 1
            event = RunEvent(
                id=new_ulid(),
                run_id=run_id,
                sequence_no=sequence_no,
                event_type=event_type,
                payload=payload,
            )
            session.add(event)
            await session.flush()
            created_at = event.created_at
        return EventRecord(event.id, sequence_no, event_type, payload, created_at)

    async def finalize_run(
        self,
        run_id: str,
        *,
        event_type: str,
        event_payload: dict[str, Any],
        status: str,
        stop_reason: str,
        error_code: str | None,
        state: dict[str, Any],
        input_tokens: int,
        output_tokens: int,
        model_rounds: int,
        tool_call_count: int,
    ) -> None:
        async with self._sessions.begin() as session:
            run = await session.get(AgentRun, run_id, with_for_update=True)
            if run is None:
                raise LookupError("Run not found")
            current = await session.scalar(
                select(func.max(RunEvent.sequence_no)).where(RunEvent.run_id == run_id)
            )
            session.add(
                RunEvent(
                    id=new_ulid(),
                    run_id=run_id,
                    sequence_no=int(current or 0) + 1,
                    event_type=event_type,
                    payload=event_payload,
                )
            )
            run.status = status
            run.stop_reason = stop_reason
            run.error_code = error_code
            run.state = state
            run.input_tokens = input_tokens
            run.output_tokens = output_tokens
            run.model_rounds = model_rounds
            run.tool_call_count = tool_call_count
            run.finished_at = datetime.now(UTC)

    async def fail_stale_active_runs(self, stale_before: datetime) -> int:
        async with self._sessions.begin() as session:
            runs = await session.scalars(
                select(AgentRun)
                .where(
                    AgentRun.status.in_(("RUNNING", "TOOL_REQUESTED")),
                    AgentRun.created_at < stale_before,
                )
                .with_for_update()
            )
            stale = list(runs)
            now = datetime.now(UTC)
            for run in stale:
                current = await session.scalar(
                    select(func.max(RunEvent.sequence_no)).where(RunEvent.run_id == run.id)
                )
                run.status = "FAILED"
                run.stop_reason = "SERVICE_RESTART"
                run.error_code = "STALE_ACTIVE_RUN"
                run.finished_at = now
                session.add(
                    RunEvent(
                        id=new_ulid(),
                        run_id=run.id,
                        sequence_no=int(current or 0) + 1,
                        event_type="run.failed",
                        payload={
                            "stopReason": "SERVICE_RESTART",
                            "errorCode": "STALE_ACTIVE_RUN",
                        },
                    )
                )
            return len(stale)

    async def delete_expired_conversations(self, inactive_before: datetime) -> int:
        active_run = exists().where(
            AgentRun.conversation_id == Conversation.id,
            AgentRun.status.in_(ACTIVE_RUN_STATUSES),
        )
        async with self._sessions.begin() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    delete(Conversation).where(
                        Conversation.updated_at < inactive_before,
                        ~active_run,
                    )
                ),
            )
            return int(result.rowcount or 0)

    async def require_run(self, run_id: str, user_id: str) -> RunRecord:
        async with self._sessions() as session:
            row = await session.execute(
                select(AgentRun)
                .join(Conversation)
                .where(
                    AgentRun.id == run_id,
                    Conversation.user_id == user_id,
                )
            )
            run = row.scalar_one_or_none()
            if run is None:
                raise LookupError("Run not found")
            return RunRecord(run.id, run.conversation_id, run.status, run.stop_reason, run.state)

    async def events_after(self, run_id: str, after: int) -> list[EventRecord]:
        async with self._sessions() as session:
            events = await session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == run_id, RunEvent.sequence_no > after)
                .order_by(RunEvent.sequence_no)
            )
            return [
                EventRecord(
                    event.id,
                    event.sequence_no,
                    event.event_type,
                    event.payload,
                    event.created_at,
                )
                for event in events
            ]

    async def active_user_memories(
        self,
        user_id: str,
        *,
        now_utc: datetime,
        limit: int,
    ) -> list[UserMemoryRecord]:
        async with self._sessions() as session:
            memories = await session.scalars(
                select(UserMemory)
                .where(
                    UserMemory.user_id == user_id,
                    UserMemory.status == "ACTIVE",
                    or_(UserMemory.expires_at.is_(None), UserMemory.expires_at > now_utc),
                )
                .order_by(UserMemory.updated_at.desc(), UserMemory.id.desc())
                .limit(limit)
            )
            return [self._user_memory_record(memory) for memory in memories]

    async def user_memory(
        self,
        user_id: str,
        memory_id: str,
        *,
        now_utc: datetime,
    ) -> UserMemoryRecord:
        async with self._sessions() as session:
            memory = await session.scalar(
                select(UserMemory).where(
                    UserMemory.id == memory_id,
                    UserMemory.user_id == user_id,
                    UserMemory.status == "ACTIVE",
                    or_(UserMemory.expires_at.is_(None), UserMemory.expires_at > now_utc),
                )
            )
            if memory is None:
                raise LookupError("User memory not found")
            return self._user_memory_record(memory)

    async def canonical_user_memory_identity(
        self,
        memory_key: MemoryKey,
        value: str,
    ) -> str:
        entity_type = {
            "preferred_brand": "brand",
            "excluded_brand": "brand",
            "preferred_equipment_role": "equipment_role",
            "preferred_use_case": "use_case",
        }.get(memory_key)
        if entity_type is None:
            return value

        async with self._sessions() as session:
            rows = await session.execute(
                select(CatalogAlias.alias, CatalogAlias.canonical_value)
                .where(
                    CatalogAlias.entity_type == entity_type,
                    CatalogAlias.active.is_(True),
                )
                .order_by(CatalogAlias.alias, CatalogAlias.canonical_value)
            )
            folded = value.casefold()
            for alias, canonical_value in rows:
                canonical = str(canonical_value)
                if str(alias).casefold() == folded or canonical.casefold() == folded:
                    return canonical
        return value

    async def upsert_user_memory(self, memory: UserMemoryWrite) -> UserMemoryRecord:
        now = datetime.now(UTC)
        async with self._sessions.begin() as session:
            await self._lock_user_memories(session, memory.user_id)
            await self._expire_user_memories(session, memory.user_id, now)
            await self._supersede_user_memory_conflicts(session, memory, now)
            stored = await self._upsert_user_memory_in_session(session, memory, now)
            return self._user_memory_record(stored)

    async def forget_user_memory(
        self,
        user_id: str,
        memory_key: MemoryKey,
        value_identity_hash: str,
        *,
        forgotten_at: datetime,
    ) -> int:
        async with self._sessions.begin() as session:
            await self._lock_user_memories(session, user_id)
            await self._expire_user_memories(session, user_id, forgotten_at)
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(UserMemory)
                    .where(
                        UserMemory.user_id == user_id,
                        UserMemory.memory_key == memory_key,
                        UserMemory.value_identity_hash == value_identity_hash,
                        UserMemory.status == "ACTIVE",
                    )
                    .values(
                        status="DELETED",
                        valid_to=forgotten_at,
                        updated_at=forgotten_at,
                    )
                ),
            )
            return int(result.rowcount or 0)

    async def replace_user_memory(
        self,
        user_id: str,
        memory_id: str,
        replacement: UserMemoryWrite,
    ) -> UserMemoryRecord:
        now = datetime.now(UTC)
        async with self._sessions.begin() as session:
            await self._lock_user_memories(session, user_id)
            await self._expire_user_memories(session, user_id, now)
            current = await session.scalar(
                select(UserMemory)
                .where(
                    UserMemory.id == memory_id,
                    UserMemory.user_id == user_id,
                    UserMemory.status == "ACTIVE",
                )
                .with_for_update()
            )
            if current is None:
                raise LookupError("User memory not found")
            current.status = "SUPERSEDED"
            current.valid_to = now
            current.updated_at = now
            await session.flush()
            await self._supersede_user_memory_conflicts(session, replacement, now)
            stored = await self._upsert_user_memory_in_session(session, replacement, now)
            return self._user_memory_record(stored)

    async def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        now = datetime.now(UTC)
        async with self._sessions.begin() as session:
            await self._lock_user_memories(session, user_id)
            await self._expire_user_memories(session, user_id, now)
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(UserMemory)
                    .where(
                        UserMemory.id == memory_id,
                        UserMemory.user_id == user_id,
                        UserMemory.status == "ACTIVE",
                    )
                    .values(status="DELETED", valid_to=now, updated_at=now)
                ),
            )
            return bool(result.rowcount)

    async def delete_user_memories(self, user_id: str) -> int:
        now = datetime.now(UTC)
        async with self._sessions.begin() as session:
            await self._lock_user_memories(session, user_id)
            await self._expire_user_memories(session, user_id, now)
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(UserMemory)
                    .where(UserMemory.user_id == user_id, UserMemory.status == "ACTIVE")
                    .values(status="DELETED", valid_to=now, updated_at=now)
                ),
            )
            return int(result.rowcount or 0)

    async def user_message_event_id(self, run_id: str) -> str | None:
        async with self._sessions() as session:
            return cast(
                str | None,
                await session.scalar(
                    select(RunEvent.id)
                    .where(RunEvent.run_id == run_id, RunEvent.event_type == "user.message")
                    .order_by(RunEvent.sequence_no)
                    .limit(1)
                ),
            )

    @staticmethod
    async def _lock_user_memories(session: AsyncSession, user_id: str) -> None:
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:user_id, 0))"),
            {"user_id": user_id},
        )

    @staticmethod
    async def _supersede_user_memory_conflicts(
        session: AsyncSession,
        memory: UserMemoryWrite,
        now: datetime,
    ) -> None:
        conflicting_keys: tuple[str, ...] = ()
        if memory.memory_key == "language":
            conflicting_keys = ("language",)
        elif memory.memory_key == "preferred_brand":
            conflicting_keys = ("excluded_brand",)
        elif memory.memory_key == "excluded_brand":
            conflicting_keys = ("preferred_brand",)
        if not conflicting_keys:
            return
        conditions = [
            UserMemory.user_id == memory.user_id,
            UserMemory.status == "ACTIVE",
            UserMemory.memory_key.in_(conflicting_keys),
        ]
        if memory.memory_key == "language":
            conditions.append(UserMemory.normalized_hash != memory.normalized_hash)
        else:
            conditions.append(UserMemory.value_identity_hash == memory.value_identity_hash)
        await session.execute(
            update(UserMemory)
            .where(*conditions)
            .values(status="SUPERSEDED", valid_to=now, updated_at=now)
        )

    @staticmethod
    async def _upsert_user_memory_in_session(
        session: AsyncSession,
        memory: UserMemoryWrite,
        now: datetime,
    ) -> UserMemory:
        insert_statement = pg_insert(UserMemory).values(
            id=new_ulid(),
            user_id=memory.user_id,
            memory_type=memory.memory_type,
            memory_key=memory.memory_key,
            value={"text": memory.value},
            summary=memory.summary,
            normalized_hash=memory.normalized_hash,
            value_identity_hash=memory.value_identity_hash,
            capture_mode=memory.capture_mode,
            confidence=memory.confidence,
            status="ACTIVE",
            source_conversation_id=memory.source_conversation_id,
            source_run_id=memory.source_run_id,
            source_event_id=memory.source_event_id,
            source_message_hash=memory.source_message_hash,
            source_created_at=memory.source_created_at,
            valid_from=memory.valid_from,
            last_confirmed_at=memory.source_created_at,
            expires_at=memory.expires_at,
            created_at=now,
            updated_at=now,
        )
        upsert_statement = insert_statement.on_conflict_do_update(
            index_elements=[
                UserMemory.user_id,
                UserMemory.memory_key,
                UserMemory.value_identity_hash,
            ],
            index_where=UserMemory.status == "ACTIVE",
            set_={
                "memory_type": insert_statement.excluded.memory_type,
                "value": insert_statement.excluded.value,
                "summary": insert_statement.excluded.summary,
                "normalized_hash": insert_statement.excluded.normalized_hash,
                "capture_mode": insert_statement.excluded.capture_mode,
                "confidence": func.greatest(
                    UserMemory.confidence,
                    insert_statement.excluded.confidence,
                ),
                "source_conversation_id": insert_statement.excluded.source_conversation_id,
                "source_run_id": insert_statement.excluded.source_run_id,
                "source_event_id": insert_statement.excluded.source_event_id,
                "source_message_hash": insert_statement.excluded.source_message_hash,
                "source_created_at": insert_statement.excluded.source_created_at,
                "last_confirmed_at": insert_statement.excluded.last_confirmed_at,
                "expires_at": insert_statement.excluded.expires_at,
                "updated_at": now,
            },
        ).returning(UserMemory)
        stored = cast(UserMemory | None, await session.scalar(upsert_statement))
        if stored is None:
            raise RuntimeError("Failed to store user memory")
        return stored

    @staticmethod
    async def _expire_user_memories(
        session: AsyncSession,
        user_id: str,
        now: datetime,
    ) -> None:
        await session.execute(
            update(UserMemory)
            .where(
                UserMemory.user_id == user_id,
                UserMemory.status == "ACTIVE",
                UserMemory.expires_at.is_not(None),
                UserMemory.expires_at <= now,
            )
            .values(
                status="EXPIRED",
                valid_to=UserMemory.expires_at,
                updated_at=now,
            )
        )

    @staticmethod
    def _user_memory_record(memory: UserMemory) -> UserMemoryRecord:
        return UserMemoryRecord(
            id=memory.id,
            user_id=memory.user_id,
            memory_type=cast(MemoryType, memory.memory_type),
            memory_key=cast(MemoryKey, memory.memory_key),
            value=str(memory.value.get("text") or ""),
            summary=memory.summary,
            value_identity_hash=memory.value_identity_hash,
            capture_mode=memory.capture_mode,
            confidence=float(memory.confidence),
            status=cast(MemoryStatus, memory.status),
            source_conversation_id=memory.source_conversation_id,
            source_run_id=memory.source_run_id,
            source_event_id=memory.source_event_id,
            source_message_hash=memory.source_message_hash,
            source_created_at=memory.source_created_at,
            valid_from=memory.valid_from,
            last_confirmed_at=memory.last_confirmed_at,
            valid_to=memory.valid_to,
            expires_at=memory.expires_at,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )

    async def upsert_conversation_rental_period(
        self,
        conversation_id: str,
        rental_period: RentalPeriodInput,
    ) -> None:
        statement = pg_insert(ConversationState).values(
            conversation_id=conversation_id,
            rental_start_date=rental_period.start_date,
            rental_end_date=rental_period.end_date,
            attributes={},
        )
        statement = statement.on_conflict_do_update(
            index_elements=[ConversationState.conversation_id],
            set_={
                "rental_start_date": rental_period.start_date,
                "rental_end_date": rental_period.end_date,
                "updated_at": func.current_timestamp(),
            },
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def clear_conversation_rental_period(self, conversation_id: str) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                update(ConversationState)
                .where(ConversationState.conversation_id == conversation_id)
                .values(
                    rental_start_date=None,
                    rental_end_date=None,
                    updated_at=func.current_timestamp(),
                )
            )

    async def upsert_pending_product_search(
        self,
        conversation_id: str,
        pending_search: PendingProductSearch,
    ) -> None:
        attributes = {
            "pendingProductSearch": pending_search.model_dump(
                mode="json",
                by_alias=True,
            )
        }
        statement = pg_insert(ConversationState).values(
            conversation_id=conversation_id,
            attributes=attributes,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[ConversationState.conversation_id],
            set_={
                "attributes": ConversationState.attributes.op("||")(statement.excluded.attributes),
                "updated_at": func.current_timestamp(),
            },
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def clear_pending_product_search(self, conversation_id: str) -> None:
        async with self._sessions.begin() as session:
            state = await session.get(
                ConversationState,
                conversation_id,
                with_for_update=True,
            )
            if state is None or "pendingProductSearch" not in state.attributes:
                return
            attributes = dict(state.attributes)
            attributes.pop("pendingProductSearch", None)
            state.attributes = attributes

    async def upsert_pending_rental_action(
        self,
        conversation_id: str,
        pending_action: PendingRentalAction,
    ) -> None:
        attributes = {
            "pendingRentalAction": pending_action.model_dump(
                mode="json",
                by_alias=True,
            )
        }
        statement = pg_insert(ConversationState).values(
            conversation_id=conversation_id,
            attributes=attributes,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[ConversationState.conversation_id],
            set_={
                "attributes": ConversationState.attributes.op("||")(statement.excluded.attributes),
                "updated_at": func.current_timestamp(),
            },
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def clear_pending_rental_action(self, conversation_id: str) -> None:
        async with self._sessions.begin() as session:
            state = await session.get(
                ConversationState,
                conversation_id,
                with_for_update=True,
            )
            if state is None or "pendingRentalAction" not in state.attributes:
                return
            attributes = dict(state.attributes)
            attributes.pop("pendingRentalAction", None)
            state.attributes = attributes

    async def upsert_recent_product_search(
        self,
        conversation_id: str,
        recent_search: RecentProductSearch,
    ) -> None:
        attributes = {"recentProductSearch": recent_search.model_dump(mode="json", by_alias=True)}
        statement = pg_insert(ConversationState).values(
            conversation_id=conversation_id,
            attributes=attributes,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[ConversationState.conversation_id],
            set_={
                "attributes": ConversationState.attributes.op("||")(statement.excluded.attributes),
                "updated_at": func.current_timestamp(),
            },
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def upsert_conversation_requirements(
        self,
        conversation_id: str,
        requirements: RentalRequirements,
    ) -> None:
        attributes = {"rentalRequirements": requirements.model_dump(mode="json", by_alias=True)}
        statement = pg_insert(ConversationState).values(
            conversation_id=conversation_id,
            attributes=attributes,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[ConversationState.conversation_id],
            set_={
                "attributes": ConversationState.attributes.op("||")(statement.excluded.attributes),
                "updated_at": func.current_timestamp(),
            },
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def conversation_timezone(self, conversation_id: str) -> str:
        async with self._sessions() as session:
            timezone = await session.scalar(
                select(Conversation.timezone).where(Conversation.id == conversation_id)
            )
            if timezone is None:
                raise LookupError("Conversation not found")
            return timezone

    async def conversation_state(self, conversation_id: str) -> ConversationStateMemory | None:
        async with self._sessions() as session:
            state = await session.get(ConversationState, conversation_id)
            if state is None:
                return None
            raw_requirements = state.attributes.get("rentalRequirements")
            raw_pending_search = state.attributes.get("pendingProductSearch")
            raw_pending_rental_action = state.attributes.get("pendingRentalAction")
            raw_recent_product_search = state.attributes.get("recentProductSearch")
            return ConversationStateMemory(
                rental_start_date=state.rental_start_date,
                rental_end_date=state.rental_end_date,
                rental_requirements=(
                    RentalRequirements.model_validate(raw_requirements)
                    if raw_requirements is not None
                    else None
                ),
                pending_product_search=(
                    PendingProductSearch.model_validate(raw_pending_search)
                    if raw_pending_search is not None
                    else None
                ),
                pending_rental_action=(
                    PendingRentalAction.model_validate(raw_pending_rental_action)
                    if raw_pending_rental_action is not None
                    else None
                ),
                recent_product_search=(
                    RecentProductSearch.model_validate(raw_recent_product_search)
                    if raw_recent_product_search is not None
                    else None
                ),
            )

    async def latest_conversation_summary(
        self, conversation_id: str
    ) -> ConversationSummaryMemory | None:
        async with self._sessions() as session:
            summary = await session.scalar(
                select(ConversationSummary)
                .where(ConversationSummary.conversation_id == conversation_id)
                .order_by(
                    ConversationSummary.created_at.desc(),
                    ConversationSummary.id.desc(),
                )
                .limit(1)
            )
            if summary is None:
                return None
            return ConversationSummaryMemory(
                content=summary.content,
                through_event_id=summary.through_event_id,
                source_message_count=summary.source_message_count,
                estimated_tokens=summary.estimated_tokens,
                created_at=summary.created_at,
            )

    async def recent_conversation_messages(
        self,
        conversation_id: str,
        limit: int,
        after_event_id: str | None = None,
    ) -> list[ConversationMessageMemory]:
        async with self._sessions() as session:
            statement = (
                select(RunEvent)
                .join(AgentRun)
                .where(
                    AgentRun.conversation_id == conversation_id,
                    RunEvent.event_type.in_(("user.message", "assistant.completed")),
                )
                .order_by(RunEvent.created_at.desc(), RunEvent.id.desc())
                .limit(limit)
            )
            if after_event_id is not None:
                boundary = await session.get(RunEvent, after_event_id)
                if boundary is not None:
                    statement = statement.where(
                        or_(
                            RunEvent.created_at > boundary.created_at,
                            (
                                (RunEvent.created_at == boundary.created_at)
                                & (RunEvent.id > boundary.id)
                            ),
                        )
                    )
            result = await session.scalars(statement)
            events = list(reversed(list(result)))
            return [self._memory_message(event) for event in events]

    async def conversation_messages_after(
        self,
        conversation_id: str,
        after_event_id: str | None,
        limit: int,
    ) -> list[ConversationMessageMemory]:
        async with self._sessions() as session:
            statement = (
                select(RunEvent)
                .join(AgentRun)
                .where(
                    AgentRun.conversation_id == conversation_id,
                    RunEvent.event_type.in_(("user.message", "assistant.completed")),
                )
                .order_by(RunEvent.created_at, RunEvent.id)
                .limit(limit)
            )
            if after_event_id is not None:
                boundary = await session.get(RunEvent, after_event_id)
                if boundary is not None:
                    statement = statement.where(
                        or_(
                            RunEvent.created_at > boundary.created_at,
                            (
                                (RunEvent.created_at == boundary.created_at)
                                & (RunEvent.id > boundary.id)
                            ),
                        )
                    )
            events = list(await session.scalars(statement))
            return [self._memory_message(event) for event in events]

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
        summary = ConversationSummary(
            id=new_ulid(),
            conversation_id=conversation_id,
            through_event_id=through_event_id,
            content=content,
            source_message_count=source_message_count,
            estimated_tokens=estimated_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        async with self._sessions.begin() as session:
            session.add(summary)

    @staticmethod
    def _memory_message(event: RunEvent) -> ConversationMessageMemory:
        return ConversationMessageMemory(
            event_id=event.id,
            role="user" if event.event_type == "user.message" else "assistant",
            content=str(event.payload.get("content") or ""),
            created_at=event.created_at,
        )

    async def conversation_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[ConversationMessageMemory]:
        async with self._sessions() as session:
            events = await session.scalars(
                select(RunEvent)
                .join(AgentRun)
                .where(
                    AgentRun.conversation_id == conversation_id,
                    RunEvent.event_type.in_(
                        ("user.message", "assistant.completed", "recommendation.presented")
                    ),
                )
                .order_by(RunEvent.created_at.desc(), RunEvent.sequence_no.desc())
                .limit(limit)
            )
            rows = list(reversed(list(events)))
            presentations = {
                event.run_id: event.payload
                for event in rows
                if event.event_type == "recommendation.presented"
            }
            return [
                ConversationMessageMemory(
                    event_id=event.id,
                    role="user" if event.event_type == "user.message" else "assistant",
                    content=str(event.payload.get("content") or ""),
                    created_at=event.created_at,
                    presentation=(
                        presentations.get(event.run_id)
                        if event.event_type == "assistant.completed"
                        else None
                    ),
                )
                for event in rows
                if event.event_type in ("user.message", "assistant.completed")
                and event.payload.get("content")
            ]
