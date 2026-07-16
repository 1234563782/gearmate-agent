from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
    AgentRun,
    Conversation,
    ConversationState,
    ConversationSummary,
    RunEvent,
)
from gearmate.requirements import RentalRequirements
from gearmate.search import RecentProductSearch
from gearmate.tools.contracts import RentalPeriodInput


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

    async def upsert_conversation_rental_period(
        self,
        conversation_id: str,
        rental_period: RentalPeriodInput,
    ) -> None:
        statement = pg_insert(ConversationState).values(
            conversation_id=conversation_id,
            rental_start_at=rental_period.start_at,
            rental_end_at=rental_period.end_at,
            attributes={},
        )
        statement = statement.on_conflict_do_update(
            index_elements=[ConversationState.conversation_id],
            set_={
                "rental_start_at": rental_period.start_at,
                "rental_end_at": rental_period.end_at,
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
                    rental_start_at=None,
                    rental_end_at=None,
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
        attributes = {
            "recentProductSearch": recent_search.model_dump(mode="json", by_alias=True)
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
                rental_start_at=state.rental_start_at,
                rental_end_at=state.rental_end_at,
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
    ) -> list[tuple[str, str]]:
        async with self._sessions() as session:
            events = await session.scalars(
                select(RunEvent)
                .join(AgentRun)
                .where(
                    AgentRun.conversation_id == conversation_id,
                    RunEvent.event_type.in_(("user.message", "assistant.completed")),
                )
                .order_by(RunEvent.created_at.desc(), RunEvent.sequence_no.desc())
                .limit(limit)
            )
            rows = list(reversed(list(events)))
            return [
                (
                    "user" if event.event_type == "user.message" else "assistant",
                    str(event.payload.get("content") or ""),
                )
                for event in rows
                if event.payload.get("content")
            ]
