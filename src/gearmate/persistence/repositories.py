from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gearmate.ids import new_ulid
from gearmate.persistence.models import AgentRun, Conversation, RunEvent


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
                select(AgentRun).join(Conversation).where(
                    AgentRun.id == run_id,
                    Conversation.user_id == user_id,
                )
            )
            run = row.scalar_one_or_none()
            if run is None:
                raise LookupError("Run not found")
            return RunRecord(
                run.id, run.conversation_id, run.status, run.stop_reason, run.state
            )

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
