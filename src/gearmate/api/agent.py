import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from time import monotonic
from typing import Annotated, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from gearmate.agent.service import RunCoordinator
from gearmate.auth.jwt import CurrentUser, current_user
from gearmate.config import Settings
from gearmate.persistence.repositories import ActiveRunConflict, AgentRepository
from gearmate.streaming.sse import encode_event, heartbeat

router = APIRouter(prefix="/api/v1", tags=["Agent"])
TERMINAL_STATUSES = {"COMPLETED", "OUTPUT_TRUNCATED", "REFUSED", "FAILED", "CANCELLED"}


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class CreateConversationRequest(ApiModel):
    title: str | None = Field(default=None, max_length=200)
    timezone: str | None = Field(default=None, max_length=64)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a valid IANA timezone") from error
        return value


class ConversationResponse(ApiModel):
    id: str
    timezone: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationMessageResponse(ApiModel):
    id: str
    role: str
    content: str
    created_at: datetime
    presentation: dict[str, object] | None = None


class CreateRunRequest(ApiModel):
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def non_blank_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class RunResponse(ApiModel):
    run_id: str
    conversation_id: str
    status: str
    stop_reason: str | None = None


def repository(request: Request) -> AgentRepository:
    return cast(AgentRepository, request.app.state.repository)


def coordinator(request: Request) -> RunCoordinator:
    return cast(RunCoordinator, request.app.state.run_coordinator)


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    body: CreateConversationRequest,
    user: Annotated[CurrentUser, Depends(current_user)],
    repo: Annotated[AgentRepository, Depends(repository)],
) -> ConversationResponse:
    conversation = await repo.create_conversation(
        user.user_id,
        body.timezone or user.timezone,
        body.title,
    )
    return ConversationResponse.model_validate(conversation, from_attributes=True)


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    user: Annotated[CurrentUser, Depends(current_user)],
    repo: Annotated[AgentRepository, Depends(repository)],
) -> list[ConversationResponse]:
    conversations = await repo.list_conversations(user.user_id)
    return [
        ConversationResponse.model_validate(item, from_attributes=True) for item in conversations
    ]


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessageResponse],
)
async def list_conversation_messages(
    conversation_id: str,
    user: Annotated[CurrentUser, Depends(current_user)],
    repo: Annotated[AgentRepository, Depends(repository)],
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[ConversationMessageResponse]:
    try:
        await repo.require_conversation(conversation_id, user.user_id)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    messages = await repo.conversation_messages(conversation_id, limit)
    return [
        ConversationMessageResponse(
            id=message.event_id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            presentation=message.presentation,
        )
        for message in messages
    ]


@router.post(
    "/conversations/{conversation_id}/runs",
    response_model=RunResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_run(
    conversation_id: str,
    body: CreateRunRequest,
    user: Annotated[CurrentUser, Depends(current_user)],
    runner: Annotated[RunCoordinator, Depends(coordinator)],
) -> RunResponse:
    try:
        run_id = await runner.start(
            conversation_id=conversation_id,
            user_id=user.user_id,
            access_token=user.access_token,
            message=body.message,
        )
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ActiveRunConflict as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return RunResponse(run_id=run_id, conversation_id=conversation_id, status="RUNNING")


@router.get("/runs/{run_id}", response_model=RunResponse, response_model_by_alias=True)
async def get_run(
    run_id: str,
    user: Annotated[CurrentUser, Depends(current_user)],
    repo: Annotated[AgentRepository, Depends(repository)],
) -> RunResponse:
    try:
        run = await repo.require_run(run_id, user.user_id)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return RunResponse(
        run_id=run.id,
        conversation_id=run.conversation_id,
        status=run.status,
        stop_reason=run.stop_reason,
    )


@router.post("/runs/{run_id}/cancel", response_model=RunResponse, response_model_by_alias=True)
async def cancel_run(
    run_id: str,
    user: Annotated[CurrentUser, Depends(current_user)],
    repo: Annotated[AgentRepository, Depends(repository)],
    runner: Annotated[RunCoordinator, Depends(coordinator)],
) -> RunResponse:
    try:
        run = await repo.require_run(run_id, user.user_id)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    if run.status not in TERMINAL_STATUSES:
        await runner.cancel(run_id)
        run = await repo.require_run(run_id, user.user_id)
    return RunResponse(
        run_id=run.id,
        conversation_id=run.conversation_id,
        status=run.status,
        stop_reason=run.stop_reason,
    )


@router.get("/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    request: Request,
    user: Annotated[CurrentUser, Depends(current_user)],
    repo: Annotated[AgentRepository, Depends(repository)],
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    try:
        await repo.require_run(run_id, user.user_id)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    cursor = after
    if last_event_id is not None:
        try:
            cursor = max(cursor, int(last_event_id))
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail="Last-Event-ID must be an integer"
            ) from error
    settings = cast(Settings, request.app.state.settings)

    async def event_stream() -> AsyncIterator[str]:
        nonlocal cursor
        last_heartbeat = monotonic()
        while True:
            if await request.is_disconnected():
                return
            events = await repo.events_after(run_id, cursor)
            for event in events:
                cursor = event.sequence_no
                yield encode_event(event)
            run = await repo.require_run(run_id, user.user_id)
            if run.status in TERMINAL_STATUSES and not events:
                return
            if monotonic() - last_heartbeat >= settings.sse_heartbeat_seconds:
                yield heartbeat()
                last_heartbeat = monotonic()
            await asyncio.sleep(settings.event_poll_interval_seconds)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
