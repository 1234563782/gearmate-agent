from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from gearmate.auth.jwt import CurrentUser, current_user
from gearmate.user_memory import UserMemoryRecord, UserMemoryService

router = APIRouter(prefix="/api/v1/me/memories", tags=["User Memory"])


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class UserMemoryResponse(ApiModel):
    id: str
    memory_type: str
    memory_key: str
    value: str
    capture_mode: str
    confidence: float
    valid_from: datetime
    last_confirmed_at: datetime
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


class UpdateUserMemoryRequest(ApiModel):
    value: str = Field(min_length=1, max_length=128)


class DeleteUserMemoriesResponse(ApiModel):
    deleted: int


def memory_service(request: Request) -> UserMemoryService:
    return cast(UserMemoryService, request.app.state.user_memory_service)


def response(memory: UserMemoryRecord) -> UserMemoryResponse:
    return UserMemoryResponse(
        id=memory.id,
        memory_type=memory.memory_type,
        memory_key=memory.memory_key,
        value=memory.value,
        capture_mode=memory.capture_mode,
        confidence=memory.confidence,
        valid_from=memory.valid_from,
        last_confirmed_at=memory.last_confirmed_at,
        expires_at=memory.expires_at,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


@router.get("", response_model=list[UserMemoryResponse], response_model_by_alias=True)
async def list_user_memories(
    user: Annotated[CurrentUser, Depends(current_user)],
    service: Annotated[UserMemoryService, Depends(memory_service)],
) -> list[UserMemoryResponse]:
    return [response(item) for item in await service.list_memories(user.user_id)]


@router.patch(
    "/{memory_id}",
    response_model=UserMemoryResponse,
    response_model_by_alias=True,
)
async def update_user_memory(
    memory_id: str,
    body: UpdateUserMemoryRequest,
    user: Annotated[CurrentUser, Depends(current_user)],
    service: Annotated[UserMemoryService, Depends(memory_service)],
) -> UserMemoryResponse:
    try:
        updated = await service.replace_memory(user.user_id, memory_id, body.value)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    return response(updated)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_memory(
    memory_id: str,
    user: Annotated[CurrentUser, Depends(current_user)],
    service: Annotated[UserMemoryService, Depends(memory_service)],
) -> None:
    if not await service.delete_memory(user.user_id, memory_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User memory not found")


@router.delete("", response_model=DeleteUserMemoriesResponse)
async def delete_all_user_memories(
    user: Annotated[CurrentUser, Depends(current_user)],
    service: Annotated[UserMemoryService, Depends(memory_service)],
) -> DeleteUserMemoriesResponse:
    deleted = await service.delete_all_memories(user.user_id)
    return DeleteUserMemoriesResponse(deleted=deleted)
