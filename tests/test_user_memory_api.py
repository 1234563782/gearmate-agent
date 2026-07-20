from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from gearmate.api.user_memory import (
    UpdateUserMemoryRequest,
    delete_all_user_memories,
    delete_user_memory,
    list_user_memories,
    update_user_memory,
)
from gearmate.auth.jwt import CurrentUser
from gearmate.user_memory import UserMemoryRecord


def user() -> CurrentUser:
    return CurrentUser(
        user_id="user-1",
        nickname="Demo User",
        timezone="Asia/Shanghai",
        roles=("USER",),
        access_token="token",
    )


def memory() -> UserMemoryRecord:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    return UserMemoryRecord(
        id="memory-1",
        user_id="user-1",
        memory_type="PREFERENCE",
        memory_key="preferred_brand",
        value="Sony",
        summary="preferred_brand: Sony",
        value_identity_hash="b" * 64,
        capture_mode="EXPLICIT",
        confidence=1.0,
        status="ACTIVE",
        source_conversation_id=None,
        source_run_id=None,
        source_event_id=None,
        source_message_hash="a" * 64,
        source_created_at=now,
        valid_from=now,
        last_confirmed_at=now,
        valid_to=None,
        expires_at=None,
        created_at=now,
        updated_at=now,
    )


async def test_list_memories_uses_authenticated_user_only() -> None:
    service = AsyncMock()
    service.list_memories.return_value = [memory()]

    result = await list_user_memories(user(), service)

    service.list_memories.assert_awaited_once_with("user-1")
    assert result[0].value == "Sony"
    assert result[0].last_confirmed_at == datetime(2026, 7, 20, tzinfo=UTC)


async def test_update_memory_uses_authenticated_user_only() -> None:
    service = AsyncMock()
    service.replace_memory.return_value = memory()

    result = await update_user_memory(
        "memory-1",
        UpdateUserMemoryRequest(value="Sony"),
        user(),
        service,
    )

    service.replace_memory.assert_awaited_once_with("user-1", "memory-1", "Sony")
    assert result.memory_key == "preferred_brand"


async def test_delete_memory_hides_other_users_memory() -> None:
    service = AsyncMock()
    service.delete_memory.return_value = False

    with pytest.raises(HTTPException) as raised:
        await delete_user_memory("memory-2", user(), service)

    assert raised.value.status_code == 404
    service.delete_memory.assert_awaited_once_with("user-1", "memory-2")


async def test_delete_all_memories_returns_count() -> None:
    service = AsyncMock()
    service.delete_all_memories.return_value = 3

    result = await delete_all_user_memories(user(), service)

    assert result.deleted == 3
    service.delete_all_memories.assert_awaited_once_with("user-1")
