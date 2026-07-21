from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from gearmate.api.agent import CreateRunRequest, list_conversation_messages
from gearmate.auth.jwt import CurrentUser
from gearmate.memory import ConversationMessageMemory


def user() -> CurrentUser:
    return CurrentUser(
        user_id="user-1",
        nickname="Demo User",
        timezone="Asia/Shanghai",
        roles=("USER",),
        access_token="token",
    )


@pytest.mark.asyncio
async def test_list_conversation_messages_returns_owned_history() -> None:
    repository = AsyncMock()
    created_at = datetime(2026, 7, 16, 8, tzinfo=UTC)
    repository.conversation_messages.return_value = [
        ConversationMessageMemory(
            event_id="event-1",
            role="user",
            content="我想买相机",
            created_at=created_at,
        ),
        ConversationMessageMemory(
            event_id="event-2",
            role="assistant",
            content="可以看看 Sony A7M4。",
            created_at=created_at,
        ),
    ]

    response = await list_conversation_messages("conversation-1", user(), repository, 100)

    repository.require_conversation.assert_awaited_once_with("conversation-1", "user-1")
    repository.conversation_messages.assert_awaited_once_with("conversation-1", 100)
    assert [message.model_dump(by_alias=True) for message in response] == [
        {
            "id": "event-1",
            "role": "user",
            "content": "我想买相机",
                "createdAt": created_at,
                "presentation": None,
        },
        {
            "id": "event-2",
            "role": "assistant",
            "content": "可以看看 Sony A7M4。",
                "createdAt": created_at,
                "presentation": None,
        },
    ]


def test_create_run_contract_has_no_rental_period() -> None:
    schema = CreateRunRequest.model_json_schema(by_alias=True)

    assert set(schema["properties"]) == {"message"}


@pytest.mark.asyncio
async def test_list_conversation_messages_hides_other_users_conversation() -> None:
    repository = AsyncMock()
    repository.require_conversation.side_effect = LookupError("Conversation not found")

    with pytest.raises(HTTPException) as raised:
        await list_conversation_messages("conversation-2", user(), repository, 100)

    assert raised.value.status_code == 404
    repository.conversation_messages.assert_not_awaited()
