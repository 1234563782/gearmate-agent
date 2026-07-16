import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from gearmate.app import (
    catalog_sync_loop,
    conversation_cleanup_loop,
    delete_expired_conversations,
)
from gearmate.config import Settings


@pytest.mark.asyncio
async def test_catalog_sync_loop_uses_initial_delay_then_regular_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", sleep)
    catalog_search = AsyncMock()
    settings = Settings(
        _env_file=None,
        catalog_sync_interval_seconds=900,
        catalog_sync_retry_seconds=30,
    )

    with pytest.raises(asyncio.CancelledError):
        await catalog_sync_loop(
            catalog_search,
            AsyncMock(),
            settings,
            initial_delay=settings.catalog_sync_retry_seconds,
        )

    assert delays == [30, 900]
    catalog_search.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_catalog_sync_loop_uses_retry_delay_after_refresh_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", sleep)
    catalog_search = AsyncMock()
    catalog_search.refresh.side_effect = RuntimeError("RentFlow is unavailable")
    settings = Settings(
        _env_file=None,
        catalog_sync_interval_seconds=900,
        catalog_sync_retry_seconds=30,
    )

    with pytest.raises(asyncio.CancelledError):
        await catalog_sync_loop(
            catalog_search,
            AsyncMock(),
            settings,
            initial_delay=settings.catalog_sync_interval_seconds,
        )

    assert delays == [900, 30]
    catalog_search.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_expired_conversations_uses_configured_idle_retention() -> None:
    repository = AsyncMock()
    settings = Settings(_env_file=None, conversation_retention_hours=24)
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    repository.delete_expired_conversations.return_value = 2

    deleted = await delete_expired_conversations(repository, settings, now_utc=now)

    assert deleted == 2
    repository.delete_expired_conversations.assert_awaited_once_with(
        now - timedelta(hours=24)
    )


@pytest.mark.asyncio
async def test_conversation_cleanup_loop_uses_configured_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", sleep)
    repository = AsyncMock()
    settings = Settings(_env_file=None, conversation_cleanup_interval_seconds=3600)

    with pytest.raises(asyncio.CancelledError):
        await conversation_cleanup_loop(repository, settings)

    assert delays == [3600, 3600]
    repository.delete_expired_conversations.assert_awaited_once()
