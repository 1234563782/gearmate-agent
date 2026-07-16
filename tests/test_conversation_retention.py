from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from gearmate.persistence.repositories import AgentRepository


class FakeSessionFactory:
    def __init__(self) -> None:
        self.session = AsyncMock(spec=AsyncSession)

    @asynccontextmanager
    async def begin(self) -> Any:
        yield self.session


@pytest.mark.asyncio
async def test_delete_expired_conversations_skips_active_runs() -> None:
    sessions = FakeSessionFactory()
    sessions.session.execute.return_value = MagicMock(rowcount=3)
    repository = AgentRepository(sessions)  # type: ignore[arg-type]
    cutoff = datetime(2026, 7, 15, 12, tzinfo=UTC)

    deleted = await repository.delete_expired_conversations(cutoff)

    assert deleted == 3
    statement = sessions.session.execute.await_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect())).replace("\n", " ")
    assert "DELETE FROM conversations" in sql
    assert "conversations.updated_at <" in sql
    assert "NOT (EXISTS" in sql
    assert "agent_runs.status IN" in sql
