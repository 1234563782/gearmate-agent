from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from gearmate.persistence.models import UserMemory
from gearmate.persistence.repositories import AgentRepository
from gearmate.user_memory import MemoryKey, UserMemoryService, UserMemoryWrite


def memory_write(
    *,
    key: MemoryKey = "preferred_brand",
    value: str = "Sony",
    source_created_at: datetime | None = None,
) -> UserMemoryWrite:
    reference = source_created_at or datetime(2026, 7, 20, tzinfo=UTC)
    identity_hash = UserMemoryService._value_identity_hash(value)
    return UserMemoryWrite(
        user_id="user-1",
        memory_type="CONSTRAINT" if key == "excluded_brand" else "PREFERENCE",
        memory_key=key,
        value=value,
        summary=f"{key}: {value}",
        normalized_hash=UserMemoryService._normalized_hash(key, value),
        value_identity_hash=identity_hash,
        capture_mode="MODEL_EXTRACTED",
        confidence=0.98,
        source_conversation_id="conversation-1",
        source_run_id="run-1",
        source_event_id="event-1",
        source_message_hash="a" * 64,
        source_created_at=reference,
        valid_from=reference,
        expires_at=reference + timedelta(days=180),
    )


async def test_brand_conflict_uses_identity_hash_without_database_lower() -> None:
    session = AsyncMock(spec=AsyncSession)
    memory = memory_write(key="excluded_brand", value="STRASSE")

    await AgentRepository._supersede_user_memory_conflicts(
        session,
        memory,
        datetime(2026, 7, 20, tzinfo=UTC),
    )

    statement = session.execute.await_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect())).casefold()
    assert "value_identity_hash" in sql
    assert "lower(" not in sql


async def test_catalog_identity_matching_uses_python_unicode_casefold() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.execute.return_value = [("Straße", "Straße")]
    sessions = MagicMock()
    sessions.return_value.__aenter__.return_value = session
    repository = AgentRepository(sessions)

    identity = await repository.canonical_user_memory_identity(
        "preferred_brand",
        "STRASSE",
    )

    assert identity == "Straße"


async def test_expiration_transitions_active_rows_before_writes() -> None:
    session = AsyncMock(spec=AsyncSession)
    now = datetime(2026, 7, 20, tzinfo=UTC)

    await AgentRepository._expire_user_memories(session, "user-1", now)

    statement = session.execute.await_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled).replace("\n", " ")
    assert "user_memories.expires_at <=" in sql
    assert "user_memories.status =" in sql
    assert "EXPIRED" in compiled.params.values()


async def test_user_memory_writes_use_transaction_scoped_user_lock() -> None:
    session = AsyncMock(spec=AsyncSession)

    await AgentRepository._lock_user_memories(session, "user-1")

    statement, parameters = session.execute.await_args.args
    assert "pg_advisory_xact_lock" in str(statement)
    assert parameters == {"user_id": "user-1"}


async def test_duplicate_confirmation_refreshes_fact_and_latest_audit_fields() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = UserMemory()
    now = datetime(2026, 7, 21, tzinfo=UTC)
    memory = memory_write(value="SONY", source_created_at=now)

    await AgentRepository._upsert_user_memory_in_session(session, memory, now)

    statement = session.scalar.await_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect())).replace("\n", " ")
    assert "ON CONFLICT (user_id, memory_key, value_identity_hash)" in sql
    assert "value = excluded.value" in sql
    assert "capture_mode = excluded.capture_mode" in sql
    assert "source_created_at = excluded.source_created_at" in sql
    assert "last_confirmed_at = excluded.last_confirmed_at" in sql
    assert "valid_from = excluded.valid_from" not in sql
