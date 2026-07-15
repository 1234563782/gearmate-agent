from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from gearmate.persistence.vector import Vector1024

ACTIVE_RUN_STATUSES = frozenset({"RUNNING", "TOOL_REQUESTED"})
# The first real Alembic revision must create this as a partial unique index.
ACTIVE_RUN_PARTIAL_UNIQUE_INDEX = "uq_agent_runs_one_active_per_conversation"


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_conversations_user_updated", "user_id", "updated_at"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(26), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('RUNNING', 'TOOL_REQUESTED', 'COMPLETED', "
            "'OUTPUT_TRUNCATED', 'REFUSED', 'FAILED', 'CANCELLED')",
            name="ck_agent_runs_status",
        ),
        Index("idx_agent_runs_conversation_created", "conversation_id", "created_at"),
        Index(
            ACTIVE_RUN_PARTIAL_UNIQUE_INDEX,
            "conversation_id",
            unique=True,
            postgresql_where=text("status IN ('RUNNING', 'TOOL_REQUESTED')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(26),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    model_provider: Mapped[str | None] = mapped_column(String(64))
    model_id: Mapped[str | None] = mapped_column(String(128))
    state: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    error_code: Mapped[str | None] = mapped_column(String(64))
    stop_reason: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))
    model_rounds: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))
    tool_call_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RunEvent(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence_no", name="uk_run_events_run_sequence"),
        Index("idx_run_events_run_created", "run_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(26),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence_no: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )


class ConversationState(Base):
    __tablename__ = "conversation_states"

    conversation_id: Mapped[str] = mapped_column(
        String(26),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    rental_start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rental_end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "through_event_id",
            name="uk_conversation_summaries_boundary",
        ),
        Index(
            "idx_conversation_summaries_conversation_created",
            "conversation_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(26),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    through_event_id: Mapped[str] = mapped_column(String(26), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_message_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    estimated_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )


class ProductSearchDocument(Base):
    __tablename__ = "product_search_documents"
    __table_args__ = (
        Index("idx_product_search_documents_role", "equipment_role"),
        Index("idx_product_search_documents_brand", "brand"),
        Index(
            "idx_product_search_documents_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    product_id: Mapped[str] = mapped_column(String(26), primary_key=True)
    category_id: Mapped[str] = mapped_column(String(26), nullable=False)
    equipment_role: Mapped[str] = mapped_column(String(64), nullable=False)
    brand: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector1024(), nullable=False)
    active: Mapped[bool] = mapped_column(nullable=False, server_default=text("true"))
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
