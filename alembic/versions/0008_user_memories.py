"""add user-scoped long-term memories

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_memories",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("user_id", sa.String(length=26), nullable=False),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column("memory_key", sa.String(length=64), nullable=False),
        sa.Column(
            "value",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("normalized_hash", sa.String(length=64), nullable=False),
        sa.Column("capture_mode", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source_conversation_id", sa.String(length=26), nullable=True),
        sa.Column("source_run_id", sa.String(length=26), nullable=True),
        sa.Column("source_event_id", sa.String(length=26), nullable=True),
        sa.Column("source_message_hash", sa.String(length=64), nullable=False),
        sa.Column("source_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "memory_type IN ('PREFERENCE', 'CONSTRAINT')",
            name="ck_user_memories_type",
        ),
        sa.CheckConstraint(
            "memory_key IN ('preferred_brand', 'excluded_brand', "
            "'preferred_equipment_role', 'preferred_use_case', 'language')",
            name="ck_user_memories_key",
        ),
        sa.CheckConstraint(
            "capture_mode IN ('EXPLICIT', 'MODEL_EXTRACTED')",
            name="ck_user_memories_capture_mode",
        ),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'SUPERSEDED', 'DELETED')",
            name="ck_user_memories_status",
        ),
        sa.ForeignKeyConstraint(
            ["source_conversation_id"],
            ["conversations.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_run_id"],
            ["agent_runs.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_event_id"],
            ["run_events.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_user_memories_user_active_type",
        "user_memories",
        ["user_id", "status", "memory_type"],
    )
    op.create_index(
        "idx_user_memories_user_key_updated",
        "user_memories",
        ["user_id", "memory_key", "updated_at"],
    )
    op.create_index(
        "idx_user_memories_expires_at",
        "user_memories",
        ["expires_at"],
    )
    op.create_index(
        "uq_user_memories_active_fact",
        "user_memories",
        ["user_id", "memory_key", "normalized_hash"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )


def downgrade() -> None:
    op.drop_index("uq_user_memories_active_fact", table_name="user_memories")
    op.drop_index("idx_user_memories_expires_at", table_name="user_memories")
    op.drop_index("idx_user_memories_user_key_updated", table_name="user_memories")
    op.drop_index("idx_user_memories_user_active_type", table_name="user_memories")
    op.drop_table("user_memories")
