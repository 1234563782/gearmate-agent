"""create GearMate conversation, run, and event store

Revision ID: 0001
Revises:
Create Date: 2026-07-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("user_id", sa.String(length=26), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_conversations_user_updated", "conversations", ["user_id", "updated_at"])
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("conversation_id", sa.String(length=26), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("model_provider", sa.String(length=64), nullable=True),
        sa.Column("model_id", sa.String(length=128), nullable=True),
        sa.Column(
            "state",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("stop_reason", sa.String(length=64), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("prompt_hash", sa.String(length=64), nullable=True),
        sa.Column("input_tokens", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("model_rounds", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("tool_call_count", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('RUNNING', 'TOOL_REQUESTED', 'COMPLETED', "
            "'OUTPUT_TRUNCATED', 'REFUSED', 'FAILED', 'CANCELLED')",
            name="ck_agent_runs_status",
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_agent_runs_conversation_created",
        "agent_runs",
        ["conversation_id", "created_at"],
    )
    op.create_index(
        "uq_agent_runs_one_active_per_conversation",
        "agent_runs",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('RUNNING', 'TOOL_REQUESTED')"),
    )
    op.create_table(
        "run_events",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("run_id", sa.String(length=26), nullable=False),
        sa.Column("sequence_no", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sequence_no", name="uk_run_events_run_sequence"),
    )
    op.create_index("idx_run_events_run_created", "run_events", ["run_id", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_run_events_run_created", table_name="run_events")
    op.drop_table("run_events")
    op.drop_index("uq_agent_runs_one_active_per_conversation", table_name="agent_runs")
    op.drop_index("idx_agent_runs_conversation_created", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_index("idx_conversations_user_updated", table_name="conversations")
    op.drop_table("conversations")
