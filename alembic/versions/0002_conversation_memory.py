"""add conversation state and rolling summaries

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_states",
        sa.Column("conversation_id", sa.String(length=26), nullable=False),
        sa.Column("rental_start_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rental_end_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "attributes",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("conversation_id", sa.String(length=26), nullable=False),
        sa.Column("through_event_id", sa.String(length=26), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_message_count", sa.BigInteger(), nullable=False),
        sa.Column("estimated_tokens", sa.BigInteger(), nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "through_event_id",
            name="uk_conversation_summaries_boundary",
        ),
    )
    op.create_index(
        "idx_conversation_summaries_conversation_created",
        "conversation_summaries",
        ["conversation_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_conversation_summaries_conversation_created",
        table_name="conversation_summaries",
    )
    op.drop_table("conversation_summaries")
    op.drop_table("conversation_states")
