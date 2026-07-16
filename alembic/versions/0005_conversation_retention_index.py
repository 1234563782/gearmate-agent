"""add conversation retention index

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-16
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("idx_conversations_updated_at", "conversations", ["updated_at"])


def downgrade() -> None:
    op.drop_index("idx_conversations_updated_at", table_name="conversations")
