"""store conversation rental periods as dates

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-18
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DELETE FROM conversations")
    op.drop_column("conversation_states", "rental_start_at")
    op.drop_column("conversation_states", "rental_end_at")
    op.add_column(
        "conversation_states",
        sa.Column("rental_start_date", sa.Date(), nullable=True),
    )
    op.add_column(
        "conversation_states",
        sa.Column("rental_end_date", sa.Date(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_states", "rental_end_date")
    op.drop_column("conversation_states", "rental_start_date")
    op.add_column(
        "conversation_states",
        sa.Column("rental_start_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "conversation_states",
        sa.Column("rental_end_at", sa.DateTime(timezone=True), nullable=True),
    )
