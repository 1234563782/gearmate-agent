"""remove rental conversation state

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-21
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE conversation_states
        SET attributes = attributes - 'pendingRentalAction' - 'rentalRequirements'
        WHERE attributes ? 'pendingRentalAction'
           OR attributes ? 'rentalRequirements'
        """
    )
    op.drop_column("conversation_states", "rental_end_date")
    op.drop_column("conversation_states", "rental_start_date")


def downgrade() -> None:
    op.add_column(
        "conversation_states",
        sa.Column("rental_start_date", sa.Date(), nullable=True),
    )
    op.add_column(
        "conversation_states",
        sa.Column("rental_end_date", sa.Date(), nullable=True),
    )
