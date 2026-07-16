"""add dynamic catalog use cases

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_catalog_aliases_entity_type", "catalog_aliases", type_="check")
    op.create_check_constraint(
        "ck_catalog_aliases_entity_type",
        "catalog_aliases",
        "entity_type IN ('equipment_role', 'brand', 'model', 'use_case')",
    )
    op.add_column(
        "product_search_documents",
        sa.Column(
            "use_case_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_product_search_documents_use_cases",
        "product_search_documents",
        ["use_case_ids"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_product_search_documents_use_cases",
        table_name="product_search_documents",
    )
    op.drop_column("product_search_documents", "use_case_ids")
    op.drop_constraint("ck_catalog_aliases_entity_type", "catalog_aliases", type_="check")
    op.create_check_constraint(
        "ck_catalog_aliases_entity_type",
        "catalog_aliases",
        "entity_type IN ('equipment_role', 'brand', 'model')",
    )
