"""add pgvector product semantic index

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-15
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from gearmate.persistence.vector import Vector1024

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "product_search_documents",
        sa.Column("product_id", sa.String(length=26), nullable=False),
        sa.Column("category_id", sa.String(length=26), nullable=False),
        sa.Column("equipment_role", sa.String(length=64), nullable=False),
        sa.Column("brand", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("embedding", Vector1024(), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("product_id"),
    )
    op.create_index(
        "idx_product_search_documents_role",
        "product_search_documents",
        ["equipment_role"],
    )
    op.create_index(
        "idx_product_search_documents_brand",
        "product_search_documents",
        ["brand"],
    )
    op.create_index(
        "idx_product_search_documents_embedding_hnsw",
        "product_search_documents",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(
        "idx_product_search_documents_embedding_hnsw",
        table_name="product_search_documents",
    )
    op.drop_index("idx_product_search_documents_brand", table_name="product_search_documents")
    op.drop_index("idx_product_search_documents_role", table_name="product_search_documents")
    op.drop_table("product_search_documents")
