"""add catalog aliases

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-15
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "catalog_aliases",
        sa.Column("alias", sa.String(length=128), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("canonical_value", sa.String(length=128), nullable=False),
        sa.Column("locale", sa.String(length=16), server_default="und", nullable=False),
        sa.Column("source", sa.String(length=32), server_default="manual", nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "entity_type IN ('equipment_role', 'brand', 'model')",
            name="ck_catalog_aliases_entity_type",
        ),
        sa.PrimaryKeyConstraint("alias", "entity_type"),
    )
    op.create_index("idx_catalog_aliases_active", "catalog_aliases", ["active"])
    aliases = sa.table(
        "catalog_aliases",
        sa.column("alias", sa.String()),
        sa.column("entity_type", sa.String()),
        sa.column("canonical_value", sa.String()),
        sa.column("locale", sa.String()),
        sa.column("source", sa.String()),
    )
    op.bulk_insert(
        aliases,
        [
            {
                "alias": "电脑",
                "entity_type": "equipment_role",
                "canonical_value": "laptop",
                "locale": "zh-CN",
                "source": "seed",
            },
            {
                "alias": "笔记本",
                "entity_type": "equipment_role",
                "canonical_value": "laptop",
                "locale": "zh-CN",
                "source": "seed",
            },
            {
                "alias": "苹果电脑",
                "entity_type": "equipment_role",
                "canonical_value": "laptop",
                "locale": "zh-CN",
                "source": "seed",
            },
            {
                "alias": "苹果电脑",
                "entity_type": "brand",
                "canonical_value": "Apple",
                "locale": "zh-CN",
                "source": "seed",
            },
            {
                "alias": "苹果",
                "entity_type": "brand",
                "canonical_value": "Apple",
                "locale": "zh-CN",
                "source": "seed",
            },
            {
                "alias": "Mac",
                "entity_type": "brand",
                "canonical_value": "Apple",
                "locale": "en",
                "source": "seed",
            },
            {
                "alias": "大疆",
                "entity_type": "brand",
                "canonical_value": "DJI",
                "locale": "zh-CN",
                "source": "seed",
            },
            {
                "alias": "微单",
                "entity_type": "equipment_role",
                "canonical_value": "camera",
                "locale": "zh-CN",
                "source": "seed",
            },
            {
                "alias": "无人机",
                "entity_type": "equipment_role",
                "canonical_value": "drone",
                "locale": "zh-CN",
                "source": "seed",
            },
            {
                "alias": "话筒",
                "entity_type": "equipment_role",
                "canonical_value": "microphone",
                "locale": "zh-CN",
                "source": "seed",
            },
        ],
    )


def downgrade() -> None:
    op.drop_index("idx_catalog_aliases_active", table_name="catalog_aliases")
    op.drop_table("catalog_aliases")
