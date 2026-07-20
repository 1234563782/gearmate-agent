"""add canonical user-memory identity and audit timestamps

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-20
"""

from collections.abc import Sequence
from hashlib import sha256

import sqlalchemy as sa

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _hash(value: str) -> str:
    return sha256(value.casefold().encode("utf-8")).hexdigest()


def upgrade() -> None:
    op.add_column(
        "user_memories",
        sa.Column("value_identity_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "user_memories",
        sa.Column("last_confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )

    bind = op.get_bind()
    op.drop_constraint("ck_user_memories_status", "user_memories", type_="check")
    bind.execute(
        sa.text(
            """
            INSERT INTO catalog_aliases (
                alias, entity_type, canonical_value, locale, source, active, updated_at
            ) VALUES (
                :alias, 'brand', 'Sony', 'zh-CN', 'user_memory_seed', true,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (alias, entity_type) DO NOTHING
            """
        ),
        {"alias": "索尼"},
    )

    alias_rows = bind.execute(
        sa.text(
            """
            SELECT alias, entity_type, canonical_value
            FROM catalog_aliases
            WHERE active = true
            """
        )
    )
    aliases: dict[tuple[str, str], str] = {}
    for row in alias_rows:
        alias = str(row.alias)
        entity_type = str(row.entity_type)
        canonical = str(row.canonical_value)
        aliases[(entity_type, alias.casefold())] = canonical
        aliases.setdefault((entity_type, canonical.casefold()), canonical)

    key_entity_types = {
        "preferred_brand": "brand",
        "excluded_brand": "brand",
        "preferred_equipment_role": "equipment_role",
        "preferred_use_case": "use_case",
    }
    memory_rows = bind.execute(
        sa.text(
            """
            SELECT id, memory_key, value, source_created_at, valid_from
            FROM user_memories
            """
        )
    )
    for row in memory_rows:
        payload = row.value
        value = str(payload.get("text") or "") if isinstance(payload, dict) else str(payload)
        entity_type = key_entity_types.get(str(row.memory_key))
        identity = aliases.get((entity_type, value.casefold()), value) if entity_type else value
        bind.execute(
            sa.text(
                """
                UPDATE user_memories
                SET value_identity_hash = :value_identity_hash,
                    normalized_hash = :normalized_hash,
                    last_confirmed_at = COALESCE(source_created_at, valid_from)
                WHERE id = :id
                """
            ),
            {
                "id": row.id,
                "value_identity_hash": _hash(identity),
                "normalized_hash": _hash(f"{row.memory_key}\n{identity}"),
            },
        )

    bind.execute(
        sa.text(
            """
            UPDATE user_memories
            SET status = 'EXPIRED',
                valid_to = expires_at,
                updated_at = CURRENT_TIMESTAMP
            WHERE status = 'ACTIVE'
              AND expires_at IS NOT NULL
              AND expires_at <= CURRENT_TIMESTAMP
            """
        )
    )
    op.drop_index("uq_user_memories_active_fact", table_name="user_memories")
    bind.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT id,
                       row_number() OVER (
                           PARTITION BY user_id, memory_key, value_identity_hash
                           ORDER BY updated_at DESC, id DESC
                       ) AS position
                FROM user_memories
                WHERE status = 'ACTIVE'
            )
            UPDATE user_memories AS memory
            SET status = 'SUPERSEDED',
                valid_to = COALESCE(memory.valid_to, CURRENT_TIMESTAMP),
                updated_at = CURRENT_TIMESTAMP
            FROM ranked
            WHERE memory.id = ranked.id AND ranked.position > 1
            """
        )
    )
    bind.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT id,
                       row_number() OVER (
                           PARTITION BY user_id, value_identity_hash
                           ORDER BY updated_at DESC, id DESC
                       ) AS position
                FROM user_memories
                WHERE status = 'ACTIVE'
                  AND memory_key IN ('preferred_brand', 'excluded_brand')
            )
            UPDATE user_memories AS memory
            SET status = 'SUPERSEDED',
                valid_to = COALESCE(memory.valid_to, CURRENT_TIMESTAMP),
                updated_at = CURRENT_TIMESTAMP
            FROM ranked
            WHERE memory.id = ranked.id AND ranked.position > 1
            """
        )
    )
    op.create_check_constraint(
        "ck_user_memories_status",
        "user_memories",
        "status IN ('ACTIVE', 'SUPERSEDED', 'DELETED', 'EXPIRED')",
    )
    op.alter_column("user_memories", "value_identity_hash", nullable=False)
    op.alter_column("user_memories", "last_confirmed_at", nullable=False)
    op.create_index(
        "idx_user_memories_user_active_identity",
        "user_memories",
        ["user_id", "status", "value_identity_hash"],
    )
    op.create_index(
        "uq_user_memories_active_identity",
        "user_memories",
        ["user_id", "memory_key", "value_identity_hash"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            UPDATE user_memories
            SET status = 'SUPERSEDED'
            WHERE status = 'EXPIRED'
            """
        )
    )
    op.drop_index("uq_user_memories_active_identity", table_name="user_memories")
    op.drop_index("idx_user_memories_user_active_identity", table_name="user_memories")
    op.create_index(
        "uq_user_memories_active_fact",
        "user_memories",
        ["user_id", "memory_key", "normalized_hash"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )
    op.drop_constraint("ck_user_memories_status", "user_memories", type_="check")
    op.create_check_constraint(
        "ck_user_memories_status",
        "user_memories",
        "status IN ('ACTIVE', 'SUPERSEDED', 'DELETED')",
    )
    op.drop_column("user_memories", "last_confirmed_at")
    op.drop_column("user_memories", "value_identity_hash")
    bind.execute(
        sa.text(
            """
            DELETE FROM catalog_aliases
            WHERE alias = :alias
              AND entity_type = 'brand'
              AND source = 'user_memory_seed'
            """
        ),
        {"alias": "索尼"},
    )
