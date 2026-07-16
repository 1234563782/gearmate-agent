import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, cast

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gearmate.embeddings import EmbeddingPort
from gearmate.persistence.models import CatalogAlias, ProductSearchDocument
from gearmate.rentflow.client import RentFlowClient
from gearmate.tools.contracts import ProductDetail, ProductSearchInput, ProductSummary


@dataclass(frozen=True, slots=True)
class CatalogDocument:
    product_id: str
    category_id: str
    equipment_role: str
    brand: str
    model: str
    name: str
    description: str
    search_text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class SemanticProductCandidate:
    product_id: str
    score: float
    vector_score: float
    lexical_score: float


@dataclass(frozen=True, slots=True)
class CatalogVocabulary:
    equipment_roles: tuple[str, ...] = ()
    brands: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    aliases: tuple["CatalogAliasTerm", ...] = ()


@dataclass(frozen=True, slots=True)
class CatalogAliasTerm:
    alias: str
    entity_type: str
    canonical_value: str


@dataclass(frozen=True, slots=True)
class CatalogIndexStats:
    discovered: int
    indexed: int
    unchanged: int
    deactivated: int


class CatalogSearchRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def content_hashes(self) -> dict[str, str]:
        async with self._sessions() as session:
            rows = await session.execute(
                select(ProductSearchDocument.product_id, ProductSearchDocument.content_hash).where(
                    ProductSearchDocument.active.is_(True)
                )
            )
            return {str(product_id): str(content_hash) for product_id, content_hash in rows}

    async def upsert(
        self,
        document: CatalogDocument,
        embedding_model: str,
        embedding: tuple[float, ...],
    ) -> None:
        statement = text(
            """
            INSERT INTO product_search_documents (
                product_id, category_id, equipment_role, brand, model, name, description,
                search_text, content_hash, embedding_model, embedding, active, indexed_at
            ) VALUES (
                :product_id, :category_id, :equipment_role, :brand, :model, :name, :description,
                :search_text, :content_hash, :embedding_model, CAST(:embedding AS vector),
                true, CURRENT_TIMESTAMP
            )
            ON CONFLICT (product_id) DO UPDATE SET
                category_id = EXCLUDED.category_id,
                equipment_role = EXCLUDED.equipment_role,
                brand = EXCLUDED.brand,
                model = EXCLUDED.model,
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                search_text = EXCLUDED.search_text,
                content_hash = EXCLUDED.content_hash,
                embedding_model = EXCLUDED.embedding_model,
                embedding = EXCLUDED.embedding,
                active = true,
                indexed_at = CURRENT_TIMESTAMP
            """
        )
        parameters = {
            **asdict(document),
            "embedding_model": embedding_model,
            "embedding": json.dumps(embedding, separators=(",", ":")),
        }
        async with self._sessions.begin() as session:
            await session.execute(statement, parameters)

    async def deactivate_missing(self, active_product_ids: set[str]) -> int:
        statement = update(ProductSearchDocument).where(ProductSearchDocument.active.is_(True))
        if active_product_ids:
            statement = statement.where(
                ProductSearchDocument.product_id.not_in(active_product_ids)
            )
        statement = statement.values(active=False)
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
            return int(cast(Any, result).rowcount or 0)

    async def vocabulary(self, limit: int = 200) -> CatalogVocabulary:
        async with self._sessions() as session:
            role_rows = await session.scalars(
                select(ProductSearchDocument.equipment_role)
                .where(ProductSearchDocument.active.is_(True))
                .distinct()
                .order_by(ProductSearchDocument.equipment_role)
                .limit(limit)
            )
            brand_rows = await session.scalars(
                select(ProductSearchDocument.brand)
                .where(ProductSearchDocument.active.is_(True))
                .distinct()
                .order_by(ProductSearchDocument.brand)
                .limit(limit)
            )
            model_rows = await session.scalars(
                select(ProductSearchDocument.model)
                .where(ProductSearchDocument.active.is_(True))
                .distinct()
                .order_by(ProductSearchDocument.model)
                .limit(limit)
            )
            alias_rows = await session.execute(
                select(
                    CatalogAlias.alias,
                    CatalogAlias.entity_type,
                    CatalogAlias.canonical_value,
                )
                .where(CatalogAlias.active.is_(True))
                .order_by(CatalogAlias.alias, CatalogAlias.entity_type)
                .limit(limit)
            )
            return CatalogVocabulary(
                equipment_roles=tuple(role_rows),
                brands=tuple(brand_rows),
                models=tuple(model_rows),
                aliases=tuple(
                    CatalogAliasTerm(
                        alias=str(alias),
                        entity_type=str(entity_type),
                        canonical_value=str(canonical_value),
                    )
                    for alias, entity_type, canonical_value in alias_rows
                ),
            )

    async def semantic_search(
        self,
        embedding: tuple[float, ...],
        *,
        query: str,
        equipment_role: str | None,
        brand: str | None,
        model: str | None,
        limit: int,
    ) -> tuple[SemanticProductCandidate, ...]:
        filters = ["active = true"]
        parameters: dict[str, object] = {
            "embedding": json.dumps(embedding, separators=(",", ":")),
            "limit": limit,
            "query": query,
            "pattern": f"%{query.casefold()}%",
        }
        if equipment_role is not None:
            filters.append("equipment_role = :equipment_role")
            parameters["equipment_role"] = equipment_role
        if brand is not None:
            filters.append("LOWER(brand) = LOWER(:brand)")
            parameters["brand"] = brand
        if model is not None:
            filters.append("LOWER(model) = LOWER(:model)")
            parameters["model"] = model
        statement = text(
            "SELECT product_id, "
            "1 - (embedding <=> CAST(:embedding AS vector)) AS vector_score, "
            "CASE "
            "WHEN LOWER(model) = LOWER(:query) THEN 1.0 "
            "WHEN LOWER(name) = LOWER(:query) THEN 0.9 "
            "WHEN LOWER(model) LIKE :pattern OR LOWER(name) LIKE :pattern THEN 0.7 "
            "WHEN LOWER(brand) = LOWER(:query) THEN 0.5 "
            "ELSE 0.0 END AS lexical_score "
            "FROM product_search_documents WHERE "
            + " AND ".join(filters)
            + " ORDER BY embedding <=> CAST(:embedding AS vector), product_id LIMIT :limit"
        )
        async with self._sessions() as session:
            rows = await session.execute(statement, parameters)
            return tuple(
                SemanticProductCandidate(
                    product_id=str(row.product_id),
                    score=0.0,
                    vector_score=float(row.vector_score),
                    lexical_score=float(row.lexical_score),
                )
                for row in rows
            )


class CatalogSearchService:
    def __init__(
        self,
        repository: CatalogSearchRepository,
        embeddings: EmbeddingPort,
        *,
        batch_size: int,
        top_k: int,
        max_concurrency: int,
        min_score: float,
        vector_weight: float,
        lexical_weight: float,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._batch_size = batch_size
        self._top_k = top_k
        self._max_concurrency = max_concurrency
        self._min_score = min_score
        self._vector_weight = vector_weight
        self._lexical_weight = lexical_weight

    async def refresh(self, rentflow: RentFlowClient) -> CatalogIndexStats:
        summaries: list[ProductSummary] = []
        page = 0
        while True:
            result = await rentflow.search_products(ProductSearchInput(page=page, size=100))
            summaries.extend(result.items)
            if page + 1 >= result.total_pages:
                break
            page += 1

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def load_detail(product_id: str) -> ProductDetail:
            async with semaphore:
                return await rentflow.get_product(product_id)

        details = await asyncio.gather(
            *(load_detail(summary.product_id) for summary in summaries)
        )
        documents = tuple(self._document(detail) for detail in details)
        existing = await self._repository.content_hashes()
        changed = tuple(
            item for item in documents if existing.get(item.product_id) != item.content_hash
        )
        for offset in range(0, len(changed), self._batch_size):
            batch = changed[offset : offset + self._batch_size]
            vectors = await self._embeddings.embed(tuple(item.search_text for item in batch))
            for document, embedding in zip(batch, vectors, strict=True):
                await self._repository.upsert(
                    document,
                    self._embeddings.model_id,
                    embedding,
                )
        active_ids = {item.product_id for item in documents}
        deactivated = await self._repository.deactivate_missing(active_ids)
        return CatalogIndexStats(
            discovered=len(documents),
            indexed=len(changed),
            unchanged=len(documents) - len(changed),
            deactivated=deactivated,
        )

    async def vocabulary(self) -> CatalogVocabulary:
        return await self._repository.vocabulary()

    async def search(
        self,
        query: str,
        *,
        equipment_role: str | None,
        brand: str | None,
        model: str | None,
    ) -> tuple[SemanticProductCandidate, ...]:
        embeddings = await self._embeddings.embed((query,))
        candidates = await self._repository.semantic_search(
            embeddings[0],
            query=query,
            equipment_role=equipment_role,
            brand=brand,
            model=model,
            limit=min(100, self._top_k * 3),
        )
        ranked = [
            SemanticProductCandidate(
                product_id=item.product_id,
                vector_score=item.vector_score,
                lexical_score=item.lexical_score,
                score=(
                    self._vector_weight * item.vector_score
                    + self._lexical_weight * item.lexical_score
                ),
            )
            for item in candidates
            if item.vector_score >= self._min_score
        ]
        ranked.sort(key=lambda item: (-item.score, item.product_id))
        return tuple(ranked[: self._top_k])

    def _document(self, product: ProductDetail) -> CatalogDocument:
        search_text = "\n".join(
            (
                f"equipment role: {product.equipment_role}",
                f"brand: {product.brand}",
                f"model: {product.model}",
                f"name: {product.name}",
                f"description: {product.description}",
            )
        )
        content_hash = hashlib.sha256(
            f"v1\n{self._embeddings.model_id}\n{search_text}".encode()
        ).hexdigest()
        return CatalogDocument(
            product_id=product.product_id,
            category_id=product.category_id,
            equipment_role=product.equipment_role,
            brand=product.brand,
            model=product.model,
            name=product.name,
            description=product.description,
            search_text=search_text,
            content_hash=content_hash,
        )
