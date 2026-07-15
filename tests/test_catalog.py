from gearmate.catalog import (
    CatalogIndexStats,
    CatalogSearchService,
    CatalogVocabulary,
    SemanticProductCandidate,
)
from gearmate.tools.contracts import ProductDetail, ProductSearchResult, ProductSummary


class FakeEmbeddings:
    model_id = "test-embedding"
    dimensions = 1024

    def __init__(self) -> None:
        self.requests: list[tuple[str, ...]] = []

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.requests.append(texts)
        return tuple((float(index + 1), 0.0) for index, _ in enumerate(texts))

    async def close(self) -> None:
        return None


class FakeCatalogRepository:
    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}
        self.upserts = []
        self.search_arguments: dict[str, object] | None = None

    async def content_hashes(self) -> dict[str, str]:
        return dict(self.hashes)

    async def upsert(self, document, embedding_model, embedding) -> None:
        self.upserts.append((document, embedding_model, embedding))
        self.hashes[document.product_id] = document.content_hash

    async def deactivate_missing(self, active_product_ids: set[str]) -> int:
        return 0

    async def vocabulary(self, limit: int = 200) -> CatalogVocabulary:
        return CatalogVocabulary(
            equipment_roles=("laptop",),
            brands=("Apple",),
            models=("MacBook Pro 14",),
        )

    async def semantic_search(
        self,
        embedding,
        *,
        equipment_role,
        brand,
        model,
        limit,
    ) -> tuple[SemanticProductCandidate, ...]:
        self.search_arguments = {
            "embedding": embedding,
            "equipment_role": equipment_role,
            "brand": brand,
            "model": model,
            "limit": limit,
        }
        return (SemanticProductCandidate("01J00000000000000000000105", 0.91),)


class FakeRentFlowCatalog:
    async def search_products(self, request) -> ProductSearchResult:
        return ProductSearchResult(
            items=(
                ProductSummary(
                    product_id="01J00000000000000000000105",
                    category_id="01J00000000000000000000002",
                    equipment_role="laptop",
                    name="MacBook Pro 14",
                    brand="Apple",
                    model="MacBook Pro 14",
                    daily_rate="160.00",
                    fixed_deposit="1200.00",
                ),
            ),
            page=0,
            size=100,
            total_elements=1,
            total_pages=1,
        )

    async def get_product(self, product_id: str) -> ProductDetail:
        return ProductDetail(
            product_id=product_id,
            category_id="01J00000000000000000000002",
            equipment_role="laptop",
            name="MacBook Pro 14",
            brand="Apple",
            model="MacBook Pro 14",
            description="Portable computer for video editing",
            daily_rate="160.00",
            fixed_deposit="1200.00",
        )


async def test_catalog_refresh_embeds_only_changed_products() -> None:
    repository = FakeCatalogRepository()
    embeddings = FakeEmbeddings()
    service = CatalogSearchService(
        repository,  # type: ignore[arg-type]
        embeddings,
        batch_size=32,
        top_k=20,
        max_concurrency=4,
    )
    rentflow = FakeRentFlowCatalog()

    first = await service.refresh(rentflow)  # type: ignore[arg-type]
    second = await service.refresh(rentflow)  # type: ignore[arg-type]

    assert first == CatalogIndexStats(discovered=1, indexed=1, unchanged=0, deactivated=0)
    assert second == CatalogIndexStats(discovered=1, indexed=0, unchanged=1, deactivated=0)
    assert len(embeddings.requests) == 1
    assert repository.upserts[0][1] == "test-embedding"
    assert "MacBook Pro 14" in repository.upserts[0][0].search_text


async def test_semantic_search_keeps_structured_filters() -> None:
    repository = FakeCatalogRepository()
    embeddings = FakeEmbeddings()
    service = CatalogSearchService(
        repository,  # type: ignore[arg-type]
        embeddings,
        batch_size=32,
        top_k=12,
        max_concurrency=4,
    )

    candidates = await service.search(
        "适合剪辑视频的电脑",
        equipment_role="laptop",
        brand="Apple",
        model=None,
    )

    assert candidates[0].product_id == "01J00000000000000000000105"
    assert repository.search_arguments == {
        "embedding": (1.0, 0.0),
        "equipment_role": "laptop",
        "brand": "Apple",
        "model": None,
        "limit": 12,
    }
