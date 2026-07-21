from gearmate.catalog import SemanticProductCandidate
from gearmate.llm.types import ModelToolCall
from gearmate.tools.contracts import (
    ProductDetail,
    ProductSearchInput,
    ProductSearchResult,
    ProductSummary,
    StoreSku,
)
from gearmate.tools.registry import ToolRegistry
from gearmate.validation.facts import FactSnapshot


class FakeSemanticCatalog:
    async def search(self, query, *, equipment_role, brand, model, use_case_id=None):
        assert query == "适合剪视频的电脑"
        assert equipment_role == "laptop"
        assert brand == "Apple"
        assert model is None
        return (
            SemanticProductCandidate(
                product_id="01J00000000000000000000105",
                score=0.93,
                vector_score=0.91,
                lexical_score=0.2,
            ),
        )


class FailingSemanticCatalog:
    async def search(self, query, *, equipment_role, brand, model, use_case_id=None):
        raise RuntimeError("embedding service unavailable")


class FakeRentFlow:
    def __init__(self) -> None:
        self.structured_calls = 0

    async def search_products(self, request: ProductSearchInput) -> ProductSearchResult:
        self.structured_calls += 1
        return ProductSearchResult(
            items=(
                ProductSummary(
                    product_id="01J00000000000000000000105",
                    category_id="01J00000000000000000000002",
                    equipment_role="laptop",
                    name="MacBook Pro 14",
                    brand="Apple",
                    model="MacBook Pro 14",
                ),
            ),
            page=0,
            size=20,
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
            description="Portable computer for editing",
        )

async def test_semantic_candidates_are_hydrated_from_rentflow() -> None:
    rentflow = FakeRentFlow()
    registry = ToolRegistry(
        rentflow,  # type: ignore[arg-type]
        timeout_seconds=5,
        max_result_items=20,
        max_concurrency=4,
        catalog_search=FakeSemanticCatalog(),  # type: ignore[arg-type]
    )

    results = await registry.execute_all(
        [
            ModelToolCall(
                id="semantic-search",
                name="search_products",
                arguments={
                    "equipmentRole": "laptop",
                    "brand": "Apple",
                    "semanticQuery": "适合剪视频的电脑",
                },
            )
        ],
        FactSnapshot(),
        _ignore_event,
    )

    assert not results[0].is_error
    assert results[0].result is not None
    assert results[0].result.items[0].name == "MacBook Pro 14"  # type: ignore[attr-defined]
    assert rentflow.structured_calls == 0
    assert registry.last_search_diagnostics == {
        "mode": "semantic",
        "semanticQuery": "适合剪视频的电脑",
        "candidateCount": 1,
        "candidates": [
            {
                "productId": "01J00000000000000000000105",
                "score": 0.93,
                "vectorScore": 0.91,
                "lexicalScore": 0.2,
            }
        ],
    }


async def test_semantic_failure_falls_back_to_structured_search() -> None:
    rentflow = FakeRentFlow()
    registry = ToolRegistry(
        rentflow,  # type: ignore[arg-type]
        timeout_seconds=5,
        max_result_items=20,
        max_concurrency=4,
        catalog_search=FailingSemanticCatalog(),  # type: ignore[arg-type]
    )

    results = await registry.execute_all(
        [
            ModelToolCall(
                id="semantic-fallback",
                name="search_products",
                arguments={
                    "equipmentRole": "laptop",
                    "semanticQuery": "适合剪视频的电脑",
                },
            )
        ],
        FactSnapshot(),
        _ignore_event,
    )

    assert not results[0].is_error
    assert rentflow.structured_calls == 1
    assert registry.last_search_diagnostics == {
        "mode": "structured_fallback",
        "reason": "SEMANTIC_SEARCH_FAILED",
        "semanticQuery": "适合剪视频的电脑",
        "resultCount": 1,
    }


def test_price_preference_filters_hard_maximum_and_ranks_nearest_target() -> None:
    def product(index: int, name: str, sale_price: str) -> ProductSummary:
        product_id = f"01J0000000000000000000010{index}"
        return ProductSummary(
            product_id=product_id,
            category_id="01J00000000000000000000001",
            equipment_role="camera",
            name=name,
            brand="Demo",
            model=name,
            store_skus=(
                StoreSku(
                    sku_id=f"01J0000000000000000000020{index}",
                    product_id=product_id,
                    sku_code=f"SKU-{index}",
                    sku_name="标准版",
                    specs={},
                    sale_price=sale_price,
                    available_quantity=2,
                    enabled=True,
                ),
            ),
        )

    result = ProductSearchResult(
        items=(
            product(1, "Budget Camera", "4500.00"),
            product(2, "Target Camera", "5200.00"),
            product(3, "Expensive Camera", "9000.00"),
        ),
        page=0,
        size=20,
        total_elements=3,
        total_pages=1,
    )

    ranked = ToolRegistry._apply_price_preference(
        result,
        ProductSearchInput(max_price="6000", target_price="5000"),
    )

    assert [item.name for item in ranked.items] == ["Target Camera", "Budget Camera"]
    assert ranked.total_elements == 2


async def _ignore_event(event_type, payload) -> None:
    return None
