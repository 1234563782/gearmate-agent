from gearmate.actions import AgentAction
from gearmate.recommendations import RecommendationPlanner
from gearmate.search import RecentProductReference
from gearmate.tools.contracts import (
    ProductSearchResult,
    ProductSummary,
    ProductUseCase,
    StoreSku,
    StoreSkuList,
)

PRODUCT_ID = "01J00000000000000000000105"
SKU_ID = "01J00000000000000000000305"
USE_CASE_ID = "01J00000000000000000000205"


def sku() -> StoreSku:
    return StoreSku(
        sku_id=SKU_ID,
        product_id=PRODUCT_ID,
        sku_code="MAC-16-512",
        sku_name="16GB + 512GB",
        specs={"memory": "16GB"},
        sale_price="7999.00",
        available_quantity=4,
        enabled=True,
    )


def result() -> ProductSearchResult:
    return ProductSearchResult(
        items=(
            ProductSummary(
                product_id=PRODUCT_ID,
                category_id="01J00000000000000000000405",
                equipment_role="laptop",
                name="MacBook Pro 14",
                brand="Apple",
                model="MacBook Pro 14",
                use_cases=(
                    ProductUseCase(
                        id=USE_CASE_ID,
                        code="editing",
                        name="视频剪辑",
                        weight="0.90",
                    ),
                ),
                store_skus=(sku(),),
            ),
        ),
        page=0,
        size=20,
        total_elements=1,
        total_pages=1,
    )


def test_explore_presentation_asks_dynamic_use_case_followup() -> None:
    presentation = RecommendationPlanner().plan(
        result(),
        AgentAction(action="product_search", equipment_role="laptop"),
    )

    assert presentation.mode == "explore"
    assert presentation.follow_up is not None
    assert presentation.follow_up.field == "use_case"
    assert presentation.sections[0].products[0].sale_price == "7999.00"
    payload = presentation.model_dump(mode="json", by_alias=True)
    assert "rentalPeriod" not in payload
    assert "dailyRate" not in payload["sections"][0]["products"][0]


def test_target_price_presentation_uses_purchase_price() -> None:
    presentation = RecommendationPlanner().plan(
        result(),
        AgentAction(action="product_search", target_price="8000"),
    )

    assert presentation.sections[0].title == "接近售价 ¥8000"


def test_purchase_presentation_contains_skus_and_quantity() -> None:
    presentation = RecommendationPlanner().plan_purchase(
        RecentProductReference(
            position=1,
            product_id=PRODUCT_ID,
            name="MacBook Pro 14",
            brand="Apple",
            model="MacBook Pro 14",
            equipment_role="laptop",
        ),
        StoreSkuList(product_id=PRODUCT_ID, items=(sku(),)),
        2,
    )

    assert presentation.mode == "purchase"
    assert presentation.purchase_quantity == 2
    assert presentation.sections[0].products[0].store_skus == (sku(),)
