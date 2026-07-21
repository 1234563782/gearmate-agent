from decimal import Decimal

from gearmate.actions import AgentAction
from gearmate.responses import UserResponseComposer
from gearmate.tools.contracts import (
    ProductSearchResult,
    ProductSummary,
    ProductUseCase,
    StoreSku,
)
from gearmate.validation.facts import FactSnapshot


def product(product_id: str, name: str, sale_price: str, stock: int) -> ProductSummary:
    sku = StoreSku(
        sku_id=product_id[:-1] + "9",
        product_id=product_id,
        sku_code=name.upper().replace(" ", "-"),
        sku_name="标准版",
        specs={},
        sale_price=sale_price,
        available_quantity=stock,
        enabled=True,
    )
    return ProductSummary(
        product_id=product_id,
        category_id="01J00000000000000000000007",
        equipment_role="laptop",
        name=name,
        brand="Demo",
        model=name,
        use_cases=(
            ProductUseCase(
                id="01J00000000000000000000202",
                code="video_editing",
                name="视频剪辑",
                weight="1.0",
            ),
        ),
        store_skus=(sku,),
    )


def test_product_search_response_is_grounded_rich_and_hides_internal_ids() -> None:
    facts = FactSnapshot()
    facts.add_constraint_amount(Decimal("8000"))
    facts.add(
        ProductSearchResult(
            items=(
                product("01J00000000000000000000105", "MacBook Pro 14", "7999.00", 2),
                product("01J00000000000000000000111", "Dell XPS 15", "7499.00", 1),
            ),
            page=0,
            size=20,
            total_elements=2,
            total_pages=1,
        )
    )

    text = UserResponseComposer().compose(
        action=AgentAction(action="product_search", target_price="8000"),
        facts=facts,
    )

    assert "目标购买价 ¥8000" in text
    assert "适合视频剪辑" in text
    assert "MacBook Pro 14" in text
    assert "Dell XPS 15" in text
    assert "01J000" not in text
    assert "RentFlow" not in text
    assert facts.validate(text).valid


def test_fallback_text_never_exposes_product_ids_or_rental_fields() -> None:
    facts = FactSnapshot()
    facts.add(
        ProductSearchResult(
            items=(product("01J00000000000000000000105", "MacBook Pro 14", "7999.00", 2),),
            page=0,
            size=20,
            total_elements=1,
            total_pages=1,
        )
    )

    text = facts.fallback_text()

    assert "01J00000000000000000000105" not in text
    assert "日租" not in text
    assert "押金" not in text
