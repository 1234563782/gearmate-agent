from decimal import Decimal

from gearmate.actions import AgentAction
from gearmate.responses import UserResponseComposer
from gearmate.tools.contracts import ProductSearchResult, ProductSummary, ProductUseCase
from gearmate.validation.facts import FactSnapshot


def product(
    product_id: str,
    name: str,
    daily_rate: str,
    available_count: int,
) -> ProductSummary:
    return ProductSummary(
        product_id=product_id,
        category_id="01J00000000000000000000007",
        equipment_role="laptop",
        name=name,
        brand="Demo",
        model=name,
        daily_rate=daily_rate,
        fixed_deposit="3000.00",
        available_count=available_count,
        use_cases=(
            ProductUseCase(
                id="01J00000000000000000000202",
                code="video_editing",
                name="视频剪辑",
                weight="1.0",
            ),
        ),
    )


def test_product_search_response_is_grounded_rich_and_hides_internal_ids() -> None:
    facts = FactSnapshot()
    facts.add_constraint_amount(Decimal("150"))
    facts.add(
        ProductSearchResult(
            items=(
                product("01J00000000000000000000105", "MacBook Pro 14", "160.00", 2),
                product("01J00000000000000000000111", "Dell XPS 15", "140.00", 1),
            ),
            page=0,
            size=20,
            total_elements=2,
            total_pages=1,
        )
    )

    text = UserResponseComposer().compose(
        action=AgentAction(action="product_search", target_daily_rate="150"),
        facts=facts,
        rental_period=None,
    )

    assert "目标日租 ¥150" in text
    assert "适合视频剪辑" in text
    assert "MacBook Pro 14" in text
    assert "Dell XPS 15" in text
    assert "01J000" not in text
    assert "RentFlow" not in text
    assert facts.validate(text).valid


def test_fallback_text_never_exposes_product_ids_or_internal_source_name() -> None:
    facts = FactSnapshot()
    facts.add(
        ProductSearchResult(
            items=(product("01J00000000000000000000105", "MacBook Pro 14", "160.00", 2),),
            page=0,
            size=20,
            total_elements=1,
            total_pages=1,
        )
    )

    text = facts.fallback_text()

    assert "01J00000000000000000000105" not in text
    assert "RentFlow" not in text
