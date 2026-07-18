from datetime import UTC, date, datetime

from gearmate.actions import AgentAction
from gearmate.recommendations import RecommendationPlanner
from gearmate.search import RecentProductReference
from gearmate.tools.contracts import (
    PriceSnapshot,
    ProductSearchResult,
    ProductSummary,
    ProductUseCase,
    QuoteResult,
    RentalPeriodInput,
)


def product(
    product_id: str,
    name: str,
    use_cases: tuple[ProductUseCase, ...],
) -> ProductSummary:
    return ProductSummary(
        product_id=product_id,
        category_id="01J00000000000000000000003",
        equipment_role="laptop",
        name=name,
        brand="Apple",
        model=name,
        daily_rate="160.00",
        fixed_deposit="5000.00",
        use_cases=use_cases,
    )


def test_explore_presentation_uses_dynamic_use_cases_for_sections_and_question() -> None:
    editing = ProductUseCase(
        id="01J00000000000000000000202",
        code="video_editing",
        name="视频剪辑",
        weight="0.98",
    )
    office = ProductUseCase(
        id="01J00000000000000000000201",
        code="mobile_office",
        name="移动办公",
        weight="0.95",
    )
    result = ProductSearchResult(
        items=(
            product("01J00000000000000000000105", "MacBook Pro 14", (editing, office)),
        ),
        page=0,
        size=20,
        total_elements=1,
        total_pages=1,
    )

    presentation = RecommendationPlanner().plan(
        result,
        AgentAction(action="product_search", equipment_role="laptop"),
        None,
    )

    assert presentation.mode == "explore"
    assert "具体用途" in presentation.intro
    assert presentation.sections[0].title == "视频剪辑"
    assert "视频剪辑" in presentation.sections[0].description
    assert presentation.sections[0].products[0].product_id == result.items[0].product_id
    assert presentation.follow_up is not None
    assert [option.label for option in presentation.follow_up.options] == [
        "视频剪辑",
        "移动办公",
    ]


def test_selected_use_case_asks_only_for_missing_rental_period() -> None:
    editing = ProductUseCase(
        id="01J00000000000000000000202",
        code="video_editing",
        name="视频剪辑",
        weight="0.98",
    )
    result = ProductSearchResult(
        items=(product("01J00000000000000000000105", "MacBook Pro 14", (editing,)),),
        page=0,
        size=20,
        total_elements=1,
        total_pages=1,
    )

    presentation = RecommendationPlanner().plan(
        result,
        AgentAction(
            action="product_search",
            equipment_role="laptop",
            use_case_id=editing.id,
        ),
        None,
    )

    assert presentation.mode == "recommend"
    assert presentation.follow_up is not None
    assert presentation.follow_up.field == "rental_period"
    assert presentation.follow_up.options == ()
    assert presentation.rental_period is None
    assert presentation.closing is not None


def test_exact_result_keeps_period_and_live_availability() -> None:
    period = RentalPeriodInput(
        start_date=date(2026, 7, 20),
        end_date=date(2026, 7, 21),
    )
    presentation = RecommendationPlanner().plan_exact(
        RecentProductReference(
            position=1,
            product_id="01J00000000000000000000105",
            name="MacBook Pro 14",
            brand="Apple",
            model="MacBook Pro 14",
            equipment_role="laptop",
            daily_rate="160.00",
            fixed_deposit="5000.00",
        ),
        period,
        2,
    )

    assert presentation is not None
    assert presentation.rental_period == period
    assert presentation.follow_up is None
    assert presentation.sections[0].products[0].available_count == 2
    assert presentation.closing == "点开卡片可以查看完整报价并继续预订。"


def test_exact_quote_presentation_exposes_price_and_availability() -> None:
    period = RentalPeriodInput(
        start_date=date(2026, 7, 20),
        end_date=date(2026, 7, 21),
    )
    quote = QuoteResult(
        quote_id="01J00000000000000000000901",
        product_id="01J00000000000000000000101",
        start_date=period.start_date,
        end_date=period.end_date,
        expires_at=datetime(2026, 7, 18, 3, tzinfo=UTC),
        price_snapshot=PriceSnapshot(
            currency="CNY",
            pricing_version=1,
            pricing_rule="CEIL_24H_FIXED_DEPOSIT",
            billing_days=2,
            daily_rate="200.00",
            rental_amount="400.00",
            deposit_amount="3000.00",
            total_amount="3400.00",
            rounding_mode="HALF_UP",
        ),
    )

    presentation = RecommendationPlanner().plan_exact(
        RecentProductReference(
            position=1,
            product_id=quote.product_id,
            name="Sony A7M4 相机机身",
            brand="Sony",
            model="A7M4",
            equipment_role="camera",
            daily_rate="200.00",
            fixed_deposit="3000.00",
        ),
        period,
        1,
        quote,
    )

    assert presentation is not None
    assert "正式报价已生成" in presentation.intro
    assert "合计 ¥3400.00" in presentation.intro
    assert "当前租期可租 1 台" in presentation.intro
    assert presentation.sections[0].products[0].available_count == 1
    assert presentation.closing == "报价有有效期，点开卡片可以继续预订。"


def test_target_price_presentation_preserves_price_distance_order() -> None:
    result = ProductSearchResult(
        items=(
            product("01J00000000000000000000105", "MacBook Pro 14", ()),
            product("01J00000000000000000000111", "Dell XPS 15", ()),
            product("01J00000000000000000000112", "Lenovo Legion", ()),
        ),
        page=0,
        size=20,
        total_elements=3,
        total_pages=1,
    )

    presentation = RecommendationPlanner().plan(
        result,
        AgentAction(
            action="product_search",
            equipment_role="laptop",
            target_daily_rate="150",
        ),
        None,
    )

    assert presentation.sections[0].title == "接近日租 ¥150"
    assert "目标日租" in presentation.intro
    assert [item.name for item in presentation.sections[0].products] == [
        "MacBook Pro 14",
        "Dell XPS 15",
        "Lenovo Legion",
    ]
