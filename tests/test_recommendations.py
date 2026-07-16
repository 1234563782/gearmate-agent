from datetime import UTC, datetime

from gearmate.actions import AgentAction
from gearmate.recommendations import RecommendationPlanner
from gearmate.search import RecentProductReference
from gearmate.tools.contracts import (
    ProductSearchResult,
    ProductSummary,
    ProductUseCase,
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
    assert presentation.sections[0].title == "视频剪辑"
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


def test_exact_result_keeps_period_and_live_availability() -> None:
    period = RentalPeriodInput(
        start_at=datetime(2026, 7, 20, tzinfo=UTC),
        end_at=datetime(2026, 7, 21, tzinfo=UTC),
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
