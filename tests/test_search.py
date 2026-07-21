import pytest
from pydantic import ValidationError

from gearmate.actions import AgentAction
from gearmate.search import ProductSearchPlanner
from gearmate.tools.contracts import ProductSearchInput


def test_generic_role_keyword_is_not_sent_twice() -> None:
    plan = ProductSearchPlanner().plan(
        AgentAction(
            action="product_search",
            keyword="电脑",
            keyword_specificity="generic",
            equipment_role="laptop",
        )
    )

    assert plan.equipment_role == "laptop"
    assert plan.keyword is None


def test_unclassified_keyword_with_role_defaults_to_safe_role_search() -> None:
    plan = ProductSearchPlanner().plan(
        AgentAction(
            action="product_search",
            keyword="苹果电脑",
            equipment_role="laptop",
            brand="Apple",
        )
    )

    assert plan.keyword is None
    assert plan.brand == "Apple"


def test_specific_model_keyword_is_preserved() -> None:
    plan = ProductSearchPlanner().plan(
        AgentAction(
            action="product_search",
            keyword="MacBook Pro",
            keyword_specificity="specific",
            equipment_role="laptop",
            brand="Apple",
        )
    )

    assert plan.keyword == "MacBook Pro"
    assert plan.brand == "Apple"


def test_target_purchase_price_is_preserved_for_server_side_ranking() -> None:
    plan = ProductSearchPlanner().plan(
        AgentAction(
            action="product_search",
            equipment_role="laptop",
            target_price="8000",
        )
    )

    assert plan.target_price == 8000


@pytest.mark.parametrize("field", ("brand", "model"))
def test_catalog_exact_filters_match_rentflow_column_limit(field: str) -> None:
    with pytest.raises(ValidationError):
        AgentAction(action="product_search", **{field: "x" * 65})
    with pytest.raises(ValidationError):
        ProductSearchInput(**{field: "x" * 65})
