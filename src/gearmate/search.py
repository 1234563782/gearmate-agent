from dataclasses import dataclass
from decimal import Decimal

from gearmate.actions import AgentAction


@dataclass(frozen=True, slots=True)
class ProductSearchPlan:
    keyword: str | None
    equipment_role: str | None
    brand: str | None
    model: str | None
    semantic_query: str | None
    category_id: str | None
    max_daily_rate: Decimal | None


class ProductSearchPlanner:
    def plan(self, action: AgentAction) -> ProductSearchPlan:
        if action.action != "product_search":
            raise ValueError("A product_search action is required")
        keyword = action.keyword
        if action.equipment_role is not None and action.keyword_specificity != "specific":
            keyword = None
        return ProductSearchPlan(
            keyword=keyword,
            equipment_role=action.equipment_role,
            brand=action.brand,
            model=action.model,
            semantic_query=action.semantic_query,
            category_id=action.category_id,
            max_daily_rate=action.max_daily_rate,
        )
