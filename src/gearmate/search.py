from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from gearmate.actions import AgentAction
from gearmate.tools.contracts import ProductSearchResult, ProductUseCase


class RecentProductReference(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    position: int = Field(ge=1, le=100)
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    name: str
    brand: str
    model: str
    equipment_role: str
    daily_rate: str | None = None
    fixed_deposit: str | None = None
    use_cases: tuple[ProductUseCase, ...] = ()


class RecentProductSearch(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    items: tuple[RecentProductReference, ...] = ()

    @classmethod
    def from_result(cls, result: ProductSearchResult) -> "RecentProductSearch":
        return cls(
            items=tuple(
                RecentProductReference(
                    position=index,
                    product_id=item.product_id,
                    name=item.name,
                    brand=item.brand,
                    model=item.model,
                    equipment_role=item.equipment_role,
                    daily_rate=item.daily_rate,
                    fixed_deposit=item.fixed_deposit,
                    use_cases=item.use_cases,
                )
                for index, item in enumerate(result.items, start=1)
            )
        )


@dataclass(frozen=True, slots=True)
class ProductSearchPlan:
    keyword: str | None
    equipment_role: str | None
    brand: str | None
    model: str | None
    semantic_query: str | None
    use_case_id: str | None
    category_id: str | None
    max_daily_rate: Decimal | None
    target_daily_rate: Decimal | None


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
            use_case_id=action.use_case_id,
            category_id=action.category_id,
            max_daily_rate=action.max_daily_rate,
            target_daily_rate=action.target_daily_rate,
        )
