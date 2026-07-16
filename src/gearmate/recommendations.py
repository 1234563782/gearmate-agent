from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from gearmate.actions import AgentAction
from gearmate.search import RecentProductReference
from gearmate.tools.contracts import ProductSearchResult, ProductUseCase, RentalPeriodInput


class RecommendationModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class RecommendationCard(RecommendationModel):
    product_id: str
    name: str
    brand: str
    model: str
    daily_rate: str
    fixed_deposit: str
    available_count: int | None
    use_cases: tuple[ProductUseCase, ...]


class RecommendationSection(RecommendationModel):
    use_case_id: str | None
    title: str
    description: str
    products: tuple[RecommendationCard, ...]


class FollowUpOption(RecommendationModel):
    value: str
    label: str


class FollowUpQuestion(RecommendationModel):
    field: Literal["use_case", "rental_period"]
    text: str
    options: tuple[FollowUpOption, ...] = ()


class RecommendationPresentation(RecommendationModel):
    mode: Literal["explore", "recommend"]
    intro: str
    sections: tuple[RecommendationSection, ...]
    rental_period: RentalPeriodInput | None = None
    follow_up: FollowUpQuestion | None = None
    closing: str | None = None


class RecommendationPlanner:
    def plan(
        self,
        result: ProductSearchResult,
        action: AgentAction,
        rental_period: RentalPeriodInput | None,
    ) -> RecommendationPresentation:
        cards = tuple(
            RecommendationCard(
                product_id=item.product_id,
                name=item.name,
                brand=item.brand,
                model=item.model,
                daily_rate=item.daily_rate,
                fixed_deposit=item.fixed_deposit,
                available_count=item.available_count,
                use_cases=item.use_cases,
            )
            for item in result.items[:4]
        )
        sections = (
            (
                RecommendationSection(
                    use_case_id=None,
                    title=f"接近日租 ¥{action.target_daily_rate}",
                    description="这些设备按与目标日租的接近程度排列，优先看前面的候选。",
                    products=cards,
                ),
            )
            if action.target_daily_rate is not None
            else self._sections(cards, action.use_case_id)
        )
        if action.use_case_id is None:
            options = self._use_case_options(cards)
            follow_up = (
                FollowUpQuestion(
                    field="use_case",
                    text="主要用于哪种场景？",
                    options=options,
                )
                if options
                else None
            )
            return RecommendationPresentation(
                mode="explore",
                intro=self._intro(action, sections),
                sections=sections,
                rental_period=rental_period,
                follow_up=follow_up,
                closing=self._closing(rental_period),
            )
        follow_up = None
        if rental_period is None:
            follow_up = FollowUpQuestion(
                field="rental_period",
                text="计划什么时候租用？",
            )
        return RecommendationPresentation(
            mode="recommend",
            intro=self._intro(action, sections),
            sections=sections,
            rental_period=rental_period,
            follow_up=follow_up,
            closing=self._closing(rental_period),
        )

    def plan_exact(
        self,
        product: RecentProductReference,
        rental_period: RentalPeriodInput,
        available_count: int | None,
    ) -> RecommendationPresentation | None:
        if product.daily_rate is None or product.fixed_deposit is None:
            return None
        card = RecommendationCard(
            product_id=product.product_id,
            name=product.name,
            brand=product.brand,
            model=product.model,
            daily_rate=product.daily_rate,
            fixed_deposit=product.fixed_deposit,
            available_count=available_count,
            use_cases=product.use_cases,
        )
        return RecommendationPresentation(
            mode="recommend",
            intro="这个租期的实时库存已经核验，下面是你选中的设备。",
            sections=self._sections((card,), None),
            rental_period=rental_period,
            closing="点开卡片可以查看完整报价并继续预订。",
        )

    @staticmethod
    def _sections(
        cards: tuple[RecommendationCard, ...],
        selected_use_case_id: str | None,
    ) -> tuple[RecommendationSection, ...]:
        grouped: dict[tuple[str | None, str], list[RecommendationCard]] = defaultdict(list)
        for card in cards:
            primary = next(
                (
                    item
                    for item in card.use_cases
                    if selected_use_case_id is not None and item.id == selected_use_case_id
                ),
                card.use_cases[0] if card.use_cases else None,
            )
            key = (primary.id, primary.name) if primary is not None else (None, "推荐设备")
            grouped[key].append(card)
        return tuple(
            RecommendationSection(
                use_case_id=use_case_id,
                title=title,
                description=(
                    f"这些设备在目录中与“{title}”场景匹配度较高。"
                    if use_case_id is not None
                    else "下面是与当前条件匹配度较高的设备。"
                ),
                products=tuple(items),
            )
            for (use_case_id, title), items in grouped.items()
        )

    @staticmethod
    def _use_case_options(
        cards: tuple[RecommendationCard, ...],
    ) -> tuple[FollowUpOption, ...]:
        scores: dict[tuple[str, str], float] = defaultdict(float)
        for card in cards:
            for use_case in card.use_cases:
                scores[(use_case.id, use_case.name)] += float(use_case.weight)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0][0]))[:3]
        return tuple(
            FollowUpOption(value=use_case_id, label=name)
            for (use_case_id, name), _score in ranked
        )

    @staticmethod
    def _intro(
        action: AgentAction,
        sections: tuple[RecommendationSection, ...],
    ) -> str:
        if action.target_daily_rate is not None:
            return f"我按你的用途和目标日租 ¥{action.target_daily_rate} 整理了这些候选。"
        if action.max_daily_rate is not None:
            return f"我按你的用途和日租不超过 ¥{action.max_daily_rate} 整理了这些候选。"
        if action.use_case_id is not None and sections:
            return f"我按“{sections[0].title}”场景进一步缩小了范围。"
        return "如果你还没确定具体用途，可以先从下面几个场景了解合适的设备。"

    @staticmethod
    def _closing(rental_period: RentalPeriodInput | None) -> str:
        if rental_period is not None:
            return "这些设备已按你提供的租期核验库存，点开卡片可以查看报价并预订。"
        return "可以先选择主要用途，也可以直接点开卡片填写租期并查询报价。"
