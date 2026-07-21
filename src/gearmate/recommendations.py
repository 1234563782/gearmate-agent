from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from gearmate.actions import AgentAction
from gearmate.search import RecentProductReference
from gearmate.tools.contracts import (
    ProductSearchResult,
    ProductSummary,
    ProductUseCase,
    StoreSku,
    StoreSkuList,
)


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
    sale_price: str | None = None
    available_quantity: int | None = None
    store_skus: tuple[StoreSku, ...] = ()
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
    field: Literal["use_case"]
    text: str
    options: tuple[FollowUpOption, ...] = ()


class RecommendationPresentation(RecommendationModel):
    mode: Literal["explore", "recommend", "purchase"]
    intro: str
    sections: tuple[RecommendationSection, ...]
    follow_up: FollowUpQuestion | None = None
    closing: str | None = None
    purchase_quantity: int | None = None


class RecommendationPlanner:
    def plan(
        self,
        result: ProductSearchResult,
        action: AgentAction,
    ) -> RecommendationPresentation:
        cards = tuple(self._card(item) for item in result.items[:4])
        sections = (
            (
                RecommendationSection(
                    use_case_id=None,
                    title=f"接近售价 ¥{action.target_price}",
                    description="这些商品按与目标价格的接近程度排列。",
                    products=cards,
                ),
            )
            if action.target_price is not None
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
                follow_up=follow_up,
                closing=self._closing(),
            )
        return RecommendationPresentation(
            mode="recommend",
            intro=self._intro(action, sections),
            sections=sections,
            closing=self._closing(),
        )

    def plan_purchase(
        self,
        product: RecentProductReference,
        skus: StoreSkuList,
        quantity: int,
    ) -> RecommendationPresentation:
        available = tuple(sku for sku in skus.items if sku.enabled)
        card = RecommendationCard(
            product_id=product.product_id,
            name=product.name,
            brand=product.brand,
            model=product.model,
            sale_price=(
                min(available, key=lambda sku: float(sku.sale_price)).sale_price
                if available
                else None
            ),
            available_quantity=sum(sku.available_quantity for sku in available),
            store_skus=available,
            use_cases=product.use_cases,
        )
        return RecommendationPresentation(
            mode="purchase",
            intro="我已经查到这款商品的可售规格，请在卡片中确认规格和数量。",
            sections=(
                RecommendationSection(
                    use_case_id=None,
                    title="购买选择",
                    description="库存与售价来自 RentFlow 当前 SKU；确认前不会创建订单。",
                    products=(card,),
                ),
            ),
            closing="确认商品后会进入结算页填写收货信息，Agent 不会自动支付。",
            purchase_quantity=quantity,
        )

    @staticmethod
    def _card(product: ProductSummary) -> RecommendationCard:
        skus = product.store_skus
        return RecommendationCard(
            product_id=product.product_id,
            name=product.name,
            brand=product.brand,
            model=product.model,
            sale_price=(
                min(skus, key=lambda sku: float(sku.sale_price)).sale_price if skus else None
            ),
            available_quantity=sum(sku.available_quantity for sku in skus),
            store_skus=skus,
            use_cases=product.use_cases,
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
            key = (primary.id, primary.name) if primary is not None else (None, "推荐商品")
            grouped[key].append(card)
        return tuple(
            RecommendationSection(
                use_case_id=use_case_id,
                title=title,
                description=(
                    f"这些商品在目录中与“{title}”场景匹配度较高。"
                    if use_case_id is not None
                    else "下面是与当前条件匹配度较高的商品。"
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
            FollowUpOption(value=use_case_id, label=name) for (use_case_id, name), _score in ranked
        )

    @staticmethod
    def _intro(
        action: AgentAction,
        sections: tuple[RecommendationSection, ...],
    ) -> str:
        if action.target_price is not None:
            return f"我按你的用途和目标购买价 ¥{action.target_price} 整理了这些候选。"
        if action.max_price is not None:
            return f"我按你的用途和售价不超过 ¥{action.max_price} 整理了这些候选。"
        if action.use_case_id is not None and sections:
            return f"我按“{sections[0].title}”场景进一步缩小了范围。"
        return "如果你还没有确定具体用途，可以先从下面几个场景了解合适的商品。"

    @staticmethod
    def _closing() -> str:
        return "可以继续选择主要用途，也可以直接点开卡片选择 SKU 和数量后进入结算。"
