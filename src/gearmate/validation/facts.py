import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from gearmate.tools.contracts import (
    AvailabilityResult,
    OrderPage,
    OrderSummary,
    ProductDetail,
    ProductSearchResult,
    ProductSummary,
    QuoteResult,
    ScenarioKitResult,
)

ULID_PATTERN = re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b")
MONEY_PATTERN = re.compile(r"(?:CNY\s*|¥\s*|￥\s*)(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*元")
COUNT_PATTERN = re.compile(r"(?<!\d)(\d+)\s*台")


@dataclass(frozen=True, slots=True)
class FactValidationResult:
    valid: bool
    unsupported_ids: tuple[str, ...] = ()
    unsupported_amounts: tuple[str, ...] = ()
    unsupported_counts: tuple[int, ...] = ()
    mismatched_product_ids: tuple[str, ...] = ()
    missing_fact_citation: bool = False


@dataclass(slots=True)
class FactSnapshot:
    products: dict[str, ProductSummary] = field(default_factory=dict)
    availability: dict[str, AvailabilityResult] = field(default_factory=dict)
    quotes: dict[str, QuoteResult] = field(default_factory=dict)
    scenario_kits: list[ScenarioKitResult] = field(default_factory=list)
    orders: dict[str, OrderSummary] = field(default_factory=dict)
    product_search_performed: bool = False
    order_list_performed: bool = False
    constraint_amounts: set[Decimal] = field(default_factory=set)
    constraint_counts: set[int] = field(default_factory=set)

    def add_constraint_amount(self, value: Decimal) -> None:
        self.constraint_amounts.add(value.quantize(Decimal("0.01")))

    def add_constraint_count(self, value: int) -> None:
        self.constraint_counts.add(value)

    def add(self, result: BaseModel) -> None:
        if isinstance(result, ProductSearchResult):
            self.product_search_performed = True
            self.products.update((item.product_id, item) for item in result.items)
        elif isinstance(result, ProductDetail):
            self.products[result.product_id] = ProductSummary(
                product_id=result.product_id,
                category_id=result.category_id,
                equipment_role=result.equipment_role,
                name=result.name,
                brand=result.brand,
                model=result.model,
                daily_rate=result.daily_rate,
                fixed_deposit=result.fixed_deposit,
                available_count=None,
                use_cases=result.use_cases,
            )
        elif isinstance(result, AvailabilityResult):
            self.availability[result.product_id] = result
        elif isinstance(result, QuoteResult):
            self.quotes[result.quote_id] = result
        elif isinstance(result, ScenarioKitResult):
            self.scenario_kits.append(result)
            self.products.update((item.product.product_id, item.product) for item in result.items)
        elif isinstance(result, OrderPage):
            self.order_list_performed = True
            self.orders.update((item.order_id, item) for item in result.items)
        else:
            raise TypeError(f"Unsupported fact result: {type(result).__name__}")

    def validate(self, text: str) -> FactValidationResult:
        stated_ids = set(ULID_PATTERN.findall(text))
        unsupported_ids = tuple(sorted(stated_ids))
        missing_fact_citation = False
        allowed_amounts = self._allowed_amounts()
        stated_amounts = {
            self._money_value(first or second) for first, second in MONEY_PATTERN.findall(text)
        }
        unsupported_amounts = tuple(
            sorted(str(amount) for amount in stated_amounts if amount not in allowed_amounts)
        )
        allowed_counts = set(self.constraint_counts)
        allowed_counts.update(result.available_count for result in self.availability.values())
        allowed_counts.update(
            product.available_count
            for product in self.products.values()
            if product.available_count is not None
        )
        allowed_counts.update(item.quantity for kit in self.scenario_kits for item in kit.items)
        stated_counts = {int(value) for value in COUNT_PATTERN.findall(text)}
        unsupported_counts = tuple(sorted(stated_counts - allowed_counts))
        mismatched_product_ids = tuple(
            sorted(
                product_id
                for product_id in stated_ids & self.products.keys()
                if self.products[product_id].name not in text
            )
        )
        return FactValidationResult(
            valid=(
                not unsupported_ids
                and not unsupported_amounts
                and not unsupported_counts
                and not mismatched_product_ids
                and not missing_fact_citation
            ),
            unsupported_ids=unsupported_ids,
            unsupported_amounts=unsupported_amounts,
            unsupported_counts=unsupported_counts,
            mismatched_product_ids=mismatched_product_ids,
            missing_fact_citation=missing_fact_citation,
        )

    def fallback_text(self) -> str:
        lines: list[str] = []
        for kit in self.scenario_kits:
            for item in kit.items:
                lines.append(
                    f"- {item.product.name} × {item.quantity}："
                    f"小计 {item.subtotal_daily_rate} 元/天"
                )
            if kit.missing_roles:
                lines.append("- 当前目录还缺少部分必要设备")
            lines.append(
                f"- 组合日租合计 {kit.total_daily_rate} 元，"
                f"预算 {kit.max_daily_budget} 元，"
                + ("预算内" if kit.within_budget else "未满足完整性或预算约束")
            )
            if not kit.availability_checked:
                lines.append("- 尚未提供完整租期，本组合未核验实时库存")
        kit_product_ids = {
            item.product.product_id for kit in self.scenario_kits for item in kit.items
        }
        for product in list(self.products.values())[:5]:
            if product.product_id in kit_product_ids:
                continue
            line = f"- {product.name}（{product.brand} {product.model}）"
            available = self.availability.get(product.product_id)
            if available is not None:
                line += (
                    f"：可租 {available.available_count} 台"
                    if available.available
                    else "：当前租期不可租"
                )
            lines.append(line)
        for product_id, available in self.availability.items():
            if product_id in self.products:
                continue
            availability_text = (
                f"可租 {available.available_count} 台" if available.available else "当前租期不可租"
            )
            lines.append(f"- 这款设备：{availability_text}")
        for quote in self.quotes.values():
            snapshot = quote.price_snapshot
            lines.append(
                f"- 正式报价：租金 {snapshot.rental_amount} 元，"
                f"押金 {snapshot.deposit_amount} 元，总计 {snapshot.total_amount} 元"
            )
        for order in self.orders.values():
            lines.append(
                f"- {order.product_name}：状态 {order.effective_status}，"
                f"合计 {order.price_snapshot.total_amount} 元"
            )
        if not lines:
            if self.product_search_performed:
                return (
                    "当前没有找到符合这些搜索条件的商品。"
                    "你可以调整商品类型、预算或租期后重试。"
                )
            if self.order_list_performed:
                return "当前筛选条件下没有订单。"
            return "暂时没有取得可核验的商品、库存或报价信息，请补充商品和租期后重试。"
        return "我查到的结果如下：\n" + "\n".join(lines)

    def _allowed_amounts(self) -> set[Decimal]:
        values = set(self.constraint_amounts)
        for product in self.products.values():
            values.add(self._money_value(product.daily_rate))
            values.add(self._money_value(product.fixed_deposit))
        for quote in self.quotes.values():
            snapshot = quote.price_snapshot
            values.update(
                self._money_value(value)
                for value in (
                    snapshot.daily_rate,
                    snapshot.rental_amount,
                    snapshot.deposit_amount,
                    snapshot.total_amount,
                )
            )
        for kit in self.scenario_kits:
            values.add(self._money_value(kit.total_daily_rate))
            values.add(self._money_value(kit.max_daily_budget))
            values.update(self._money_value(item.subtotal_daily_rate) for item in kit.items)
        for order in self.orders.values():
            snapshot = order.price_snapshot
            values.update(
                self._money_value(value)
                for value in (
                    snapshot.daily_rate,
                    snapshot.rental_amount,
                    snapshot.deposit_amount,
                    snapshot.total_amount,
                )
            )
        return values

    @staticmethod
    def _money_value(value: str) -> Decimal:
        try:
            return Decimal(value).quantize(Decimal("0.01"))
        except InvalidOperation:
            return Decimal("NaN")
