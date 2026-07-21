import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from gearmate.tools.contracts import (
    ProductDetail,
    ProductSearchResult,
    ProductSummary,
    StoreOrder,
    StoreOrderPage,
    StoreSku,
    StoreSkuList,
)

ULID_PATTERN = re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b")
MONEY_PATTERN = re.compile(
    r"(?:CNY\s*|[¥￥]\s*)(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*元"
)
COUNT_PATTERN = re.compile(r"(?<!\d)(\d+)\s*(?:件|台)")


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
    product_search_performed: bool = False
    store_skus: dict[str, StoreSku] = field(default_factory=dict)
    store_orders: dict[str, StoreOrder] = field(default_factory=dict)
    store_order_list_performed: bool = False
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
                use_cases=result.use_cases,
            )
        elif isinstance(result, StoreSkuList):
            self.store_skus.update((item.sku_id, item) for item in result.items)
        elif isinstance(result, StoreSku):
            self.store_skus[result.sku_id] = result
        elif isinstance(result, StoreOrderPage):
            self.store_order_list_performed = True
            self.store_orders.update((item.order_id, item) for item in result.items)
        elif isinstance(result, StoreOrder):
            self.store_orders[result.order_id] = result
        else:
            raise TypeError(f"Unsupported fact result: {type(result).__name__}")

    def validate(self, text: str) -> FactValidationResult:
        stated_ids = set(ULID_PATTERN.findall(text))
        unsupported_ids = tuple(sorted(stated_ids))
        allowed_amounts = self._allowed_amounts()
        stated_amounts = {
            self._money_value(first or second) for first, second in MONEY_PATTERN.findall(text)
        }
        unsupported_amounts = tuple(
            sorted(str(amount) for amount in stated_amounts if amount not in allowed_amounts)
        )
        allowed_counts = set(self.constraint_counts)
        allowed_counts.update(sku.available_quantity for sku in self.store_skus.values())
        for product in self.products.values():
            allowed_counts.update(sku.available_quantity for sku in product.store_skus)
            if product.store_skus:
                allowed_counts.add(sum(sku.available_quantity for sku in product.store_skus))
        allowed_counts.update(
            item.quantity for order in self.store_orders.values() for item in order.items
        )
        allowed_counts.update(
            sum(item.quantity for item in order.items) for order in self.store_orders.values()
        )
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
            ),
            unsupported_ids=unsupported_ids,
            unsupported_amounts=unsupported_amounts,
            unsupported_counts=unsupported_counts,
            mismatched_product_ids=mismatched_product_ids,
        )

    def fallback_text(self) -> str:
        lines: list[str] = []
        for product in list(self.products.values())[:5]:
            skus = [sku for sku in product.store_skus if sku.enabled]
            line = f"- {product.name}（{product.brand} {product.model}）"
            if skus:
                line += (
                    f"：售价 {min(Decimal(sku.sale_price) for sku in skus):.2f} 元起，"
                    f"可售库存 {sum(sku.available_quantity for sku in skus)} 件"
                )
            lines.append(line)
        for sku in self.store_skus.values():
            lines.append(
                f"- {sku.sku_name}：售价 {sku.sale_price} 元，可售库存 {sku.available_quantity} 件"
            )
        for order in self.store_orders.values():
            names = "、".join(item.product_name for item in order.items[:2])
            lines.append(f"- {names or '商城订单'}：{order.status}，合计 {order.payable_amount} 元")
        if not lines:
            if self.product_search_performed:
                return "当前没有找到符合搜索条件的商品，可以调整商品类型或预算后重试。"
            if self.store_order_list_performed:
                return "当前筛选条件下没有商城订单。"
            return "暂时没有取得可核验的商品、库存或订单信息，请补充条件后重试。"
        return "我查到的结果如下：\n" + "\n".join(lines)

    def _allowed_amounts(self) -> set[Decimal]:
        values = set(self.constraint_amounts)
        values.update(self._money_value(sku.sale_price) for sku in self.store_skus.values())
        for product in self.products.values():
            values.update(self._money_value(sku.sale_price) for sku in product.store_skus)
        for order in self.store_orders.values():
            values.update(
                self._money_value(value)
                for value in (
                    order.item_amount,
                    order.shipping_amount,
                    order.payable_amount,
                )
            )
            values.update(self._money_value(item.subtotal) for item in order.items)
        return values

    @staticmethod
    def _money_value(value: str) -> Decimal:
        try:
            return Decimal(value).quantize(Decimal("0.01"))
        except InvalidOperation:
            return Decimal("NaN")
