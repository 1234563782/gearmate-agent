from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ToolModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class ProductSearchInput(ToolModel):
    keyword: str | None = Field(default=None, max_length=128)
    equipment_role: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    brand: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=64)
    semantic_query: str | None = Field(default=None, max_length=512)
    use_case_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    category_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    max_price: Decimal | None = Field(default=None, gt=0, max_digits=12)
    target_price: Decimal | None = Field(default=None, gt=0, max_digits=12)
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


class ProductUseCase(ToolModel):
    id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    code: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    name: str = Field(max_length=64)
    weight: Decimal = Field(gt=0, le=1)


class CatalogUseCase(ToolModel):
    id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    code: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    name: str = Field(max_length=64)
    description: str = Field(max_length=512)
    aliases: tuple[str, ...] = ()


class ProductSummary(ToolModel):
    product_id: str
    category_id: str
    equipment_role: str
    name: str
    brand: str
    model: str
    use_cases: tuple[ProductUseCase, ...] = ()
    store_skus: tuple["StoreSku", ...] = ()


class ProductDetail(ToolModel):
    product_id: str
    category_id: str
    equipment_role: str
    name: str
    brand: str
    model: str
    description: str
    use_cases: tuple[ProductUseCase, ...] = ()


class StoreSku(ToolModel):
    sku_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    sku_code: str
    sku_name: str
    specs: dict[str, object]
    sale_price: str
    available_quantity: int = Field(ge=0)
    enabled: bool


class StoreSkuListInput(ToolModel):
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class StoreSkuInput(ToolModel):
    sku_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class StoreSkuList(ToolModel):
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    items: tuple[StoreSku, ...]


class ProductDetailInput(ToolModel):
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class ProductSearchResult(ToolModel):
    items: tuple[ProductSummary, ...]
    page: int = Field(ge=0)
    size: int = Field(ge=1, le=100)
    total_elements: int = Field(ge=0)
    total_pages: int = Field(ge=0)


StoreOrderStatus = Literal[
    "PENDING_PAYMENT",
    "PAID",
    "SHIPPED",
    "RECEIVED",
    "CANCELLED",
    "CLOSED",
]


class StoreOrderListInput(ToolModel):
    status: StoreOrderStatus | None = None
    page: int = Field(default=0, ge=0)
    size: int = Field(default=5, ge=1, le=20)


class StoreOrderDetailInput(ToolModel):
    order_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class StoreOrderItem(ToolModel):
    order_item_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    sku_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    product_name: str
    sku_name: str
    specs: dict[str, object]
    unit_price: str
    quantity: int = Field(ge=1)
    subtotal: str


class StoreOrder(ToolModel):
    order_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    status: StoreOrderStatus
    currency: str
    item_amount: str
    shipping_amount: str
    payable_amount: str
    payment_expires_at: datetime
    created_at: datetime
    paid_at: datetime | None
    shipped_at: datetime | None
    received_at: datetime | None
    cancelled_at: datetime | None
    closed_at: datetime | None
    carrier: str | None
    tracking_number: str | None
    items: tuple[StoreOrderItem, ...]


class StoreOrderPage(ToolModel):
    items: tuple[StoreOrder, ...]
    page: int = Field(ge=0)
    size: int = Field(ge=1)
    total_elements: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class CatalogSearchTool(Protocol):
    async def search_products(self, request: ProductSearchInput) -> ProductSearchResult: ...
