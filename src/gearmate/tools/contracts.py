from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from gearmate.requirements import EquipmentRole


class ToolModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class RentalPeriodInput(ToolModel):
    start_at: datetime
    end_at: datetime

    @model_validator(mode="after")
    def validate_period(self) -> "RentalPeriodInput":
        if self.start_at.tzinfo is None or self.end_at.tzinfo is None:
            raise ValueError("rental timestamps must include a timezone offset")
        if self.start_at >= self.end_at:
            raise ValueError("start_at must be before end_at")
        return self


class ProductSearchInput(ToolModel):
    keyword: str | None = Field(default=None, max_length=128)
    equipment_role: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    brand: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=64)
    semantic_query: str | None = Field(default=None, max_length=512)
    use_case_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    category_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    rental_period: RentalPeriodInput | None = None
    max_daily_rate: Decimal | None = Field(default=None, gt=0, max_digits=10)
    target_daily_rate: Decimal | None = Field(default=None, gt=0, max_digits=10)
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
    daily_rate: str
    fixed_deposit: str
    available_count: int | None = Field(default=None, ge=0)
    use_cases: tuple[ProductUseCase, ...] = ()


class ProductDetail(ToolModel):
    product_id: str
    category_id: str
    equipment_role: str
    name: str
    brand: str
    model: str
    description: str
    daily_rate: str
    fixed_deposit: str
    use_cases: tuple[ProductUseCase, ...] = ()


class ProductDetailInput(ToolModel):
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class ProductSearchResult(ToolModel):
    items: tuple[ProductSummary, ...]
    page: int = Field(ge=0)
    size: int = Field(ge=1, le=100)
    total_elements: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class ScenarioKitInput(ToolModel):
    rental_period: RentalPeriodInput | None = None


class ScenarioKitItem(ToolModel):
    role: EquipmentRole
    quantity: int = Field(ge=1, le=8)
    product: ProductSummary
    subtotal_daily_rate: str


class ScenarioKitResult(ToolModel):
    scenario: str
    items: tuple[ScenarioKitItem, ...]
    total_daily_rate: str
    max_daily_budget: str
    within_budget: bool
    availability_checked: bool
    missing_roles: tuple[EquipmentRole, ...] = ()


class AvailabilityInput(RentalPeriodInput):
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class AvailabilityResult(ToolModel):
    product_id: str
    start_at: datetime
    end_at: datetime
    available: bool
    available_count: int = Field(ge=0)
    checked_at: datetime


class QuoteInput(RentalPeriodInput):
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class PriceSnapshot(ToolModel):
    currency: str
    pricing_version: int = Field(ge=1)
    pricing_rule: str
    billing_days: int = Field(ge=1)
    daily_rate: str
    rental_amount: str
    deposit_amount: str
    total_amount: str
    rounding_mode: str


class QuoteResult(ToolModel):
    quote_id: str
    product_id: str
    start_at: datetime
    end_at: datetime
    expires_at: datetime
    price_snapshot: PriceSnapshot


OrderStatus = Literal[
    "PENDING_CONFIRMATION",
    "CONFIRMED",
    "RECEIVED",
    "CANCELLED",
    "EXPIRED",
]


class OrderListInput(ToolModel):
    status: OrderStatus | None = None
    page: int = Field(default=0, ge=0)
    size: int = Field(default=5, ge=1, le=20)


class OrderSummary(ToolModel):
    order_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    source_reservation_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    product_name: str
    product_model: str
    equipment_display_code: str | None
    status: OrderStatus
    effective_status: OrderStatus
    start_at: datetime
    end_at: datetime
    expires_at: datetime
    price_snapshot: PriceSnapshot
    created_at: datetime
    confirmed_at: datetime | None
    received_at: datetime | None
    cancelled_at: datetime | None
    expired_at: datetime | None


class OrderPage(ToolModel):
    items: tuple[OrderSummary, ...]
    page: int = Field(ge=0)
    size: int = Field(ge=1)
    total_elements: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class CatalogSearchTool(Protocol):
    async def search_products(self, request: ProductSearchInput) -> ProductSearchResult: ...


class AvailabilityTool(Protocol):
    async def search_availability(self, request: AvailabilityInput) -> AvailabilityResult: ...


class QuoteTool(Protocol):
    async def create_quote(self, request: QuoteInput) -> QuoteResult: ...


@dataclass(frozen=True, slots=True)
class ToolPorts:
    catalog: CatalogSearchTool
    availability: AvailabilityTool
    quote: QuoteTool
