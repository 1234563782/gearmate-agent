from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RentalPeriodInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_at: datetime
    end_at: datetime

    @model_validator(mode="after")
    def validate_period(self) -> "RentalPeriodInput":
        if self.start_at.tzinfo is None or self.end_at.tzinfo is None:
            raise ValueError("rental timestamps must include a timezone offset")
        if self.start_at >= self.end_at:
            raise ValueError("start_at must be before end_at")
        return self


class ProductSearchInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    keyword: str | None = Field(default=None, max_length=128)
    category_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    rental_period: RentalPeriodInput | None = None
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


class ProductSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    product_id: str
    category_id: str
    name: str
    brand: str
    model: str
    daily_rate: str
    fixed_deposit: str
    available_count: int | None = Field(default=None, ge=0)


class ProductSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[ProductSummary, ...]
    page: int = Field(ge=0)
    size: int = Field(ge=1, le=100)
    total_elements: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class AvailabilityInput(RentalPeriodInput):
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class AvailabilityResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    product_id: str
    start_at: datetime
    end_at: datetime
    available: bool
    available_count: int = Field(ge=0)
    checked_at: datetime


class QuoteInput(RentalPeriodInput):
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")


class PriceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: str
    pricing_version: int = Field(ge=1)
    pricing_rule: str
    billing_days: int = Field(ge=1)
    daily_rate: str
    rental_amount: str
    deposit_amount: str
    total_amount: str
    rounding_mode: str


class QuoteResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    quote_id: str
    product_id: str
    start_at: datetime
    end_at: datetime
    expires_at: datetime
    price_snapshot: PriceSnapshot


class CatalogSearchTool(Protocol):
    async def search_products(self, request: ProductSearchInput) -> ProductSearchResult:
        ...


class AvailabilityTool(Protocol):
    async def search_availability(self, request: AvailabilityInput) -> AvailabilityResult:
        ...


class QuoteTool(Protocol):
    async def create_quote(self, request: QuoteInput) -> QuoteResult:
        ...


@dataclass(frozen=True, slots=True)
class ToolPorts:
    catalog: CatalogSearchTool
    availability: AvailabilityTool
    quote: QuoteTool
