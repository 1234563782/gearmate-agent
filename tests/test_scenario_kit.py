from decimal import Decimal

from gearmate.llm.types import ModelToolCall
from gearmate.requirements import RentalRequirements, ScenarioCatalog
from gearmate.tools.contracts import (
    AvailabilityInput,
    AvailabilityResult,
    ProductSearchInput,
    ProductSearchResult,
    ProductSummary,
    QuoteInput,
    QuoteResult,
)
from gearmate.tools.registry import ToolRegistry
from gearmate.validation.facts import FactSnapshot

ROLE_PRODUCTS = {
    "相机": ("01J00000000000000000000101", "Sony A7M4 相机机身", "200.00"),
    "镜头": ("01J00000000000000000000103", "Sony 24-70 镜头", "120.00"),
    "麦克风": ("01J00000000000000000000107", "Rode 无线麦克风", "40.00"),
    "补光灯": ("01J00000000000000000000108", "Aputure 补光灯", "40.00"),
    "采集卡": ("01J00000000000000000000109", "Elgato 采集卡", "40.00"),
    "三脚架": ("01J00000000000000000000110", "Manfrotto 三脚架", "20.00"),
}


class FakeRentFlow:
    def __init__(self) -> None:
        self.searches: list[ProductSearchInput] = []

    async def search_products(self, request: ProductSearchInput) -> ProductSearchResult:
        self.searches.append(request)
        product_id, name, rate = ROLE_PRODUCTS[request.keyword or ""]
        return ProductSearchResult(
            items=(
                ProductSummary(
                    product_id=product_id,
                    category_id="01J00000000000000000000001",
                    equipment_role=(request.equipment_role or "unknown"),
                    name=name,
                    brand=name.split()[0],
                    model=name,
                    daily_rate=rate,
                    fixed_deposit="500.00",
                    available_count=1,
                ),
            ),
            page=0,
            size=20,
            total_elements=1,
            total_pages=1,
        )

    async def search_availability(self, request: AvailabilityInput) -> AvailabilityResult:
        raise AssertionError("availability is not called directly")

    async def create_quote(self, request: QuoteInput) -> QuoteResult:
        raise AssertionError("quote is not called")


async def test_scenario_kit_is_complete_auditable_and_within_budget() -> None:
    plan = ScenarioCatalog.load_default().build_plan(
        RentalRequirements(
            scenario_id="live_streaming",
            daily_budget=Decimal("500"),
            answers={
                "streaming_mode": "camera",
                "camera_count": 1,
                "needs_audio": True,
                "needs_lighting": True,
            },
        )
    )
    rentflow = FakeRentFlow()
    registry = ToolRegistry(
        rentflow,  # type: ignore[arg-type]
        timeout_seconds=5,
        max_result_items=20,
        max_concurrency=4,
        scenario_plan=plan,
    )
    facts = FactSnapshot()
    events: list[tuple[str, dict[str, object]]] = []

    async def write_event(event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    results = await registry.execute_all(
        [ModelToolCall(id="kit-1", name="recommend_scenario_kit", arguments={})],
        facts,
        write_event,  # type: ignore[arg-type]
    )

    assert not results[0].is_error
    assert results[0].result is not None
    assert results[0].result.total_daily_rate == "460.00"  # type: ignore[attr-defined]
    assert results[0].result.within_budget is True  # type: ignore[attr-defined]
    assert results[0].result.availability_checked is False  # type: ignore[attr-defined]
    assert len(rentflow.searches) == 6
    assert all(item.max_daily_rate == Decimal("500") for item in rentflow.searches)
    assert [item.equipment_role for item in rentflow.searches] == [
        "camera",
        "lens",
        "capture_card",
        "tripod",
        "microphone",
        "lighting",
    ]
    assert events[0][0] == "tool.started"
    assert events[0][1]["arguments"] == {}

    text = (
        " ".join(f"{name} ID: {product_id}" for product_id, name, _ in ROLE_PRODUCTS.values())
        + "，组合日租合计 460.00 元"
    )
    assert facts.validate(text).valid
