from datetime import UTC, datetime

from gearmate.actions import AgentAction
from gearmate.agent.graph import GearMateAgent
from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelUsage
from gearmate.prompts.loader import RenderedPrompt
from gearmate.tools.contracts import ProductSearchResult, ProductSummary, RentalPeriodInput
from gearmate.tools.registry import ToolExecutionResult


class FakeModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text="不应调用主模型",
            finish_reason="stop",
            usage=ModelUsage(),
        )

    async def close(self) -> None:
        return None


class FakeTools:
    def __init__(self) -> None:
        self.calls = []

    def model_definitions(self):
        return ()

    async def execute_all(self, calls, facts, write_event):
        self.calls.extend(calls)
        result = ProductSearchResult(
            items=(
                ProductSummary(
                    product_id="01J00000000000000000000101",
                    category_id="01J00000000000000000000001",
                    equipment_role="camera",
                    name="Sony A7M4 相机机身",
                    brand="Sony",
                    model="A7M4",
                    daily_rate="200.00",
                    fixed_deposit="1000.00",
                    available_count=2,
                ),
            ),
            page=0,
            size=20,
            total_elements=1,
            total_pages=1,
        )
        facts.add(result)
        return [
            ToolExecutionResult(
                call=calls[0],
                content=result.model_dump_json(by_alias=True),
                is_error=False,
                result=result,
            )
        ]


async def test_product_search_is_routed_without_main_model_tool_choice() -> None:
    model = FakeModel()
    tools = FakeTools()
    period = RentalPeriodInput(
        start_at=datetime(2026, 7, 20, tzinfo=UTC),
        end_at=datetime(2026, 7, 22, tzinfo=UTC),
    )

    result = await GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    ).run(
        message="有哪些相机可以租？",
        history=[],
        rental_period=period,
        scenario_plan=None,
        action=AgentAction(
            action="product_search",
            keyword="相机",
            keyword_specificity="specific",
            equipment_role="camera",
        ),
        write_event=_ignore_event,
    )

    assert model.requests == []
    assert len(tools.calls) == 1
    assert tools.calls[0].name == "search_products"
    assert tools.calls[0].arguments["keyword"] == "相机"
    assert tools.calls[0].arguments["equipmentRole"] == "camera"
    assert tools.calls[0].arguments["rentalPeriod"]["startAt"] == ("2026-07-20T00:00:00Z")
    assert "Sony A7M4" in result.text


async def test_generic_laptop_search_drops_redundant_keyword() -> None:
    model = FakeModel()
    tools = FakeTools()

    await GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    ).run(
        message="我想租苹果电脑",
        history=[],
        rental_period=None,
        scenario_plan=None,
        action=AgentAction(
            action="product_search",
            keyword="苹果电脑",
            keyword_specificity="generic",
            equipment_role="laptop",
            brand="Apple",
        ),
        write_event=_ignore_event,
    )

    assert len(tools.calls) == 1
    assert "keyword" not in tools.calls[0].arguments
    assert tools.calls[0].arguments["equipmentRole"] == "laptop"
    assert tools.calls[0].arguments["brand"] == "Apple"


async def test_availability_without_product_id_clarifies_without_tools() -> None:
    model = FakeModel()
    tools = FakeTools()

    result = await GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    ).run(
        message="这段时间有货吗？",
        history=[],
        rental_period=RentalPeriodInput(
            start_at=datetime(2026, 7, 20, tzinfo=UTC),
            end_at=datetime(2026, 7, 22, tzinfo=UTC),
        ),
        scenario_plan=None,
        action=AgentAction(action="availability"),
        write_event=_ignore_event,
    )

    assert result.stop_reason == "NEED_CLARIFICATION"
    assert "商品 ID" in result.text
    assert tools.calls == []
    assert model.requests == []


class EmptySearchTools(FakeTools):
    async def execute_all(self, calls, facts, write_event):
        self.calls.extend(calls)
        result = ProductSearchResult(
            items=(),
            page=0,
            size=20,
            total_elements=0,
            total_pages=0,
        )
        facts.add(result)
        return [
            ToolExecutionResult(
                call=calls[0],
                content=result.model_dump_json(by_alias=True),
                is_error=False,
                result=result,
            )
        ]


async def test_empty_exact_search_does_not_broaden_results() -> None:
    model = FakeModel()
    tools = EmptySearchTools()

    result = await GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    ).run(
        message="找单反",
        history=[],
        rental_period=None,
        scenario_plan=None,
        action=AgentAction(
            action="product_search",
            keyword="单反",
            keyword_specificity="specific",
            equipment_role="camera",
        ),
        write_event=_ignore_event,
    )

    assert model.requests == []
    assert "没有返回符合这些搜索条件的商品" in result.text


async def _ignore_event(event_type, payload):
    return None
