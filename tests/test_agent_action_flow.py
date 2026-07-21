from gearmate.actions import AgentAction
from gearmate.agent import GearMateAgent
from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelUsage
from gearmate.prompts.loader import RenderedPrompt
from gearmate.tools.contracts import ProductSearchResult, ProductSummary
from gearmate.tools.registry import ToolExecutionResult


class FakeModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(text="不应调用主模型", finish_reason="stop", usage=ModelUsage())

    async def close(self) -> None:
        return None


class FakeTools:
    def __init__(self, *, empty: bool = False) -> None:
        self.calls = []
        self.empty = empty

    def model_definitions(self):
        return ()

    async def execute_all(self, calls, facts, write_event):
        self.calls.extend(calls)
        result = ProductSearchResult(
            items=(
                ()
                if self.empty
                else (
                    ProductSummary(
                        product_id="01J00000000000000000000101",
                        category_id="01J00000000000000000000001",
                        equipment_role="camera",
                        name="Sony A7M4 相机机身",
                        brand="Sony",
                        model="A7M4",
                    ),
                )
            ),
            page=0,
            size=20,
            total_elements=0 if self.empty else 1,
            total_pages=0 if self.empty else 1,
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


def agent(model: FakeModel, tools: FakeTools) -> GearMateAgent:
    return GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    )


async def test_product_search_is_routed_without_main_model() -> None:
    model = FakeModel()
    tools = FakeTools()

    result = await agent(model, tools).run(
        message="有哪些相机？",
        history=[],
        action=AgentAction(
            action="product_search",
            keyword="单反",
            keyword_specificity="specific",
            equipment_role="camera",
        ),
        write_event=_ignore_event,
    )

    assert model.requests == []
    assert tools.calls[0].name == "search_products"
    assert tools.calls[0].arguments == {
        "keyword": "单反",
        "equipmentRole": "camera",
    }
    assert "Sony A7M4" in result.text


async def test_generic_category_search_drops_redundant_keyword() -> None:
    tools = FakeTools()

    await agent(FakeModel(), tools).run(
        message="我想买苹果电脑",
        history=[],
        action=AgentAction(
            action="product_search",
            keyword="苹果电脑",
            keyword_specificity="generic",
            equipment_role="laptop",
            brand="Apple",
        ),
        write_event=_ignore_event,
    )

    assert "keyword" not in tools.calls[0].arguments
    assert tools.calls[0].arguments["equipmentRole"] == "laptop"
    assert tools.calls[0].arguments["brand"] == "Apple"


async def test_product_search_routes_target_purchase_price() -> None:
    tools = FakeTools()

    await agent(FakeModel(), tools).run(
        message="我想买 5000 元左右的电脑",
        history=[],
        action=AgentAction(
            action="product_search",
            equipment_role="laptop",
            target_price="5000",
        ),
        write_event=_ignore_event,
    )

    assert tools.calls[0].arguments["targetPrice"] == "5000"
    assert "maxDailyRate" not in tools.calls[0].arguments


async def test_stock_without_product_id_clarifies_without_tools() -> None:
    model = FakeModel()
    tools = FakeTools()

    result = await agent(model, tools).run(
        message="这款有货吗？",
        history=[],
        action=AgentAction(action="sku_stock"),
        write_event=_ignore_event,
    )

    assert result.stop_reason == "NEED_CLARIFICATION"
    assert "指定一款商品" in result.text
    assert tools.calls == []
    assert model.requests == []


async def test_product_detail_uses_product_and_sku_tools() -> None:
    tools = FakeTools()

    await agent(FakeModel(), tools).run(
        message="看看第一个",
        history=[],
        action=AgentAction(
            action="product_detail",
            product_id="01J00000000000000000000101",
        ),
        write_event=_ignore_event,
    )

    assert [call.name for call in tools.calls] == ["get_product", "list_product_skus"]


async def test_empty_exact_search_does_not_broaden_results() -> None:
    result = await agent(FakeModel(), FakeTools(empty=True)).run(
        message="找单反",
        history=[],
        action=AgentAction(
            action="product_search",
            keyword="单反",
            keyword_specificity="specific",
            equipment_role="camera",
        ),
        write_event=_ignore_event,
    )

    assert "没有找到符合条件的商品" in result.text


async def _ignore_event(event_type, payload):
    return None
