import httpx

from gearmate.actions import AgentAction
from gearmate.agent import GearMateAgent
from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelUsage
from gearmate.prompts.loader import RenderedPrompt
from gearmate.rentflow.client import RentFlowClient
from gearmate.tools.contracts import ProductDetail, StoreSkuList, StoreSkuListInput
from gearmate.tools.registry import ToolExecutionResult

PRODUCT_ID = "01J00000000000000000000101"
SKU_ID = "01J00000000000000000000201"


def sku_payload() -> dict[str, object]:
    return {
        "skuId": SKU_ID,
        "productId": PRODUCT_ID,
        "skuCode": "CAM-BODY",
        "skuName": "单机身",
        "specs": {"color": "black"},
        "salePrice": "8999.00",
        "availableQuantity": 3,
        "enabled": True,
    }


async def test_store_sku_client_uses_public_read_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[sku_payload()])

    async with httpx.AsyncClient(
        base_url="http://localhost:8080", transport=httpx.MockTransport(handle)
    ) as http:
        result = await RentFlowClient(http, "").list_store_skus(
            StoreSkuListInput(product_id=PRODUCT_ID)
        )

    assert requests[0].url.path == f"/api/v1/store/products/{PRODUCT_ID}/skus"
    assert result.items[0].sale_price == "8999.00"
    assert result.items[0].available_quantity == 3


class NoCallModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(text="不应调用主模型", usage=ModelUsage())

    async def close(self) -> None:
        return None


class PurchaseTools:
    def __init__(self) -> None:
        self.calls = []

    def model_definitions(self):
        return ()

    async def execute_all(self, calls, facts, write_event):
        self.calls.extend(calls)
        detail = ProductDetail.model_validate({
            "productId": PRODUCT_ID,
            "categoryId": "01J00000000000000000000001",
            "equipmentRole": "camera",
            "name": "Sony A7M4",
            "brand": "Sony",
            "model": "A7M4",
            "description": "Full-frame camera",
        })
        skus = StoreSkuList.model_validate({
            "productId": PRODUCT_ID,
            "items": [sku_payload()],
        })
        facts.add(detail)
        facts.add(skus)
        return [
            ToolExecutionResult(
                call=calls[0], content=detail.model_dump_json(by_alias=True),
                is_error=False, result=detail,
            ),
            ToolExecutionResult(
                call=calls[1], content=skus.model_dump_json(by_alias=True),
                is_error=False, result=skus,
            ),
        ]


async def test_purchase_prepare_queries_product_and_skus_without_writing_order() -> None:
    model = NoCallModel()
    tools = PurchaseTools()

    result = await GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    ).run(
        message="第一个买两件",
        history=[],
        action=AgentAction(
            action="purchase_prepare", product_id=PRODUCT_ID, quantity=2
        ),
        write_event=_ignore_event,
    )

    assert model.requests == []
    assert [call.name for call in tools.calls] == ["get_product", "list_product_skus"]
    assert all("checkout" not in call.name and "pay" not in call.name for call in tools.calls)
    assert "单机身" in result.text
    assert "2 件" in result.text
    assert "01J000" not in result.text


async def _ignore_event(event_type, payload) -> None:
    return None
