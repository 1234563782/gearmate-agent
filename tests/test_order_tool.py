from datetime import UTC, datetime

import httpx

from gearmate.actions import AgentAction
from gearmate.agent import GearMateAgent
from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelUsage
from gearmate.prompts.loader import RenderedPrompt
from gearmate.rentflow.client import RentFlowClient
from gearmate.responses import UserResponseComposer
from gearmate.tools.contracts import StoreOrderListInput, StoreOrderPage
from gearmate.tools.registry import ToolExecutionResult
from gearmate.validation.facts import FactSnapshot


def order_payload() -> dict[str, object]:
    return {
        "orderId": "01J00000000000000000000401",
        "status": "SHIPPED",
        "currency": "CNY",
        "itemAmount": "6999.00",
        "shippingAmount": "0.00",
        "payableAmount": "6999.00",
        "paymentExpiresAt": "2026-07-20T09:00:00Z",
        "createdAt": "2026-07-20T08:30:00Z",
        "paidAt": "2026-07-20T08:35:00Z",
        "shippedAt": "2026-07-20T10:00:00Z",
        "receivedAt": None,
        "cancelledAt": None,
        "closedAt": None,
        "carrier": "SF",
        "trackingNumber": "SF123456",
        "items": [
            {
                "orderItemId": "01J00000000000000000000402",
                "productId": "01J00000000000000000000121",
                "skuId": "01J00000000000000000000221",
                "productName": "iPhone 15 Pro Max",
                "skuName": "256GB 原色钛金属",
                "specs": {"storage": "256GB"},
                "unitPrice": "6999.00",
                "quantity": 1,
                "subtotal": "6999.00",
            }
        ],
    }


async def test_store_order_client_uses_current_access_token_and_status_filter() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={
            "items": [order_payload()], "page": 0, "size": 5,
            "totalElements": 1, "totalPages": 1,
        })

    async with httpx.AsyncClient(
        base_url="http://localhost:8080", transport=httpx.MockTransport(handle)
    ) as http:
        result = await RentFlowClient(http, "current-user-token").list_store_orders(
            StoreOrderListInput(status="SHIPPED")
        )

    assert requests[0].url.path == "/api/v1/store/orders"
    assert requests[0].url.params["status"] == "SHIPPED"
    assert requests[0].headers["Authorization"] == "Bearer current-user-token"
    assert result.items[0].items[0].product_name == "iPhone 15 Pro Max"


def test_store_order_response_is_grounded_and_hides_internal_ids() -> None:
    facts = FactSnapshot()
    facts.add(StoreOrderPage.model_validate({
        "items": [order_payload()], "page": 0, "size": 5,
        "totalElements": 1, "totalPages": 1,
    }))

    text = UserResponseComposer().compose(
        action=AgentAction(action="order_list", order_status="SHIPPED"),
        facts=facts,
    )

    assert "iPhone 15 Pro Max" in text
    assert "待收货" in text
    assert "¥6999.00" in text
    assert "01J000" not in text
    assert facts.validate(text).valid


def test_empty_store_order_result_is_explicit() -> None:
    facts = FactSnapshot()
    facts.add(StoreOrderPage(items=(), page=0, size=5, total_elements=0, total_pages=0))

    text = UserResponseComposer().compose(action=AgentAction(action="order_list"), facts=facts)

    assert text == "当前筛选条件下没有商城订单。"


class NoCallModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(text="不应调用主模型", usage=ModelUsage())

    async def close(self) -> None:
        return None


class OrderTools:
    def __init__(self) -> None:
        self.calls = []

    def model_definitions(self):
        return ()

    async def execute_all(self, calls, facts, write_event):
        self.calls.extend(calls)
        result = StoreOrderPage.model_validate({
            "items": [order_payload()], "page": 0, "size": 5,
            "totalElements": 1, "totalPages": 1,
        })
        facts.add(result)
        return [ToolExecutionResult(
            call=calls[0], content=result.model_dump_json(by_alias=True),
            is_error=False, result=result,
        )]


async def test_store_order_action_is_routed_without_main_model() -> None:
    model = NoCallModel()
    tools = OrderTools()

    result = await GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    ).run(
        message="查看我的待收货订单",
        history=[],
        action=AgentAction(action="order_list", order_status="SHIPPED"),
        write_event=_ignore_event,
    )

    assert model.requests == []
    assert tools.calls[0].name == "list_store_orders"
    assert tools.calls[0].arguments == {"page": 0, "size": 5, "status": "SHIPPED"}
    assert "iPhone 15 Pro Max" in result.text
    assert "01J000" not in result.text


def test_store_order_lifecycle_timestamps_keep_timezone() -> None:
    order = StoreOrderPage.model_validate({
        "items": [order_payload()], "page": 0, "size": 5,
        "totalElements": 1, "totalPages": 1,
    }).items[0]

    assert order.shipped_at == datetime(2026, 7, 20, 10, 0, tzinfo=UTC)


async def _ignore_event(event_type, payload) -> None:
    return None
