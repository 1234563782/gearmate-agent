from datetime import UTC, date, datetime

import httpx

from gearmate.actions import AgentAction
from gearmate.agent import GearMateAgent
from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelUsage
from gearmate.prompts.loader import RenderedPrompt
from gearmate.rentflow.client import RentFlowClient
from gearmate.responses import UserResponseComposer
from gearmate.tools.contracts import OrderListInput, OrderPage
from gearmate.tools.registry import ToolExecutionResult
from gearmate.validation.facts import FactSnapshot


def order_payload() -> dict[str, object]:
    return {
        "orderId": "01J00000000000000000000401",
        "sourceReservationId": "01J00000000000000000000501",
        "productId": "01J00000000000000000000121",
        "productName": "iPhone 15 Pro Max",
        "productModel": "iPhone 15 Pro Max",
        "equipmentDisplayCode": None,
        "status": "CONFIRMED",
        "effectiveStatus": "CONFIRMED",
        "startDate": "2026-07-20",
        "endDate": "2026-07-22",
        "expiresAt": "2026-07-20T09:00:00Z",
        "priceSnapshot": {
            "currency": "CNY",
            "pricingVersion": 1,
            "pricingRule": "CALENDAR_DAY",
            "billingDays": 3,
            "dailyRate": "120.00",
            "rentalAmount": "360.00",
            "depositAmount": "4500.00",
            "totalAmount": "4860.00",
            "roundingMode": "CEILING",
        },
        "createdAt": "2026-07-16T10:00:00Z",
        "confirmedAt": "2026-07-16T10:05:00Z",
        "receivedAt": None,
        "cancelledAt": None,
        "expiredAt": None,
    }


async def test_order_client_uses_current_access_token_and_status_filter() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "items": [order_payload()],
                "page": 0,
                "size": 5,
                "totalElements": 1,
                "totalPages": 1,
            },
        )

    async with httpx.AsyncClient(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(handle),
    ) as http:
        result = await RentFlowClient(http, "current-user-token").list_orders(
            OrderListInput(status="CONFIRMED")
        )

    assert len(requests) == 1
    assert requests[0].url.path == "/api/v1/orders"
    assert requests[0].url.params["status"] == "CONFIRMED"
    assert requests[0].url.params["page"] == "0"
    assert requests[0].url.params["size"] == "5"
    assert requests[0].headers["Authorization"] == "Bearer current-user-token"
    assert result.items[0].product_name == "iPhone 15 Pro Max"


def test_order_response_is_localized_grounded_and_hides_internal_ids() -> None:
    facts = FactSnapshot()
    facts.add(
        OrderPage.model_validate(
            {
                "items": [order_payload()],
                "page": 0,
                "size": 5,
                "totalElements": 1,
                "totalPages": 1,
            }
        )
    )

    text = UserResponseComposer().compose(
        action=AgentAction(action="order_list", order_status="CONFIRMED"),
        facts=facts,
        rental_period=None,
        timezone="Asia/Shanghai",
    )

    assert "iPhone 15 Pro Max" in text
    assert "已确认" in text
    assert "2026-07-20 至 2026-07-22（结束日包含）" in text
    assert "¥4860.00" in text
    assert "01J000" not in text
    assert facts.validate(text).valid


def test_empty_order_result_is_explicit() -> None:
    facts = FactSnapshot()
    facts.add(OrderPage(items=(), page=0, size=5, total_elements=0, total_pages=0))

    text = UserResponseComposer().compose(
        action=AgentAction(action="order_list"),
        facts=facts,
        rental_period=None,
    )

    assert text == "当前筛选条件下没有订单。"


def test_order_rental_dates_are_date_only_while_lifecycle_timestamps_keep_timezone() -> None:
    order = OrderPage.model_validate(
        {
            "items": [order_payload()],
            "page": 0,
            "size": 5,
            "totalElements": 1,
            "totalPages": 1,
        }
    ).items[0]

    assert order.start_date == date(2026, 7, 20)
    assert order.end_date == date(2026, 7, 22)
    assert order.confirmed_at == datetime(2026, 7, 16, 10, 5, tzinfo=UTC)


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
        result = OrderPage.model_validate(
            {
                "items": [order_payload()],
                "page": 0,
                "size": 5,
                "totalElements": 1,
                "totalPages": 1,
            }
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


async def test_order_action_is_routed_without_main_model_tool_choice() -> None:
    model = NoCallModel()
    tools = OrderTools()

    result = await GearMateAgent(
        model,
        tools,  # type: ignore[arg-type]
        Settings(_env_file=None),
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    ).run(
        message="查看我的已确认订单",
        history=[],
        rental_period=None,
        scenario_plan=None,
        action=AgentAction(action="order_list", order_status="CONFIRMED"),
        write_event=_ignore_event,
        timezone="Asia/Shanghai",
    )

    assert model.requests == []
    assert len(tools.calls) == 1
    assert tools.calls[0].name == "list_orders"
    assert tools.calls[0].arguments == {"page": 0, "size": 5, "status": "CONFIRMED"}
    assert "iPhone 15 Pro Max" in result.text
    assert "01J000" not in result.text


async def _ignore_event(event_type, payload) -> None:
    return None
