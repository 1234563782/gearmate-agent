import json
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from gearmate.actions import PendingProductSearch, PendingRentalAction
from gearmate.agent.service import RunCoordinator
from gearmate.config import Settings
from gearmate.llm.types import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
)
from gearmate.memory import ConversationMessageMemory, ConversationStateMemory
from gearmate.prompts.loader import RenderedPrompt
from gearmate.requirements import RentalRequirements
from gearmate.tools.contracts import RentalPeriodInput


class FakeRepository:
    def __init__(self, message: str) -> None:
        self.message = message
        self.remembered: RentalPeriodInput | None = None
        self.remembered_requirements: RentalRequirements | None = None
        self.remembered_pending_search: PendingProductSearch | None = None
        self.remembered_pending_rental_action: PendingRentalAction | None = None
        self.events: list[tuple[str, dict[str, object]]] = []
        self.finalized: dict[str, object] | None = None
        self.state: ConversationStateMemory | None = None
        self.run_created = False

    async def require_conversation(self, conversation_id: str, user_id: str) -> object:
        return object()

    async def create_run(self, *args: object, **kwargs: object) -> object:
        self.run_created = True
        raise AssertionError("a run must not be created in this test")

    async def conversation_timezone(self, conversation_id: str) -> str:
        return "Asia/Shanghai"

    async def conversation_state(self, conversation_id: str) -> ConversationStateMemory | None:
        return self.state

    async def latest_conversation_summary(self, conversation_id: str) -> None:
        return None

    async def recent_conversation_messages(
        self,
        conversation_id: str,
        limit: int,
        after_event_id: str | None = None,
    ) -> list[ConversationMessageMemory]:
        return [
            ConversationMessageMemory(
                event_id="01J00000000000000000000001",
                role="user",
                content=self.message,
                created_at=datetime.now(UTC),
            )
        ]

    async def conversation_messages_after(
        self, conversation_id: str, after_event_id: str | None, limit: int
    ) -> list[ConversationMessageMemory]:
        return []

    async def upsert_conversation_rental_period(
        self, conversation_id: str, rental_period: RentalPeriodInput
    ) -> None:
        self.remembered = rental_period
        state = self.state or ConversationStateMemory(None, None)
        self.state = ConversationStateMemory(
            rental_period.start_at,
            rental_period.end_at,
            state.rental_requirements,
            state.pending_product_search,
            state.pending_rental_action,
        )

    async def clear_conversation_rental_period(self, conversation_id: str) -> None:
        self.remembered = None
        if self.state is not None:
            self.state = ConversationStateMemory(
                None,
                None,
                self.state.rental_requirements,
                self.state.pending_product_search,
                self.state.pending_rental_action,
            )

    async def upsert_pending_product_search(
        self,
        conversation_id: str,
        pending_search: PendingProductSearch,
    ) -> None:
        self.remembered_pending_search = pending_search
        state = self.state or ConversationStateMemory(None, None)
        self.state = ConversationStateMemory(
            state.rental_start_at,
            state.rental_end_at,
            state.rental_requirements,
            pending_search,
            state.pending_rental_action,
        )

    async def clear_pending_product_search(self, conversation_id: str) -> None:
        self.remembered_pending_search = None
        if self.state is not None:
            self.state = ConversationStateMemory(
                self.state.rental_start_at,
                self.state.rental_end_at,
                self.state.rental_requirements,
                None,
                self.state.pending_rental_action,
            )

    async def upsert_pending_rental_action(
        self,
        conversation_id: str,
        pending_action: PendingRentalAction,
    ) -> None:
        self.remembered_pending_rental_action = pending_action
        state = self.state or ConversationStateMemory(None, None)
        self.state = ConversationStateMemory(
            state.rental_start_at,
            state.rental_end_at,
            state.rental_requirements,
            state.pending_product_search,
            pending_action,
        )

    async def clear_pending_rental_action(self, conversation_id: str) -> None:
        self.remembered_pending_rental_action = None
        if self.state is not None:
            self.state = ConversationStateMemory(
                self.state.rental_start_at,
                self.state.rental_end_at,
                self.state.rental_requirements,
                self.state.pending_product_search,
                None,
            )

    async def upsert_conversation_requirements(
        self,
        conversation_id: str,
        requirements: RentalRequirements,
    ) -> None:
        self.remembered_requirements = requirements

    async def save_conversation_summary(self, **kwargs: object) -> None:
        raise AssertionError("summary should not run for one message")

    async def append_event(self, run_id: str, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))

    async def finalize_run(self, run_id: str, **kwargs: object) -> None:
        self.finalized = kwargs


class SequenceModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def close(self) -> None:
        return None


async def test_run_resolves_and_remembers_natural_language_period() -> None:
    message = "2026 年 7 月 20 日 9 点到 7 月 22 日 18 点租一台相机"
    repository = FakeRepository(message)
    period_arguments = {
        "startAt": "2026-07-20T09:00:00+08:00",
        "endAt": "2026-07-22T18:00:00+08:00",
    }
    model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=10, output_tokens=3),
                tool_calls=(
                    ModelToolCall(
                        id="action-1",
                        name="resolve_agent_action",
                        arguments={"action": "chat"},
                    ),
                ),
            ),
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=50, output_tokens=10),
                tool_calls=(
                    ModelToolCall(
                        id="extract-1",
                        name="set_rental_period",
                        arguments=period_arguments,
                    ),
                ),
            ),
            ModelResponse(
                text="租期已经确认。",
                finish_reason="stop",
                usage=ModelUsage(input_tokens=30, output_tokens=7),
            ),
        ]
    )
    settings = Settings(_env_file=None, context_summary_trigger_tokens=8000)
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = model

        await coordinator._execute(
            run_id="run-1",
            conversation_id="conversation-1",
            access_token="token",
            message=message,
            rental_period=None,
        )

    assert repository.remembered == RentalPeriodInput.model_validate(period_arguments)
    assert repository.finalized is not None
    assert repository.finalized["status"] == "COMPLETED"
    assert repository.finalized["input_tokens"] == 90
    assert repository.finalized["output_tokens"] == 20
    assert repository.finalized["model_rounds"] == 3
    assert any(event_type == "assistant.completed" for event_type, _ in repository.events)
    assert len(model.requests) == 3


async def test_pending_specific_search_survives_rental_period_clarification() -> None:
    first_message = "单反，没有预算限制，计划从今天下午两点开始租，租一天"
    repository = FakeRepository(first_message)
    first_model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=20, output_tokens=5),
                tool_calls=(
                    ModelToolCall(
                        id="action-specific-search",
                        name="resolve_agent_action",
                        arguments={
                            "action": "product_search",
                            "keyword": "单反",
                            "equipmentRole": "camera",
                        },
                    ),
                ),
            ),
            ModelResponse(
                text="今天下午两点已经过去了，你是想从明天下午两点开始租吗？",
                finish_reason="stop",
                usage=ModelUsage(input_tokens=30, output_tokens=8),
            ),
        ]
    )
    settings = Settings(_env_file=None)
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = first_model
        await coordinator._execute(
            run_id="run-search-clarification",
            conversation_id="conversation-specific-search",
            access_token="token",
            message=first_message,
            rental_period=None,
        )

    pending = repository.remembered_pending_search
    assert pending is not None
    assert pending.keyword == "单反"
    assert pending.equipment_role == "camera"
    assert pending.waiting_for_rental_period is True

    repository.message = "对"
    repository.events = []
    repository.finalized = None
    second_model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=15, output_tokens=4),
                tool_calls=(
                    ModelToolCall(
                        id="action-continue-search",
                        name="resolve_agent_action",
                        arguments={
                            "action": "product_search",
                            "continuesPending": True,
                        },
                    ),
                ),
            ),
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=25, output_tokens=6),
                tool_calls=(
                    ModelToolCall(
                        id="period-confirmed",
                        name="set_rental_period",
                        arguments={
                            "startAt": "2026-07-16T14:00:00+08:00",
                            "endAt": "2026-07-17T14:00:00+08:00",
                        },
                    ),
                ),
            ),
        ]
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "productId": "01J00000000000000000000101",
                        "categoryId": "01J00000000000000000000001",
                        "equipmentRole": "camera",
                        "name": "Canon DSLR 单反相机",
                        "brand": "Canon",
                        "model": "DSLR",
                        "dailyRate": "200.00",
                        "fixedDeposit": "1000.00",
                    },
                    {
                        "productId": "01J00000000000000000000110",
                        "categoryId": "01J00000000000000000000002",
                        "equipmentRole": "tripod",
                        "name": "Manfrotto Befree 三脚架",
                        "brand": "Manfrotto",
                        "model": "Befree",
                        "dailyRate": "20.00",
                        "fixedDeposit": "100.00",
                    },
                ],
                "page": 0,
                "size": 20,
                "totalElements": 2,
                "totalPages": 1,
            },
        )

    async with httpx.AsyncClient(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(handle),
    ) as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = second_model
        await coordinator._execute(
            run_id="run-search-confirmed",
            conversation_id="conversation-specific-search",
            access_token="token",
            message="对",
            rental_period=None,
        )

    assert len(requests) == 1
    assert requests[0].url.params["keyword"] == "单反"
    assert requests[0].url.params["equipmentRole"] == "camera"
    assert repository.finalized is not None
    reply = repository.finalized["state"]["reply"]  # type: ignore[index]
    assert "Canon DSLR" in reply
    assert "Manfrotto" not in reply
    assert repository.remembered_pending_search is None


async def test_run_clarifies_and_remembers_vague_live_streaming_request() -> None:
    message = "我需要直播设备，预算每天 500 元以内"
    repository = FakeRepository(message)
    model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=10, output_tokens=3),
                tool_calls=(
                    ModelToolCall(
                        id="action-2",
                        name="resolve_agent_action",
                        arguments={"action": "scenario_continue"},
                    ),
                ),
            ),
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=40, output_tokens=8),
                tool_calls=(
                    ModelToolCall(
                        id="requirements-1",
                        name="set_rental_requirements",
                        arguments={
                            "scenarioId": "live_streaming",
                            "dailyBudget": "500",
                            "answers": {},
                        },
                    ),
                ),
            ),
        ]
    )
    settings = Settings(_env_file=None)
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = model

        await coordinator._execute(
            run_id="run-2",
            conversation_id="conversation-2",
            access_token="token",
            message=message,
            rental_period=None,
        )

    assert repository.remembered_requirements == RentalRequirements(
        scenario_id="live_streaming", daily_budget=Decimal("500")
    )
    assert repository.finalized is not None
    assert repository.finalized["stop_reason"] == "NEED_CLARIFICATION"
    assert repository.finalized["model_rounds"] == 2
    assert any(event_type == "requirements.resolved" for event_type, _ in repository.events)


async def test_structured_period_outside_window_does_not_create_run_or_memory() -> None:
    repository = FakeRepository("搜索相机")
    settings = Settings(_env_file=None)
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )

        with pytest.raises(ValueError, match="未来 90 天"):
            await coordinator.start(
                conversation_id="conversation-invalid",
                user_id="user-1",
                access_token="token",
                message="搜索相机",
                rental_period=RentalPeriodInput(
                    start_at=datetime(2035, 7, 20, tzinfo=UTC),
                    end_at=datetime(2035, 7, 22, tzinfo=UTC),
                ),
            )

    assert repository.remembered is None
    assert repository.run_created is False


def complete_scenario_state(
    rental_period: RentalPeriodInput | None = None,
) -> ConversationStateMemory:
    return ConversationStateMemory(
        rental_period.start_at if rental_period else None,
        rental_period.end_at if rental_period else None,
        RentalRequirements(
            scenario_id="live_streaming",
            daily_budget=Decimal("500"),
            answers={
                "streaming_mode": "camera",
                "camera_count": 1,
                "needs_audio": True,
                "needs_lighting": True,
            },
        ),
    )


async def test_thanks_does_not_replay_complete_saved_scenario() -> None:
    repository = FakeRepository("谢谢")
    repository.state = complete_scenario_state()
    model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=10, output_tokens=2),
                tool_calls=(
                    ModelToolCall(
                        id="action-chat",
                        name="resolve_agent_action",
                        arguments={"action": "chat"},
                    ),
                ),
            ),
            ModelResponse(
                text="不客气。",
                finish_reason="stop",
                usage=ModelUsage(input_tokens=12, output_tokens=3),
            ),
        ]
    )
    settings = Settings(_env_file=None)
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = model

        await coordinator._execute(
            run_id="run-chat",
            conversation_id="conversation-chat",
            access_token="token",
            message="谢谢",
            rental_period=None,
        )

    assert repository.finalized is not None
    assert repository.finalized["state"]["reply"] == "不客气。"  # type: ignore[index]
    assert not any(event == "requirements.resolved" for event, _ in repository.events)
    assert not any(event == "tool.started" for event, _ in repository.events)
    assert model.requests[1].tools == ()


async def test_new_product_search_does_not_replay_saved_scenario() -> None:
    repository = FakeRepository("有哪些相机可以租？")
    repository.state = complete_scenario_state()
    model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=10, output_tokens=2),
                tool_calls=(
                    ModelToolCall(
                        id="action-search",
                        name="resolve_agent_action",
                        arguments={
                            "action": "product_search",
                            "keyword": "相机",
                        },
                    ),
                ),
            )
        ]
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "productId": "01J00000000000000000000101",
                        "categoryId": "01J00000000000000000000001",
                        "equipmentRole": "camera",
                        "name": "Sony A7M4 相机机身",
                        "brand": "Sony",
                        "model": "A7M4",
                        "dailyRate": "200.00",
                        "fixedDeposit": "1000.00",
                    }
                ],
                "page": 0,
                "size": 20,
                "totalElements": 1,
                "totalPages": 1,
            },
        )

    settings = Settings(_env_file=None)
    async with httpx.AsyncClient(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(handle),
    ) as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = model

        await coordinator._execute(
            run_id="run-search",
            conversation_id="conversation-search",
            access_token="token",
            message="有哪些相机可以租？",
            rental_period=None,
        )

    assert len(requests) == 1
    assert requests[0].url.path == "/api/v1/products"
    assert requests[0].url.params["keyword"] == "相机"
    assert not any(event == "requirements.resolved" for event, _ in repository.events)
    tool_events = [payload for event, payload in repository.events if event == "tool.started"]
    assert [payload["tool"] for payload in tool_events] == ["search_products"]


async def test_saved_valid_period_is_reused_for_availability() -> None:
    product_id = "01J00000000000000000000101"
    period = RentalPeriodInput(
        start_at=datetime(2026, 7, 20, tzinfo=UTC),
        end_at=datetime(2026, 7, 22, tzinfo=UTC),
    )
    repository = FakeRepository("这台有货吗？")
    repository.state = complete_scenario_state(period)
    model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=10, output_tokens=2),
                tool_calls=(
                    ModelToolCall(
                        id="action-availability",
                        name="resolve_agent_action",
                        arguments={
                            "action": "availability",
                            "productId": product_id,
                        },
                    ),
                ),
            )
        ]
    )
    bodies: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "productId": product_id,
                "startAt": "2026-07-20T00:00:00Z",
                "endAt": "2026-07-22T00:00:00Z",
                "available": True,
                "availableCount": 2,
                "checkedAt": "2026-07-15T08:00:00Z",
            },
        )

    settings = Settings(_env_file=None)
    async with httpx.AsyncClient(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(handle),
    ) as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = model

        await coordinator._execute(
            run_id="run-availability",
            conversation_id="conversation-availability",
            access_token="token",
            message="这台有货吗？",
            rental_period=None,
        )

    assert bodies == [
        {
            "startAt": "2026-07-20T00:00:00Z",
            "endAt": "2026-07-22T00:00:00Z",
            "productId": product_id,
        }
    ]
    assert repository.finalized is not None
    assert "可租 2 台" in repository.finalized["state"]["reply"]  # type: ignore[index]


async def test_selected_product_survives_period_confirmation_rounds() -> None:
    product_id = "01J00000000000000000000101"
    repository = FakeRepository("帮我看看第一个")
    settings = Settings(_env_file=None)
    first_model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=10, output_tokens=2),
                tool_calls=(
                    ModelToolCall(
                        id="select-first-product",
                        name="resolve_agent_action",
                        arguments={
                            "action": "availability",
                            "productId": product_id,
                        },
                    ),
                ),
            )
        ]
    )
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = first_model
        await coordinator._execute(
            run_id="run-select-first",
            conversation_id="conversation-confirm-period",
            access_token="token",
            message="帮我看看第一个",
            rental_period=None,
        )

    assert repository.remembered_pending_rental_action == PendingRentalAction(
        action="availability",
        product_id=product_id,
    )
    assert repository.finalized is not None
    assert repository.finalized["stop_reason"] == "NEED_CLARIFICATION"

    repository.message = "从明天下午六点到后天下午六点"
    repository.events = []
    repository.finalized = None
    date_model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=10, output_tokens=2),
                tool_calls=(
                    ModelToolCall(
                        id="continue-with-dates",
                        name="resolve_agent_action",
                        arguments={
                            "action": "availability",
                            "continuesPending": True,
                        },
                    ),
                ),
            ),
            ModelResponse(
                text="您确认这个租赁时间正确吗？",
                finish_reason="stop",
                usage=ModelUsage(input_tokens=20, output_tokens=5),
            ),
        ]
    )
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = date_model
        await coordinator._execute(
            run_id="run-propose-period",
            conversation_id="conversation-confirm-period",
            access_token="token",
            message=repository.message,
            rental_period=None,
        )

    assert repository.remembered_pending_rental_action is not None
    assert repository.finalized is not None
    assert repository.finalized["stop_reason"] == "NEED_CLARIFICATION"

    repository.message = "是"
    repository.events = []
    repository.finalized = None
    confirmation_model = SequenceModel(
        [
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=10, output_tokens=2),
                tool_calls=(
                    ModelToolCall(
                        id="confirm-period-action",
                        name="resolve_agent_action",
                        arguments={
                            "action": "chat",
                            "continuesPending": True,
                        },
                    ),
                ),
            ),
            ModelResponse(
                text="",
                finish_reason="tool_calls",
                usage=ModelUsage(input_tokens=20, output_tokens=5),
                tool_calls=(
                    ModelToolCall(
                        id="confirmed-period",
                        name="set_rental_period",
                        arguments={
                            "startAt": "2026-07-16T18:00:00+08:00",
                            "endAt": "2026-07-17T18:00:00+08:00",
                        },
                    ),
                ),
            ),
        ]
    )
    bodies: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "productId": product_id,
                "startAt": "2026-07-16T10:00:00Z",
                "endAt": "2026-07-17T10:00:00Z",
                "available": True,
                "availableCount": 1,
                "checkedAt": "2026-07-15T09:00:00Z",
            },
        )

    async with httpx.AsyncClient(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(handle),
    ) as rentflow_http:
        coordinator = RunCoordinator(
            settings,
            repository,  # type: ignore[arg-type]
            rentflow_http,
            RenderedPrompt(version="test", content_hash="hash", content="system"),
        )
        coordinator._model = confirmation_model
        await coordinator._execute(
            run_id="run-confirm-period",
            conversation_id="conversation-confirm-period",
            access_token="token",
            message=repository.message,
            rental_period=None,
        )

    assert bodies == [
        {
            "startAt": "2026-07-16T18:00:00+08:00",
            "endAt": "2026-07-17T18:00:00+08:00",
            "productId": product_id,
        }
    ]
    assert repository.finalized is not None
    assert repository.finalized["stop_reason"] == "COMPLETED"
    assert repository.remembered_pending_rental_action is None
