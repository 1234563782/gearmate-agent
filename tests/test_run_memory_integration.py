from datetime import UTC, datetime

import httpx

from gearmate.actions import PendingProductSearch
from gearmate.agent.service import RunCoordinator
from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelToolCall, ModelUsage
from gearmate.memory import ConversationMessageMemory, ConversationStateMemory
from gearmate.prompts.loader import RenderedPrompt
from gearmate.search import RecentProductSearch


class FakeRepository:
    def __init__(self, message: str) -> None:
        self.message = message
        self.state: ConversationStateMemory | None = None
        self.events: list[tuple[str, dict[str, object]]] = []
        self.finalized: dict[str, object] | None = None
        self.pending_search: PendingProductSearch | None = None
        self.recent_search: RecentProductSearch | None = None

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

    async def upsert_pending_product_search(
        self,
        conversation_id: str,
        pending_search: PendingProductSearch,
    ) -> None:
        self.pending_search = pending_search
        recent = self.state.recent_product_search if self.state is not None else None
        self.state = ConversationStateMemory(
            pending_product_search=pending_search,
            recent_product_search=recent,
        )

    async def clear_pending_product_search(self, conversation_id: str) -> None:
        self.pending_search = None
        if self.state is not None:
            self.state = ConversationStateMemory(
                recent_product_search=self.state.recent_product_search
            )

    async def upsert_recent_product_search(
        self,
        conversation_id: str,
        recent_search: RecentProductSearch,
    ) -> None:
        self.recent_search = recent_search
        pending = self.state.pending_product_search if self.state is not None else None
        self.state = ConversationStateMemory(
            pending_product_search=pending,
            recent_product_search=recent_search,
        )

    async def save_conversation_summary(self, **kwargs: object) -> None:
        raise AssertionError("summary should not run for one message")

    async def append_event(
        self, run_id: str, event_type: str, payload: dict[str, object]
    ) -> None:
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


def coordinator(
    repository: FakeRepository,
    model: SequenceModel,
    http: httpx.AsyncClient,
    **settings: object,
) -> RunCoordinator:
    result = RunCoordinator(
        Settings(_env_file=None, **settings),
        repository,  # type: ignore[arg-type]
        http,
        RenderedPrompt(version="test", content_hash="hash", content="system"),
    )
    result._model = model
    return result


def action_response(arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        text="",
        finish_reason="tool_calls",
        usage=ModelUsage(input_tokens=10, output_tokens=2),
        tool_calls=(
            ModelToolCall(
                id="action-1",
                name="resolve_agent_action",
                arguments=arguments,
            ),
        ),
    )


async def test_enforced_pure_social_route_skips_action_model() -> None:
    repository = FakeRepository("谢谢")
    model = SequenceModel(
        [
            ModelResponse(
                text="不客气。",
                finish_reason="stop",
                usage=ModelUsage(input_tokens=12, output_tokens=3),
            )
        ]
    )
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        await coordinator(
            repository,
            model,
            rentflow_http,
            intent_pre_router_mode="enforce",
        )._execute(
            run_id="run-chat",
            conversation_id="conversation-chat",
            user_id="user-1",
            access_token="token",
            message="谢谢",
        )

    assert len(model.requests) == 1
    assert model.requests[0].tools == ()
    event = next(payload for name, payload in repository.events if name == "action.resolution")
    assert event == {
        "source": "deterministic",
        "rule": "pure_social",
        "actionModelCalled": False,
    }


async def test_shadow_mode_records_candidate_but_uses_llm_action() -> None:
    repository = FakeRepository("谢谢")
    model = SequenceModel(
        [
            action_response({"action": "chat"}),
            ModelResponse(text="不客气。", finish_reason="stop"),
        ]
    )
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        await coordinator(
            repository,
            model,
            rentflow_http,
            intent_pre_router_mode="shadow",
        )._execute(
            run_id="run-shadow",
            conversation_id="conversation-shadow",
            user_id="user-1",
            access_token="token",
            message="谢谢",
        )

    event = next(payload for name, payload in repository.events if name == "action.resolution")
    assert event["source"] == "llm_tool_call"
    assert event["candidateRule"] == "pure_social"
    assert event["matched"] is True


async def test_enforced_pre_router_miss_calls_action_model() -> None:
    message = "你能做什么"
    repository = FakeRepository(message)
    model = SequenceModel(
        [
            action_response({"action": "chat"}),
            ModelResponse(text="我可以帮你选购商品和查询订单。", finish_reason="stop"),
        ]
    )
    async with httpx.AsyncClient(base_url="http://localhost:8080") as rentflow_http:
        await coordinator(
            repository,
            model,
            rentflow_http,
            intent_pre_router_mode="enforce",
        )._execute(
            run_id="run-miss",
            conversation_id="conversation-miss",
            user_id="user-1",
            access_token="token",
            message=message,
        )

    assert len(model.requests) == 2
    assert model.requests[0].tools[0].name == "resolve_agent_action"


async def test_changed_product_category_does_not_inherit_pending_use_case() -> None:
    repository = FakeRepository("我想买手机")
    repository.state = ConversationStateMemory(
        pending_product_search=PendingProductSearch(
            equipment_role="laptop",
            use_case_id="01J00000000000000000000202",
        )
    )
    model = SequenceModel(
        [
            action_response(
                {
                    "action": "product_search",
                    "equipmentRole": "smartphone",
                    "continuesPending": True,
                }
            )
        ]
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/skus"):
            return httpx.Response(200, json=[])
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "productId": "01J00000000000000000000121",
                        "categoryId": "01J00000000000000000000010",
                        "equipmentRole": "smartphone",
                        "name": "iPhone 15 Pro Max",
                        "brand": "Apple",
                        "model": "iPhone 15 Pro Max",
                        "dailyRate": "0.00",
                        "fixedDeposit": "0.00",
                    }
                ],
                "page": 0,
                "size": 20,
                "totalElements": 1,
                "totalPages": 1,
            },
        )

    async with httpx.AsyncClient(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(handle),
    ) as rentflow_http:
        await coordinator(repository, model, rentflow_http)._execute(
            run_id="run-smartphone",
            conversation_id="conversation-smartphone",
            user_id="user-1",
            access_token="token",
            message="我想买手机",
        )

    assert requests[0].url.params["equipmentRole"] == "smartphone"
    assert "useCaseId" not in requests[0].url.params
    action = next(payload for name, payload in repository.events if name == "action.resolved")
    assert action["equipmentRole"] == "smartphone"
    assert action["useCaseId"] is None
    assert action["continuesPending"] is False


async def test_product_search_remembers_authoritative_result_positions() -> None:
    repository = FakeRepository("我想买相机")
    model = SequenceModel(
        [action_response({"action": "product_search", "equipmentRole": "camera"})]
    )

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/skus"):
            return httpx.Response(200, json=[])
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "productId": "01J00000000000000000000101",
                        "categoryId": "01J00000000000000000000001",
                        "equipmentRole": "camera",
                        "name": "Sony A7M4",
                        "brand": "Sony",
                        "model": "A7M4",
                        "dailyRate": "0.00",
                        "fixedDeposit": "0.00",
                    }
                ],
                "page": 0,
                "size": 20,
                "totalElements": 1,
                "totalPages": 1,
            },
        )

    async with httpx.AsyncClient(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(handle),
    ) as rentflow_http:
        await coordinator(repository, model, rentflow_http)._execute(
            run_id="run-search",
            conversation_id="conversation-search",
            user_id="user-1",
            access_token="token",
            message="我想买相机",
        )

    assert repository.recent_search is not None
    assert repository.recent_search.items[0].position == 1
    assert repository.recent_search.items[0].product_id == "01J00000000000000000000101"
