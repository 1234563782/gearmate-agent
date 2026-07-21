from gearmate.actions import (
    AgentAction,
    AgentActionResolver,
    PendingProductSearch,
    merge_pending_product_search,
    normalize_price_intent,
    preserve_semantic_query_language,
)
from gearmate.catalog import CatalogAliasTerm, CatalogVocabulary
from gearmate.llm.types import ModelRequest, ModelResponse, ModelToolCall, ModelUsage


class FakeModel:
    def __init__(self, arguments: dict[str, object]) -> None:
        self.arguments = arguments
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text="",
            finish_reason="tool_calls",
            usage=ModelUsage(input_tokens=20, output_tokens=5),
            tool_calls=(
                ModelToolCall(
                    id="action-1",
                    name="resolve_agent_action",
                    arguments=self.arguments,
                ),
            ),
        )

    async def close(self) -> None:
        return None


async def resolve(model: FakeModel, message: str = "我想买电脑"):
    return await AgentActionResolver(("camera", "laptop", "tripod")).resolve(
        message=message,
        history=(),
        pending_product_search=None,
        model=model,
        max_output_tokens=128,
    )


async def test_resolver_returns_structured_product_search() -> None:
    model = FakeModel(
        {
            "action": "product_search",
            "keyword": "单反",
            "keywordSpecificity": "specific",
            "equipmentRole": "camera",
        }
    )

    result = await resolve(model, "有哪些单反相机？")

    assert result.action is not None
    assert result.action.action == "product_search"
    assert result.action.keyword == "单反"
    assert result.action.equipment_role == "camera"
    request = model.requests[0]
    assert request.temperature == 0
    assert request.enable_thinking is False
    assert request.tool_choice == "resolve_agent_action"


async def test_action_tool_schema_exposes_only_commerce_actions_and_prices() -> None:
    model = FakeModel({"action": "chat"})

    await resolve(model, "你好")

    properties = model.requests[0].tools[0].parameters["properties"]
    assert properties["action"]["enum"] == [
        "chat",
        "product_search",
        "product_detail",
        "order_list",
        "order_detail",
        "sku_stock",
        "purchase_prepare",
    ]
    assert "maxPrice" in properties
    assert "targetPrice" in properties
    assert "maxDailyRate" not in properties
    assert "targetDailyRate" not in properties


async def test_resolver_rejects_legacy_rental_action() -> None:
    result = await resolve(FakeModel({"action": "quote"}), "给我报价")

    assert result.action is None
    assert result.clarification is not None


def test_price_intent_distinguishes_target_and_upper_limit() -> None:
    approximate = normalize_price_intent(
        "我想买 5000 元左右的电脑",
        AgentAction(action="product_search", max_price="5000"),
    )
    hard_limit = normalize_price_intent(
        "预算不超过 5000 元",
        AgentAction(action="product_search", target_price="5000"),
    )

    assert approximate.target_price == 5000
    assert approximate.max_price is None
    assert hard_limit.max_price == 5000
    assert hard_limit.target_price is None


def test_new_search_without_price_drops_model_carried_price() -> None:
    action = normalize_price_intent(
        "我想买电脑",
        AgentAction(action="product_search", target_price="5000"),
    )

    assert action.max_price is None
    assert action.target_price is None


def test_semantic_query_keeps_user_language_when_model_translates() -> None:
    action = preserve_semantic_query_language(
        "我想拍旅行 vlog，想要轻便、手持稳定的设备",
        AgentAction(
            action="product_search",
            semantic_query="travel vlog, lightweight, handheld stable",
        ),
    )

    assert action.semantic_query == "我想拍旅行 vlog，想要轻便、手持稳定的设备"


async def test_resolver_returns_store_order_status() -> None:
    result = await resolve(
        FakeModel({"action": "order_list", "orderStatus": "SHIPPED"}),
        "查看待收货订单",
    )

    assert result.action is not None
    assert result.action.order_status == "SHIPPED"


async def test_resolver_maps_recent_product_position_to_authoritative_id() -> None:
    model = FakeModel({"action": "purchase_prepare", "productPosition": 1, "quantity": 2})
    result = await AgentActionResolver(("camera",)).resolve(
        message="第一个买两件",
        history=(),
        pending_product_search=None,
        model=model,
        max_output_tokens=128,
        recent_product_ids=("01J00000000000000000000101",),
    )

    assert result.action is not None
    assert result.action.product_id == "01J00000000000000000000101"
    assert result.action.quantity == 2


async def test_resolver_rejects_missing_recent_product_position() -> None:
    model = FakeModel({"action": "sku_stock", "productPosition": 2})
    result = await AgentActionResolver(("camera",)).resolve(
        message="第二个有货吗",
        history=(),
        pending_product_search=None,
        model=model,
        max_output_tokens=128,
        recent_product_ids=("01J00000000000000000000101",),
    )

    assert result.action is None
    assert result.clarification is not None


async def test_resolver_uses_dynamic_catalog_aliases() -> None:
    model = FakeModel(
        {"action": "product_search", "equipmentRole": "laptop", "brand": "Apple"}
    )
    await AgentActionResolver(("laptop",)).resolve(
        message="苹果电脑",
        history=(),
        pending_product_search=None,
        model=model,
        max_output_tokens=128,
        catalog_vocabulary=CatalogVocabulary(
            aliases=(
                CatalogAliasTerm("苹果电脑", "equipment_role", "laptop"),
                CatalogAliasTerm("苹果电脑", "brand", "Apple"),
            )
        ),
    )

    prompt = model.requests[0].messages[0].content
    assert '"canonicalValue": "Apple"' in prompt


def test_pending_search_merges_only_missing_followup_fields() -> None:
    pending = PendingProductSearch(equipment_role="laptop", max_price="8000")
    merged = pending.merge_into(
        AgentAction(action="product_search", target_price="7000", continues_pending=True)
    )

    assert merged.equipment_role == "laptop"
    assert merged.max_price == 8000
    assert merged.target_price == 7000


def test_changed_equipment_role_does_not_inherit_pending_search() -> None:
    pending = PendingProductSearch(
        equipment_role="laptop",
        use_case_id="01J00000000000000000000202",
    )
    new_search = AgentAction(
        action="product_search",
        equipment_role="smartphone",
        continues_pending=True,
    )

    resolved = merge_pending_product_search(new_search, pending)

    assert resolved.equipment_role == "smartphone"
    assert resolved.use_case_id is None
    assert resolved.continues_pending is False
