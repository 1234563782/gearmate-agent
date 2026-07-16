from gearmate.actions import (
    AgentAction,
    AgentActionResolver,
    PendingProductSearch,
    PendingRentalAction,
    merge_pending_product_search,
    merge_pending_rental_action,
)
from gearmate.catalog import CatalogAliasTerm, CatalogVocabulary
from gearmate.llm.types import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
)


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


async def test_resolver_returns_structured_product_search() -> None:
    model = FakeModel(
        {
            "action": "product_search",
            "keyword": "单反",
            "keywordSpecificity": "specific",
            "equipmentRole": "camera",
        }
    )

    result = await AgentActionResolver(("camera", "tripod")).resolve(
        message="有哪些相机可以租？",
        history=(),
        current_scenario_id=None,
        pending_product_search=None,
        pending_rental_action=None,
        model=model,
        max_output_tokens=128,
    )

    assert result.action is not None
    assert result.action.action == "product_search"
    assert result.action.keyword == "单反"
    assert result.action.keyword_specificity == "specific"
    assert result.action.equipment_role == "camera"
    assert model.requests[0].temperature == 0
    assert model.requests[0].enable_thinking is False
    assert model.requests[0].tools[0].name == "resolve_agent_action"
    equipment_role_schema = model.requests[0].tools[0].parameters["properties"]["equipmentRole"]
    assert equipment_role_schema["anyOf"][0]["enum"] == ["camera", "tripod"]
    assert model.requests[0].tool_choice == "resolve_agent_action"


async def test_resolver_keeps_thanks_as_chat_with_saved_scenario() -> None:
    model = FakeModel({"action": "chat"})

    result = await AgentActionResolver(("camera", "tripod")).resolve(
        message="谢谢",
        history=(),
        current_scenario_id="live_streaming",
        pending_product_search=None,
        pending_rental_action=None,
        model=model,
        max_output_tokens=128,
    )

    assert result.action is not None
    assert result.action.action == "chat"
    assert "must not turn thanks" in model.requests[0].messages[0].content


async def test_resolver_receives_database_catalog_aliases() -> None:
    model = FakeModel(
        {
            "action": "product_search",
            "equipmentRole": "laptop",
            "brand": "Apple",
        }
    )

    await AgentActionResolver(("laptop",)).resolve(
        message="苹果电脑",
        history=(),
        current_scenario_id=None,
        pending_product_search=None,
        pending_rental_action=None,
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
    assert '"alias": "苹果电脑"' in prompt
    assert '"canonicalValue": "Apple"' in prompt


async def test_resolver_maps_dynamic_use_case_alias_without_hardcoded_values() -> None:
    model = FakeModel(
        {
            "action": "product_search",
            "equipmentRole": "laptop",
            "semanticQuery": "适合做后期的电脑",
        }
    )

    result = await AgentActionResolver(("laptop",)).resolve(
        message="我想租一台做后期的电脑",
        history=(),
        current_scenario_id=None,
        pending_product_search=None,
        pending_rental_action=None,
        model=model,
        max_output_tokens=128,
        catalog_vocabulary=CatalogVocabulary(
            aliases=(
                CatalogAliasTerm(
                    "后期",
                    "use_case",
                    "01J00000000000000000000202",
                ),
            )
        ),
    )

    assert result.action is not None
    assert result.action.use_case_id == "01J00000000000000000000202"
    assert result.action.semantic_query == "适合做后期的电脑"


async def test_resolver_maps_recent_product_position_to_authoritative_id() -> None:
    model = FakeModel(
        {
            "action": "product_detail",
            "productPosition": 1,
        }
    )

    result = await AgentActionResolver(("laptop",)).resolve(
        message="看看第一个",
        history=(),
        current_scenario_id=None,
        pending_product_search=None,
        pending_rental_action=None,
        model=model,
        max_output_tokens=128,
        recent_product_search_json='{"items": [{"position": 1}]}',
        recent_product_ids=("01J00000000000000000000105",),
    )

    assert result.action is not None
    assert result.action.product_id == "01J00000000000000000000105"


async def test_resolver_rejects_missing_recent_product_position() -> None:
    model = FakeModel(
        {
            "action": "availability",
            "productPosition": 2,
        }
    )

    result = await AgentActionResolver(("laptop",)).resolve(
        message="第二个有货吗",
        history=(),
        current_scenario_id=None,
        pending_product_search=None,
        pending_rental_action=None,
        model=model,
        max_output_tokens=128,
        recent_product_ids=("01J00000000000000000000105",),
    )

    assert result.action is None
    assert result.clarification == "最近搜索结果中没有这个位置，请重新选择商品。"


def test_pending_search_merges_only_missing_followup_fields() -> None:
    pending = PendingProductSearch(
        keyword="单反",
        keyword_specificity="specific",
        equipment_role="camera",
        max_daily_rate="500",
        waiting_for_rental_period=True,
    )

    merged = pending.merge_into(
        AgentAction(
            action="product_search",
            max_daily_rate="600",
            continues_pending=True,
        )
    )

    assert merged.keyword == "单反"
    assert merged.keyword_specificity == "specific"
    assert merged.equipment_role == "camera"
    assert str(merged.max_daily_rate) == "600"
    assert merged.continues_pending is True


def test_new_search_does_not_inherit_pending_search_fields() -> None:
    pending = PendingProductSearch(
        keyword="单反",
        equipment_role="camera",
        waiting_for_rental_period=True,
    )
    new_search = AgentAction(
        action="product_search",
        keyword="麦克风",
        equipment_role="microphone",
    )

    resolved = merge_pending_product_search(new_search, pending)

    assert resolved == new_search


def test_pending_availability_keeps_selected_product_during_confirmation() -> None:
    pending = PendingRentalAction(
        action="availability",
        product_id="01J00000000000000000000101",
    )

    resolved = merge_pending_rental_action(
        AgentAction(action="chat", continues_pending=True),
        pending,
    )

    assert resolved.action == "availability"
    assert resolved.product_id == "01J00000000000000000000101"
