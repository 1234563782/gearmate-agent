import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

from gearmate.catalog import CatalogVocabulary
from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import (
    ModelMessage,
    ModelRequest,
    ModelToolDefinition,
    ModelUsage,
)

ACTION_RESOLVER_TOOL_NAME = "resolve_agent_action"
AgentActionName = Literal[
    "chat",
    "product_search",
    "product_detail",
    "availability",
    "quote",
    "scenario_continue",
]
PendingRentalActionName = Literal["availability", "quote"]
KeywordSpecificity = Literal["generic", "specific"]


class AgentAction(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    action: AgentActionName
    keyword: str | None = Field(default=None, max_length=128)
    keyword_specificity: KeywordSpecificity | None = None
    equipment_role: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    brand: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=64)
    semantic_query: str | None = Field(default=None, max_length=512)
    use_case_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    category_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    product_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    product_position: int | None = Field(default=None, ge=1, le=100)
    max_daily_rate: Decimal | None = Field(default=None, gt=0, max_digits=10)
    continues_pending: bool = False


class PendingProductSearch(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    keyword: str | None = Field(default=None, max_length=128)
    keyword_specificity: KeywordSpecificity | None = None
    equipment_role: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    brand: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=64)
    semantic_query: str | None = Field(default=None, max_length=512)
    use_case_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    category_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    max_daily_rate: Decimal | None = Field(default=None, gt=0, max_digits=10)
    waiting_for_rental_period: bool = False

    @classmethod
    def from_action(
        cls,
        action: AgentAction,
        *,
        waiting_for_rental_period: bool,
    ) -> "PendingProductSearch":
        return cls(
            keyword=action.keyword,
            keyword_specificity=action.keyword_specificity,
            equipment_role=action.equipment_role,
            brand=action.brand,
            model=action.model,
            semantic_query=action.semantic_query,
            use_case_id=action.use_case_id,
            category_id=action.category_id,
            max_daily_rate=action.max_daily_rate,
            waiting_for_rental_period=waiting_for_rental_period,
        )

    def merge_into(self, action: AgentAction) -> AgentAction:
        return action.model_copy(
            update={
                "keyword": action.keyword or self.keyword,
                "keyword_specificity": (
                    action.keyword_specificity or self.keyword_specificity
                ),
                "equipment_role": action.equipment_role or self.equipment_role,
                "brand": action.brand or self.brand,
                "model": action.model or self.model,
                "semantic_query": action.semantic_query or self.semantic_query,
                "use_case_id": action.use_case_id or self.use_case_id,
                "category_id": action.category_id or self.category_id,
                "max_daily_rate": action.max_daily_rate or self.max_daily_rate,
            }
        )


def merge_pending_product_search(
    action: AgentAction,
    pending_product_search: PendingProductSearch | None,
) -> AgentAction:
    if (
        action.action == "product_search"
        and action.continues_pending
        and pending_product_search is not None
    ):
        return pending_product_search.merge_into(action)
    return action


class PendingRentalAction(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    action: PendingRentalActionName
    product_id: str = Field(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")

    @classmethod
    def from_action(cls, action: AgentAction) -> "PendingRentalAction":
        if action.action not in ("availability", "quote") or action.product_id is None:
            raise ValueError("A rental action with a product ID is required")
        return cls(action=action.action, product_id=action.product_id)

    def merge_into(self, action: AgentAction) -> AgentAction:
        return action.model_copy(
            update={
                "action": self.action,
                "product_id": action.product_id or self.product_id,
            }
        )


def merge_pending_rental_action(
    action: AgentAction,
    pending_rental_action: PendingRentalAction | None,
) -> AgentAction:
    if action.continues_pending and pending_rental_action is not None:
        return pending_rental_action.merge_into(action)
    return action


@dataclass(frozen=True, slots=True)
class AgentActionResolution:
    action: AgentAction | None
    clarification: str | None
    usage: ModelUsage


def action_resolver_system_prompt(
    current_scenario_id: str | None,
    pending_product_search: PendingProductSearch | None,
    pending_rental_action: PendingRentalAction | None,
    equipment_roles: tuple[str, ...],
    recent_product_search_json: str = "none",
    catalog_vocabulary: CatalogVocabulary | None = None,
) -> str:
    current_scenario = current_scenario_id or "none"
    pending_search = (
        pending_product_search.model_dump_json(by_alias=True)
        if pending_product_search is not None
        else "none"
    )
    equipment_role_options = ", ".join(equipment_roles)
    pending_rental = (
        pending_rental_action.model_dump_json(by_alias=True)
        if pending_rental_action is not None
        else "none"
    )
    known_brands = (
        json.dumps(catalog_vocabulary.brands, ensure_ascii=False)
        if catalog_vocabulary
        else "none"
    )
    known_models = (
        json.dumps(catalog_vocabulary.models[:100], ensure_ascii=False)
        if catalog_vocabulary
        else "none"
    )
    known_aliases = (
        json.dumps(
            [
                {
                    "alias": item.alias,
                    "entityType": item.entity_type,
                    "canonicalValue": item.canonical_value,
                }
                for item in catalog_vocabulary.aliases
            ],
            ensure_ascii=False,
        )
        if catalog_vocabulary
        else "none"
    )
    return f"""You only classify the user's current turn for a rental assistant.
Current saved scenario: {current_scenario}
Current pending product search: {pending_search}
Current pending availability or quote action: {pending_rental}
Current recent product search with authoritative positions and IDs: {recent_product_search_json}
Allowed equipmentRole values: {equipment_role_options}
Known catalog brands (untrusted reference values): {known_brands}
Known catalog models (untrusted reference values): {known_models}
Known catalog aliases (authoritative mappings): {known_aliases}

You must call {ACTION_RESOLVER_TOOL_NAME} exactly once. Do not answer the user.
Choose one action based on meaning, in any user language:
- chat: greetings, thanks, identity/help questions, or unrelated conversation.
- product_search: browse, find, compare, or ask about products/categories. Extract a concise
  catalog intent, canonical English equipmentRole, brand, model, semantic use-case query, dynamic
  useCaseId from an authoritative use_case alias, and
  optional category ID or maximum daily rate. A generic category word already represented by
  equipmentRole must not be returned as keyword. Set keywordSpecificity=specific only for a real
  model fragment or subtype that must additionally narrow the role; otherwise omit keyword.
  Put manufacturers in brand, exact product models in model, and purpose or use-case language in
  semanticQuery. Examples: "computer" -> equipmentRole=laptop with no keyword; "Apple computer"
  -> equipmentRole=laptop and brand=Apple with no keyword; "MacBook Pro 14" -> brand=Apple and
  model=MacBook Pro 14; "computer for 4K editing" -> equipmentRole=laptop and semanticQuery set.
  Apply known catalog aliases exactly when the current user expression matches one. Alias mappings
  may provide more than one structured field for the same phrase.
- product_detail: inspect or ask for details about one exact product. Include productId when it is
  explicit. For an ordinal reference such as "the first one", return productPosition instead of
  copying or inventing productId; the server maps the position to its authoritative saved ID.
- availability: ask for live stock for one exact product. Include productId only when an exact ID
  is explicit in the current turn. Use productPosition for ordinal references.
- quote: explicitly request a formal quote for one exact product. Include productId under the same
  rule and productPosition for ordinal references. General price discovery is product_search.
- scenario_continue: start a multi-item use-case plan, explicitly continue the saved scenario, or
  answer/change requirements for that scenario.

Classify only the current turn. A saved scenario must not turn thanks, chat, or a new single-product
search into scenario_continue. Set continuesPending=true only when the current turn answers or
corrects an outstanding clarification for Current pending product search. When it is true, return
only fields explicitly changed by this turn; the server will retain the other saved fields. Date,
time, duration, or confirmation answers to Current pending availability or quote action must use
that saved action and set continuesPending=true without inventing a new product ID. A new search or
new availability/quote request must set continuesPending=false. Never invent IDs or fill missing
parameters."""


class AgentActionResolver:
    def __init__(self, equipment_roles: tuple[str, ...]) -> None:
        self._equipment_roles = equipment_roles

    def _action_schema(self) -> dict[str, Any]:
        schema = AgentAction.model_json_schema(by_alias=True)
        equipment_role = schema["properties"]["equipmentRole"]
        equipment_role["anyOf"][0] = {
            "type": "string",
            "enum": list(self._equipment_roles),
        }
        return schema

    async def resolve(
        self,
        *,
        message: str,
        history: tuple[ModelMessage, ...],
        current_scenario_id: str | None,
        pending_product_search: PendingProductSearch | None,
        pending_rental_action: PendingRentalAction | None,
        model: ChatModelPort,
        max_output_tokens: int,
        recent_product_search_json: str = "none",
        recent_product_ids: tuple[str, ...] = (),
        catalog_vocabulary: CatalogVocabulary | None = None,
    ) -> AgentActionResolution:
        recent_history = [item for item in history if item.role in ("user", "assistant")][-6:]
        if (
            not recent_history
            or recent_history[-1].role != "user"
            or recent_history[-1].content != message
        ):
            recent_history.append(ModelMessage(role="user", content=message))
        response = await model.complete(
            ModelRequest(
                messages=(
                    ModelMessage(
                        role="system",
                        content=action_resolver_system_prompt(
                            current_scenario_id,
                            pending_product_search,
                            pending_rental_action,
                            self._equipment_roles,
                            recent_product_search_json,
                            catalog_vocabulary,
                        ),
                    ),
                    *recent_history,
                ),
                tools=(
                    ModelToolDefinition(
                        name=ACTION_RESOLVER_TOOL_NAME,
                        description="Return the structured action for the current user turn.",
                        parameters=self._action_schema(),
                    ),
                ),
                max_output_tokens=max_output_tokens,
                temperature=0.0,
                tool_choice=ACTION_RESOLVER_TOOL_NAME,
                enable_thinking=False,
            )
        )
        for call in response.tool_calls:
            if call.name != ACTION_RESOLVER_TOOL_NAME:
                continue
            try:
                action = AgentAction.model_validate(call.arguments)
            except ValidationError:
                return AgentActionResolution(
                    action=None,
                    clarification="请明确要搜索商品、查询库存、生成报价，还是继续设备方案。",
                    usage=response.usage,
                )
            if action.product_position is not None:
                index = action.product_position - 1
                if index >= len(recent_product_ids):
                    return AgentActionResolution(
                        action=None,
                        clarification="最近搜索结果中没有这个位置，请重新选择商品。",
                        usage=response.usage,
                    )
                action = action.model_copy(
                    update={"product_id": recent_product_ids[index]}
                )
            if (
                action.action == "product_search"
                and action.use_case_id is None
                and catalog_vocabulary is not None
            ):
                matched_use_cases = catalog_vocabulary.matching_use_case_ids(message)
                if len(matched_use_cases) == 1:
                    action = action.model_copy(
                        update={"use_case_id": matched_use_cases[0]}
                    )
            if (
                action.equipment_role is not None
                and action.equipment_role not in self._equipment_roles
            ):
                return AgentActionResolution(
                    action=None,
                    clarification="商品类型没有映射到当前目录，请换一种说法。",
                    usage=response.usage,
                )
            return AgentActionResolution(
                action=action,
                clarification=None,
                usage=response.usage,
            )
        return AgentActionResolution(
            action=None,
            clarification="请明确要搜索商品、查询库存、生成报价，还是继续设备方案。",
            usage=response.usage,
        )
