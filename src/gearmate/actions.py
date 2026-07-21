import json
import re
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
from gearmate.tools.contracts import StoreOrderStatus

ACTION_RESOLVER_TOOL_NAME = "resolve_agent_action"
PRICE_AMOUNT = r"(?P<amount>\d+(?:\.\d{1,2})?)"
HARD_MAX_PRICE = re.compile(
    rf"(?:不超过|不高于|最高|最多|以内|以下|预算(?:上限)?|at\s+most|up\s+to|under)"
    rf"[^\d]{{0,12}}{PRICE_AMOUNT}",
    re.IGNORECASE,
)
APPROXIMATE_PRICE_PREFIX = re.compile(
    rf"(?:大约|大概|约|接近|around|about)[^\d]{{0,8}}{PRICE_AMOUNT}",
    re.IGNORECASE,
)
APPROXIMATE_PRICE_SUFFIX = re.compile(
    rf"{PRICE_AMOUNT}\s*(?:元)?\s*(?:左右|上下|附近)",
    re.IGNORECASE,
)
CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
COMMERCE_ACTION_NAMES = (
    "chat",
    "product_search",
    "product_detail",
    "order_list",
    "order_detail",
    "sku_stock",
    "purchase_prepare",
)
AgentActionName = Literal[
    "chat",
    "product_search",
    "product_detail",
    "order_list",
    "order_detail",
    "sku_stock",
    "purchase_prepare",
]
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
    max_price: Decimal | None = Field(default=None, gt=0, max_digits=12)
    target_price: Decimal | None = Field(default=None, gt=0, max_digits=12)
    sku_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    quantity: int | None = Field(default=None, ge=1, le=99)
    order_id: str | None = Field(default=None, pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")
    order_status: StoreOrderStatus | None = None
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
    max_price: Decimal | None = Field(default=None, gt=0, max_digits=12)
    target_price: Decimal | None = Field(default=None, gt=0, max_digits=12)

    @classmethod
    def from_action(cls, action: AgentAction) -> "PendingProductSearch":
        return cls(
            keyword=action.keyword,
            keyword_specificity=action.keyword_specificity,
            equipment_role=action.equipment_role,
            brand=action.brand,
            model=action.model,
            semantic_query=action.semantic_query,
            use_case_id=action.use_case_id,
            category_id=action.category_id,
            max_price=action.max_price,
            target_price=action.target_price,
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
                "max_price": action.max_price or self.max_price,
                "target_price": action.target_price or self.target_price,
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
        if (
            action.equipment_role is not None
            and pending_product_search.equipment_role is not None
            and action.equipment_role != pending_product_search.equipment_role
        ):
            return action.model_copy(update={"continues_pending": False})
        return pending_product_search.merge_into(action)
    return action


@dataclass(frozen=True, slots=True)
class AgentActionResolution:
    action: AgentAction | None
    clarification: str | None
    usage: ModelUsage


def normalize_price_intent(message: str, action: AgentAction) -> AgentAction:
    if action.action != "product_search":
        return action
    hard_max = HARD_MAX_PRICE.search(message)
    if hard_max is not None:
        return action.model_copy(
            update={
                "max_price": Decimal(hard_max.group("amount")),
                "target_price": None,
            }
        )
    preferred = (
        APPROXIMATE_PRICE_PREFIX.search(message)
        or APPROXIMATE_PRICE_SUFFIX.search(message)
    )
    if preferred is None:
        return action.model_copy(
            update={
                "max_price": None,
                "target_price": None,
            }
        )
    return action.model_copy(
        update={
            "max_price": None,
            "target_price": Decimal(preferred.group("amount")),
        }
    )


def preserve_semantic_query_language(message: str, action: AgentAction) -> AgentAction:
    if action.action != "product_search" or not action.semantic_query:
        return action
    if CJK_CHARACTER.search(message) and not CJK_CHARACTER.search(action.semantic_query):
        return action.model_copy(update={"semantic_query": message.strip()[:512]})
    return action


def action_resolver_system_prompt(
    pending_product_search: PendingProductSearch | None,
    equipment_roles: tuple[str, ...],
    recent_product_search_json: str = "none",
    catalog_vocabulary: CatalogVocabulary | None = None,
    user_memory_context: str = "none",
) -> str:
    pending_search = (
        pending_product_search.model_dump_json(by_alias=True)
        if pending_product_search is not None
        else "none"
    )
    equipment_role_options = ", ".join(equipment_roles)
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
    return f"""You only classify the user's current turn for an electronics commerce assistant.
Current pending product search: {pending_search}
Current recent product search with authoritative positions and IDs: {recent_product_search_json}
User long-term preferences (soft context only): {user_memory_context}
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
  optional category ID or purchase-price constraint. Use maxPrice only for a hard upper limit such
  as "under 5000" or "at most 5000". Use targetPrice for a desired/approximate purchase price such
  as "around 5000"; do not turn an approximate
  target into a hard maximum. A generic category word already represented by
  equipmentRole must not be returned as keyword. Set keywordSpecificity=specific only for a real
  model fragment or subtype that must additionally narrow the role; otherwise omit keyword.
  Put manufacturers in brand, exact product models in model, and purpose or use-case language in
  semanticQuery. semanticQuery must preserve the user's current language and original domain terms;
  never translate it into another language. Examples: "computer" -> equipmentRole=laptop with no
  keyword; "Apple computer" -> equipmentRole=laptop and brand=Apple with no keyword;
  "MacBook Pro 14" -> brand=Apple and
  model=MacBook Pro 14; "computer for 4K editing" -> equipmentRole=laptop and semanticQuery set.
  Apply known catalog aliases exactly when the current user expression matches one. Alias mappings
  may provide more than one structured field for the same phrase.
- product_detail: inspect or ask for details about one exact product. Include productId when it is
  explicit. For an ordinal reference such as "the first one", return productPosition instead of
  copying or inventing productId; the server maps the position to its authoritative saved ID.
- sku_stock: ask for current purchasable SKU stock for one exact product. Include productId only
  when an exact ID is explicit in the current turn. Use productPosition for ordinal references.
- purchase_prepare: the user wants to buy one exact product or asks to prepare checkout. Include
  productId or skuId only when authoritative, and quantity only when explicit. This action prepares
  SKU and quantity choices but never places or pays an order.
- order_list: view or list the current signed-in user's orders. Use orderStatus only when the user
  explicitly asks for pending payment, paid, shipped, received, cancelled, or closed orders.
  Never request or invent a user ID, and never expose internal order IDs.
- order_detail: inspect one exact commerce order. Include orderId only when supplied by a trusted
  structured client context; never copy or invent an ID from assistant-visible prose.

Classify only the current turn. Set continuesPending=true only when the current turn answers or
corrects an outstanding clarification for Current pending product search. When it is true, return
only fields explicitly changed by this turn; the server will retain the other saved fields. A new
search, stock/purchase request, or order query must set continuesPending=false. Long-term
preferences must not populate current-turn brand, model, equipmentRole, useCaseId, or price fields
unless the current user message explicitly states them. Never invent IDs or fill missing
parameters."""


class AgentActionResolver:
    def __init__(self, equipment_roles: tuple[str, ...]) -> None:
        self._equipment_roles = equipment_roles

    def _action_schema(self, equipment_roles: tuple[str, ...]) -> dict[str, Any]:
        schema = AgentAction.model_json_schema(by_alias=True)
        schema["properties"]["action"] = {
            "type": "string",
            "enum": list(COMMERCE_ACTION_NAMES),
        }
        equipment_role = schema["properties"]["equipmentRole"]
        equipment_role["anyOf"][0] = {
            "type": "string",
            "enum": list(equipment_roles),
        }
        return schema

    async def resolve(
        self,
        *,
        message: str,
        history: tuple[ModelMessage, ...],
        pending_product_search: PendingProductSearch | None,
        model: ChatModelPort,
        max_output_tokens: int,
        recent_product_search_json: str = "none",
        recent_product_ids: tuple[str, ...] = (),
        catalog_vocabulary: CatalogVocabulary | None = None,
        user_memory_context: str = "none",
    ) -> AgentActionResolution:
        equipment_roles = tuple(
            dict.fromkeys(
                (
                    *self._equipment_roles,
                    *(catalog_vocabulary.equipment_roles if catalog_vocabulary else ()),
                )
            )
        )
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
                            pending_product_search,
                            equipment_roles,
                            recent_product_search_json,
                            catalog_vocabulary,
                            user_memory_context,
                        ),
                    ),
                    *recent_history,
                ),
                tools=(
                    ModelToolDefinition(
                        name=ACTION_RESOLVER_TOOL_NAME,
                        description="Return the structured action for the current user turn.",
                        parameters=self._action_schema(equipment_roles),
                    ),
                ),
                max_output_tokens=max_output_tokens,
                temperature=0.0,
                tool_choice=ACTION_RESOLVER_TOOL_NAME,
                enable_thinking=False,
                workload="action",
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
                    clarification="请明确要搜索商品、查看商品或库存、准备购买，还是查询订单。",
                    usage=response.usage,
                )
            action = normalize_price_intent(message, action)
            action = preserve_semantic_query_language(message, action)
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
                and action.equipment_role not in equipment_roles
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
            clarification="请明确要搜索商品、查看商品或库存、准备购买，还是查询订单。",
            usage=response.usage,
        )
