import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, cast

from gearmate.actions import AgentAction
from gearmate.agent.state import AgentState
from gearmate.config import Settings
from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import ModelMessage, ModelRequest, ModelToolCall
from gearmate.prompts.loader import RenderedPrompt
from gearmate.responses import UserResponseComposer
from gearmate.search import ProductSearchPlanner
from gearmate.tools.registry import ToolRegistry
from gearmate.validation.facts import FactSnapshot

EventWriter = Callable[[str, dict[str, Any]], Awaitable[None]]
AUTOMATIC_ACTION_CALL_ID = "automatic-action"


@dataclass(frozen=True, slots=True)
class AgentResult:
    text: str
    stop_reason: str
    error_code: str | None
    model_rounds: int
    tool_call_count: int
    input_tokens: int
    output_tokens: int


class GearMateAgent:
    def __init__(
        self,
        model: ChatModelPort,
        tools: ToolRegistry,
        settings: Settings,
        prompt: RenderedPrompt,
    ) -> None:
        self._model = model
        self._tools = tools
        self._settings = settings
        self._prompt = prompt

    async def run(
        self,
        *,
        message: str,
        history: list[ModelMessage],
        action: AgentAction,
        write_event: EventWriter,
        user_memory_context: str | None = None,
    ) -> AgentResult:
        facts = FactSnapshot()
        if action.max_price is not None:
            facts.add_constraint_amount(action.max_price)
        if action.target_price is not None:
            facts.add_constraint_amount(action.target_price)
        if action.quantity is not None:
            facts.add_constraint_count(action.quantity)
        response_composer = UserResponseComposer()

        def grounded_response() -> str:
            return response_composer.compose(action=action, facts=facts)

        messages = [ModelMessage(role="system", content=self._prompt.content)]
        if user_memory_context:
            messages.append(ModelMessage(role="system", content=user_memory_context))
        messages.extend(history)
        if not history or history[-1].role != "user" or history[-1].content != message:
            messages.append(ModelMessage(role="user", content=message))

        automatic_tool_calls: list[ModelToolCall] = []
        if action.action == "product_search":
            plan = ProductSearchPlanner().plan(action)
            arguments: dict[str, Any] = {}
            if plan.keyword:
                arguments["keyword"] = plan.keyword
            if plan.equipment_role:
                arguments["equipmentRole"] = plan.equipment_role
            if plan.brand:
                arguments["brand"] = plan.brand
            if plan.model:
                arguments["model"] = plan.model
            if plan.semantic_query:
                arguments["semanticQuery"] = plan.semantic_query
            if plan.use_case_id:
                arguments["useCaseId"] = plan.use_case_id
            if plan.category_id:
                arguments["categoryId"] = plan.category_id
            if plan.max_price is not None:
                arguments["maxPrice"] = str(plan.max_price)
            if plan.target_price is not None:
                arguments["targetPrice"] = str(plan.target_price)
            automatic_tool_calls.append(
                ModelToolCall(
                    id=AUTOMATIC_ACTION_CALL_ID,
                    name="search_products",
                    arguments=arguments,
                )
            )
        elif action.action == "product_detail" and action.product_id is not None:
            automatic_tool_calls.extend(
                (
                    ModelToolCall(
                        id=AUTOMATIC_ACTION_CALL_ID,
                        name="get_product",
                        arguments={"productId": action.product_id},
                    ),
                    ModelToolCall(
                        id="automatic-product-skus",
                        name="list_product_skus",
                        arguments={"productId": action.product_id},
                    ),
                )
            )
        elif action.action in ("sku_stock", "purchase_prepare"):
            if action.sku_id is not None:
                automatic_tool_calls.append(
                    ModelToolCall(
                        id=AUTOMATIC_ACTION_CALL_ID,
                        name="get_store_sku",
                        arguments={"skuId": action.sku_id},
                    )
                )
            elif action.product_id is not None:
                automatic_tool_calls.extend(
                    (
                        ModelToolCall(
                            id="automatic-purchase-product",
                            name="get_product",
                            arguments={"productId": action.product_id},
                        ),
                        ModelToolCall(
                            id=AUTOMATIC_ACTION_CALL_ID,
                            name="list_product_skus",
                            arguments={"productId": action.product_id},
                        ),
                    )
                )
        elif action.action == "order_list":
            arguments = {
                "page": 0,
                "size": max(1, min(5, self._settings.max_tool_result_items)),
            }
            if action.order_status is not None:
                arguments["status"] = action.order_status
            automatic_tool_calls.append(
                ModelToolCall(
                    id=AUTOMATIC_ACTION_CALL_ID,
                    name="list_store_orders",
                    arguments=arguments,
                )
            )
        elif action.action == "order_detail" and action.order_id is not None:
            automatic_tool_calls.append(
                ModelToolCall(
                    id=AUTOMATIC_ACTION_CALL_ID,
                    name="get_store_order",
                    arguments={"orderId": action.order_id},
                )
            )
        if automatic_tool_calls and not any(item.tool_calls for item in messages[-1:]):
            messages.append(
                ModelMessage(
                    role="assistant",
                    content="",
                    tool_calls=tuple(automatic_tool_calls),
                )
            )

        async def preprocess(state: AgentState) -> dict[str, Any]:
            if action.action == "product_detail" and action.product_id is None:
                text = "请先指定一款商品，或告诉我是最近搜索结果中的第几个。"
                await write_event(
                    "decision.made",
                    {"outcome": "NEED_CLARIFICATION", "field": "productId"},
                )
                return {"final_text": text, "stop_reason": "NEED_CLARIFICATION"}
            if (
                action.action in ("sku_stock", "purchase_prepare")
                and action.product_id is None
                and action.sku_id is None
            ):
                text = "请先指定一款商品，例如点击最近推荐的商品卡片，或告诉我是第几个。"
                await write_event(
                    "decision.made",
                    {"outcome": "NEED_CLARIFICATION", "field": "productId"},
                )
                return {"final_text": text, "stop_reason": "NEED_CLARIFICATION"}
            if action.action == "order_detail" and action.order_id is None:
                text = "请从订单列表中选择一笔订单查看详情。"
                await write_event(
                    "decision.made",
                    {"outcome": "NEED_CLARIFICATION", "field": "orderId"},
                )
                return {"final_text": text, "stop_reason": "NEED_CLARIFICATION"}
            if state["pending_tool_calls"]:
                await write_event(
                    "decision.made",
                    {
                        "outcome": "RUN_TOOLS",
                        "action": action.action,
                        "tools": [call.name for call in state["pending_tool_calls"]],
                    },
                )
                return {}
            await write_event("decision.made", {"outcome": "RUN_AGENT"})
            return {}

        async def call_model(state: AgentState) -> dict[str, Any]:
            if state["model_rounds"] >= self._settings.max_model_rounds:
                return {
                    "final_text": grounded_response(),
                    "stop_reason": "MAX_MODEL_ROUNDS",
                }
            round_no = state["model_rounds"] + 1
            await write_event("model.started", {"round": round_no})
            started = monotonic()
            request = ModelRequest(
                messages=tuple(state["messages"]),
                tools=(() if action.action == "chat" else self._tools.model_definitions()),
                max_output_tokens=self._settings.model_max_output_tokens,
                workload="main",
            )
            async with asyncio.timeout(self._settings.model_request_timeout_seconds):
                response = await self._model.complete(request)
            await write_event(
                "model.completed",
                {
                    "round": round_no,
                    "finishReason": response.finish_reason,
                    "inputTokens": response.usage.input_tokens,
                    "outputTokens": response.usage.output_tokens,
                    "toolCallCount": len(response.tool_calls),
                    "durationMs": round((monotonic() - started) * 1000),
                },
            )
            assistant = ModelMessage(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
            )
            return {
                "messages": [*state["messages"], assistant],
                "pending_tool_calls": list(response.tool_calls),
                "final_text": response.text if not response.tool_calls else None,
                "model_rounds": round_no,
                "input_tokens": state["input_tokens"] + response.usage.input_tokens,
                "output_tokens": state["output_tokens"] + response.usage.output_tokens,
            }

        async def execute_tools(state: AgentState) -> dict[str, Any]:
            pending = state["pending_tool_calls"]
            next_count = state["tool_call_count"] + len(pending)
            if next_count > self._settings.max_tool_calls:
                return {
                    "pending_tool_calls": [],
                    "final_text": grounded_response(),
                    "stop_reason": "MAX_TOOL_CALLS",
                    "tool_call_count": state["tool_call_count"],
                }
            results = await self._tools.execute_all(pending, facts, write_event)
            tool_messages = [
                ModelMessage(
                    role="tool",
                    name=result.call.name,
                    tool_call_id=result.call.id,
                    content=result.content,
                )
                for result in results
            ]
            update: dict[str, Any] = {
                "messages": [*state["messages"], *tool_messages],
                "pending_tool_calls": [],
                "tool_call_count": next_count,
            }
            if pending and all(call.id.startswith("automatic-") for call in pending):
                update["final_text"] = grounded_response()
            return update

        async def validate_output(state: AgentState) -> dict[str, Any]:
            text = (state["final_text"] or "").strip() or grounded_response()
            validation = facts.validate(text)
            await write_event(
                "output.validated",
                {
                    "valid": validation.valid,
                    "unsupportedIds": list(validation.unsupported_ids),
                    "unsupportedAmounts": list(validation.unsupported_amounts),
                    "unsupportedCounts": list(validation.unsupported_counts),
                    "mismatchedProductIds": list(validation.mismatched_product_ids),
                    "missingFactCitation": validation.missing_fact_citation,
                },
            )
            if not validation.valid:
                return {
                    "final_text": grounded_response(),
                    "stop_reason": "FACT_VALIDATION_FALLBACK",
                }
            return {"final_text": text, "stop_reason": state["stop_reason"] or "COMPLETED"}

        async def finalize(state: AgentState) -> dict[str, Any]:
            text = state["final_text"] or grounded_response()
            await write_event("assistant.delta", {"content": text})
            await write_event(
                "assistant.completed",
                {"content": text, "stopReason": state["stop_reason"] or "COMPLETED"},
            )
            return {"final_text": text, "stop_reason": state["stop_reason"] or "COMPLETED"}

        state: AgentState = {
            "messages": messages,
            "pending_tool_calls": automatic_tool_calls,
            "final_text": None,
            "stop_reason": None,
            "error_code": None,
            "model_rounds": 0,
            "tool_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        step = "preprocess"
        while True:
            if step == "preprocess":
                state.update(cast(AgentState, await preprocess(state)))
                step = (
                    "finalize"
                    if state["final_text"]
                    else ("tools" if state["pending_tool_calls"] else "model")
                )
                continue
            if step == "model":
                state.update(cast(AgentState, await call_model(state)))
                step = "tools" if state["pending_tool_calls"] else "validate"
                continue
            if step == "tools":
                state.update(cast(AgentState, await execute_tools(state)))
                step = "validate" if state["final_text"] else "model"
                continue
            if step == "validate":
                state.update(cast(AgentState, await validate_output(state)))
                step = "finalize"
                continue
            state.update(cast(AgentState, await finalize(state)))
            break
        return AgentResult(
            text=state["final_text"] or grounded_response(),
            stop_reason=state["stop_reason"] or "COMPLETED",
            error_code=state["error_code"],
            model_rounds=state["model_rounds"],
            tool_call_count=state["tool_call_count"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
        )
