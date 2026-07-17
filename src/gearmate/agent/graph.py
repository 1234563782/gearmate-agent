import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from gearmate.actions import AgentAction
from gearmate.config import Settings
from gearmate.graph_state import GearMateGraphState
from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import ModelMessage, ModelRequest, ModelToolCall
from gearmate.prompts.loader import RenderedPrompt
from gearmate.requirements import ScenarioPlan
from gearmate.responses import UserResponseComposer
from gearmate.search import ProductSearchPlanner
from gearmate.tools.contracts import RentalPeriodInput
from gearmate.tools.registry import ToolRegistry
from gearmate.validation.facts import FactSnapshot

EventWriter = Callable[[str, dict[str, Any]], Awaitable[None]]
AUTOMATIC_SCENARIO_KIT_CALL_ID = "automatic-scenario-kit"
AUTOMATIC_ACTION_CALL_ID = "automatic-action"


AgentState = GearMateGraphState


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
        rental_period: RentalPeriodInput | None,
        scenario_plan: ScenarioPlan | None,
        action: AgentAction,
        write_event: EventWriter,
        timezone: str = "Asia/Shanghai",
    ) -> AgentResult:
        facts = FactSnapshot()
        if action.max_daily_rate is not None:
            facts.add_constraint_amount(action.max_daily_rate)
        if action.target_daily_rate is not None:
            facts.add_constraint_amount(action.target_daily_rate)
        response_composer = UserResponseComposer()

        def grounded_response() -> str:
            return response_composer.compose(
                action=action,
                facts=facts,
                rental_period=rental_period,
                timezone=timezone,
            )
        if scenario_plan is not None and scenario_plan.requirements.daily_budget is not None:
            facts.add_constraint_amount(scenario_plan.requirements.daily_budget)
            for need in scenario_plan.equipment_needs:
                facts.add_constraint_count(need.quantity)
        messages = [ModelMessage(role="system", content=self._prompt.content)]
        messages.extend(history)
        if not history or history[-1].role != "user" or history[-1].content != message:
            messages.append(ModelMessage(role="user", content=message))
        if rental_period is not None:
            messages.append(
                ModelMessage(
                    role="system",
                    content=(
                        "本轮已确认租期："
                        f"startAt={rental_period.start_at.isoformat()}, "
                        f"endAt={rental_period.end_at.isoformat()}。"
                    ),
                )
            )
        if scenario_plan is not None and scenario_plan.ready:
            messages.append(
                ModelMessage(
                    role="system",
                    content=(
                        "本轮已确认的结构化场景需求如下:\n"
                        + json.dumps(scenario_plan.model_context(), ensure_ascii=False)
                        + "\n不得把整个场景缩减成单一商品关键词。"
                        "有每日预算时必须优先调用 recommend_scenario_kit，"
                        "由工具选择商品并计算组合总价；"
                        "没有预算时按 equipmentNeeds 中每个角色分别搜索。"
                    ),
                )
            )
        automatic_tool_calls: list[ModelToolCall] = []
        if (
            scenario_plan is not None
            and scenario_plan.ready
            and scenario_plan.requirements.daily_budget is not None
        ):
            arguments: dict[str, Any] = {}
            if rental_period is not None:
                arguments["rentalPeriod"] = rental_period.model_dump(mode="json", by_alias=True)
            automatic_call = ModelToolCall(
                id=AUTOMATIC_SCENARIO_KIT_CALL_ID,
                name="recommend_scenario_kit",
                arguments=arguments,
            )
            automatic_tool_calls.append(automatic_call)
            messages.append(
                ModelMessage(
                    role="assistant",
                    content="",
                    tool_calls=(automatic_call,),
                )
            )
        elif scenario_plan is not None and scenario_plan.ready:
            for index, need in enumerate(scenario_plan.equipment_needs):
                arguments = {
                    "keyword": need.keyword,
                    "equipmentRole": need.role,
                }
                if rental_period is not None:
                    arguments["rentalPeriod"] = rental_period.model_dump(mode="json", by_alias=True)
                automatic_tool_calls.append(
                    ModelToolCall(
                        id=f"automatic-scenario-search-{index}",
                        name="search_products",
                        arguments=arguments,
                    )
                )
        elif action.action == "product_search":
            plan = ProductSearchPlanner().plan(action)
            arguments = {}
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
            if plan.max_daily_rate is not None:
                arguments["maxDailyRate"] = str(plan.max_daily_rate)
            if plan.target_daily_rate is not None:
                arguments["targetDailyRate"] = str(plan.target_daily_rate)
            if rental_period is not None:
                arguments["rentalPeriod"] = rental_period.model_dump(mode="json", by_alias=True)
            automatic_tool_calls.append(
                ModelToolCall(
                    id=AUTOMATIC_ACTION_CALL_ID,
                    name="search_products",
                    arguments=arguments,
                )
            )
        elif action.action == "product_detail" and action.product_id is not None:
            automatic_tool_calls.append(
                ModelToolCall(
                    id=AUTOMATIC_ACTION_CALL_ID,
                    name="get_product",
                    arguments={"productId": action.product_id},
                )
            )
        elif (
            action.action in ("availability", "quote")
            and action.product_id is not None
            and rental_period is not None
        ):
            automatic_tool_calls.append(
                ModelToolCall(
                    id=AUTOMATIC_ACTION_CALL_ID,
                    name=(
                        "check_availability" if action.action == "availability" else "create_quote"
                    ),
                    arguments={
                        "productId": action.product_id,
                        "startAt": rental_period.start_at.isoformat(),
                        "endAt": rental_period.end_at.isoformat(),
                    },
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
                    name="list_orders",
                    arguments=arguments,
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
                text = "请先指定一个准确商品或最近搜索结果中的位置。"
                await write_event(
                    "decision.made",
                    {"outcome": "NEED_CLARIFICATION", "field": "productId"},
                )
                return {"final_text": text, "stop_reason": "NEED_CLARIFICATION"}
            if action.action in ("availability", "quote") and action.product_id is None:
                text = (
                    "请先指定具体商品，例如点击卡片，"
                    "或告诉我是最近搜索结果中的第几个，再查询库存或生成报价。"
                )
                await write_event(
                    "decision.made",
                    {"outcome": "NEED_CLARIFICATION", "field": "productId"},
                )
                return {"final_text": text, "stop_reason": "NEED_CLARIFICATION"}
            if action.action in ("availability", "quote") and rental_period is None:
                text = "请提供完整租期，包括开始时间、结束时间和时区，我再为你查询库存或生成报价。"
                await write_event(
                    "decision.made",
                    {"outcome": "NEED_CLARIFICATION", "field": "rentalPeriod"},
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
            text = (state["final_text"] or "").strip()
            if not text:
                text = grounded_response()
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

        def after_preprocess(
            state: AgentState,
        ) -> Literal["model", "tools", "finalize"]:
            if state["final_text"]:
                return "finalize"
            return "tools" if state["pending_tool_calls"] else "model"

        def after_model(state: AgentState) -> Literal["tools", "validate"]:
            return "tools" if state["pending_tool_calls"] else "validate"

        def after_tools(state: AgentState) -> Literal["model", "validate"]:
            return "validate" if state["final_text"] else "model"

        graph = StateGraph(AgentState)
        graph.add_node("preprocess", preprocess)
        graph.add_node("model", call_model)
        graph.add_node("tools", execute_tools)
        graph.add_node("validate", validate_output)
        graph.add_node("finalize", finalize)
        graph.add_edge(START, "preprocess")
        graph.add_conditional_edges("preprocess", after_preprocess)
        graph.add_conditional_edges("model", after_model)
        graph.add_conditional_edges("tools", after_tools)
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)
        compiled = graph.compile()
        initial: AgentState = {
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
        final = await compiled.ainvoke(initial)
        return AgentResult(
            text=final["final_text"] or grounded_response(),
            stop_reason=final["stop_reason"] or "COMPLETED",
            error_code=final["error_code"],
            model_rounds=final["model_rounds"],
            tool_call_count=final["tool_call_count"],
            input_tokens=final["input_tokens"],
            output_tokens=final["output_tokens"],
        )
