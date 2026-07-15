import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from gearmate.config import Settings
from gearmate.graph_state import GearMateGraphState
from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import ModelMessage, ModelRequest
from gearmate.prompts.loader import RenderedPrompt
from gearmate.tools.contracts import RentalPeriodInput
from gearmate.tools.registry import ToolRegistry
from gearmate.validation.facts import FactSnapshot

EventWriter = Callable[[str, dict[str, Any]], Awaitable[None]]


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
        write_event: EventWriter,
    ) -> AgentResult:
        facts = FactSnapshot()
        messages = [ModelMessage(role="system", content=self._prompt.content)]
        messages.extend(history)
        if (
            not history
            or history[-1].role != "user"
            or history[-1].content != message
        ):
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

        async def preprocess(state: AgentState) -> dict[str, Any]:
            needs_period = any(
                signal in message.lower()
                for signal in (
                    "有货",
                    "库存",
                    "可租",
                    "能租",
                    "报价",
                    "多少钱",
                    "价格",
                    "费用",
                    "租金",
                    "押金",
                    "quote",
                    "available",
                )
            )
            if needs_period and rental_period is None:
                text = "请提供完整租期，包括开始时间、结束时间和时区，我再为你查询库存或生成报价。"
                await write_event(
                    "decision.made",
                    {"outcome": "NEED_CLARIFICATION", "field": "rentalPeriod"},
                )
                return {"final_text": text, "stop_reason": "NEED_CLARIFICATION"}
            await write_event("decision.made", {"outcome": "RUN_AGENT"})
            return {}

        async def call_model(state: AgentState) -> dict[str, Any]:
            if state["model_rounds"] >= self._settings.max_model_rounds:
                return {
                    "final_text": facts.fallback_text(),
                    "stop_reason": "MAX_MODEL_ROUNDS",
                }
            round_no = state["model_rounds"] + 1
            await write_event("model.started", {"round": round_no})
            started = monotonic()
            request = ModelRequest(
                messages=tuple(state["messages"]),
                tools=self._tools.model_definitions(),
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
                    "final_text": facts.fallback_text(),
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
            return {
                "messages": [*state["messages"], *tool_messages],
                "pending_tool_calls": [],
                "tool_call_count": next_count,
            }

        async def validate_output(state: AgentState) -> dict[str, Any]:
            text = (state["final_text"] or "").strip()
            if not text:
                text = facts.fallback_text()
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
                    "final_text": facts.fallback_text(),
                    "stop_reason": "FACT_VALIDATION_FALLBACK",
                }
            return {"final_text": text, "stop_reason": state["stop_reason"] or "COMPLETED"}

        async def finalize(state: AgentState) -> dict[str, Any]:
            text = state["final_text"] or facts.fallback_text()
            await write_event("assistant.delta", {"content": text})
            await write_event(
                "assistant.completed",
                {"content": text, "stopReason": state["stop_reason"] or "COMPLETED"},
            )
            return {"final_text": text, "stop_reason": state["stop_reason"] or "COMPLETED"}

        def after_preprocess(state: AgentState) -> Literal["model", "finalize"]:
            return "finalize" if state["final_text"] else "model"

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
            "pending_tool_calls": [],
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
            text=final["final_text"] or facts.fallback_text(),
            stop_reason=final["stop_reason"] or "COMPLETED",
            error_code=final["error_code"],
            model_rounds=final["model_rounds"],
            tool_call_count=final["tool_call_count"],
            input_tokens=final["input_tokens"],
            output_tokens=final["output_tokens"],
        )
