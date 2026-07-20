from typing import Literal, NotRequired, TypedDict

from gearmate.llm.types import ModelMessage, ModelToolCall

RunStatus = Literal[
    "RUNNING",
    "TOOL_REQUESTED",
    "COMPLETED",
    "OUTPUT_TRUNCATED",
    "REFUSED",
    "FAILED",
    "CANCELLED",
]


class AgentState(TypedDict):
    conversation_id: NotRequired[str]
    run_id: NotRequired[str]
    user_id: NotRequired[str]
    timezone: NotRequired[str]
    status: NotRequired[RunStatus]
    messages: list[ModelMessage]
    selected_tool: NotRequired[str | None]
    error_code: str | None
    stop_reason: str | None
    model_rounds: int
    tool_call_count: int
    pending_tool_calls: list[ModelToolCall]
    final_text: str | None
    input_tokens: int
    output_tokens: int
