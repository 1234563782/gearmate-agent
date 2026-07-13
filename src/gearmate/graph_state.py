from typing import Literal, TypedDict

from gearmate.llm.types import ModelMessage

RunStatus = Literal[
    "RUNNING",
    "TOOL_REQUESTED",
    "COMPLETED",
    "OUTPUT_TRUNCATED",
    "REFUSED",
    "FAILED",
    "CANCELLED",
]


class GearMateGraphState(TypedDict):
    conversation_id: str
    run_id: str
    user_id: str
    timezone: str
    status: RunStatus
    messages: list[ModelMessage]
    selected_tool: str | None
    error_code: str | None
