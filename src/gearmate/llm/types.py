from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelRole = Literal["system", "user", "assistant", "tool"]


class ModelToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any]


class ModelMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ModelRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()


class ModelToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[ModelMessage, ...]
    max_output_tokens: int = Field(ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    tools: tuple[ModelToolDefinition, ...] = ()
    tool_choice: str = "auto"
    enable_thinking: bool | None = None


class ModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    finish_reason: str
    usage: ModelUsage = Field(default_factory=ModelUsage)
    tool_calls: tuple[ModelToolCall, ...] = ()
