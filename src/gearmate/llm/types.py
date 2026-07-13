from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelRole = Literal["system", "user", "assistant", "tool"]


class ModelMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ModelRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[ModelMessage, ...]
    max_output_tokens: int = Field(ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class ModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    finish_reason: str
    usage: ModelUsage = Field(default_factory=ModelUsage)
