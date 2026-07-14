from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

ToolHandler = Callable[[Any], Awaitable[BaseModel]]


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    read_only: bool
    concurrency_safe: bool
    timeout_seconds: float
    max_result_items: int | None = None

    def schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema(by_alias=True)
