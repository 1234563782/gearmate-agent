from typing import Protocol

from gearmate.llm.types import ModelRequest, ModelResponse


class ChatModelPort(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one provider-neutral model request."""
        ...

    async def close(self) -> None:
        """Release provider resources."""
        ...
