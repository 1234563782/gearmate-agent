from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import ModelRequest, ModelResponse
from gearmate.resilience import AsyncModelGovernor, GovernorSnapshot


class GovernedChatModel:
    def __init__(self, inner: ChatModelPort, governor: AsyncModelGovernor) -> None:
        self._inner = inner
        self._governor = governor

    @property
    def governor_snapshot(self) -> GovernorSnapshot:
        return self._governor.snapshot

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await self._governor.run(
            lambda: self._inner.complete(request),
            lane=request.workload,
            estimated_tokens=estimate_request_tokens(request),
        )

    async def close(self) -> None:
        await self._inner.close()


def estimate_request_tokens(request: ModelRequest) -> int:
    input_characters = sum(len(message.content) for message in request.messages)
    tool_characters = sum(
        len(tool.name) + len(tool.description) + len(str(tool.parameters))
        for tool in request.tools
    )
    estimated_input = max(1, (input_characters + tool_characters + 3) // 4)
    return estimated_input + request.max_output_tokens
