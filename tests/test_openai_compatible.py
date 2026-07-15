from types import SimpleNamespace

from pydantic import SecretStr

from gearmate.llm.openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleConfig,
)
from gearmate.llm.types import (
    ModelMessage,
    ModelRequest,
    ModelToolDefinition,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                ),
            ),
            usage=None,
        )


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    async def close(self) -> None:
        return None


async def test_forced_tool_disables_thinking_in_provider_request() -> None:
    model = OpenAICompatibleChatModel(
        OpenAICompatibleConfig(
            base_url="http://model.example/v1",
            model_id="test-model",
            api_key=SecretStr("test-key"),
            connect_timeout_seconds=1,
            first_token_timeout_seconds=1,
            request_timeout_seconds=1,
            max_output_tokens=128,
        )
    )
    client = FakeClient()
    model._client = client  # type: ignore[assignment]

    await model.complete(
        ModelRequest(
            messages=(ModelMessage(role="user", content="search"),),
            tools=(
                ModelToolDefinition(
                    name="resolve_agent_action",
                    description="resolve",
                    parameters={"type": "object"},
                ),
            ),
            tool_choice="resolve_agent_action",
            enable_thinking=False,
            max_output_tokens=128,
        )
    )

    assert client.completions.kwargs is not None
    assert client.completions.kwargs["extra_body"] == {"enable_thinking": False}
    assert client.completions.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "resolve_agent_action"},
    }
    await model.close()
