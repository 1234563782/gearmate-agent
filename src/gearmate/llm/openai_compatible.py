import json
from dataclasses import dataclass
from typing import Any, cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import SecretStr

from gearmate.config import Settings
from gearmate.llm.types import ModelRequest, ModelResponse, ModelToolCall, ModelUsage


class ModelConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    base_url: str
    model_id: str
    api_key: SecretStr
    connect_timeout_seconds: float
    first_token_timeout_seconds: float
    request_timeout_seconds: float
    max_output_tokens: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleConfig":
        if settings.model_base_url is None:
            raise ModelConfigurationError("GEARMATE_MODEL_BASE_URL is required")
        if settings.model_id is None:
            raise ModelConfigurationError("GEARMATE_MODEL_ID is required")
        if settings.model_api_key is None:
            raise ModelConfigurationError("GEARMATE_MODEL_API_KEY is required")
        return cls(
            base_url=settings.model_base_url,
            model_id=settings.model_id,
            api_key=settings.model_api_key,
            connect_timeout_seconds=settings.model_connect_timeout_seconds,
            first_token_timeout_seconds=settings.model_first_token_timeout_seconds,
            request_timeout_seconds=settings.model_request_timeout_seconds,
            max_output_tokens=settings.model_max_output_tokens,
        )


class OpenAICompatibleChatModel:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self._config = config
        timeout = httpx.Timeout(
            timeout=config.request_timeout_seconds,
            connect=config.connect_timeout_seconds,
        )
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value(),
            timeout=timeout,
        )

    @property
    def config(self) -> OpenAICompatibleConfig:
        return self._config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        messages: list[ChatCompletionMessageParam] = []
        for message in request.messages:
            item: dict[str, object] = {"role": message.role, "content": message.content}
            if message.name is not None:
                item["name"] = message.name
            if message.tool_call_id is not None:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(cast(ChatCompletionMessageParam, item))

        tools = [
            cast(
                ChatCompletionToolParam,
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                },
            )
            for tool in request.tools
        ]
        extra_body = (
            {"enable_thinking": request.enable_thinking}
            if request.enable_thinking is not None
            else None
        )
        if tools:
            tool_choice: object
            if request.tool_choice in ("auto", "required", "none"):
                tool_choice = request.tool_choice
            else:
                tool_choice = {
                    "type": "function",
                    "function": {"name": request.tool_choice},
                }
            response = await self._client.chat.completions.create(
                model=self._config.model_id,
                messages=messages,
                tools=tools,
                tool_choice=cast(Any, tool_choice),
                max_tokens=request.max_output_tokens,
                temperature=request.temperature,
                extra_body=extra_body,
            )
        else:
            response = await self._client.chat.completions.create(
                model=self._config.model_id,
                messages=messages,
                max_tokens=request.max_output_tokens,
                temperature=request.temperature,
                extra_body=extra_body,
            )
        choice = response.choices[0]
        tool_calls: list[ModelToolCall] = []
        for call in choice.message.tool_calls or []:
            if call.type != "function":
                raise ValueError(f"Model returned unsupported tool call type: {call.type}")
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Model returned invalid tool arguments for {call.function.name}"
                ) from error
            if not isinstance(arguments, dict):
                raise ValueError(f"Tool arguments for {call.function.name} must be an object")
            tool_calls.append(
                ModelToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )
        usage = response.usage
        return ModelResponse(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=ModelUsage(
                input_tokens=usage.prompt_tokens if usage is not None else 0,
                output_tokens=usage.completion_tokens if usage is not None else 0,
            ),
            tool_calls=tuple(tool_calls),
        )

    async def close(self) -> None:
        await self._client.close()
