import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from gearmate.llm.types import ModelToolCall, ModelToolDefinition
from gearmate.rentflow.client import RentFlowClient, RentFlowError
from gearmate.tools.contracts import AvailabilityInput, ProductSearchInput, QuoteInput
from gearmate.tools.metadata import ToolDescriptor
from gearmate.validation.facts import FactSnapshot

EventWriter = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    call: ModelToolCall
    content: str
    is_error: bool
    result: BaseModel | None = None


class ToolRegistry:
    def __init__(
        self,
        rentflow: RentFlowClient,
        *,
        timeout_seconds: float,
        max_result_items: int,
        max_concurrency: int,
    ) -> None:
        self._max_concurrency = max_concurrency
        self._cache: dict[str, ToolExecutionResult] = {}
        self._tools = {
            "search_products": ToolDescriptor(
                name="search_products",
                description="按关键词、类目和可选租期搜索 RentFlow 商品。",
                input_model=ProductSearchInput,
                handler=rentflow.search_products,
                read_only=True,
                concurrency_safe=True,
                timeout_seconds=timeout_seconds,
                max_result_items=max_result_items,
            ),
            "check_availability": ToolDescriptor(
                name="check_availability",
                description="查询一个商品在明确租期内的实时可租数量。",
                input_model=AvailabilityInput,
                handler=rentflow.search_availability,
                read_only=True,
                concurrency_safe=True,
                timeout_seconds=timeout_seconds,
            ),
            "create_quote": ToolDescriptor(
                name="create_quote",
                description="为一个商品和明确租期生成不可变正式报价；不会锁定库存。",
                input_model=QuoteInput,
                handler=rentflow.create_quote,
                read_only=False,
                concurrency_safe=False,
                timeout_seconds=timeout_seconds,
            ),
        }

    def model_definitions(self) -> tuple[ModelToolDefinition, ...]:
        return tuple(
            ModelToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.schema(),
            )
            for tool in self._tools.values()
        )

    async def execute_all(
        self,
        calls: list[ModelToolCall],
        facts: FactSnapshot,
        write_event: EventWriter,
    ) -> list[ToolExecutionResult]:
        results: list[ToolExecutionResult] = []
        index = 0
        while index < len(calls):
            call = calls[index]
            descriptor = self._tools.get(call.name)
            if descriptor is None or not descriptor.concurrency_safe:
                result = await self._execute(call, write_event)
                results.append(result)
                if result.result is not None:
                    facts.add(result.result)
                index += 1
                continue
            batch: list[ModelToolCall] = []
            while index < len(calls):
                candidate = calls[index]
                candidate_tool = self._tools.get(candidate.name)
                if candidate_tool is None or not candidate_tool.concurrency_safe:
                    break
                batch.append(candidate)
                index += 1
            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def limited(item: ModelToolCall) -> ToolExecutionResult:
                async with semaphore:
                    return await self._execute(item, write_event)

            batch_results = await asyncio.gather(*(limited(item) for item in batch))
            results.extend(batch_results)
            for result in batch_results:
                if result.result is not None:
                    facts.add(result.result)
        return results

    async def _execute(
        self,
        call: ModelToolCall,
        write_event: EventWriter,
    ) -> ToolExecutionResult:
        descriptor = self._tools.get(call.name)
        started = monotonic()
        await write_event("tool.started", {"toolCallId": call.id, "tool": call.name})
        if descriptor is None:
            content = json.dumps({"error": "UNKNOWN_TOOL", "tool": call.name})
            await write_event(
                "tool.failed",
                {"toolCallId": call.id, "tool": call.name, "errorCode": "UNKNOWN_TOOL"},
            )
            return ToolExecutionResult(call=call, content=content, is_error=True)
        cache_key = json.dumps(
            {"tool": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            await write_event(
                "tool.completed",
                {
                    "toolCallId": call.id,
                    "tool": call.name,
                    "cached": True,
                    "durationMs": round((monotonic() - started) * 1000),
                },
            )
            return ToolExecutionResult(
                call=call,
                content=cached.content,
                is_error=cached.is_error,
                result=cached.result,
            )
        try:
            request = descriptor.input_model.model_validate(call.arguments)
            async with asyncio.timeout(descriptor.timeout_seconds):
                result = await descriptor.handler(request)
            full_payload = result.model_dump(mode="json", by_alias=True)
            visible_payload = dict(full_payload)
            truncated = False
            if descriptor.max_result_items is not None and isinstance(
                visible_payload.get("items"), list
            ):
                items = visible_payload["items"]
                visible_payload["items"] = items[: descriptor.max_result_items]
                truncated = len(items) > descriptor.max_result_items
            visible_result = type(result).model_validate(visible_payload)
            await write_event(
                "tool.completed",
                {
                    "toolCallId": call.id,
                    "tool": call.name,
                    "result": full_payload,
                    "modelResultTruncated": truncated,
                    "durationMs": round((monotonic() - started) * 1000),
                    "resultSizeBytes": len(
                        json.dumps(full_payload, ensure_ascii=False).encode("utf-8")
                    ),
                },
            )
            execution = ToolExecutionResult(
                call=call,
                content=json.dumps(visible_payload, ensure_ascii=False),
                is_error=False,
                result=visible_result,
            )
            self._cache[cache_key] = execution
            return execution
        except (ValidationError, TimeoutError, RentFlowError, httpx.HTTPError) as error:
            code = getattr(error, "code", None) or (
                "TOOL_TIMEOUT"
                if isinstance(error, TimeoutError)
                else (
                    "RENTFLOW_TRANSPORT_ERROR"
                    if isinstance(error, httpx.HTTPError)
                    else "INVALID_TOOL_ARGUMENTS"
                )
            )
            await write_event(
                "tool.failed",
                {
                    "toolCallId": call.id,
                    "tool": call.name,
                    "errorCode": code,
                    "durationMs": round((monotonic() - started) * 1000),
                },
            )
            content = json.dumps({"error": code, "message": str(error)}, ensure_ascii=False)
            return ToolExecutionResult(call=call, content=content, is_error=True)
