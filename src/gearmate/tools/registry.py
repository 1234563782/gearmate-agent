import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from time import monotonic
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from gearmate.catalog import CatalogSearchService
from gearmate.llm.types import ModelToolCall, ModelToolDefinition
from gearmate.rentflow.client import RentFlowClient, RentFlowError
from gearmate.requirements import EquipmentNeed, EquipmentRole, ScenarioPlan
from gearmate.tools.contracts import (
    AvailabilityInput,
    ProductSearchInput,
    ProductSearchResult,
    ProductSummary,
    QuoteInput,
    ScenarioKitInput,
    ScenarioKitItem,
    ScenarioKitResult,
)
from gearmate.tools.metadata import ToolDescriptor
from gearmate.validation.facts import FactSnapshot

EventWriter = Callable[[str, dict[str, Any]], Awaitable[None]]
logger = logging.getLogger(__name__)


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
        scenario_plan: ScenarioPlan | None = None,
        catalog_search: CatalogSearchService | None = None,
    ) -> None:
        self._rentflow = rentflow
        self._max_result_items = max_result_items
        self._max_concurrency = max_concurrency
        self._scenario_plan = scenario_plan
        self._catalog_search = catalog_search
        self._cache: dict[str, ToolExecutionResult] = {}
        self._tools = {
            "search_products": ToolDescriptor(
                name="search_products",
                description="按关键词、类目、最高日租金和可选租期搜索 RentFlow 商品。",
                input_model=ProductSearchInput,
                handler=self._search_products,
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
        if (
            scenario_plan is not None
            and scenario_plan.ready
            and scenario_plan.requirements.daily_budget is not None
        ):
            self._tools["recommend_scenario_kit"] = ToolDescriptor(
                name="recommend_scenario_kit",
                description=("根据本轮已确认的场景设备角色和每日总预算，确定性选择完整设备组合。"),
                input_model=ScenarioKitInput,
                handler=self._recommend_scenario_kit,
                read_only=True,
                concurrency_safe=False,
                timeout_seconds=timeout_seconds,
            )

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
                async with semaphore:  # noqa: B023
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
        await write_event(
            "tool.started",
            {
                "toolCallId": call.id,
                "tool": call.name,
                "arguments": call.arguments,
            },
        )
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
        except (
            ValidationError,
            ValueError,
            TimeoutError,
            RentFlowError,
            httpx.HTTPError,
        ) as error:
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

    async def _search_products(
        self,
        request: ProductSearchInput,
    ) -> ProductSearchResult:
        if request.semantic_query and self._catalog_search is not None:
            try:
                semantic_result = await self._semantic_products(request)
                if semantic_result.items:
                    return semantic_result
            except Exception:
                logger.exception("Semantic product search failed; using structured fallback")
        result = await self._rentflow.search_products(request)
        if (
            request.equipment_role is None
            and request.brand is None
            and request.model is None
        ):
            return result
        matching_items = tuple(
            item
            for item in result.items
            if (
                request.equipment_role is None
                or item.equipment_role == request.equipment_role
            )
            and (request.brand is None or item.brand.casefold() == request.brand.casefold())
            and (request.model is None or item.model.casefold() == request.model.casefold())
        )
        if len(matching_items) == len(result.items):
            return result
        return result.model_copy(
            update={
                "items": matching_items,
                "total_elements": len(matching_items),
                "total_pages": 1 if matching_items else 0,
            }
        )

    async def _semantic_products(self, request: ProductSearchInput) -> ProductSearchResult:
        catalog_search = self._catalog_search
        if catalog_search is None or request.semantic_query is None:
            raise ValueError("Semantic catalog search is not configured")
        candidates = await catalog_search.search(
            request.semantic_query,
            equipment_role=request.equipment_role,
            brand=request.brand,
            model=request.model,
        )
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def hydrate(product_id: str) -> ProductSummary:
            async with semaphore:
                detail = await self._rentflow.get_product(product_id)
                available_count: int | None = None
                if request.rental_period is not None:
                    availability = await self._rentflow.search_availability(
                        AvailabilityInput(
                            product_id=product_id,
                            start_at=request.rental_period.start_at,
                            end_at=request.rental_period.end_at,
                        )
                    )
                    available_count = availability.available_count
                return ProductSummary(
                    product_id=detail.product_id,
                    category_id=detail.category_id,
                    equipment_role=detail.equipment_role,
                    name=detail.name,
                    brand=detail.brand,
                    model=detail.model,
                    daily_rate=detail.daily_rate,
                    fixed_deposit=detail.fixed_deposit,
                    available_count=available_count,
                )

        items = tuple(
            await asyncio.gather(*(hydrate(candidate.product_id) for candidate in candidates))
        )
        return ProductSearchResult(
            items=items,
            page=0,
            size=max(1, min(100, self._max_result_items)),
            total_elements=len(items),
            total_pages=1 if items else 0,
        )

    async def _recommend_scenario_kit(self, request: ScenarioKitInput) -> ScenarioKitResult:
        plan = self._scenario_plan
        if plan is None or not plan.ready or plan.requirements.daily_budget is None:
            raise ValueError("A complete scenario plan with a budget is required")
        budget = plan.requirements.daily_budget

        async def search(
            need: EquipmentNeed,
        ) -> tuple[EquipmentNeed, ProductSummary | None]:
            result = await self._search_products(
                ProductSearchInput(
                    keyword=need.keyword,
                    equipment_role=need.role,
                    rental_period=request.rental_period,
                    max_daily_rate=budget,
                    size=self._max_result_items,
                )
            )
            eligible = [
                product
                for product in result.items
                if product.available_count is None or product.available_count >= need.quantity
            ]
            eligible.sort(key=lambda product: (Decimal(product.daily_rate), product.product_id))
            return need, eligible[0] if eligible else None

        matches = await asyncio.gather(*(search(need) for need in plan.equipment_needs))
        items: list[ScenarioKitItem] = []
        missing_roles: list[EquipmentRole] = []
        total = Decimal("0")
        for need, product in matches:
            if product is None:
                missing_roles.append(need.role)
                continue
            subtotal = Decimal(product.daily_rate) * need.quantity
            total += subtotal
            items.append(
                ScenarioKitItem(
                    role=need.role,
                    quantity=need.quantity,
                    product=product,
                    subtotal_daily_rate=f"{subtotal:.2f}",
                )
            )
        return ScenarioKitResult(
            scenario=plan.requirements.scenario_id,
            items=tuple(items),
            total_daily_rate=f"{total:.2f}",
            max_daily_budget=f"{budget:.2f}",
            within_budget=not missing_roles and total <= budget,
            availability_checked=request.rental_period is not None,
            missing_roles=tuple(missing_roles),
        )
