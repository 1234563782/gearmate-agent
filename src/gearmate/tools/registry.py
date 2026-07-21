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
from gearmate.tools.contracts import (
    ProductDetailInput,
    ProductSearchInput,
    ProductSearchResult,
    ProductSummary,
    StoreOrder,
    StoreOrderDetailInput,
    StoreOrderListInput,
    StoreOrderPage,
    StoreSku,
    StoreSkuInput,
    StoreSkuList,
    StoreSkuListInput,
)
from gearmate.tools.metadata import ToolDescriptor
from gearmate.validation.facts import FactSnapshot

EventWriter = Callable[[str, dict[str, Any]], Awaitable[None]]
MODEL_VISIBLE_TOOL_NAMES = frozenset(
    {
        "search_products",
        "get_product",
        "list_product_skus",
        "get_store_sku",
        "list_store_orders",
        "get_store_order",
    }
)
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
        catalog_search: CatalogSearchService | None = None,
        preferred_brands: tuple[str, ...] = (),
        excluded_brands: tuple[str, ...] = (),
    ) -> None:
        self._rentflow = rentflow
        self._max_result_items = max_result_items
        self._max_concurrency = max_concurrency
        self._catalog_search = catalog_search
        self._preferred_brands = frozenset(brand.casefold() for brand in preferred_brands)
        self._excluded_brands = frozenset(brand.casefold() for brand in excluded_brands)
        self._last_search_diagnostics: dict[str, Any] | None = None
        self._last_product_search_result: ProductSearchResult | None = None
        self._last_store_skus: StoreSkuList | None = None
        self._last_store_order_page: StoreOrderPage | None = None
        self._last_store_order: StoreOrder | None = None
        self._cache: dict[str, ToolExecutionResult] = {}
        self._tools = {
            "search_products": ToolDescriptor(
                name="search_products",
                description="按关键词、品类、品牌、型号、用途和购买价格搜索商城商品。",
                input_model=ProductSearchInput,
                handler=self._search_products,
                read_only=True,
                concurrency_safe=True,
                timeout_seconds=timeout_seconds,
                max_result_items=max_result_items,
            ),
            "get_product": ToolDescriptor(
                name="get_product",
                description="按精确商品 ID 获取 RentFlow 商品详情。",
                input_model=ProductDetailInput,
                handler=self._get_product,
                read_only=True,
                concurrency_safe=True,
                timeout_seconds=timeout_seconds,
            ),
            "list_product_skus": ToolDescriptor(
                name="list_product_skus",
                description="查询一个商品当前可购买的 SKU、规格、售价和可售库存。",
                input_model=StoreSkuListInput,
                handler=self._list_product_skus,
                read_only=True,
                concurrency_safe=True,
                timeout_seconds=timeout_seconds,
            ),
            "get_store_sku": ToolDescriptor(
                name="get_store_sku",
                description="按可信 SKU ID 查询规格、售价和可售库存。",
                input_model=StoreSkuInput,
                handler=self._get_store_sku,
                read_only=True,
                concurrency_safe=True,
                timeout_seconds=timeout_seconds,
            ),
            "list_store_orders": ToolDescriptor(
                name="list_store_orders",
                description="查询当前登录用户的商城订单，可按支付和物流状态筛选。",
                input_model=StoreOrderListInput,
                handler=self._list_store_orders,
                read_only=True,
                concurrency_safe=True,
                timeout_seconds=timeout_seconds,
                max_result_items=max_result_items,
            ),
            "get_store_order": ToolDescriptor(
                name="get_store_order",
                description="查询当前登录用户的一笔商城订单详情。",
                input_model=StoreOrderDetailInput,
                handler=self._get_store_order,
                read_only=True,
                concurrency_safe=True,
                timeout_seconds=timeout_seconds,
            ),
        }
    @property
    def last_search_diagnostics(self) -> dict[str, Any] | None:
        if self._last_search_diagnostics is None:
            return None
        return dict(self._last_search_diagnostics)

    @property
    def last_product_search_result(self) -> ProductSearchResult | None:
        return self._last_product_search_result

    @property
    def last_store_skus(self) -> StoreSkuList | None:
        return self._last_store_skus

    @property
    def last_store_order_page(self) -> StoreOrderPage | None:
        return self._last_store_order_page

    @property
    def last_store_order(self) -> StoreOrder | None:
        return self._last_store_order

    def model_definitions(self) -> tuple[ModelToolDefinition, ...]:
        return tuple(
            ModelToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.schema(),
            )
            for tool in self._tools.values()
            if tool.name in MODEL_VISIBLE_TOOL_NAMES
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
            if isinstance(visible_result, StoreSkuList):
                self._last_store_skus = visible_result
            elif isinstance(visible_result, StoreSku):
                self._last_store_skus = StoreSkuList(
                    product_id=visible_result.product_id,
                    items=(visible_result,),
                )
            elif isinstance(visible_result, StoreOrderPage):
                self._last_store_order_page = visible_result
            elif isinstance(visible_result, StoreOrder):
                self._last_store_order = visible_result
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
        semantic_attempted = bool(request.semantic_query and self._catalog_search is not None)
        if request.semantic_query and self._catalog_search is not None:
            try:
                semantic_result = await self._semantic_products(request)
                semantic_result = await self._attach_store_skus(semantic_result)
                semantic_result = self._apply_price_preference(semantic_result, request)
                semantic_result = self._apply_user_preferences(semantic_result, request)
                if semantic_result.items:
                    self._last_product_search_result = semantic_result
                    return semantic_result
                self._last_search_diagnostics = {
                    "mode": "structured_fallback",
                    "reason": "NO_SEMANTIC_CANDIDATE_ABOVE_THRESHOLD",
                    "semanticQuery": request.semantic_query,
                }
            except Exception:
                logger.exception("Semantic product search failed; using structured fallback")
                self._last_search_diagnostics = {
                    "mode": "structured_fallback",
                    "reason": "SEMANTIC_SEARCH_FAILED",
                    "semanticQuery": request.semantic_query,
                }
        result = await self._rentflow.search_products(request)
        result = await self._attach_store_skus(result)
        result = self._apply_price_preference(result, request)
        result = self._apply_user_preferences(result, request)
        diagnostics = self._last_search_diagnostics or {}
        self._last_search_diagnostics = {
            **diagnostics,
            "mode": "structured_fallback" if semantic_attempted else "structured",
            "resultCount": len(result.items),
        }
        if (
            request.equipment_role is None
            and request.brand is None
            and request.model is None
            and request.use_case_id is None
        ):
            self._last_product_search_result = result
            return result
        matching_items = tuple(
            item
            for item in result.items
            if (request.equipment_role is None or item.equipment_role == request.equipment_role)
            and (request.brand is None or item.brand.casefold() == request.brand.casefold())
            and (request.model is None or item.model.casefold() == request.model.casefold())
            and (
                request.use_case_id is None
                or any(use_case.id == request.use_case_id for use_case in item.use_cases)
            )
        )
        if len(matching_items) == len(result.items):
            self._last_product_search_result = result
            return result
        filtered_result = result.model_copy(
            update={
                "items": matching_items,
                "total_elements": len(matching_items),
                "total_pages": 1 if matching_items else 0,
            }
        )
        self._last_product_search_result = filtered_result
        return filtered_result

    async def _list_product_skus(self, request: StoreSkuListInput) -> StoreSkuList:
        return await self._rentflow.list_store_skus(request)

    async def _get_store_sku(self, request: StoreSkuInput) -> StoreSku:
        return await self._rentflow.get_store_sku(request)

    async def _list_store_orders(self, request: StoreOrderListInput) -> StoreOrderPage:
        return await self._rentflow.list_store_orders(request)

    async def _get_store_order(self, request: StoreOrderDetailInput) -> StoreOrder:
        return await self._rentflow.get_store_order(request)

    async def _attach_store_skus(self, result: ProductSearchResult) -> ProductSearchResult:
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def attach(item: ProductSummary) -> ProductSummary:
            try:
                async with semaphore:
                    skus = await self._rentflow.list_store_skus(
                        StoreSkuListInput(product_id=item.product_id)
                    )
                return item.model_copy(update={"store_skus": skus.items})
            except (AttributeError, RentFlowError, ValidationError, TypeError, httpx.HTTPError):
                logger.warning("Store SKU enrichment failed for product %s", item.product_id)
                return item

        if not result.items:
            return result
        return result.model_copy(
            update={"items": tuple(await asyncio.gather(*(attach(item) for item in result.items)))}
        )

    @staticmethod
    def _apply_price_preference(
        result: ProductSearchResult,
        request: ProductSearchInput,
    ) -> ProductSearchResult:
        items = result.items
        if request.max_price is not None:
            items = tuple(
                item
                for item in items
                if item.store_skus
                and min(Decimal(sku.sale_price) for sku in item.store_skus) <= request.max_price
            )
        if request.target_price is not None:
            target_price = request.target_price
            original_positions = {item.product_id: index for index, item in enumerate(items)}
            items = tuple(
                sorted(
                    items,
                    key=lambda item: (
                        abs(
                            min(Decimal(sku.sale_price) for sku in item.store_skus)
                            - target_price
                        )
                        if item.store_skus
                        else Decimal("Infinity"),
                        original_positions[item.product_id],
                    ),
                )
            )
        if items == result.items:
            return result
        return result.model_copy(
            update={
                "items": items,
                "total_elements": len(items),
                "total_pages": 1 if items else 0,
            }
        )

    def _apply_user_preferences(
        self,
        result: ProductSearchResult,
        request: ProductSearchInput,
    ) -> ProductSearchResult:
        if request.brand is not None:
            return result
        items = result.items
        if self._excluded_brands:
            items = tuple(
                item for item in items if item.brand.casefold() not in self._excluded_brands
            )
        if self._preferred_brands:
            original_positions = {item.product_id: index for index, item in enumerate(items)}
            items = tuple(
                sorted(
                    items,
                    key=lambda item: (
                        0 if item.brand.casefold() in self._preferred_brands else 1,
                        original_positions[item.product_id],
                    ),
                )
            )
        if items == result.items:
            return result
        return result.model_copy(
            update={
                "items": items,
                "total_elements": len(items),
                "total_pages": 1 if items else 0,
            }
        )

    async def _get_product(self, request: ProductDetailInput) -> BaseModel:
        return await self._rentflow.get_product(request.product_id)

    async def _semantic_products(self, request: ProductSearchInput) -> ProductSearchResult:
        catalog_search = self._catalog_search
        if catalog_search is None or request.semantic_query is None:
            raise ValueError("Semantic catalog search is not configured")
        candidates = await catalog_search.search(
            request.semantic_query,
            equipment_role=request.equipment_role,
            brand=request.brand,
            model=request.model,
            use_case_id=request.use_case_id,
        )
        self._last_search_diagnostics = {
            "mode": "semantic",
            "semanticQuery": request.semantic_query,
            **({"useCaseId": request.use_case_id} if request.use_case_id else {}),
            "candidateCount": len(candidates),
            "candidates": [
                {
                    "productId": candidate.product_id,
                    "score": round(candidate.score, 6),
                    "vectorScore": round(candidate.vector_score, 6),
                    "lexicalScore": round(candidate.lexical_score, 6),
                }
                for candidate in candidates
            ],
        }
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def hydrate(product_id: str) -> ProductSummary:
            async with semaphore:
                detail = await self._rentflow.get_product(product_id)
                return ProductSummary(
                    product_id=detail.product_id,
                    category_id=detail.category_id,
                    equipment_role=detail.equipment_role,
                    name=detail.name,
                    brand=detail.brand,
                    model=detail.model,
                    use_cases=detail.use_cases,
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
