from typing import Any

import httpx

from gearmate.tools.contracts import (
    AvailabilityInput,
    AvailabilityResult,
    CatalogUseCase,
    ProductDetail,
    ProductSearchInput,
    ProductSearchResult,
    QuoteInput,
    QuoteResult,
)


class RentFlowError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class RentFlowClient:
    def __init__(self, client: httpx.AsyncClient, access_token: str) -> None:
        self._client = client
        self._access_token = access_token

    async def search_products(self, request: ProductSearchInput) -> ProductSearchResult:
        params: dict[str, Any] = {
            "keyword": request.keyword,
            "equipmentRole": request.equipment_role,
            "brand": request.brand,
            "model": request.model,
            "useCaseId": request.use_case_id,
            "categoryId": request.category_id,
            "maxDailyRate": request.max_daily_rate,
            "page": request.page,
            "size": request.size,
        }
        if request.rental_period is not None:
            params["startAt"] = request.rental_period.start_at.isoformat()
            params["endAt"] = request.rental_period.end_at.isoformat()
        response = await self._client.get(
            "/api/v1/products",
            params={key: value for key, value in params.items() if value is not None},
        )
        return ProductSearchResult.model_validate(self._payload(response))

    async def list_use_cases(self) -> tuple[CatalogUseCase, ...]:
        response = await self._client.get("/api/v1/catalog/use-cases")
        return tuple(CatalogUseCase.model_validate(item) for item in self._payload(response))

    async def search_availability(self, request: AvailabilityInput) -> AvailabilityResult:
        response = await self._client.post(
            "/api/v1/availability/search",
            json=request.model_dump(mode="json", by_alias=True),
        )
        return AvailabilityResult.model_validate(self._payload(response))

    async def get_product(self, product_id: str) -> ProductDetail:
        response = await self._client.get(f"/api/v1/products/{product_id}")
        return ProductDetail.model_validate(self._payload(response))

    async def create_quote(self, request: QuoteInput) -> QuoteResult:
        response = await self._client.post(
            "/api/v1/quotes",
            json=request.model_dump(mode="json", by_alias=True),
            headers={"Authorization": f"Bearer {self._access_token}"},
        )
        return QuoteResult.model_validate(self._payload(response))

    @staticmethod
    def _payload(response: httpx.Response) -> Any:
        if response.is_success:
            return response.json()
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        code = str(payload.get("code") or "RENTFLOW_REQUEST_FAILED")
        message = str(payload.get("message") or f"RentFlow returned HTTP {response.status_code}")
        raise RentFlowError(code, message, response.status_code)
