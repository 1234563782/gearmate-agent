from typing import Any

import httpx

from gearmate.tools.contracts import (
    CatalogUseCase,
    ProductDetail,
    ProductSearchInput,
    ProductSearchResult,
    StoreOrder,
    StoreOrderDetailInput,
    StoreOrderListInput,
    StoreOrderPage,
    StoreSku,
    StoreSkuInput,
    StoreSkuList,
    StoreSkuListInput,
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
            "page": request.page,
            "size": request.size,
        }
        response = await self._client.get(
            "/api/v1/products",
            params={key: value for key, value in params.items() if value is not None},
        )
        payload = self._payload(response)
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            payload = {
                **payload,
                "items": [self._commerce_product(item) for item in payload["items"]],
            }
        return ProductSearchResult.model_validate(payload)

    async def list_use_cases(self) -> tuple[CatalogUseCase, ...]:
        response = await self._client.get("/api/v1/catalog/use-cases")
        return tuple(CatalogUseCase.model_validate(item) for item in self._payload(response))

    async def get_product(self, product_id: str) -> ProductDetail:
        response = await self._client.get(f"/api/v1/products/{product_id}")
        return ProductDetail.model_validate(self._commerce_product(self._payload(response)))

    async def list_store_skus(self, request: StoreSkuListInput) -> StoreSkuList:
        response = await self._client.get(
            f"/api/v1/store/products/{request.product_id}/skus"
        )
        return StoreSkuList(
            product_id=request.product_id,
            items=tuple(StoreSku.model_validate(item) for item in self._payload(response)),
        )

    async def get_store_sku(self, request: StoreSkuInput) -> StoreSku:
        response = await self._client.get(f"/api/v1/store/skus/{request.sku_id}")
        return StoreSku.model_validate(self._payload(response))

    async def list_store_orders(self, request: StoreOrderListInput) -> StoreOrderPage:
        response = await self._client.get(
            "/api/v1/store/orders",
            params={
                key: value
                for key, value in {
                    "status": request.status,
                    "page": request.page,
                    "size": request.size,
                }.items()
                if value is not None
            },
            headers={"Authorization": f"Bearer {self._access_token}"},
        )
        return StoreOrderPage.model_validate(self._payload(response))

    async def get_store_order(self, request: StoreOrderDetailInput) -> StoreOrder:
        response = await self._client.get(
            f"/api/v1/store/orders/{request.order_id}",
            headers={"Authorization": f"Bearer {self._access_token}"},
        )
        return StoreOrder.model_validate(self._payload(response))

    @staticmethod
    def _commerce_product(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        allowed = {
            "productId",
            "categoryId",
            "equipmentRole",
            "name",
            "brand",
            "model",
            "description",
            "useCases",
            "storeSkus",
        }
        return {key: value for key, value in payload.items() if key in allowed}

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
