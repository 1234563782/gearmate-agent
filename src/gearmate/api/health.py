from typing import Literal, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from gearmate.config import Settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["gearmate-agent"]
    environment: Literal["development", "test", "production"]


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = cast(Settings, request.app.state.settings)
    return HealthResponse(
        status="ok",
        service="gearmate-agent",
        environment=settings.environment,
    )
