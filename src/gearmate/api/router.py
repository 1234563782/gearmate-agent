from fastapi import APIRouter

from gearmate.api.agent import router as agent_router
from gearmate.api.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(agent_router)
