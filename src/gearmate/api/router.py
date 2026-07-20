from fastapi import APIRouter

from gearmate.api.agent import router as agent_router
from gearmate.api.health import router as health_router
from gearmate.api.user_memory import router as user_memory_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(agent_router)
api_router.include_router(user_memory_router)
