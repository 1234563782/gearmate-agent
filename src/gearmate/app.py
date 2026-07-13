from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gearmate import __version__
from gearmate.api.router import api_router
from gearmate.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="GearMate API",
        version=__version__,
        description="Read-only rental assistant service",
    )
    app.state.settings = resolved_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID", "X-Correlation-ID"],
        expose_headers=["X-Correlation-ID"],
    )
    app.include_router(api_router)
    return app
