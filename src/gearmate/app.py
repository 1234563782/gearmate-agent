from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gearmate import __version__
from gearmate.agent.service import RunCoordinator
from gearmate.api.router import api_router
from gearmate.config import Settings, get_settings
from gearmate.persistence.database import Database
from gearmate.persistence.repositories import AgentRepository
from gearmate.prompts.loader import load_system_prompt


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(resolved_settings)
        repository = AgentRepository(database.session_factory)
        stale_before = datetime.now(UTC) - timedelta(
            seconds=resolved_settings.run_timeout_seconds * 2
        )
        await repository.fail_stale_active_runs(stale_before)
        timeout = httpx.Timeout(
            connect=resolved_settings.rentflow_connect_timeout_seconds,
            read=resolved_settings.rentflow_read_timeout_seconds,
            write=resolved_settings.rentflow_read_timeout_seconds,
            pool=resolved_settings.rentflow_connect_timeout_seconds,
        )
        rentflow_http = httpx.AsyncClient(
            base_url=resolved_settings.rentflow_base_url.rstrip("/"),
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        run_coordinator = RunCoordinator(
            resolved_settings,
            repository,
            rentflow_http,
            load_system_prompt(),
        )
        app.state.database = database
        app.state.repository = repository
        app.state.run_coordinator = run_coordinator
        try:
            yield
        finally:
            await run_coordinator.close()
            await rentflow_http.aclose()
            await database.dispose()

    app = FastAPI(
        title="GearMate API",
        version=__version__,
        description="Read-only rental assistant service",
        lifespan=lifespan,
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
