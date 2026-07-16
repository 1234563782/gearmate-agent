import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gearmate import __version__
from gearmate.agent.service import RunCoordinator
from gearmate.api.router import api_router
from gearmate.catalog import CatalogSearchRepository, CatalogSearchService
from gearmate.config import Settings, get_settings
from gearmate.embeddings import build_embedding_model
from gearmate.persistence.database import Database
from gearmate.persistence.repositories import AgentRepository
from gearmate.prompts.loader import load_system_prompt
from gearmate.rentflow.client import RentFlowClient

logger = logging.getLogger(__name__)


async def catalog_sync_loop(
    catalog_search: CatalogSearchService,
    rentflow: RentFlowClient,
    settings: Settings,
) -> None:
    delay = settings.catalog_sync_interval_seconds
    while True:
        await asyncio.sleep(delay)
        try:
            stats = await catalog_search.refresh(rentflow)
            logger.info("Catalog semantic index refreshed: %s", stats)
            delay = settings.catalog_sync_interval_seconds
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Catalog semantic index refresh failed")
            delay = settings.catalog_sync_retry_seconds


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
        embedding_model = build_embedding_model(resolved_settings)
        catalog_search = None
        catalog_sync_task: asyncio.Task[None] | None = None
        if embedding_model is not None:
            catalog_search = CatalogSearchService(
                CatalogSearchRepository(database.session_factory),
                embedding_model,
                batch_size=resolved_settings.embedding_batch_size,
                top_k=resolved_settings.semantic_search_top_k,
                max_concurrency=resolved_settings.max_tool_concurrency,
                min_score=resolved_settings.semantic_search_min_score,
                vector_weight=resolved_settings.semantic_vector_weight,
                lexical_weight=resolved_settings.semantic_lexical_weight,
            )
            if resolved_settings.catalog_sync_on_startup:
                try:
                    stats = await catalog_search.refresh(RentFlowClient(rentflow_http, ""))
                    logger.info("Catalog semantic index refreshed: %s", stats)
                except Exception:
                    logger.exception("Catalog semantic index refresh failed")
            catalog_sync_task = asyncio.create_task(
                catalog_sync_loop(
                    catalog_search,
                    RentFlowClient(rentflow_http, ""),
                    resolved_settings,
                ),
                name="gearmate-catalog-sync",
            )
        run_coordinator = RunCoordinator(
            resolved_settings,
            repository,
            rentflow_http,
            load_system_prompt(),
            catalog_search,
        )
        app.state.database = database
        app.state.repository = repository
        app.state.run_coordinator = run_coordinator
        app.state.catalog_search = catalog_search
        try:
            yield
        finally:
            await run_coordinator.close()
            if catalog_sync_task is not None:
                catalog_sync_task.cancel()
                await asyncio.gather(catalog_sync_task, return_exceptions=True)
            if embedding_model is not None:
                await embedding_model.close()
            await rentflow_http.aclose()
            await database.dispose()

    app = FastAPI(
        title="GearMate API",
        version=__version__,
        description="Electronic equipment rental selection and quote assistant service",
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
