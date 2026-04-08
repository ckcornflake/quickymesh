"""
FastAPI application factory for quickymesh.

Usage
-----
    from src.api.app import create_app
    app = create_app(agent, cfg, concept_worker, restyle_worker)
    # then: uvicorn.run(app, ...)
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from src.api.auth import load_users
from src.api.event_bus import event_bus

if TYPE_CHECKING:
    from src.agent.pipeline_agent import PipelineAgent
    from src.config import Config
    from src.workers.concept_art import ConceptArtWorker

log = logging.getLogger(__name__)


def create_app(
    agent: "PipelineAgent",
    cfg: "Config",
    concept_worker: "ConceptArtWorker",
    restyle_worker=None,
    *,
    users_file: str | None = None,
) -> FastAPI:
    """
    Build and return the FastAPI application.

    Parameters
    ----------
    agent:          The PipelineAgent (workers started separately by the caller).
    cfg:            Application config.
    concept_worker: The active concept art worker (Gemini or FLUX).
    restyle_worker: Optional ControlNetRestyleWorker.
    users_file:     Path to users.yaml.  Defaults to auto-detection in auth.py.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Give the event bus the running loop so worker threads can post to it.
        event_bus.set_loop(asyncio.get_running_loop())
        log.info("Event bus initialised")
        yield
        log.info("API server shutting down")

    app = FastAPI(
        title="quickymesh API",
        version="1.0.0",
        description="3-D asset generation pipeline API",
        lifespan=lifespan,
    )

    # Load authentication users
    load_users(users_file)

    # Store shared state on app so routers can access it via request.app.state
    app.state.agent = agent
    app.state.cfg = cfg
    app.state.concept_worker = concept_worker
    app.state.restyle_worker = restyle_worker

    # Register routers
    from src.api.routers import assets, events, pipelines, pipelines_3d, review, status

    app.include_router(pipelines.router, prefix="/api/v1")
    app.include_router(pipelines_3d.router, prefix="/api/v1")
    app.include_router(review.router, prefix="/api/v1")
    app.include_router(assets.router, prefix="/api/v1")
    app.include_router(events.router, prefix="/api/v1")
    app.include_router(status.router, prefix="/api/v1")

    return app
