"""
quickymesh API server entry point.

Usage
-----
    python api_server.py                          # default: localhost:8000
    python api_server.py --host 0.0.0.0 --port 8080
    uvicorn api_server:app --reload               # dev mode

Environment variables (same as main.py, plus):
    API_KEY              Single-user API key (if users.yaml not present)
    QUICKYMESH_USERS_FILE  Path to users.yaml (default: ./users.yaml)
"""
from __future__ import annotations

import argparse
import logging
import os

from src.logging_config import configure_logging
configure_logging()

from src.broker import Broker
from src.config import config
from src.agent.pipeline_agent import PipelineAgent
from src.api.app import create_app
from src.vram_arbiter import VRAMArbiter
from src.workers.comfyui_client import ComfyUIClient
from src.workers.concept_art import (
    ControlNetRestyleWorker,
    FluxComfyUIConceptArtWorker,
    GeminiConceptArtWorker,
)
from src.workers.screenshot import BlenderScreenshotWorker
from src.workers.trellis import ComfyUITrellisWorker

log = logging.getLogger(__name__)


def build_app():
    """Build the FastAPI app with all workers initialised."""
    config.output_root.mkdir(parents=True, exist_ok=True)
    broker = Broker(config.output_root / "tasks.db")
    arbiter = VRAMArbiter()

    comfyui = ComfyUIClient(
        base_url=config.comfyui_url,
        poll_interval=config.comfyui_poll_interval,
        timeout=config.comfyui_timeout,
    )

    concept_worker = GeminiConceptArtWorker(
        api_key=os.environ.get("GEMINI_API_KEY"),
        model=config.gemini_model,
    )

    flux_concept_worker = FluxComfyUIConceptArtWorker(
        client=comfyui,
        comfyui_output_dir=config.comfyui_output_dir,
        workflow_path=config.workflow_flux_generate,
        image_size=config.concept_art_image_size,
        arbiter=arbiter,
        vram_lock_timeout=config.vram_lock_timeout,
    )

    restyle_worker = ControlNetRestyleWorker(
        client=comfyui,
        workflow_path=config.workflow_controlnet_restyle,
        arbiter=arbiter,
        vram_lock_timeout=config.vram_lock_timeout,
    )

    trellis_worker = ComfyUITrellisWorker(
        client=comfyui,
        comfyui_output_dir=config.comfyui_output_dir,
        workflow_generate=config.workflow_generate,
        workflow_texture=config.workflow_texture,
        arbiter=arbiter,
    )

    screenshot_worker = BlenderScreenshotWorker(
        blender_path=config.blender_path,
    )

    agent = PipelineAgent(
        broker=broker,
        arbiter=arbiter,
        cfg=config,
        concept_worker=concept_worker,
        flux_concept_worker=flux_concept_worker,
        trellis_worker=trellis_worker,
        screenshot_worker=screenshot_worker,
    )

    app = create_app(
        agent=agent,
        cfg=config,
        concept_worker=concept_worker,
        restyle_worker=restyle_worker,
    )

    # Start workers inside the app lifespan — attach agent to app so we can
    # start/stop in the lifespan context manager if desired.  For simplicity
    # in Phase 1, start workers immediately and rely on the broker for recovery.
    agent.start_workers()
    log.info("Worker threads started")

    return app


# Build the app at module level so `uvicorn api_server:app` works
app = build_app()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="quickymesh API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (dev only)")
    args = parser.parse_args()

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,  # use our logging_config instead of uvicorn's default
    )
