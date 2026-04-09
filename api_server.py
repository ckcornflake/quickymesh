"""
quickymesh API server entry point.

Usage
-----
    python api_server.py                          # default: localhost:8000, auth OFF
    python api_server.py --host 0.0.0.0 --port 8080
    python api_server.py --auth-file users.yaml   # enable bearer-token auth
    uvicorn api_server:app --reload               # dev mode

Auth is OFF by default — every request is treated as a local admin user.
This is the right default for single-user / localhost setups.  Pass
``--auth-file PATH`` to enable bearer-token auth backed by a users.yaml file.

Environment variables (same as main.py, plus):
    QUICKYMESH_AUTH_FILE   Path to users.yaml — if set, auth is enabled
                           (equivalent to passing --auth-file)
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


def build_app(*, auth_file: str | None = None):
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
        restyle_worker=restyle_worker,
        trellis_worker=trellis_worker,
        screenshot_worker=screenshot_worker,
    )

    app = create_app(
        agent=agent,
        cfg=config,
        concept_worker=concept_worker,
        restyle_worker=restyle_worker,
        users_file=auth_file,
    )

    # Start workers inside the app lifespan — attach agent to app so we can
    # start/stop in the lifespan context manager if desired.  For simplicity
    # in Phase 1, start workers immediately and rely on the broker for recovery.
    agent.start_workers()
    log.info("Worker threads started")

    return app


# Build the app at module level so `uvicorn api_server:app` works.
# Honour QUICKYMESH_AUTH_FILE so module-level loading still respects auth.
app = build_app(auth_file=os.environ.get("QUICKYMESH_AUTH_FILE"))

# Single-user fallback: if API_KEY is set (e.g. in Docker via docker/.env) but
# no users.yaml was provided, enable auth with that key.  load_users() picks
# up API_KEY from the environment when no users file exists.
if not os.environ.get("QUICKYMESH_AUTH_FILE") and os.environ.get("API_KEY"):
    from src.api.auth import load_users, set_auth_enabled
    load_users(None)
    set_auth_enabled(True)
    log.info("Auth ENABLED via API_KEY env var (single-user mode)")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="quickymesh API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (dev only)")
    parser.add_argument(
        "--auth-file",
        default=None,
        help="Path to users.yaml.  If supplied, bearer-token auth is enabled. "
             "Default is no auth (all requests treated as local admin).",
    )
    args = parser.parse_args()

    if args.auth_file:
        # Apply to the already-built module-level app.
        from src.api.auth import load_users, set_auth_enabled
        load_users(args.auth_file)
        set_auth_enabled(True)
        log.info("Auth ENABLED via --auth-file %s", args.auth_file)

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,  # use our logging_config instead of uvicorn's default
    )
