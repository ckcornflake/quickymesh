"""
quickymesh — entry point.

Usage:
    python main.py

Environment variables (override defaults.yaml):
    GEMINI_API_KEY      Required when using Gemini Flash for concept art.
    GEMINI_MODEL        Which Gemini model to use.
    OUTPUT_ROOT         Where to write pipeline data.
    COMFYUI_URL         ComfyUI server URL.
    BLENDER_PATH        Path to Blender executable.
"""

from __future__ import annotations

import logging
import os
import sys

from src.logging_config import configure_logging
configure_logging()

from src.broker import Broker
from src.config import config
from src.agent.pipeline_agent import PipelineAgent
from src.agent.cli_main import run_cli
from src.prompt_interface.cli import CLIPromptInterface
from src.vram_arbiter import VRAMArbiter
from src.workers.comfyui_client import ComfyUIClient
from src.workers.concept_art import (
    ControlNetRestyleWorker,
    FluxComfyUIConceptArtWorker,
    GeminiConceptArtWorker,
)
from src.workers.screenshot import BlenderScreenshotWorker
from src.workers.trellis import ComfyUITrellisWorker


def main() -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    broker = Broker(config.output_root / "tasks.db")
    arbiter = VRAMArbiter()
    ui = CLIPromptInterface()

    comfyui = ComfyUIClient(
        base_url=config.comfyui_url,
        poll_interval=config.comfyui_poll_interval,
        timeout=config.comfyui_timeout,
    )

    # Gemini worker — created lazily: no API key required at startup.
    # If the user selects Gemini for a pipeline and no key is set, the CLI
    # will prompt for it and store it in os.environ before generation begins.
    concept_worker = GeminiConceptArtWorker(
        api_key=os.environ.get("GEMINI_API_KEY"),
        model=config.gemini_model,
    )

    # FLUX.1 [dev] worker — requires ComfyUI + flux1-dev-fp8.safetensors.
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

    try:
        run_cli(
            agent,
            ui,
            config,
            concept_worker=concept_worker,
            trellis_worker=trellis_worker,
            screenshot_worker=screenshot_worker,
            restyle_worker=restyle_worker,
        )
    finally:
        broker.close()


if __name__ == "__main__":
    main()
