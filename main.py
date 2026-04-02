"""
quickymesh — entry point.

Usage:
    python main.py

Environment variables (override defaults.yaml):
    GEMINI_API_KEY      Required.
    GEMINI_MODEL        Which Gemini model to use.
    OUTPUT_ROOT         Where to write pipeline data.
    COMFYUI_URL         ComfyUI server URL.
    BLENDER_PATH        Path to Blender executable.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

from src.broker import Broker
from src.config import config
from src.agent.pipeline_agent import PipelineAgent
from src.agent.cli_main import run_cli
from src.prompt_interface.cli import CLIPromptInterface
from src.vram_arbiter import VRAMArbiter
from src.workers.comfyui_client import ComfyUIClient
from src.workers.concept_art import GeminiConceptArtWorker
from src.workers.screenshot import BlenderScreenshotWorker
from src.workers.trellis import ComfyUITrellisWorker


def main() -> None:
    broker = Broker(config.output_root / "tasks.db")
    arbiter = VRAMArbiter()
    ui = CLIPromptInterface()

    comfyui = ComfyUIClient(
        base_url=config.comfyui_url,
        poll_interval=config.comfyui_poll_interval,
        timeout=config.comfyui_timeout,
    )
    concept_worker = GeminiConceptArtWorker(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
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
        )
    finally:
        broker.close()


if __name__ == "__main__":
    main()
