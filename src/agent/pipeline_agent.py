"""
Pipeline agent — orchestrates multiple concurrent pipelines.

Responsibilities
----------------
- Start / stop worker threads.
- Start new pipelines (create dir, initial state, enqueue first task).
- Determine which pipeline needs the user's attention most urgently (priority
  prompting).
- Provide the CLI main loop with a list of pipelines and their status.

The agent is intentionally thin: all domain logic lives in the pipeline
modules (concept_art_pipeline, mesh_pipeline, etc.).  The agent's job is to
wire everything together and keep the broker populated.

Pipeline directory layout (relative to cfg.uncompleted_pipelines_dir)
----------------------------------------------------------------------
<pipeline_name>/
    state.json          ← PipelineState saved here
    concept_arts/       ← generated images
    meshes/             ← .glb files
    screenshots/        ← per-mesh subdirs

"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.broker import Broker
    from src.config import Config
    from src.prompt_interface.base import PromptInterface
    from src.vram_arbiter import VRAMArbiter
    from src.workers.concept_art import ConceptArtWorker
    from src.workers.screenshot import ScreenshotWorker
    from src.workers.trellis import TrellisWorker

from src.agent.worker_threads import (
    ConceptArtWorkerThread,
    ScreenshotWorkerThread,
    TrellisWorkerThread,
)
from src.state import (
    ConceptArtStatus,
    MeshStatus,
    PipelineState,
    PipelineStatus,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PipelineAgent
# ---------------------------------------------------------------------------


class PipelineAgent:
    """
    Manages worker threads and pipeline state files.

    Parameters
    ----------
    broker:         The shared SQLite task queue.
    arbiter:        GPU-lock for Trellis / Blender tasks.
    cfg:            Application configuration.
    concept_worker: Worker used to generate / modify concept art.
    trellis_worker: Worker used for mesh gen + texturing.
    screenshot_worker: Worker used for Blender renders.
    poll_interval:  Seconds between worker-thread task-queue polls.
    """

    def __init__(
        self,
        broker: "Broker",
        arbiter: "VRAMArbiter",
        cfg: "Config",
        concept_worker: "ConceptArtWorker",
        trellis_worker: "TrellisWorker",
        screenshot_worker: "ScreenshotWorker",
        *,
        poll_interval: float = 1.0,
    ) -> None:
        self._broker = broker
        self._arbiter = arbiter
        self._cfg = cfg
        self._concept_worker = concept_worker
        self._trellis_worker = trellis_worker
        self._screenshot_worker = screenshot_worker
        self._poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start_workers(self) -> None:
        """Spawn all worker threads (idempotent — safe to call once)."""
        if self._threads:
            return
        self._stop_event.clear()

        self._threads = [
            ConceptArtWorkerThread(
                self._broker,
                self._stop_event,
                self._concept_worker,
                self._cfg,
                poll_interval=self._poll_interval,
            ),
            TrellisWorkerThread(
                self._broker,
                self._stop_event,
                self._trellis_worker,
                self._arbiter,
                self._cfg,
                poll_interval=self._poll_interval,
            ),
            ScreenshotWorkerThread(
                self._broker,
                self._stop_event,
                self._screenshot_worker,
                self._arbiter,
                self._cfg,
                poll_interval=self._poll_interval,
            ),
        ]
        for t in self._threads:
            t.start()
        log.info("Worker threads started (%d threads)", len(self._threads))

    def stop_workers(self, timeout: float = 5.0) -> None:
        """Signal all threads to stop and wait for them to exit."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        log.info("Worker threads stopped")

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def start_pipeline(
        self,
        name: str,
        description: str,
        num_polys: int | None = None,
    ) -> PipelineState:
        """
        Create a new pipeline directory, initial state, and enqueue the
        first concept-art generation task.

        Returns the freshly created PipelineState.
        """
        pipeline_dir = self._cfg.uncompleted_pipelines_dir / name
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        state = PipelineState(
            name=name,
            description=description,
            num_polys=num_polys or self._cfg.num_polys,
            pipeline_dir=str(
                pipeline_dir.relative_to(self._cfg.output_root)
            ),
        )
        state_path = pipeline_dir / "state.json"
        state.save(state_path)

        self._broker.enqueue(
            name,
            "concept_art_generate",
            {"pipeline_name": name, "state_path": str(state_path)},
        )
        log.info("Pipeline '%s' started", name)
        return state

    def get_pipeline_state(self, name: str) -> PipelineState | None:
        """Load and return the current state for a pipeline, or None."""
        state_path = self._cfg.uncompleted_pipelines_dir / name / "state.json"
        if not state_path.exists():
            return None
        return PipelineState.load(state_path)

    def list_pipeline_names(self) -> list[str]:
        """Return names of all pipelines that have a state.json."""
        root = self._cfg.uncompleted_pipelines_dir
        if not root.exists():
            return []
        return sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / "state.json").exists()
        )

    def enqueue_mesh_generation(self, pipeline_name: str) -> None:
        """Enqueue a mesh_generate task for a pipeline (if not already queued)."""
        if self._broker.has_pending_or_running(pipeline_name, "mesh_generate"):
            return
        state_path = self._cfg.uncompleted_pipelines_dir / pipeline_name / "state.json"
        self._broker.enqueue(
            pipeline_name,
            "mesh_generate",
            {"pipeline_name": pipeline_name, "state_path": str(state_path)},
        )

    def enqueue_mesh_texturing(self, pipeline_name: str) -> None:
        """Enqueue a mesh_texture task (if not already queued)."""
        if self._broker.has_pending_or_running(pipeline_name, "mesh_texture"):
            return
        state_path = self._cfg.uncompleted_pipelines_dir / pipeline_name / "state.json"
        self._broker.enqueue(
            pipeline_name,
            "mesh_texture",
            {"pipeline_name": pipeline_name, "state_path": str(state_path)},
        )

    def enqueue_screenshots(self, pipeline_name: str) -> None:
        """Enqueue a screenshot task (if not already queued)."""
        if self._broker.has_pending_or_running(pipeline_name, "screenshot"):
            return
        state_path = self._cfg.uncompleted_pipelines_dir / pipeline_name / "state.json"
        self._broker.enqueue(
            pipeline_name,
            "screenshot",
            {"pipeline_name": pipeline_name, "state_path": str(state_path)},
        )

    def cancel_pipeline(self, name: str) -> int:
        """Cancel all pending/running broker tasks for a pipeline."""
        return self._broker.cancel_pipeline_tasks(name)

    # ------------------------------------------------------------------
    # Priority prompting
    # ------------------------------------------------------------------

    def highest_priority_pipeline(self) -> str | None:
        """
        Return the name of the pipeline that most urgently needs user input,
        or None if no pipeline is waiting for the user.

        Priority order (highest first):
        1. CONCEPT_ART_REVIEW — user needs to approve concept art.
        2. MESH_REVIEW        — user needs to approve a mesh.
        3. None               — all pipelines are either running or done.
        """
        names = self.list_pipeline_names()
        for status in (PipelineStatus.CONCEPT_ART_REVIEW, PipelineStatus.MESH_REVIEW):
            for name in names:
                state = self.get_pipeline_state(name)
                if state and state.status == status:
                    return name
        return None

    def pipelines_needing_attention(self) -> list[str]:
        """Return all pipeline names that are waiting for user input."""
        names = self.list_pipeline_names()
        result = []
        review_statuses = {PipelineStatus.CONCEPT_ART_REVIEW, PipelineStatus.MESH_REVIEW}
        for name in names:
            state = self.get_pipeline_state(name)
            if state and state.status in review_statuses:
                result.append(name)
        return result
