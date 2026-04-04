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
        flux_concept_worker: "ConceptArtWorker | None" = None,
        poll_interval: float = 1.0,
    ) -> None:
        self._broker = broker
        self._arbiter = arbiter
        self._cfg = cfg
        self._concept_worker = concept_worker
        self._flux_concept_worker = flux_concept_worker
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
                flux_worker=self._flux_concept_worker,
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
        self.recover_stalled_pipelines()

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
        *,
        input_image_path: str | None = None,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
        concept_art_backend: str = "gemini",
    ) -> PipelineState:
        """
        Create a new pipeline directory, initial state, and enqueue the
        first concept-art generation task.

        Returns the freshly created PipelineState.
        """
        from src.state import SymmetryAxis
        pipeline_dir = self._cfg.uncompleted_pipelines_dir / name
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        state = PipelineState(
            name=name,
            description=description,
            num_polys=num_polys or self._cfg.num_polys,
            input_image_path=input_image_path,
            symmetrize=symmetrize,
            symmetry_axis=SymmetryAxis(symmetry_axis),
            concept_art_backend=concept_art_backend,
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

    def pause_pipeline(self, name: str, cfg=None) -> None:
        """
        Pause a pipeline: cancel in-flight broker tasks and set status to PAUSED.
        The state file is preserved so the pipeline can be resumed.
        """
        self._broker.cancel_pipeline_tasks(name)
        cfg = cfg or self._cfg
        state_path = cfg.uncompleted_pipelines_dir / name / "state.json"
        if state_path.exists():
            state = PipelineState.load(state_path)
            state.status = PipelineStatus.PAUSED
            state.save(state_path)
            log.info("Pipeline '%s' paused", name)

    def resume_pipeline(self, name: str) -> None:
        """
        Resume a paused pipeline by re-enqueueing the appropriate next task,
        using the same logic as recover_stalled_pipelines.
        """
        state = self.get_pipeline_state(name)
        if state is None:
            return
        if state.status != PipelineStatus.PAUSED:
            log.warning("resume_pipeline called on '%s' which is not paused (status=%s)", name, state.status)
            return
        # Restore to the status it had before pausing by inspecting mesh/art progress
        state_path = self._cfg.uncompleted_pipelines_dir / name / "state.json"
        payload = {"pipeline_name": name, "state_path": str(state_path)}

        # Figure out where the pipeline was based on mesh item statuses
        if not state.meshes:
            # No meshes — was in concept art or mesh-not-started phase
            from src.state import ConceptArtStatus
            any_approved = any(ca.status == ConceptArtStatus.APPROVED for ca in state.concept_arts)
            if any_approved:
                state.status = PipelineStatus.MESH_GENERATING
                state.save(state_path)
                if not self._broker.has_pending_or_running(name, "mesh_generate"):
                    self._broker.enqueue(name, "mesh_generate", payload)
            else:
                state.status = PipelineStatus.CONCEPT_ART_GENERATING
                state.save(state_path)
                if not self._broker.has_pending_or_running(name, "concept_art_generate"):
                    self._broker.enqueue(name, "concept_art_generate", payload)
        else:
            # Meshes exist — restore mesh_generating and re-enqueue the right step
            state.status = PipelineStatus.MESH_GENERATING
            state.save(state_path)
            mesh_statuses = {m.status for m in state.meshes}
            if MeshStatus.CLEANUP_DONE in mesh_statuses or MeshStatus.SCREENSHOT_DONE in mesh_statuses:
                if not self._broker.has_pending_or_running(name, "screenshot"):
                    self._broker.enqueue(name, "screenshot", payload)
            elif MeshStatus.TEXTURE_DONE in mesh_statuses:
                if not self._broker.has_pending_or_running(name, "mesh_cleanup"):
                    self._broker.enqueue(name, "mesh_cleanup", payload)
            elif MeshStatus.MESH_DONE in mesh_statuses:
                if not self._broker.has_pending_or_running(name, "mesh_texture"):
                    self._broker.enqueue(name, "mesh_texture", payload)
            else:
                if not self._broker.has_pending_or_running(name, "mesh_generate"):
                    self._broker.enqueue(name, "mesh_generate", payload)
        log.info("Pipeline '%s' resumed", name)

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

    def recover_stalled_pipelines(self) -> None:
        """
        At startup, inspect each pipeline's state.json and re-enqueue the
        appropriate broker task if none is already in flight.

        This handles two crash scenarios:
          1. tasks.db was lost/wiped — pipeline state exists but broker has no
             record of any task.
          2. A task completed (status='done') but the worker crashed before
             enqueueing the next step.
        """
        for name in self.list_pipeline_names():
            state = self.get_pipeline_state(name)
            if state is None:
                continue
            state_path = self._cfg.uncompleted_pipelines_dir / name / "state.json"
            payload = {"pipeline_name": name, "state_path": str(state_path)}

            if state.status in (PipelineStatus.PAUSED, PipelineStatus.CANCELLED):
                continue  # paused/cancelled — leave alone until user acts

            if state.status == PipelineStatus.CONCEPT_ART_GENERATING:
                if not self._broker.has_pending_or_running(name, "concept_art_generate"):
                    log.info("Recovery: re-enqueueing concept_art_generate for '%s'", name)
                    self._broker.enqueue(name, "concept_art_generate", payload)

            elif state.status == PipelineStatus.MESH_GENERATING:
                if not state.meshes:
                    # No meshes at all — re-enqueue mesh_generate
                    if not self._broker.has_pending_or_running(name, "mesh_generate"):
                        log.info("Recovery: re-enqueueing mesh_generate for '%s'", name)
                        self._broker.enqueue(name, "mesh_generate", payload)
                else:
                    mesh_statuses = {m.status for m in state.meshes}
                    if MeshStatus.CLEANUP_DONE in mesh_statuses or MeshStatus.SCREENSHOT_DONE in mesh_statuses:
                        # Cleanup done, screenshots missing
                        if not self._broker.has_pending_or_running(name, "screenshot"):
                            log.info("Recovery: re-enqueueing screenshot for '%s'", name)
                            self._broker.enqueue(name, "screenshot", payload)
                    elif MeshStatus.TEXTURE_DONE in mesh_statuses:
                        # Texturing done, cleanup missing
                        if not self._broker.has_pending_or_running(name, "mesh_cleanup"):
                            log.info("Recovery: re-enqueueing mesh_cleanup for '%s'", name)
                            self._broker.enqueue(name, "mesh_cleanup", payload)
                    elif MeshStatus.MESH_DONE in mesh_statuses:
                        # Mesh generated, texturing missing
                        if not self._broker.has_pending_or_running(name, "mesh_texture"):
                            log.info("Recovery: re-enqueueing mesh_texture for '%s'", name)
                            self._broker.enqueue(name, "mesh_texture", payload)
                    else:
                        # No meshes with any completion status — re-run mesh_generate
                        if not self._broker.has_pending_or_running(name, "mesh_generate"):
                            log.info("Recovery: re-enqueueing mesh_generate for '%s'", name)
                            self._broker.enqueue(name, "mesh_generate", payload)

            elif state.status == PipelineStatus.APPROVED:
                # Review completed but export never ran (e.g. process killed after review)
                if any(m.status == MeshStatus.APPROVED for m in state.meshes):
                    from src.mesh_pipeline import run_mesh_export
                    pipeline_dir = self._cfg.uncompleted_pipelines_dir / name
                    log.info("Recovery: exporting approved meshes for '%s'", name)
                    run_mesh_export(state, pipeline_dir, self._cfg)
