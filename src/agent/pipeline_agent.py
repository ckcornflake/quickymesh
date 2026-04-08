"""
Pipeline agent — orchestrates multiple concurrent 2D and 3D pipelines.

Responsibilities
----------------
- Start / stop worker threads.
- Start new 2D pipelines (create dir, initial state, enqueue first task).
- Start new 3D pipelines (from a 2D concept art image or an uploaded image).
- Provide the CLI main loop with a list of pipelines and their status.

The agent is intentionally thin: all domain logic lives in the pipeline
modules (concept_art_pipeline, mesh_pipeline, etc.).

Pipeline directory layout (all under cfg.pipelines_dir)
--------------------------------------------------------
{pipeline_name}/
    state.json          ← PipelineState (2D) or Pipeline3DState (3D)
    concept_arts/       ← 2D pipelines only: generated images
    meshes/             ← 3D pipelines: .glb files
    screenshots/        ← 3D pipelines: per-pipeline screenshots

3D pipeline naming:
    Derived from 2D:  {2d_name}_{ca_index}_{ca_version}  e.g. "ship_1_0"
    From file upload: u_{user_name}                        e.g. "u_myship"
"""

from __future__ import annotations

import logging
import threading
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
    Pipeline3DState,
    Pipeline3DStatus,
    PipelineState,
    PipelineStatus,
    SymmetryAxis,
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

        # Session-only set of pipelines the CLI has already surfaced for review
        # this session but the user declined to act on (e.g. typed a non-action
        # in the review loop, or approved without submitting for 3D). Prevents
        # the priority loop from re-surfacing the same pipeline on every tick.
        # Cleared on process restart; the user can re-enter a dismissed pipeline
        # via the "Return to a pipeline" menu option.
        self._dismissed_from_priority: set[str] = set()

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
    # 2D pipeline management
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
        Create a new 2D pipeline directory, initial state, and enqueue the
        first concept-art generation task.

        Returns the freshly created PipelineState.
        """
        pipeline_dir = self._cfg.pipelines_dir / name
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
        log.info("2D pipeline '%s' started", name)
        return state

    def get_pipeline_state(self, name: str) -> PipelineState | None:
        """Load and return the current 2D state for a pipeline, or None."""
        state_path = self._cfg.pipelines_dir / name / "state.json"
        if not state_path.exists():
            return None
        try:
            state = PipelineState.load(state_path)
            # Filter out 3D pipelines (they also have state.json but are Pipeline3DState)
            if isinstance(state, PipelineState) and not hasattr(state, "source_2d_pipeline"):
                return state
            return state
        except Exception:
            return None

    def list_pipeline_names(self) -> list[str]:
        """Return names of all 2D pipelines (have PipelineState)."""
        root = self._cfg.pipelines_dir
        if not root.exists():
            return []
        results = []
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            state_path = p / "state.json"
            if not state_path.exists():
                continue
            try:
                import json
                data = json.loads(state_path.read_text(encoding="utf-8"))
                # 2D pipelines have concept_arts; 3D have input_image_path at top level
                # Distinguish by presence of 'concept_art_backend' field
                if "concept_art_backend" in data:
                    results.append(p.name)
            except Exception:
                pass
        return results

    def pipeline_name_exists(self, name: str) -> bool:
        """Return True if any pipeline (2D or 3D) uses this folder name."""
        return (self._cfg.pipelines_dir / name).exists()

    # ------------------------------------------------------------------
    # 3D pipeline management
    # ------------------------------------------------------------------

    def start_3d_pipeline(
        self,
        name: str,
        input_image_path: str,
        num_polys: int | None = None,
        *,
        source_2d_pipeline: str | None = None,
        source_concept_art_index: int | None = None,
        source_concept_art_version: int | None = None,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
    ) -> Pipeline3DState:
        """
        Create a new 3D pipeline directory, initial state, and enqueue
        the first mesh_generate task.

        `name` should follow the naming convention:
          - Derived from 2D:  "{2d_name}_{ca_index}_{ca_version}"
          - From file upload: "u_{user_name}"
        """
        pipeline_dir = self._cfg.pipelines_dir / name
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        state = Pipeline3DState(
            name=name,
            input_image_path=input_image_path,
            num_polys=num_polys or self._cfg.num_polys,
            source_2d_pipeline=source_2d_pipeline,
            source_concept_art_index=source_concept_art_index,
            source_concept_art_version=source_concept_art_version,
            symmetrize=symmetrize,
            symmetry_axis=SymmetryAxis(symmetry_axis),
            pipeline_dir=str(
                pipeline_dir.relative_to(self._cfg.output_root)
            ),
        )
        state_path = pipeline_dir / "state.json"
        state.save(state_path)

        self._broker.enqueue(
            name,
            "mesh_generate",
            {"pipeline_name": name, "state_path": str(state_path)},
        )
        log.info("3D pipeline '%s' started", name)
        return state

    def get_3d_pipeline_state(self, name: str) -> Pipeline3DState | None:
        """Load and return the current state for a 3D pipeline, or None."""
        state_path = self._cfg.pipelines_dir / name / "state.json"
        if not state_path.exists():
            return None
        try:
            return Pipeline3DState.load(state_path)
        except Exception as exc:
            log.warning(
                "get_3d_pipeline_state('%s'): failed to load state — %s: %s",
                name, type(exc).__name__, exc,
            )
            return None

    def list_3d_pipeline_names(self) -> list[str]:
        """Return names of all 3D pipelines (have Pipeline3DState)."""
        root = self._cfg.pipelines_dir
        if not root.exists():
            return []
        results = []
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            state_path = p / "state.json"
            if not state_path.exists():
                continue
            try:
                import json
                data = json.loads(state_path.read_text(encoding="utf-8"))
                if "concept_art_backend" not in data and "input_image_path" in data:
                    results.append(p.name)
            except Exception:
                pass
        return results

    def enqueue_mesh_generation(self, pipeline_name: str) -> None:
        """Enqueue a mesh_generate task for a 3D pipeline (if not already queued)."""
        if self._broker.has_pending_or_running(pipeline_name, "mesh_generate"):
            return
        state_path = self._cfg.pipelines_dir / pipeline_name / "state.json"
        self._broker.enqueue(
            pipeline_name,
            "mesh_generate",
            {"pipeline_name": pipeline_name, "state_path": str(state_path)},
        )

    def enqueue_mesh_texturing(self, pipeline_name: str) -> None:
        """Enqueue a mesh_texture task (if not already queued)."""
        if self._broker.has_pending_or_running(pipeline_name, "mesh_texture"):
            return
        state_path = self._cfg.pipelines_dir / pipeline_name / "state.json"
        self._broker.enqueue(
            pipeline_name,
            "mesh_texture",
            {"pipeline_name": pipeline_name, "state_path": str(state_path)},
        )

    def enqueue_screenshots(self, pipeline_name: str) -> None:
        """Enqueue a screenshot task (if not already queued)."""
        if self._broker.has_pending_or_running(pipeline_name, "screenshot"):
            return
        state_path = self._cfg.pipelines_dir / pipeline_name / "state.json"
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
        Pause a 2D pipeline: cancel in-flight broker tasks and set status PAUSED.
        """
        self._broker.cancel_pipeline_tasks(name)
        cfg = cfg or self._cfg
        state_path = cfg.pipelines_dir / name / "state.json"
        if state_path.exists():
            state = PipelineState.load(state_path)
            state.status = PipelineStatus.PAUSED
            state.save(state_path)
            log.info("Pipeline '%s' paused", name)

    def resume_pipeline(self, name: str) -> None:
        """
        Resume a paused 2D pipeline by re-enqueueing the concept art task.
        """
        state = self.get_pipeline_state(name)
        if state is None:
            return
        if state.status != PipelineStatus.PAUSED:
            log.warning("resume_pipeline called on '%s' which is not paused", name)
            return
        state_path = self._cfg.pipelines_dir / name / "state.json"
        payload = {"pipeline_name": name, "state_path": str(state_path)}
        state.status = PipelineStatus.CONCEPT_ART_GENERATING
        state.save(state_path)
        if not self._broker.has_pending_or_running(name, "concept_art_generate"):
            self._broker.enqueue(name, "concept_art_generate", payload)
        log.info("Pipeline '%s' resumed", name)

    # ------------------------------------------------------------------
    # Priority prompting (2D pipelines)
    # ------------------------------------------------------------------

    def dismiss_from_priority(self, name: str) -> None:
        """
        Mark a pipeline as handled for this session so the priority loop
        stops re-surfacing it. The user can still revisit it via the
        "Return to a pipeline" menu option. Cleared on restart.
        """
        self._dismissed_from_priority.add(name)

    def undismiss_from_priority(self, name: str) -> None:
        """Remove a pipeline from the dismissed set (e.g. on explicit Return)."""
        self._dismissed_from_priority.discard(name)

    def is_dismissed_from_priority(self, name: str) -> bool:
        """Return True if this pipeline has been dismissed for the session."""
        return name in self._dismissed_from_priority

    def highest_priority_pipeline(self) -> str | None:
        """
        Return the name of the 2D pipeline that most urgently needs user input,
        or None if no pipeline is waiting for the user.
        """
        names = self.list_pipeline_names()
        for name in names:
            state = self.get_pipeline_state(name)
            if state and state.status == PipelineStatus.CONCEPT_ART_REVIEW:
                return name
        return None

    def pipelines_needing_attention(self) -> list[str]:
        """Return all 2D pipeline names waiting for user input."""
        names = self.list_pipeline_names()
        return [
            name for name in names
            if (s := self.get_pipeline_state(name))
            and s.status == PipelineStatus.CONCEPT_ART_REVIEW
        ]

    def recover_stalled_pipelines(self) -> None:
        """
        At startup, inspect each pipeline's state.json and re-enqueue the
        appropriate broker task if none is already in flight.

        Handles 2D pipelines (concept art phase) and 3D pipelines (mesh phase).
        """
        # 2D pipelines
        for name in self.list_pipeline_names():
            state = self.get_pipeline_state(name)
            if state is None:
                continue
            state_path = self._cfg.pipelines_dir / name / "state.json"
            payload = {"pipeline_name": name, "state_path": str(state_path)}

            if state.status in (PipelineStatus.PAUSED, PipelineStatus.CANCELLED):
                continue

            if state.status in (
                PipelineStatus.INITIALIZING,
                PipelineStatus.CONCEPT_ART_GENERATING,
            ):
                if not self._broker.has_pending_or_running(name, "concept_art_generate"):
                    log.info("Recovery: re-enqueueing concept_art_generate for '%s'", name)
                    self._broker.enqueue(name, "concept_art_generate", payload)

        # 3D pipelines
        for name in self.list_3d_pipeline_names():
            state = self.get_3d_pipeline_state(name)
            if state is None:
                continue
            if state.status in (Pipeline3DStatus.IDLE, Pipeline3DStatus.CANCELLED,
                                 Pipeline3DStatus.AWAITING_APPROVAL):
                continue

            state_path = self._cfg.pipelines_dir / name / "state.json"
            payload = {"pipeline_name": name, "state_path": str(state_path)}

            if state.status == Pipeline3DStatus.QUEUED:
                if not self._broker.has_pending_or_running(name, "mesh_generate"):
                    log.info("Recovery: re-enqueueing mesh_generate for '%s'", name)
                    self._broker.enqueue(name, "mesh_generate", payload)
            elif state.status in (Pipeline3DStatus.GENERATING_MESH, Pipeline3DStatus.MESH_DONE):
                if not self._broker.has_pending_or_running(name, "mesh_texture"):
                    log.info("Recovery: re-enqueueing mesh_texture for '%s'", name)
                    self._broker.enqueue(name, "mesh_texture", payload)
            elif state.status in (Pipeline3DStatus.GENERATING_TEXTURE, Pipeline3DStatus.TEXTURE_DONE):
                if not self._broker.has_pending_or_running(name, "mesh_cleanup"):
                    log.info("Recovery: re-enqueueing mesh_cleanup for '%s'", name)
                    self._broker.enqueue(name, "mesh_cleanup", payload)
            elif state.status in (Pipeline3DStatus.CLEANING_UP, Pipeline3DStatus.CLEANUP_DONE):
                if not self._broker.has_pending_or_running(name, "screenshot"):
                    log.info("Recovery: re-enqueueing screenshot for '%s'", name)
                    self._broker.enqueue(name, "screenshot", payload)
