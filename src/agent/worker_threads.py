"""
Worker threads for the quickymesh pipeline agent.

Each thread runs a polling loop, claiming tasks from the broker and executing
them using the appropriate worker.  The threads are designed to be started
once and stopped gracefully via a stop_event.

Thread types
------------
ConceptArtWorkerThread   — concept_art_generate          (2D pipelines)
TrellisWorkerThread      — mesh_generate, mesh_texture   (3D pipelines; needs VRAM lock)
ScreenshotWorkerThread   — mesh_cleanup, screenshot      (3D pipelines; needs VRAM lock)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.broker import Broker
    from src.config import Config
    from src.vram_arbiter import VRAMArbiter
    from src.workers.concept_art import ConceptArtWorker
    from src.workers.trellis import TrellisWorker
    from src.workers.screenshot import ScreenshotWorker

log = logging.getLogger(__name__)

_POLL_INTERVAL = 1.0   # seconds between task-queue polls when idle

# All 3D pipeline task types.  TrellisWorkerThread and ScreenshotWorkerThread
# both pass this as their `workflow_types` to claim_next so the broker can
# enforce per-pipeline FIFO across the full 3D workflow.  Without this, two
# workers running in parallel would interleave pipelines (e.g. TrellisWorker
# starts pipeline B's mesh_generate while pipeline A is still in
# mesh_cleanup/screenshot on ScreenshotWorker).
_WORKFLOW_3D = ["mesh_generate", "mesh_texture", "mesh_cleanup", "screenshot"]

# Concept art workflow: all types handled by ConceptArtWorkerThread.
_WORKFLOW_CONCEPT_ART = [
    "concept_art_generate",
    "concept_art_modify",
    "concept_art_restyle",
]


def _notify_review_ready(pipeline_name: str, review_type: str) -> None:
    """Publish an SSE event to notify connected clients that review is ready."""
    from src.api.event_bus import event_bus

    status_key = "concept_art_review" if review_type == "concept art" else "awaiting_approval"
    log.info("Pipeline '%s' %s review is ready", pipeline_name, review_type)
    event_bus.publish({
        "event": "status_change",
        "pipeline": pipeline_name,
        "status": status_key,
        "message": f"Pipeline '{pipeline_name}' {review_type} review is ready.",
    })


# ---------------------------------------------------------------------------
# Base thread
# ---------------------------------------------------------------------------


class _BaseWorkerThread(threading.Thread):
    """
    Background daemon thread that polls the broker for tasks to execute.

    Subclasses implement `_handle_task(task)`.
    """

    task_types: list[str] = []
    workflow_types: list[str] | None = None

    def __init__(
        self,
        broker: "Broker",
        stop_event: threading.Event,
        *,
        poll_interval: float = _POLL_INTERVAL,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._broker = broker
        self._stop = stop_event
        self._poll = poll_interval

    def run(self) -> None:
        log.debug("%s started", self.__class__.__name__)
        while not self._stop.is_set():
            task = self._broker.claim_next(self.task_types, self.workflow_types)
            if task is None:
                self._stop.wait(self._poll)
                continue
            log.info("Claimed task %d (%s / %s)", task.id, task.pipeline_name, task.task_type)
            try:
                self._handle_task(task)
                self._broker.mark_done(task.id)
                log.info("Task %d done", task.id)
            except Exception as exc:
                log.exception("Task %d failed: %s", task.id, exc)
                self._broker.mark_failed(task.id, str(exc))
        log.debug("%s stopped", self.__class__.__name__)

    def _handle_task(self, task) -> None:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concept-art worker thread  (2D pipelines)
# ---------------------------------------------------------------------------


class ConceptArtWorkerThread(_BaseWorkerThread):
    """
    Handles concept-art tasks for 2D pipelines.

    Payload keys
    ------------
    concept_art_generate:
        pipeline_name, state_path, indices (optional list[int])
    concept_art_modify:
        pipeline_name, state_path, index, instruction
    concept_art_restyle:
        pipeline_name, state_path, index, positive, negative, denoise
    """

    task_types = [
        "concept_art_generate",
        "concept_art_modify",
        "concept_art_restyle",
    ]
    workflow_types = _WORKFLOW_CONCEPT_ART

    def __init__(
        self,
        broker: "Broker",
        stop_event: threading.Event,
        worker: "ConceptArtWorker",
        cfg: "Config",
        *,
        flux_worker: "ConceptArtWorker | None" = None,
        restyle_worker: "ConceptArtWorker | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(broker, stop_event, **kwargs)
        self._worker = worker
        self._flux_worker = flux_worker
        self._restyle_worker = restyle_worker
        self._cfg = cfg

    def _pick_worker(self, state) -> "ConceptArtWorker":
        if getattr(state, "concept_art_backend", "gemini") == "flux" and self._flux_worker:
            return self._flux_worker
        return self._worker

    def _handle_task(self, task) -> None:
        from src.api.event_bus import event_bus
        from src.concept_art_pipeline import (
            generate_concept_arts,
            modify_concept_art,
            regenerate_concept_arts,
            restyle_concept_art,
        )
        from src.state import PipelineState

        state_path = Path(task.payload["state_path"])
        state = PipelineState.load(state_path)
        pipeline_dir = state_path.parent

        if task.task_type == "concept_art_generate":
            worker = self._pick_worker(state)
            indices = task.payload.get("indices")
            if indices is None:
                generate_concept_arts(state, worker, pipeline_dir, self._cfg)
            else:
                regenerate_concept_arts(state, worker, pipeline_dir, indices, self._cfg)
            state.concept_art_sheet_shown = False
            state.save(state_path)
            _notify_review_ready(state.name, "concept art")
            return

        if task.task_type == "concept_art_modify":
            index = int(task.payload["index"])
            instruction = task.payload["instruction"]
            source_version = task.payload.get("source_version")
            # Modify uses the same backend worker as initial generation
            # (must be Gemini — the endpoint enforces supports_modify).
            worker = self._pick_worker(state)
            modify_concept_art(
                state, worker, pipeline_dir, index, instruction,
                source_version=source_version,
            )
            state.concept_art_sheet_shown = False
            state.save(state_path)
            event_bus.publish({
                "event": "concept_art_updated",
                "pipeline": state.name,
                "index": index,
            })
            _notify_review_ready(state.name, "concept art")
            return

        if task.task_type == "concept_art_restyle":
            if self._restyle_worker is None:
                raise RuntimeError(
                    "concept_art_restyle task claimed but no restyle_worker is configured"
                )
            index = int(task.payload["index"])
            positive = task.payload["positive"]
            negative = task.payload["negative"]
            denoise = float(task.payload["denoise"])
            source_version = task.payload.get("source_version")
            restyle_concept_art(
                state, self._restyle_worker, pipeline_dir,
                index, positive, negative, denoise,
                source_version=source_version,
            )
            state.concept_art_sheet_shown = False
            state.save(state_path)
            event_bus.publish({
                "event": "concept_art_updated",
                "pipeline": state.name,
                "index": index,
            })
            _notify_review_ready(state.name, "concept art")
            return


# ---------------------------------------------------------------------------
# Trellis worker thread  (3D pipelines)
# ---------------------------------------------------------------------------


class TrellisWorkerThread(_BaseWorkerThread):
    """
    Handles mesh_generate and mesh_texture tasks for 3D pipelines.

    Payload keys
    ------------
    mesh_generate / mesh_texture:
        pipeline_name, state_path
    """

    task_types = ["mesh_texture", "mesh_generate"]  # texture preferred: finish one pipeline before starting another
    workflow_types = _WORKFLOW_3D

    def __init__(
        self,
        broker: "Broker",
        stop_event: threading.Event,
        worker: "TrellisWorker",
        arbiter: "VRAMArbiter",
        cfg: "Config",
        **kwargs,
    ) -> None:
        super().__init__(broker, stop_event, **kwargs)
        self._worker = worker
        self._arbiter = arbiter
        self._cfg = cfg

    def _handle_task(self, task) -> None:
        from src.mesh_pipeline import run_mesh_generation, run_mesh_texturing
        from src.state import Pipeline3DState, Pipeline3DStatus

        state_path = Path(task.payload["state_path"])
        pipeline_name = task.payload["pipeline_name"]
        state = Pipeline3DState.load(state_path)
        pipeline_dir = state_path.parent

        with self._arbiter.acquire(timeout=self._cfg.vram_lock_timeout):
            if task.task_type == "mesh_generate":
                run_mesh_generation(state, self._worker, pipeline_dir, self._cfg)
            elif task.task_type == "mesh_texture":
                run_mesh_texturing(state, self._worker, pipeline_dir, self._cfg)

        state.save(state_path)

        # Chain to the next step automatically
        payload = {"pipeline_name": pipeline_name, "state_path": str(state_path)}
        if task.task_type == "mesh_generate":
            if state.status == Pipeline3DStatus.MESH_DONE:
                if not self._broker.has_pending_or_running(pipeline_name, "mesh_texture"):
                    self._broker.enqueue(pipeline_name, "mesh_texture", payload)
                    log.info("Chained mesh_texture for '%s'", pipeline_name)
        elif task.task_type == "mesh_texture":
            if state.status == Pipeline3DStatus.TEXTURE_DONE:
                if not self._broker.has_pending_or_running(pipeline_name, "mesh_cleanup"):
                    self._broker.enqueue(pipeline_name, "mesh_cleanup", payload)
                    log.info("Chained mesh_cleanup for '%s'", pipeline_name)


# ---------------------------------------------------------------------------
# Screenshot worker thread  (3D pipelines)
# ---------------------------------------------------------------------------


class ScreenshotWorkerThread(_BaseWorkerThread):
    """
    Handles mesh_cleanup and screenshot tasks for 3D pipelines.

    Payload keys
    ------------
    mesh_cleanup / screenshot:
        pipeline_name, state_path
    """

    task_types = ["mesh_cleanup", "screenshot"]
    workflow_types = _WORKFLOW_3D

    def __init__(
        self,
        broker: "Broker",
        stop_event: threading.Event,
        worker: "ScreenshotWorker",
        arbiter: "VRAMArbiter",
        cfg: "Config",
        **kwargs,
    ) -> None:
        super().__init__(broker, stop_event, **kwargs)
        self._worker = worker
        self._arbiter = arbiter
        self._cfg = cfg

    def _handle_task(self, task) -> None:
        from src.screenshot_pipeline import run_cleanup, run_screenshots
        from src.state import Pipeline3DState, Pipeline3DStatus

        state_path = Path(task.payload["state_path"])
        pipeline_name = task.payload["pipeline_name"]
        state = Pipeline3DState.load(state_path)
        pipeline_dir = state_path.parent

        if task.task_type == "mesh_cleanup":
            run_cleanup(state, self._worker, pipeline_dir, self._cfg)
            state.save(state_path)

            payload = {"pipeline_name": pipeline_name, "state_path": str(state_path)}
            if state.status == Pipeline3DStatus.CLEANUP_DONE:
                if not self._broker.has_pending_or_running(pipeline_name, "screenshot"):
                    self._broker.enqueue(pipeline_name, "screenshot", payload)
                    log.info("Chained screenshot for '%s'", pipeline_name)

        elif task.task_type == "screenshot":
            run_screenshots(state, self._worker, pipeline_dir, self._cfg)

            if state.status == Pipeline3DStatus.SCREENSHOT_DONE:
                state.status = Pipeline3DStatus.AWAITING_APPROVAL

            state.save(state_path)
            log.info("3D pipeline '%s' ready for mesh review", pipeline_name)
            _notify_review_ready(pipeline_name, "mesh")
