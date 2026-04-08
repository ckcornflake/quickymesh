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
            task = self._broker.claim_next(self.task_types)
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
    Handles concept_art_generate tasks for 2D pipelines.

    Payload keys
    ------------
    concept_art_generate:
        pipeline_name, state_path, indices (optional list[int])
    """

    task_types = ["concept_art_generate", "concept_art_modify"]

    def __init__(
        self,
        broker: "Broker",
        stop_event: threading.Event,
        worker: "ConceptArtWorker",
        cfg: "Config",
        *,
        flux_worker: "ConceptArtWorker | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(broker, stop_event, **kwargs)
        self._worker = worker
        self._flux_worker = flux_worker
        self._cfg = cfg

    def _pick_worker(self, state) -> "ConceptArtWorker":
        if getattr(state, "concept_art_backend", "gemini") == "flux" and self._flux_worker:
            return self._flux_worker
        return self._worker

    def _handle_task(self, task) -> None:
        from src.concept_art_pipeline import (
            generate_concept_arts,
            regenerate_concept_arts,
        )
        from src.state import PipelineState

        state_path = Path(task.payload["state_path"])
        state = PipelineState.load(state_path)
        pipeline_dir = state_path.parent
        worker = self._pick_worker(state)

        if task.task_type == "concept_art_generate":
            indices = task.payload.get("indices")
            if indices is None:
                generate_concept_arts(state, worker, pipeline_dir, self._cfg)
            else:
                regenerate_concept_arts(state, worker, pipeline_dir, indices, self._cfg)
            _notify_review_ready(state.name, "concept art")

        state.save(state_path)


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

    task_types = ["mesh_generate", "mesh_texture"]

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
