"""
Worker threads for the quickymesh pipeline agent.

Each thread runs a polling loop, claiming tasks from the broker and executing
them using the appropriate worker.  The threads are designed to be started
once and stopped gracefully via a stop_event.

Thread types
------------
ConceptArtWorkerThread   — concept_art_generate, concept_art_modify
TrellisWorkerThread      — mesh_generate, mesh_texture   (needs VRAM lock)
ScreenshotWorkerThread   — screenshot                    (needs VRAM lock)
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

    # -- public ---------------------------------------------------------------

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

    # -- subclass interface ---------------------------------------------------

    def _handle_task(self, task) -> None:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concept-art worker thread
# ---------------------------------------------------------------------------


class ConceptArtWorkerThread(_BaseWorkerThread):
    """
    Handles concept_art_generate and concept_art_modify tasks.

    Payload keys
    ------------
    concept_art_generate:
        pipeline_name, state_path, indices (optional list[int])
    concept_art_modify:
        pipeline_name, state_path, index (int), instruction (str)
    """

    task_types = ["concept_art_generate", "concept_art_modify"]

    def __init__(
        self,
        broker: "Broker",
        stop_event: threading.Event,
        worker: "ConceptArtWorker",
        cfg: "Config",
        **kwargs,
    ) -> None:
        super().__init__(broker, stop_event, **kwargs)
        self._worker = worker
        self._cfg = cfg

    def _handle_task(self, task) -> None:
        from src.concept_art_pipeline import (
            generate_concept_arts,
            modify_concept_art,
            regenerate_concept_arts,
        )
        from src.state import PipelineState

        state_path = Path(task.payload["state_path"])
        state = PipelineState.load(state_path)
        pipeline_dir = state_path.parent

        if task.task_type == "concept_art_generate":
            indices = task.payload.get("indices")
            if indices is None:
                generate_concept_arts(state, self._worker, pipeline_dir, self._cfg)
            else:
                regenerate_concept_arts(state, self._worker, pipeline_dir, indices, self._cfg)

        elif task.task_type == "concept_art_modify":
            index = task.payload["index"]
            instruction = task.payload["instruction"]
            modify_concept_art(state, self._worker, pipeline_dir, index, instruction)

        state.save(state_path)


# ---------------------------------------------------------------------------
# Trellis worker thread
# ---------------------------------------------------------------------------


class TrellisWorkerThread(_BaseWorkerThread):
    """
    Handles mesh_generate and mesh_texture tasks.

    Both tasks are VRAM-heavy, so each execution acquires the VRAMArbiter.

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
        from src.state import MeshStatus, PipelineState

        state_path = Path(task.payload["state_path"])
        pipeline_name = task.payload["pipeline_name"]
        state = PipelineState.load(state_path)
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
            if any(m.status == MeshStatus.MESH_DONE for m in state.meshes):
                if not self._broker.has_pending_or_running(pipeline_name, "mesh_texture"):
                    self._broker.enqueue(pipeline_name, "mesh_texture", payload)
                    log.info("Chained mesh_texture for '%s'", pipeline_name)
        elif task.task_type == "mesh_texture":
            if any(m.status == MeshStatus.TEXTURE_DONE for m in state.meshes):
                if not self._broker.has_pending_or_running(pipeline_name, "screenshot"):
                    self._broker.enqueue(pipeline_name, "screenshot", payload)
                    log.info("Chained screenshot for '%s'", pipeline_name)


# ---------------------------------------------------------------------------
# Screenshot worker thread
# ---------------------------------------------------------------------------


class ScreenshotWorkerThread(_BaseWorkerThread):
    """
    Handles screenshot tasks.

    Blender rendering is VRAM-heavy, so the VRAMArbiter is acquired.

    Payload keys
    ------------
    screenshot:
        pipeline_name, state_path
    """

    task_types = ["screenshot"]

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
        from src.screenshot_pipeline import run_screenshots
        from src.state import PipelineState, PipelineStatus

        state_path = Path(task.payload["state_path"])
        state = PipelineState.load(state_path)
        pipeline_dir = state_path.parent

        with self._arbiter.acquire(timeout=self._cfg.vram_lock_timeout):
            run_screenshots(state, self._worker, pipeline_dir, self._cfg)

        # Screenshots done — surface the mesh review prompt
        state.status = PipelineStatus.MESH_REVIEW
        state.save(state_path)
        log.info("Pipeline '%s' ready for mesh review", task.payload["pipeline_name"])
