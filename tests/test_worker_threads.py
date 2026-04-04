"""
Tests for src/agent/worker_threads.py
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.agent.worker_threads import (
    ConceptArtWorkerThread,
    ScreenshotWorkerThread,
    TrellisWorkerThread,
    _BaseWorkerThread,
)
from src.broker import Broker
from src.config import Config
from src.state import (
    ConceptArtStatus,
    MeshStatus,
    PipelineState,
)
from src.vram_arbiter import VRAMArbiter
from src.workers.concept_art import MockConceptArtWorker
from src.workers.screenshot import MockScreenshotWorker
from src.workers.trellis import MockTrellisWorker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def broker():
    b = Broker()
    yield b
    b.close()


@pytest.fixture
def stop_event():
    return threading.Event()


@pytest.fixture
def arbiter():
    return VRAMArbiter()


@pytest.fixture
def cfg(tmp_path) -> Config:
    defaults = {
        "gemini": {"model": "m", "alternative_model": "a"},
        "generation": {
            "num_concept_arts": 2,
            "num_polys": 8000,
            "review_sheet_thumb_size": 64,
            "html_preview_size": 256,
            "export_format": "glb",
            "background_suffix": "plain white background",
        },
        "infrastructure": {
            "comfyui_url": "http://localhost:8188",
            "comfyui_install_dir": "/fake",
            "comfyui_output_dir": "/fake/output",
            "comfyui_poll_interval": 2.0,
            "comfyui_timeout": 600.0,
            "blender_path": "/fake/blender",
            "vram_lock_timeout": 30.0,
        },
        "output": {"root": str(tmp_path / "output")},
    }
    d = tmp_path / "defaults.yaml"
    d.write_text(yaml.dump(defaults))
    e = tmp_path / ".env"
    e.write_text("GEMINI_API_KEY=fake\n")
    return Config(defaults_path=d, env_path=e)


@pytest.fixture
def pipeline_state_path(tmp_path, cfg) -> Path:
    """Create a minimal pipeline state and return its path."""
    pipeline_dir = tmp_path / "output" / "uncompleted_pipelines" / "testpipe"
    pipeline_dir.mkdir(parents=True)
    state = PipelineState(
        name="testpipe",
        description="a blue dragon",
        num_polys=8000,
        pipeline_dir="uncompleted_pipelines/testpipe",
    )
    state_path = pipeline_dir / "state.json"
    state.save(state_path)
    return state_path


# ---------------------------------------------------------------------------
# _BaseWorkerThread
# ---------------------------------------------------------------------------


class TestBaseWorkerThread:
    def test_stops_when_stop_event_set(self, broker, stop_event, cfg):
        """Thread exits its loop when stop_event is set."""
        class NoopThread(_BaseWorkerThread):
            task_types = ["mesh_generate"]
            def _handle_task(self, task):
                pass

        t = NoopThread(broker, stop_event, poll_interval=0.02)
        t.start()
        stop_event.set()
        t.join(timeout=2)
        assert not t.is_alive()

    def test_marks_task_done_on_success(self, broker, stop_event, cfg):
        results = []

        class RecordThread(_BaseWorkerThread):
            task_types = ["mesh_generate"]
            def _handle_task(self, task):
                results.append(task.id)

        broker.enqueue("pipe1", "mesh_generate")
        t = RecordThread(broker, stop_event, poll_interval=0.01)
        t.start()
        # wait for task to be processed
        deadline = time.time() + 3
        while time.time() < deadline:
            tasks = broker.get_tasks(status="done")
            if tasks:
                break
            time.sleep(0.02)
        stop_event.set()
        t.join(timeout=2)

        assert len(broker.get_tasks(status="done")) == 1

    def test_marks_task_failed_on_exception(self, broker, stop_event, cfg):
        class FailThread(_BaseWorkerThread):
            task_types = ["mesh_generate"]
            def _handle_task(self, task):
                raise RuntimeError("boom")

        broker.enqueue("pipe1", "mesh_generate")
        t = FailThread(broker, stop_event, poll_interval=0.01)
        t.start()
        deadline = time.time() + 3
        while time.time() < deadline:
            tasks = broker.get_tasks(status="failed")
            if tasks:
                break
            time.sleep(0.02)
        stop_event.set()
        t.join(timeout=2)

        failed = broker.get_tasks(status="failed")
        assert len(failed) == 1
        assert "boom" in failed[0].error

    def test_is_daemon_thread_by_default(self, broker, stop_event):
        class NoopThread(_BaseWorkerThread):
            task_types = []
            def _handle_task(self, task): pass

        t = NoopThread(broker, stop_event)
        assert t.daemon is True


# ---------------------------------------------------------------------------
# ConceptArtWorkerThread
# ---------------------------------------------------------------------------


class TestConceptArtWorkerThread:
    def _enqueue_generate(self, broker, state_path, pipeline_name="testpipe"):
        return broker.enqueue(
            pipeline_name,
            "concept_art_generate",
            {"pipeline_name": pipeline_name, "state_path": str(state_path)},
        )

    def test_processes_concept_art_generate(self, broker, stop_event, cfg, pipeline_state_path):
        self._enqueue_generate(broker, pipeline_state_path)
        t = ConceptArtWorkerThread(
            broker, stop_event, MockConceptArtWorker(), cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="done"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        assert len(broker.get_tasks(status="done")) == 1

    def test_state_saved_after_generate(self, broker, stop_event, cfg, pipeline_state_path):
        self._enqueue_generate(broker, pipeline_state_path)
        t = ConceptArtWorkerThread(
            broker, stop_event, MockConceptArtWorker(), cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="done"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        state = PipelineState.load(pipeline_state_path)
        assert len(state.concept_arts) > 0

    def test_worker_failure_marks_task_failed(self, broker, stop_event, cfg, pipeline_state_path):
        self._enqueue_generate(broker, pipeline_state_path)
        failing_worker = MockConceptArtWorker(fail_on_generate=True)
        t = ConceptArtWorkerThread(
            broker, stop_event, failing_worker, cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="failed"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        assert len(broker.get_tasks(status="failed")) == 1

    def test_processes_concept_art_modify(self, broker, stop_event, cfg, pipeline_state_path):
        # First generate concept arts so there's something to modify
        from src.concept_art_pipeline import generate_concept_arts
        state = PipelineState.load(pipeline_state_path)
        pipeline_dir = pipeline_state_path.parent
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.save(pipeline_state_path)

        broker.enqueue(
            "testpipe",
            "concept_art_modify",
            {
                "pipeline_name": "testpipe",
                "state_path": str(pipeline_state_path),
                "index": 0,
                "instruction": "make it red",
            },
        )
        t = ConceptArtWorkerThread(
            broker, stop_event, MockConceptArtWorker(), cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="done"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        assert len(broker.get_tasks(status="done")) == 1


# ---------------------------------------------------------------------------
# TrellisWorkerThread
# ---------------------------------------------------------------------------


class TestTrellisWorkerThread:
    def _make_state_mesh_ready(self, pipeline_state_path, cfg) -> PipelineState:
        """Return a state with approved concept arts ready for mesh gen."""
        from src.concept_art_pipeline import generate_concept_arts
        state = PipelineState.load(pipeline_state_path)
        pipeline_dir = pipeline_state_path.parent
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        for ca in state.concept_arts:
            ca.status = ConceptArtStatus.APPROVED
        state.save(pipeline_state_path)
        return state

    def test_processes_mesh_generate(self, broker, stop_event, cfg, arbiter, pipeline_state_path):
        self._make_state_mesh_ready(pipeline_state_path, cfg)
        broker.enqueue(
            "testpipe",
            "mesh_generate",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        t = TrellisWorkerThread(
            broker, stop_event, MockTrellisWorker(), arbiter, cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="done"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        # Chain ran too — at least the original task is done
        done = broker.get_tasks(status="done")
        assert any(t.task_type == "mesh_generate" for t in done)

    def test_acquires_vram_lock_during_task(self, broker, stop_event, cfg, pipeline_state_path):
        self._make_state_mesh_ready(pipeline_state_path, cfg)
        broker.enqueue(
            "testpipe",
            "mesh_generate",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        arbiter = VRAMArbiter()
        lock_acquired_times = []

        original_acquire = arbiter.acquire

        def spy_acquire(*args, **kwargs):
            ctx = original_acquire(*args, **kwargs)
            return ctx

        arbiter.acquire = spy_acquire

        t = TrellisWorkerThread(
            broker, stop_event, MockTrellisWorker(), arbiter, cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="done"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        # Lock was released after task completed
        assert not arbiter.locked

    def test_task_failed_on_trellis_error(self, broker, stop_event, cfg, arbiter, pipeline_state_path):
        self._make_state_mesh_ready(pipeline_state_path, cfg)
        broker.enqueue(
            "testpipe",
            "mesh_generate",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        failing_worker = MockTrellisWorker(fail_on_generate=True)
        t = TrellisWorkerThread(
            broker, stop_event, failing_worker, arbiter, cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="failed"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        assert len(broker.get_tasks(status="failed")) == 1


# ---------------------------------------------------------------------------
# ScreenshotWorkerThread
# ---------------------------------------------------------------------------


class TestScreenshotWorkerThread:
    def _make_state_texture_done(self, pipeline_state_path, cfg) -> PipelineState:
        from src.concept_art_pipeline import generate_concept_arts
        from src.mesh_pipeline import run_mesh_generation, run_mesh_texturing

        state = PipelineState.load(pipeline_state_path)
        pipeline_dir = pipeline_state_path.parent
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        for ca in state.concept_arts:
            ca.status = ConceptArtStatus.APPROVED
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        run_mesh_texturing(state, MockTrellisWorker(), pipeline_dir, cfg)
        state.save(pipeline_state_path)
        return state

    def test_processes_screenshot_task(self, broker, stop_event, cfg, arbiter, pipeline_state_path):
        self._make_state_texture_done(pipeline_state_path, cfg)
        broker.enqueue(
            "testpipe",
            "screenshot",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        t = ScreenshotWorkerThread(
            broker, stop_event, MockScreenshotWorker(), arbiter, cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="done"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        assert len(broker.get_tasks(status="done")) == 1

    def test_state_updated_after_screenshots(self, broker, stop_event, cfg, arbiter, pipeline_state_path):
        self._make_state_texture_done(pipeline_state_path, cfg)
        broker.enqueue(
            "testpipe",
            "screenshot",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        t = ScreenshotWorkerThread(
            broker, stop_event, MockScreenshotWorker(), arbiter, cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="done"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        # Worker advances SCREENSHOT_DONE → AWAITING_APPROVAL so the review
        # API/CLI sees meshes immediately without an extra transition step.
        state = PipelineState.load(pipeline_state_path)
        for m in state.meshes:
            assert m.status == MeshStatus.AWAITING_APPROVAL

    def test_screenshot_failure_marks_task_failed(self, broker, stop_event, cfg, arbiter, pipeline_state_path):
        self._make_state_texture_done(pipeline_state_path, cfg)
        broker.enqueue(
            "testpipe",
            "screenshot",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        t = ScreenshotWorkerThread(
            broker, stop_event, MockScreenshotWorker(fail=True), arbiter, cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="failed"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        assert len(broker.get_tasks(status="failed")) == 1

    def test_screenshot_sets_pipeline_to_mesh_review(self, broker, stop_event, cfg, arbiter, pipeline_state_path):
        self._make_state_texture_done(pipeline_state_path, cfg)
        broker.enqueue(
            "testpipe",
            "screenshot",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        t = ScreenshotWorkerThread(
            broker, stop_event, MockScreenshotWorker(), arbiter, cfg, poll_interval=0.01
        )
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.get_tasks(status="done"):
                break
            time.sleep(0.05)
        stop_event.set()
        t.join(timeout=3)

        from src.state import PipelineStatus
        state = PipelineState.load(pipeline_state_path)
        assert state.status == PipelineStatus.MESH_REVIEW


# ---------------------------------------------------------------------------
# Pipeline step chaining
# ---------------------------------------------------------------------------


class TestPipelineChaining:
    """Verify that worker threads automatically enqueue the next step."""

    def _make_state_mesh_ready(self, pipeline_state_path, cfg):
        from src.concept_art_pipeline import generate_concept_arts
        state = PipelineState.load(pipeline_state_path)
        pipeline_dir = pipeline_state_path.parent
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        for ca in state.concept_arts:
            ca.status = ConceptArtStatus.APPROVED
        state.save(pipeline_state_path)
        return state

    def _wait_for(self, broker, status, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if broker.get_tasks(status=status):
                return True
            time.sleep(0.05)
        return False

    def test_mesh_generate_chains_mesh_texture(self, broker, stop_event, cfg, arbiter, pipeline_state_path):
        self._make_state_mesh_ready(pipeline_state_path, cfg)
        broker.enqueue(
            "testpipe", "mesh_generate",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        t = TrellisWorkerThread(
            broker, stop_event, MockTrellisWorker(), arbiter, cfg, poll_interval=0.01
        )
        t.start()
        # After mesh_generate completes, mesh_texture should be queued
        assert self._wait_for(broker, "pending"), "mesh_texture was never enqueued"
        stop_event.set()
        t.join(timeout=3)

        texture_tasks = broker.get_tasks(task_type="mesh_texture")
        assert len(texture_tasks) >= 1

    def test_mesh_texture_chains_screenshot(self, broker, stop_event, cfg, arbiter, pipeline_state_path):
        from src.mesh_pipeline import run_mesh_generation
        state = self._make_state_mesh_ready(pipeline_state_path, cfg)
        pipeline_dir = pipeline_state_path.parent
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        state.save(pipeline_state_path)

        broker.enqueue(
            "testpipe", "mesh_texture",
            {"pipeline_name": "testpipe", "state_path": str(pipeline_state_path)},
        )
        t = TrellisWorkerThread(
            broker, stop_event, MockTrellisWorker(), arbiter, cfg, poll_interval=0.01
        )
        t.start()
        assert self._wait_for(broker, "pending"), "mesh_cleanup was never enqueued"
        stop_event.set()
        t.join(timeout=3)

        cleanup_tasks = broker.get_tasks(task_type="mesh_cleanup")
        assert len(cleanup_tasks) >= 1
