"""
Tests for src/agent/pipeline_agent.py
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.agent.pipeline_agent import PipelineAgent
from src.broker import Broker
from src.config import Config
from src.state import Pipeline3DState, Pipeline3DStatus, PipelineState, PipelineStatus
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
def agent(broker, arbiter, cfg) -> PipelineAgent:
    return PipelineAgent(
        broker=broker,
        arbiter=arbiter,
        cfg=cfg,
        concept_worker=MockConceptArtWorker(),
        trellis_worker=MockTrellisWorker(),
        screenshot_worker=MockScreenshotWorker(),
        poll_interval=0.01,
    )


@pytest.fixture
def input_image(tmp_path) -> Path:
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    p = tmp_path / "input.png"
    img.save(str(p))
    return p


# ---------------------------------------------------------------------------
# start_pipeline
# ---------------------------------------------------------------------------


class TestStartPipeline:
    def test_creates_pipeline_directory(self, agent, cfg):
        agent.start_pipeline("mymodel", "a green dragon")
        assert (cfg.pipelines_dir / "mymodel").is_dir()

    def test_creates_state_json(self, agent, cfg):
        agent.start_pipeline("mymodel", "a green dragon")
        state_path = cfg.pipelines_dir / "mymodel" / "state.json"
        assert state_path.exists()

    def test_returned_state_has_correct_name(self, agent):
        state = agent.start_pipeline("mymodel", "a green dragon")
        assert state.name == "mymodel"

    def test_returned_state_has_correct_description(self, agent):
        state = agent.start_pipeline("mymodel", "a green dragon")
        assert state.description == "a green dragon"

    def test_enqueues_concept_art_generate_task(self, agent, broker):
        agent.start_pipeline("mymodel", "a green dragon")
        tasks = broker.get_tasks(task_type="concept_art_generate")
        assert len(tasks) == 1
        assert tasks[0].pipeline_name == "mymodel"

    def test_state_path_in_task_payload(self, agent, broker, cfg):
        agent.start_pipeline("mymodel", "a green dragon")
        tasks = broker.get_tasks(task_type="concept_art_generate")
        state_path = Path(tasks[0].payload["state_path"])
        assert state_path.exists()

    def test_custom_num_polys_stored(self, agent, cfg):
        state = agent.start_pipeline("mymodel", "a dragon", num_polys=12000)
        assert state.num_polys == 12000

    def test_default_num_polys_from_cfg(self, agent, cfg):
        state = agent.start_pipeline("mymodel", "a dragon")
        assert state.num_polys == cfg.num_polys


# ---------------------------------------------------------------------------
# get_pipeline_state
# ---------------------------------------------------------------------------


class TestGetPipelineState:
    def test_returns_state_for_existing_pipeline(self, agent):
        agent.start_pipeline("mymodel", "a dragon")
        state = agent.get_pipeline_state("mymodel")
        assert state is not None
        assert state.name == "mymodel"

    def test_returns_none_for_nonexistent_pipeline(self, agent):
        assert agent.get_pipeline_state("nonexistent") is None


# ---------------------------------------------------------------------------
# list_pipeline_names
# ---------------------------------------------------------------------------


class TestListPipelineNames:
    def test_empty_when_no_pipelines(self, agent):
        assert agent.list_pipeline_names() == []

    def test_returns_pipeline_names(self, agent):
        agent.start_pipeline("alpha", "a")
        agent.start_pipeline("beta", "b")
        names = agent.list_pipeline_names()
        assert "alpha" in names
        assert "beta" in names

    def test_returns_sorted_names(self, agent):
        agent.start_pipeline("zebra", "z")
        agent.start_pipeline("alpha", "a")
        names = agent.list_pipeline_names()
        assert names == sorted(names)

    def test_does_not_include_3d_pipelines(self, agent, input_image):
        agent.start_pipeline("twod", "a")
        agent.start_3d_pipeline("u_threed", str(input_image))
        names = agent.list_pipeline_names()
        assert "twod" in names
        assert "u_threed" not in names


# ---------------------------------------------------------------------------
# start_3d_pipeline / get_3d_pipeline_state / list_3d_pipeline_names
# ---------------------------------------------------------------------------


class TestStart3DPipeline:
    def test_creates_pipeline_directory(self, agent, cfg, input_image):
        agent.start_3d_pipeline("u_myship", str(input_image))
        assert (cfg.pipelines_dir / "u_myship").is_dir()

    def test_returns_pipeline3d_state(self, agent, input_image):
        state = agent.start_3d_pipeline("u_myship", str(input_image))
        assert isinstance(state, Pipeline3DState)
        assert state.name == "u_myship"
        assert state.status == Pipeline3DStatus.QUEUED

    def test_enqueues_mesh_generate(self, agent, broker, input_image):
        agent.start_3d_pipeline("u_myship", str(input_image))
        tasks = broker.get_tasks(task_type="mesh_generate")
        assert len(tasks) == 1

    def test_get_3d_pipeline_state(self, agent, input_image):
        agent.start_3d_pipeline("u_myship", str(input_image))
        state = agent.get_3d_pipeline_state("u_myship")
        assert state is not None
        assert state.name == "u_myship"

    def test_get_3d_pipeline_state_none_for_nonexistent(self, agent):
        assert agent.get_3d_pipeline_state("ghost") is None

    def test_list_3d_pipeline_names(self, agent, input_image):
        agent.start_3d_pipeline("u_ship1", str(input_image))
        agent.start_3d_pipeline("u_ship2", str(input_image))
        names = agent.list_3d_pipeline_names()
        assert "u_ship1" in names
        assert "u_ship2" in names

    def test_2d_pipelines_not_in_3d_list(self, agent, input_image):
        agent.start_pipeline("twodpipe", "a dragon")
        agent.start_3d_pipeline("u_threed", str(input_image))
        names = agent.list_3d_pipeline_names()
        assert "twodpipe" not in names
        assert "u_threed" in names

    def test_pipeline_name_exists(self, agent, input_image):
        agent.start_3d_pipeline("u_myship", str(input_image))
        assert agent.pipeline_name_exists("u_myship") is True
        assert agent.pipeline_name_exists("nonexistent") is False


# ---------------------------------------------------------------------------
# enqueue helpers
# ---------------------------------------------------------------------------


class TestEnqueueHelpers:
    def _make_3d_pipeline(self, agent, input_image):
        agent.start_3d_pipeline("pipe3d", str(input_image))
        # Drain the auto-enqueued mesh_generate
        return "pipe3d"

    def test_enqueue_mesh_generation_idempotent(self, agent, broker, input_image):
        self._make_3d_pipeline(agent, input_image)
        agent.enqueue_mesh_generation("pipe3d")  # already queued
        tasks = broker.get_tasks(task_type="mesh_generate")
        assert len(tasks) == 1

    def test_enqueue_mesh_texturing_adds_task(self, agent, broker, input_image):
        agent.start_3d_pipeline("pipe3d", str(input_image))
        # drain the auto-enqueued mesh_generate first
        broker.cancel_pipeline_tasks("pipe3d")
        agent.enqueue_mesh_texturing("pipe3d")
        tasks = broker.get_tasks(task_type="mesh_texture")
        assert len(tasks) == 1

    def test_enqueue_mesh_texturing_idempotent(self, agent, broker, input_image):
        agent.start_3d_pipeline("pipe3d", str(input_image))
        broker.cancel_pipeline_tasks("pipe3d")
        agent.enqueue_mesh_texturing("pipe3d")
        agent.enqueue_mesh_texturing("pipe3d")
        tasks = broker.get_tasks(task_type="mesh_texture")
        assert len(tasks) == 1

    def test_enqueue_screenshots_adds_task(self, agent, broker, input_image):
        agent.start_3d_pipeline("pipe3d", str(input_image))
        broker.cancel_pipeline_tasks("pipe3d")
        agent.enqueue_screenshots("pipe3d")
        tasks = broker.get_tasks(task_type="screenshot")
        assert len(tasks) == 1

    def test_enqueue_screenshots_idempotent(self, agent, broker, input_image):
        agent.start_3d_pipeline("pipe3d", str(input_image))
        broker.cancel_pipeline_tasks("pipe3d")
        agent.enqueue_screenshots("pipe3d")
        agent.enqueue_screenshots("pipe3d")
        tasks = broker.get_tasks(task_type="screenshot")
        assert len(tasks) == 1


# ---------------------------------------------------------------------------
# cancel_pipeline
# ---------------------------------------------------------------------------


class TestCancelPipeline:
    def test_cancels_pending_tasks(self, agent, broker):
        agent.start_pipeline("pipe1", "a dragon")
        count = agent.cancel_pipeline("pipe1")
        assert count >= 1
        for t in broker.get_tasks(pipeline_name="pipe1"):
            assert t.status == "failed"

    def test_returns_cancelled_count(self, agent, broker):
        agent.start_pipeline("pipe1", "a dragon")
        broker.enqueue("pipe1", "concept_art_generate", {})  # add extra task
        count = agent.cancel_pipeline("pipe1")
        assert count >= 1


# ---------------------------------------------------------------------------
# priority prompting
# ---------------------------------------------------------------------------


class TestPriorityPrompting:
    def _set_pipeline_status(self, agent, cfg, name, status):
        state_path = cfg.pipelines_dir / name / "state.json"
        state = PipelineState.load(state_path)
        state.status = status
        state.save(state_path)

    def test_highest_priority_none_when_no_pipelines(self, agent):
        assert agent.highest_priority_pipeline() is None

    def test_highest_priority_concept_art_review(self, agent, cfg):
        agent.start_pipeline("p1", "desc")
        self._set_pipeline_status(agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW)
        assert agent.highest_priority_pipeline() == "p1"

    def test_highest_priority_none_when_all_running(self, agent, cfg):
        agent.start_pipeline("p1", "desc")
        # status stays INITIALIZING/CONCEPT_ART_GENERATING — not a review state
        assert agent.highest_priority_pipeline() is None

    def test_pipelines_needing_attention_returns_review(self, agent, cfg):
        agent.start_pipeline("p1", "desc")
        self._set_pipeline_status(agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW)
        needing = agent.pipelines_needing_attention()
        assert "p1" in needing

    def test_pipelines_needing_attention_empty_when_none(self, agent):
        assert agent.pipelines_needing_attention() == []


# ---------------------------------------------------------------------------
# start_workers / stop_workers
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_start_and_stop_workers(self, agent):
        agent.start_workers()
        assert len(agent._threads) == 3
        agent.stop_workers(timeout=2)
        assert len(agent._threads) == 0

    def test_start_workers_idempotent(self, agent):
        agent.start_workers()
        agent.start_workers()  # second call is a no-op
        n = len(agent._threads)
        agent.stop_workers(timeout=2)
        assert n == 3

    def test_threads_are_daemon_threads(self, agent):
        agent.start_workers()
        try:
            for t in agent._threads:
                assert t.daemon is True
        finally:
            agent.stop_workers(timeout=2)


# ---------------------------------------------------------------------------
# recover_stalled_pipelines
# ---------------------------------------------------------------------------


class TestRecoverStalledPipelines:
    def _set_status_2d(self, agent, cfg, name, status):
        sp = cfg.pipelines_dir / name / "state.json"
        state = PipelineState.load(sp)
        state.status = status
        state.save(sp)
        return sp

    def _set_status_3d(self, agent, cfg, name, status):
        sp = cfg.pipelines_dir / name / "state.json"
        state = Pipeline3DState.load(sp)
        state.status = status
        state.save(sp)
        return sp

    def test_recovers_concept_art_generating(self, agent, broker, cfg):
        agent.start_pipeline("p1", "desc")
        broker.cancel_pipeline_tasks("p1")
        self._set_status_2d(agent, cfg, "p1", PipelineStatus.CONCEPT_ART_GENERATING)

        agent.recover_stalled_pipelines()

        tasks = broker.get_tasks(pipeline_name="p1", task_type="concept_art_generate")
        pending = [t for t in tasks if t.status == "pending"]
        assert len(pending) == 1

    def test_recovers_3d_queued(self, agent, broker, cfg, input_image):
        agent.start_3d_pipeline("u_ship", str(input_image))
        broker.cancel_pipeline_tasks("u_ship")
        self._set_status_3d(agent, cfg, "u_ship", Pipeline3DStatus.QUEUED)

        agent.recover_stalled_pipelines()

        tasks = broker.get_tasks(pipeline_name="u_ship", task_type="mesh_generate")
        pending = [t for t in tasks if t.status == "pending"]
        assert len(pending) == 1

    def test_recovers_3d_generating_mesh(self, agent, broker, cfg, input_image):
        agent.start_3d_pipeline("u_ship", str(input_image))
        broker.cancel_pipeline_tasks("u_ship")
        self._set_status_3d(agent, cfg, "u_ship", Pipeline3DStatus.GENERATING_MESH)

        agent.recover_stalled_pipelines()

        tasks = broker.get_tasks(pipeline_name="u_ship", task_type="mesh_texture")
        pending = [t for t in tasks if t.status == "pending"]
        assert len(pending) == 1

    def test_recovers_3d_texture_done(self, agent, broker, cfg, input_image):
        agent.start_3d_pipeline("u_ship", str(input_image))
        broker.cancel_pipeline_tasks("u_ship")
        self._set_status_3d(agent, cfg, "u_ship", Pipeline3DStatus.TEXTURE_DONE)

        agent.recover_stalled_pipelines()

        tasks = broker.get_tasks(pipeline_name="u_ship", task_type="mesh_cleanup")
        pending = [t for t in tasks if t.status == "pending"]
        assert len(pending) == 1

    def test_recovers_3d_cleanup_done(self, agent, broker, cfg, input_image):
        agent.start_3d_pipeline("u_ship", str(input_image))
        broker.cancel_pipeline_tasks("u_ship")
        self._set_status_3d(agent, cfg, "u_ship", Pipeline3DStatus.CLEANUP_DONE)

        agent.recover_stalled_pipelines()

        tasks = broker.get_tasks(pipeline_name="u_ship", task_type="screenshot")
        pending = [t for t in tasks if t.status == "pending"]
        assert len(pending) == 1

    def test_skips_3d_awaiting_approval(self, agent, broker, cfg, input_image):
        agent.start_3d_pipeline("u_ship", str(input_image))
        broker.cancel_pipeline_tasks("u_ship")
        self._set_status_3d(agent, cfg, "u_ship", Pipeline3DStatus.AWAITING_APPROVAL)

        agent.recover_stalled_pipelines()

        # Should not auto-enqueue anything for AWAITING_APPROVAL
        tasks = broker.get_tasks(pipeline_name="u_ship", status="pending")
        assert len(tasks) == 0

    def test_does_not_double_enqueue(self, agent, broker, cfg):
        agent.start_pipeline("p1", "desc")
        broker.cancel_pipeline_tasks("p1")
        self._set_status_2d(agent, cfg, "p1", PipelineStatus.CONCEPT_ART_GENERATING)
        broker.enqueue("p1", "concept_art_generate", {})

        agent.recover_stalled_pipelines()

        tasks = broker.get_tasks(pipeline_name="p1", task_type="concept_art_generate")
        pending = [t for t in tasks if t.status == "pending"]
        assert len(pending) == 1  # not 2

    def test_skips_pipelines_in_review_status(self, agent, broker, cfg):
        agent.start_pipeline("p1", "desc")
        broker.cancel_pipeline_tasks("p1")
        self._set_status_2d(agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW)

        agent.recover_stalled_pipelines()

        tasks = broker.get_tasks(pipeline_name="p1", status="pending")
        assert len(tasks) == 0

    def test_recovery_runs_automatically_on_start_workers(self, agent, broker, cfg):
        agent.start_pipeline("p1", "desc")
        broker.cancel_pipeline_tasks("p1")
        self._set_status_2d(agent, cfg, "p1", PipelineStatus.CONCEPT_ART_GENERATING)

        agent.start_workers()
        try:
            tasks = broker.get_tasks(pipeline_name="p1", task_type="concept_art_generate")
            pending = [t for t in tasks if t.status == "pending"]
            assert len(pending) == 1
        finally:
            agent.stop_workers(timeout=2)
