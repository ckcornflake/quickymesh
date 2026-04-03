"""
Tests for src/agent/cli_main.py
"""

from __future__ import annotations

import yaml

from src.agent.cli_main import (
    _edit_pipeline,
    _handle_priority_pipeline,
    _idle_menu,
    _main_loop,
    _manage_pipeline,
    _print_watch_diffs,
    _retry_failed,
    _show_status,
    _start_new_pipeline,
    _watch_mode,
    run_cli,
)
from src.agent.pipeline_agent import PipelineAgent
from src.broker import Broker
from src.config import Config
from src.concept_art_pipeline import generate_concept_arts
from src.mesh_pipeline import run_mesh_generation, run_mesh_texturing
from src.prompt_interface.mock import MockPromptInterface
from src.state import ConceptArtStatus, MeshItem, MeshStatus, PipelineState, PipelineStatus
from src.vram_arbiter import VRAMArbiter
from src.workers.concept_art import MockConceptArtWorker
from src.workers.screenshot import MockScreenshotWorker
from src.workers.trellis import MockTrellisWorker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


import pytest


@pytest.fixture
def broker():
    b = Broker()
    yield b
    b.close()


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
def agent(broker, cfg) -> PipelineAgent:
    return PipelineAgent(
        broker=broker,
        arbiter=VRAMArbiter(),
        cfg=cfg,
        concept_worker=MockConceptArtWorker(),
        trellis_worker=MockTrellisWorker(),
        screenshot_worker=MockScreenshotWorker(),
        poll_interval=0.01,
    )


# ---------------------------------------------------------------------------
# _idle_menu
# ---------------------------------------------------------------------------


class TestIdleMenu:
    def test_returns_quit_on_q(self, agent, cfg):
        ui = MockPromptInterface(responses=["q"])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "quit"

    def test_returns_status_on_s(self, agent, cfg):
        ui = MockPromptInterface(responses=["s"])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "status"

    def test_returns_new_on_n(self, agent, cfg):
        # name → backend(default) → imgpath(blank) → description → polys → sym
        ui = MockPromptInterface(responses=["n", "mymodel", "", "", "a dragon", "", ""])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "new"

    def test_returns_watch_on_w(self, agent, cfg):
        # Patch _watch_mode to a no-op so the test doesn't enter the real loop
        import src.agent.cli_main as cli_mod
        original = cli_mod._watch_mode
        cli_mod._watch_mode = lambda *a, **kw: None
        try:
            ui = MockPromptInterface(responses=["w"])
            result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
            assert result == "watch"
        finally:
            cli_mod._watch_mode = original


# ---------------------------------------------------------------------------
# _start_new_pipeline
# ---------------------------------------------------------------------------


class TestStartNewPipeline:
    # Response order: name, backend(blank=default), image-path(blank), description, poly-count, symmetrize
    def test_creates_pipeline(self, agent):
        ui = MockPromptInterface(responses=["testpipe", "", "", "a red sphere", "", ""])
        _start_new_pipeline(agent, ui)
        assert "testpipe" in agent.list_pipeline_names()

    def test_informs_user_on_success(self, agent):
        ui = MockPromptInterface(responses=["testpipe", "", "", "a red sphere", "", ""])
        _start_new_pipeline(agent, ui)
        assert any("testpipe" in m for m in ui.messages)

    def test_cancels_on_empty_name(self, agent):
        ui = MockPromptInterface(responses=["", ""])
        _start_new_pipeline(agent, ui)
        assert agent.list_pipeline_names() == []

    def test_cancels_on_empty_description(self, agent):
        # name → backend → imgpath → empty description → cancel
        ui = MockPromptInterface(responses=["testpipe", "", "", ""])
        _start_new_pipeline(agent, ui)
        assert agent.list_pipeline_names() == []

    def test_uses_custom_polys(self, agent):
        ui = MockPromptInterface(responses=["testpipe", "", "", "a dragon", "12000", ""])
        _start_new_pipeline(agent, ui)
        state = agent.get_pipeline_state("testpipe")
        assert state.num_polys == 12000

    def test_uses_default_polys_on_empty_input(self, agent, cfg):
        ui = MockPromptInterface(responses=["testpipe", "", "", "a dragon", "", ""])
        _start_new_pipeline(agent, ui)
        state = agent.get_pipeline_state("testpipe")
        assert state.num_polys == cfg.num_polys

    def test_informs_user_of_suffix(self, agent, cfg):
        ui = MockPromptInterface(responses=["testpipe", "", "", "a dragon", "", ""])
        _start_new_pipeline(agent, ui)
        assert any(cfg.background_suffix in m for m in ui.messages)

    def test_backend_gemini_saved_to_state(self, agent):
        ui = MockPromptInterface(responses=["testpipe", "1", "", "a dragon", "", ""])
        _start_new_pipeline(agent, ui)
        state = agent.get_pipeline_state("testpipe")
        assert state.concept_art_backend == "gemini"

    def test_backend_flux_saved_to_state(self, agent):
        # FLUX skips the image-path prompt
        ui = MockPromptInterface(responses=["testpipe", "2", "a dragon", "", ""])
        _start_new_pipeline(agent, ui)
        state = agent.get_pipeline_state("testpipe")
        assert state.concept_art_backend == "flux"

    def test_backend_preference_persisted(self, agent, cfg):
        ui = MockPromptInterface(responses=["testpipe", "2", "a dragon", "", ""])
        _start_new_pipeline(agent, ui)
        import json
        prefs = json.loads((cfg.output_root / ".preferences.json").read_text())
        assert prefs["concept_art_backend"] == "flux"


# ---------------------------------------------------------------------------
# _show_status (pure snapshot — no approval selection)
# ---------------------------------------------------------------------------


class TestShowStatus:
    def test_shows_worker_threads(self, agent):
        agent.start_workers()
        try:
            ui = MockPromptInterface()
            _show_status(agent, ui)
            combined = "\n".join(ui.messages)
            assert "Worker" in combined
            assert "running" in combined
        finally:
            agent.stop_workers(timeout=2)

    def test_shows_no_pipelines_when_empty(self, agent):
        ui = MockPromptInterface()
        _show_status(agent, ui)
        combined = "\n".join(ui.messages)
        assert "Pipeline" in combined
        assert "(none)" in combined

    def test_shows_pipeline_name_and_status(self, agent):
        agent.start_pipeline("pipe1", "desc1")
        ui = MockPromptInterface()
        _show_status(agent, ui)
        combined = "\n".join(ui.messages)
        assert "pipe1" in combined

    def test_returns_none(self, agent):
        ui = MockPromptInterface()
        result = _show_status(agent, ui)
        assert result is None

    def test_does_not_ask_for_approval_selection(self, agent, cfg):
        """_show_status must not prompt — approvals belong to watch mode."""
        agent.start_pipeline("pipe1", "desc")
        state_path = cfg.uncompleted_pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(state_path)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(state_path)

        ui = MockPromptInterface()  # no responses — would raise if asked anything
        _show_status(agent, ui)    # must not raise StopIteration


# ---------------------------------------------------------------------------
# _watch_mode (unit-tested via injected tick/interval)
# ---------------------------------------------------------------------------


class TestWatchMode:
    """
    Watch mode is tested by driving it through MockPromptInterface.

    MockPromptInterface.ask() is NOT used by watch mode (it never calls
    ui.ask directly during the watch loop itself — only during review prompts).
    Instead, _watch_mode calls _try_read_char() for keypress detection.
    We monkey-patch _try_read_char to feed a character sequence.
    """

    def _run_watch(self, agent, cfg, chars, concept_worker=None, trellis_worker=None,
                   extra_responses=None):
        """
        Drive watch mode by injecting a sequence of characters via
        a patched _try_read_char.  Returns the MockPromptInterface used.
        """
        import src.agent.cli_main as cli_mod

        char_iter = iter(chars)
        original_read = cli_mod._try_read_char
        original_enter = cli_mod._enter_cbreak
        original_exit = cli_mod._exit_cbreak

        def fake_read():
            try:
                return next(char_iter)
            except StopIteration:
                return None

        cli_mod._try_read_char = fake_read
        cli_mod._enter_cbreak = lambda: None   # no-op in tests
        cli_mod._exit_cbreak = lambda: None
        try:
            ui = MockPromptInterface(responses=extra_responses or [])
            _watch_mode(
                agent, ui, cfg,
                concept_worker or MockConceptArtWorker(),
                trellis_worker or MockTrellisWorker(),
                _tick=0,           # no real sleeping
                _status_interval=999,  # suppress periodic status
            )
            return ui
        finally:
            cli_mod._try_read_char = original_read
            cli_mod._enter_cbreak = original_enter
            cli_mod._exit_cbreak = original_exit

    def test_exits_on_single_q(self, agent, cfg):
        """Single 'q' keypress exits watch mode — no Enter needed."""
        ui = self._run_watch(agent, cfg, ['q'])
        assert any("menu" in m.lower() for m in ui.messages)

    def test_prints_banner_on_entry(self, agent, cfg):
        ui = self._run_watch(agent, cfg, ['q', '\n'])
        combined = "\n".join(ui.messages)
        assert "Watch mode" in combined

    def test_prints_initial_status(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = self._run_watch(agent, cfg, ['q', '\n'])
        combined = "\n".join(ui.messages)
        assert "pipe1" in combined

    def test_auto_surfaces_approval(self, agent, cfg):
        """When a pipeline enters CONCEPT_ART_REVIEW, watch mode shows the prompt."""
        agent.start_pipeline("pipe1", "desc")
        state_path = cfg.uncompleted_pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), state_path.parent, cfg)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(state_path)

        # After review is handled the 'q' keypress exits watch mode
        ui = self._run_watch(
            agent, cfg, ['q'],
            extra_responses=["approve 1 2"],
        )
        combined = "\n".join(ui.messages)
        assert "APPROVAL NEEDED" in combined

    def test_prints_diff_only_on_change(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = self._run_watch(agent, cfg, ['q', '\n'])
        # pipe1 status line should appear exactly once (initial print, no change)
        pipe1_lines = [m for m in ui.messages if "pipe1" in m and "]" in m]
        assert len(pipe1_lines) == 1


# ---------------------------------------------------------------------------
# _print_watch_diffs
# ---------------------------------------------------------------------------


class TestPrintWatchDiffs:
    def test_prints_new_pipeline(self, agent):
        agent.start_pipeline("pipe1", "desc")
        last = {}
        ui = MockPromptInterface()
        _print_watch_diffs(agent, ui, last)
        assert any("pipe1" in m for m in ui.messages)

    def test_does_not_reprint_unchanged(self, agent):
        agent.start_pipeline("pipe1", "desc")
        last = {}
        ui = MockPromptInterface()
        _print_watch_diffs(agent, ui, last)   # first call — prints
        ui2 = MockPromptInterface()
        _print_watch_diffs(agent, ui2, last)  # second call — state unchanged
        assert not any("pipe1" in m for m in ui2.messages)

    def test_reprints_after_status_change(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        last = {}
        ui = MockPromptInterface()
        _print_watch_diffs(agent, ui, last)

        # Change pipeline status
        sp = cfg.uncompleted_pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.MESH_REVIEW
        state.save(sp)

        ui2 = MockPromptInterface()
        _print_watch_diffs(agent, ui2, last)
        assert any("pipe1" in m for m in ui2.messages)


# ---------------------------------------------------------------------------
# _handle_priority_pipeline
# ---------------------------------------------------------------------------


class TestHandlePriorityPipeline:
    def _make_review_state(self, agent, cfg, name, status):
        agent.start_pipeline(name, "desc")
        state_path = cfg.uncompleted_pipelines_dir / name / "state.json"
        state = PipelineState.load(state_path)
        state.status = status
        state.save(state_path)
        return state_path

    def test_concept_art_review_calls_run_review(self, agent, cfg):
        state_path = self._make_review_state(
            agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW
        )
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), state_path.parent, cfg)
        state.save(state_path)

        ui = MockPromptInterface(responses=["approve 1 2"])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        state = PipelineState.load(state_path)
        assert any(ca.status.name == "APPROVED" for ca in state.concept_arts)

    def test_approved_concept_art_enqueues_mesh_generation(self, agent, cfg, broker):
        state_path = self._make_review_state(
            agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW
        )
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), state_path.parent, cfg)
        state.save(state_path)

        ui = MockPromptInterface(responses=["approve 1 2"])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        tasks = broker.get_tasks(task_type="mesh_generate")
        assert len(tasks) == 1

    def test_cancelled_concept_art_cancels_broker_tasks(self, agent, cfg, broker):
        state_path = self._make_review_state(
            agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW
        )
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), state_path.parent, cfg)
        state.save(state_path)

        ui = MockPromptInterface(responses=["cancel"])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        for t in broker.get_tasks(pipeline_name="p1"):
            assert t.status == "failed"

    def test_nonexistent_pipeline_does_not_raise(self, agent, cfg):
        ui = MockPromptInterface()
        _handle_priority_pipeline("no_such_pipe", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())


# ---------------------------------------------------------------------------
# run_cli
# ---------------------------------------------------------------------------


class TestRunCli:
    def test_quit_from_idle_menu_exits(self, agent, cfg):
        ui = MockPromptInterface(responses=["q"])
        run_cli(agent, ui, cfg,
                concept_worker=MockConceptArtWorker(),
                trellis_worker=MockTrellisWorker())
        assert any("stopped" in m.lower() or "goodbye" in m.lower() for m in ui.messages)

    def test_workers_started_and_stopped(self, agent, cfg):
        ui = MockPromptInterface(responses=["q"])
        run_cli(agent, ui, cfg,
                concept_worker=MockConceptArtWorker(),
                trellis_worker=MockTrellisWorker())
        assert len(agent._threads) == 0

    def test_run_cli_handles_new_pipeline_then_quit(self, agent, cfg):
        # menu=n, name, backend(blank), imgpath(blank), description, polys(blank), sym(blank), menu=q
        ui = MockPromptInterface(responses=["n", "mymodel", "", "", "a sphere", "", "", "q"])
        run_cli(agent, ui, cfg,
                concept_worker=MockConceptArtWorker(),
                trellis_worker=MockTrellisWorker())
        assert "mymodel" in agent.list_pipeline_names()


# ---------------------------------------------------------------------------
# _handle_priority_pipeline — MESH_REVIEW paths (the bug-prone dispatch layer)
# ---------------------------------------------------------------------------


class TestHandlePriorityPipelineMeshReview:
    """
    Every return value from run_mesh_review must be handled by
    _handle_priority_pipeline.  Missing branches produce silent infinite spins.
    """

    def _setup(self, agent, cfg, name="p1"):
        """
        Create a pipeline with 2 textured meshes ready for review
        (status = MESH_REVIEW, meshes = TEXTURE_DONE).
        """
        agent.start_pipeline(name, "desc")
        state_path = cfg.uncompleted_pipelines_dir / name / "state.json"
        pipeline_dir = state_path.parent
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        for ca in state.concept_arts:
            ca.status = ConceptArtStatus.APPROVED
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        run_mesh_texturing(state, MockTrellisWorker(), pipeline_dir, cfg)
        state.status = PipelineStatus.MESH_REVIEW
        state.save(state_path)
        return state_path

    # -- no_pending (the bug that was fixed) -----------------------------------

    def test_no_pending_requeues_mesh_generation(self, agent, cfg, broker):
        """
        Pipeline in MESH_REVIEW but all meshes already rejected (QUEUED):
        must enqueue mesh_generate instead of silently spinning.
        This is the exact bug that caused the program to become unresponsive.
        """
        state_path = self._setup(agent, cfg)
        state = PipelineState.load(state_path)
        for m in state.meshes:
            m.status = MeshStatus.QUEUED
        state.save(state_path)

        ui = MockPromptInterface()  # no responses — review should not show
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        tasks = broker.get_tasks(task_type="mesh_generate", pipeline_name="p1")
        assert len(tasks) == 1

    def test_no_pending_updates_pipeline_status_to_mesh_generating(self, agent, cfg):
        state_path = self._setup(agent, cfg)
        state = PipelineState.load(state_path)
        for m in state.meshes:
            m.status = MeshStatus.QUEUED
        state.save(state_path)

        ui = MockPromptInterface()
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        reloaded = PipelineState.load(state_path)
        assert reloaded.status == PipelineStatus.MESH_GENERATING

    def test_no_pending_returns_false(self, agent, cfg):
        state_path = self._setup(agent, cfg)
        state = PipelineState.load(state_path)
        for m in state.meshes:
            m.status = MeshStatus.QUEUED
        state.save(state_path)

        ui = MockPromptInterface()
        result = _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result is False

    # -- approved --------------------------------------------------------------

    def test_approved_exports_meshes(self, agent, cfg):
        self._setup(agent, cfg)
        ui = MockPromptInterface(responses=["approve asset_a", "approve asset_b"])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        # After export the pipeline is moved to completed_pipelines_dir
        completed_state = cfg.completed_pipelines_dir / "p1" / "state.json"
        reloaded = PipelineState.load(completed_state)
        exported = [m for m in reloaded.meshes if m.status == MeshStatus.EXPORTED]
        assert len(exported) == 2

    def test_approved_returns_false(self, agent, cfg):
        state_path = self._setup(agent, cfg)
        ui = MockPromptInterface(responses=["approve asset_a", "approve asset_b"])
        result = _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result is False

    # -- all_rejected ----------------------------------------------------------

    def test_all_rejected_requeues_mesh_generation(self, agent, cfg, broker):
        state_path = self._setup(agent, cfg)
        # reject both meshes; poly(blank), symmetry(blank) for each
        ui = MockPromptInterface(responses=["reject", "", "", "reject", "", ""])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        tasks = broker.get_tasks(task_type="mesh_generate", pipeline_name="p1")
        assert len(tasks) == 1

    def test_all_rejected_returns_false(self, agent, cfg):
        state_path = self._setup(agent, cfg)
        ui = MockPromptInterface(responses=["reject", "", "", "reject", "", ""])
        result = _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result is False

    # -- quit ------------------------------------------------------------------

    def test_quit_returns_true(self, agent, cfg):
        state_path = self._setup(agent, cfg)
        ui = MockPromptInterface(responses=["quit"])
        result = _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result is True

    # -- cancelled -------------------------------------------------------------

    def test_cancelled_sets_pipeline_status(self, agent, cfg):
        state_path = self._setup(agent, cfg)
        ui = MockPromptInterface(responses=["cancel"])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        reloaded = PipelineState.load(state_path)
        assert reloaded.status == PipelineStatus.CANCELLED

    def test_cancelled_returns_false(self, agent, cfg):
        state_path = self._setup(agent, cfg)
        ui = MockPromptInterface(responses=["cancel"])
        result = _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result is False


# ---------------------------------------------------------------------------
# _main_loop restart / re-entry scenarios
# ---------------------------------------------------------------------------


class TestMainLoopRestart:
    """
    Simulate starting the program with stale pipeline state from a prior session.
    These are the hardest bugs to catch without explicit restart tests.
    """

    def test_no_spin_on_mesh_review_with_all_meshes_rejected(self, agent, cfg):
        """
        Core regression test for the infinite-spin bug:
        a pipeline stuck in MESH_REVIEW with all meshes QUEUED (prior session
        rejected everything) must not cause the loop to spin without output.

        If the bug regresses, MockPromptInterface raises StopIteration because
        the loop never reaches the idle menu to consume the 'q' response.
        """
        agent.start_pipeline("stuck", "desc")
        state_path = cfg.uncompleted_pipelines_dir / "stuck" / "state.json"
        pipeline_dir = state_path.parent
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        for ca in state.concept_arts:
            ca.status = ConceptArtStatus.APPROVED
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        run_mesh_texturing(state, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state.meshes:
            m.status = MeshStatus.QUEUED         # simulate prior-session rejection
        state.status = PipelineStatus.MESH_REVIEW  # stuck here after restart
        state.save(state_path)

        # Workers NOT started — test _main_loop directly to avoid background
        # threads re-advancing the pipeline state during the test.
        ui = MockPromptInterface(responses=["q"])
        _main_loop(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        # Reaching here (without StopIteration) proves the loop exited normally.

    def test_mesh_review_resolved_state_persists(self, agent, cfg):
        """After resolving a stuck MESH_REVIEW, the persisted status is MESH_GENERATING."""
        agent.start_pipeline("stuck", "desc")
        state_path = cfg.uncompleted_pipelines_dir / "stuck" / "state.json"
        pipeline_dir = state_path.parent
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        for ca in state.concept_arts:
            ca.status = ConceptArtStatus.APPROVED
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        run_mesh_texturing(state, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state.meshes:
            m.status = MeshStatus.QUEUED
        state.status = PipelineStatus.MESH_REVIEW
        state.save(state_path)

        ui = MockPromptInterface(responses=["q"])
        _main_loop(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        reloaded = PipelineState.load(state_path)
        assert reloaded.status == PipelineStatus.MESH_GENERATING


# ---------------------------------------------------------------------------
# _manage_pipeline (pause / resume / cancel)
# ---------------------------------------------------------------------------


class TestManagePipeline:
    def test_no_pipelines_message(self, agent, cfg):
        ui = MockPromptInterface(responses=[])
        _manage_pipeline(agent, ui, cfg)
        assert any("No pipelines" in m for m in ui.messages)

    def test_cancel_on_empty_selection(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=[""])
        _manage_pipeline(agent, ui, cfg)
        assert any("Cancelled" in m for m in ui.messages)

    def test_pause_pipeline(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["1", "p"])
        _manage_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.status == PipelineStatus.PAUSED

    def test_cancel_pipeline_from_manage(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["1", "c"])
        _manage_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.status == PipelineStatus.CANCELLED

    def test_resume_paused_pipeline(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        # Pause it first directly via agent
        agent.pause_pipeline("pipe1", cfg)
        ui = MockPromptInterface(responses=["1", "r"])
        _manage_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.status != PipelineStatus.PAUSED

    def test_shows_pipeline_list(self, agent, cfg):
        agent.start_pipeline("mypipe", "desc")
        ui = MockPromptInterface(responses=[""])
        _manage_pipeline(agent, ui, cfg)
        assert any("mypipe" in m for m in ui.messages)


# ---------------------------------------------------------------------------
# _edit_pipeline
# ---------------------------------------------------------------------------


class TestEditPipeline:
    def test_no_editable_pipelines_message(self, agent, cfg):
        # Pipeline in MESH_GENERATING is not editable
        agent.start_pipeline("pipe1", "desc")
        state_path = cfg.uncompleted_pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(state_path)
        state.status = PipelineStatus.MESH_GENERATING
        state.save(state_path)

        ui = MockPromptInterface(responses=[])
        _edit_pipeline(agent, ui, cfg)
        assert any("No editable" in m for m in ui.messages)

    def test_cancel_on_empty_selection(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=[""])
        _edit_pipeline(agent, ui, cfg)
        assert any("Cancelled" in m for m in ui.messages)

    def test_edit_changes_description(self, agent, cfg):
        agent.start_pipeline("pipe1", "a sphere")
        # responses: pipeline number, new description, empty poly, empty symmetry
        ui = MockPromptInterface(responses=["1", "a cube", "", ""])
        _edit_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.description == "a cube"

    def test_edit_changes_poly_count(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        # responses: pipeline number, keep description, new poly count, empty symmetry
        ui = MockPromptInterface(responses=["1", "", "16000", ""])
        _edit_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.num_polys == 16000

    def test_edit_enables_symmetrize(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        # responses: number, keep desc, keep polys, enable sym, keep axis
        ui = MockPromptInterface(responses=["1", "", "", "y", ""])
        _edit_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.symmetrize is True

    def test_edit_disables_symmetrize(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        state_path = cfg.uncompleted_pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(state_path)
        state.symmetrize = True
        state.save(state_path)

        ui = MockPromptInterface(responses=["1", "", "", "n"])
        _edit_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.symmetrize is False

    def test_empty_description_keeps_original(self, agent, cfg):
        agent.start_pipeline("pipe1", "original desc")
        ui = MockPromptInterface(responses=["1", "", "", ""])
        _edit_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.description == "original desc"


# ---------------------------------------------------------------------------
# _retry_failed
# ---------------------------------------------------------------------------


class TestRetryFailed:
    def test_no_failures_message(self, agent, cfg):
        ui = MockPromptInterface(responses=[])
        _retry_failed(agent, ui)
        assert any("No pipelines have failed tasks" in m for m in ui.messages)

    def test_retry_resets_failed_task(self, agent, cfg, broker):
        agent.start_pipeline("pipe1", "desc")
        tasks = broker.get_tasks(pipeline_name="pipe1")
        assert tasks, "fixture should have enqueued a concept_art_generate task"
        broker.mark_failed(tasks[0].id, "test error")

        ui = MockPromptInterface(responses=["1"])
        _retry_failed(agent, ui)

        tasks_after = broker.get_tasks(pipeline_name="pipe1")
        assert any(t.status == "pending" for t in tasks_after)

    def test_retry_all_resets_all_pipelines(self, agent, cfg, broker):
        agent.start_pipeline("pipe1", "desc")
        agent.start_pipeline("pipe2", "desc")
        for name in ("pipe1", "pipe2"):
            tasks = broker.get_tasks(pipeline_name=name)
            broker.mark_failed(tasks[0].id, "error")

        ui = MockPromptInterface(responses=["a"])
        _retry_failed(agent, ui)

        for name in ("pipe1", "pipe2"):
            tasks = broker.get_tasks(pipeline_name=name)
            assert any(t.status == "pending" for t in tasks)

    def test_cancel_on_empty_input(self, agent, cfg, broker):
        agent.start_pipeline("pipe1", "desc")
        tasks = broker.get_tasks(pipeline_name="pipe1")
        broker.mark_failed(tasks[0].id, "error")

        ui = MockPromptInterface(responses=[""])
        _retry_failed(agent, ui)
        # task should remain failed (no retry happened)
        tasks_after = broker.get_tasks(pipeline_name="pipe1")
        assert all(t.status == "failed" for t in tasks_after)
