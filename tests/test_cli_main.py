"""
Tests for src/agent/cli_main.py
"""

from __future__ import annotations

import yaml

from src.agent.cli_main import (
    _handle_priority_pipeline,
    _idle_menu,
    _print_watch_diffs,
    _show_status,
    _start_new_pipeline,
    _watch_mode,
    run_cli,
)
from src.agent.pipeline_agent import PipelineAgent
from src.broker import Broker
from src.config import Config
from src.concept_art_pipeline import generate_concept_arts
from src.prompt_interface.mock import MockPromptInterface
from src.state import PipelineState, PipelineStatus
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
        ui = MockPromptInterface(responses=["n", "mymodel", "a dragon", ""])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "new"

    def test_returns_watch_on_w(self, agent, cfg):
        # Watch mode exits immediately when given 'q' + newline via mock
        ui = MockPromptInterface(responses=["w", "q"])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker(),
                            )
        assert result == "watch"


# ---------------------------------------------------------------------------
# _start_new_pipeline
# ---------------------------------------------------------------------------


class TestStartNewPipeline:
    def test_creates_pipeline(self, agent):
        ui = MockPromptInterface(responses=["testpipe", "a red sphere", ""])
        _start_new_pipeline(agent, ui)
        assert "testpipe" in agent.list_pipeline_names()

    def test_informs_user_on_success(self, agent):
        ui = MockPromptInterface(responses=["testpipe", "a red sphere", ""])
        _start_new_pipeline(agent, ui)
        assert any("testpipe" in m for m in ui.messages)

    def test_cancels_on_empty_name(self, agent):
        ui = MockPromptInterface(responses=["", ""])
        _start_new_pipeline(agent, ui)
        assert agent.list_pipeline_names() == []

    def test_cancels_on_empty_description(self, agent):
        ui = MockPromptInterface(responses=["testpipe", ""])
        _start_new_pipeline(agent, ui)
        assert agent.list_pipeline_names() == []

    def test_uses_custom_polys(self, agent):
        ui = MockPromptInterface(responses=["testpipe", "a dragon", "12000"])
        _start_new_pipeline(agent, ui)
        state = agent.get_pipeline_state("testpipe")
        assert state.num_polys == 12000

    def test_uses_default_polys_on_empty_input(self, agent, cfg):
        ui = MockPromptInterface(responses=["testpipe", "a dragon", ""])
        _start_new_pipeline(agent, ui)
        state = agent.get_pipeline_state("testpipe")
        assert state.num_polys == cfg.num_polys


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

    def _run_watch(self, agent, cfg, chars, concept_worker=None, trellis_worker=None):
        """
        Drive watch mode by injecting a sequence of characters via
        a patched _try_read_char.  Returns the MockPromptInterface used.
        """
        import src.agent.cli_main as cli_mod

        char_iter = iter(chars)
        original = cli_mod._try_read_char

        def fake_read():
            try:
                return next(char_iter)
            except StopIteration:
                return None

        cli_mod._try_read_char = fake_read
        try:
            ui = MockPromptInterface()
            _watch_mode(
                agent, ui, cfg,
                concept_worker or MockConceptArtWorker(),
                trellis_worker or MockTrellisWorker(),
                _tick=0,           # no real sleeping
                _status_interval=999,  # suppress periodic status
            )
            return ui
        finally:
            cli_mod._try_read_char = original

    def test_exits_on_q_enter(self, agent, cfg):
        ui = self._run_watch(agent, cfg, ['q', '\n'])
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

        import src.agent.cli_main as cli_mod
        original = cli_mod._try_read_char

        # Simulate: no keypress until after the review surfaces and is handled
        # The review itself uses ui.ask() which goes through MockPromptInterface
        char_iter = iter(['q', '\n'])

        def fake_read():
            try:
                return next(char_iter)
            except StopIteration:
                return None

        cli_mod._try_read_char = fake_read
        try:
            # The mock will handle: approve review → then 'q' exits watch
            ui = MockPromptInterface(responses=["approve 1 2"])
            _watch_mode(
                agent, ui, cfg,
                MockConceptArtWorker(), MockTrellisWorker(),
                _tick=0,
                _status_interval=999,
            )
        finally:
            cli_mod._try_read_char = original

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
        ui = MockPromptInterface(responses=["n", "mymodel", "a sphere", "", "q"])
        run_cli(agent, ui, cfg,
                concept_worker=MockConceptArtWorker(),
                trellis_worker=MockTrellisWorker())
        assert "mymodel" in agent.list_pipeline_names()
