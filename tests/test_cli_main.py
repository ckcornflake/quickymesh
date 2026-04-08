"""
Tests for src/agent/cli_main.py
"""

from __future__ import annotations

import shutil

import yaml

from src.agent.cli_main import (
    _edit_pipeline,
    _handle_3d_approval,
    _handle_priority_pipeline,
    _idle_menu,
    _kill_pipeline_folder,
    _main_loop,
    _manage_pipeline,
    _print_watch_diffs,
    _retry_failed,
    _return_to_pipeline,
    _show_status,
    _start_3d_pipeline_from_file,
    _start_new_pipeline,
    _submit_approved_for_3d,
    _unhide_pipeline,
    _watch_mode,
    run_cli,
)
from src.agent.pipeline_agent import PipelineAgent
from src.broker import Broker
from src.config import Config
from src.concept_art_pipeline import generate_concept_arts
from src.prompt_interface.mock import MockPromptInterface
from src.state import (
    ConceptArtStatus,
    Pipeline3DState,
    Pipeline3DStatus,
    PipelineState,
    PipelineStatus,
)
from src.vram_arbiter import VRAMArbiter
from src.workers.concept_art import MockConceptArtWorker
from src.workers.screenshot import MockScreenshotWorker
from src.workers.trellis import MockTrellisWorker


import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _make_fake_image(cfg, filename="img.png") -> str:
    """Create a tiny fake image file and return its absolute path."""
    path = cfg.output_root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"FAKEIMAGE")
    return str(path)


def _make_3d_awaiting(agent, cfg, name="model_1_0") -> "Path":
    """Create a 3D pipeline in AWAITING_APPROVAL status. Returns state_path."""
    from pathlib import Path
    img_path = _make_fake_image(cfg)
    pipeline_dir = cfg.pipelines_dir / name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = pipeline_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    glb = mesh_dir / "textured.glb"
    glb.write_bytes(b"GLB")
    state = Pipeline3DState(
        name=name,
        input_image_path=img_path,
        num_polys=8000,
        status=Pipeline3DStatus.AWAITING_APPROVAL,
        textured_mesh_path=str(glb),
        pipeline_dir=str(pipeline_dir.relative_to(cfg.output_root)),
    )
    state_path = pipeline_dir / "state.json"
    state.save(state_path)
    return state_path


# ---------------------------------------------------------------------------
# _idle_menu
# ---------------------------------------------------------------------------


class TestIdleMenu:
    def test_returns_quit_on_q(self, agent, cfg):
        ui = MockPromptInterface(responses=["q"])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "quit"

    def test_returns_watch_on_w_numeric(self, agent, cfg):
        # [w] is now option 5 numerically (after [u] was added)
        import src.agent.cli_main as cli_mod
        original = cli_mod._watch_mode
        cli_mod._watch_mode = lambda *a, **kw: None
        try:
            ui = MockPromptInterface(responses=["5"])
            result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
            assert result == "watch"
        finally:
            cli_mod._watch_mode = original

    def test_returns_new_on_n(self, agent, cfg):
        # name → backend(default) → imgpath(blank) → description → polys → sym
        ui = MockPromptInterface(responses=["n", "mymodel", "", "", "a dragon", "", ""])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "new"

    def test_returns_watch_on_w(self, agent, cfg):
        import src.agent.cli_main as cli_mod
        original = cli_mod._watch_mode
        cli_mod._watch_mode = lambda *a, **kw: None
        try:
            ui = MockPromptInterface(responses=["w"])
            result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
            assert result == "watch"
        finally:
            cli_mod._watch_mode = original

    def test_returns_3d_on_3(self, agent, cfg):
        # empty name → cancel
        ui = MockPromptInterface(responses=["3", ""])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "3d"

    def test_returns_manage_on_m_with_no_pipelines(self, agent, cfg):
        # [m] with no pipelines → "No pipelines" message, returns "manage"
        ui = MockPromptInterface(responses=["m"])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "manage"

    def test_returns_manage_on_m(self, agent, cfg):
        # [m] with a pipeline, then cancel from action menu
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["m", "1", ""])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "manage"

    def test_returns_retry_on_t(self, agent, cfg):
        ui = MockPromptInterface(responses=["t"])
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "retry"

    def test_attention_note_shown_when_2d_needs_review(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)
        # q to exit — the attention note is embedded in the ask() prompt
        ui = MockPromptInterface(responses=["q"])
        _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        combined = "\n".join(ui.asked)
        assert "waiting for review" in combined

    def test_attention_note_shown_for_3d_awaiting_approval(self, agent, cfg):
        _make_3d_awaiting(agent, cfg, "m_1_0")
        ui = MockPromptInterface(responses=["q"])
        _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        combined = "\n".join(ui.asked)
        assert "waiting for review" in combined

    def test_hidden_2d_pipeline_not_shown_in_note(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.hidden = True
        state.save(sp)
        ui = MockPromptInterface(responses=["q"])
        _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        combined = "\n".join(ui.asked)
        assert "waiting for approval" not in combined


# ---------------------------------------------------------------------------
# _start_new_pipeline
# ---------------------------------------------------------------------------


class TestStartNewPipeline:
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
# _start_3d_pipeline_from_file
# ---------------------------------------------------------------------------


class TestStart3DPipelineFromFile:
    def test_creates_3d_pipeline(self, agent, cfg):
        img_path = _make_fake_image(cfg)
        ui = MockPromptInterface(responses=["myship", img_path, "", ""])
        _start_3d_pipeline_from_file(agent, ui, cfg)
        assert "u_myship" in agent.list_3d_pipeline_names()

    def test_prefixes_name_with_u(self, agent, cfg):
        img_path = _make_fake_image(cfg)
        ui = MockPromptInterface(responses=["myship", img_path, "", ""])
        _start_3d_pipeline_from_file(agent, ui, cfg)
        assert "u_myship" in agent.list_3d_pipeline_names()
        assert "myship" not in agent.list_pipeline_names()

    def test_cancels_on_empty_name(self, agent, cfg):
        ui = MockPromptInterface(responses=[""])
        _start_3d_pipeline_from_file(agent, ui, cfg)
        assert agent.list_3d_pipeline_names() == []

    def test_cancels_on_empty_path(self, agent, cfg):
        ui = MockPromptInterface(responses=["myship", ""])
        _start_3d_pipeline_from_file(agent, ui, cfg)
        assert agent.list_3d_pipeline_names() == []

    def test_informs_on_success(self, agent, cfg):
        img_path = _make_fake_image(cfg)
        ui = MockPromptInterface(responses=["myship", img_path, "", ""])
        _start_3d_pipeline_from_file(agent, ui, cfg)
        assert any("u_myship" in m for m in ui.messages)

    def test_uses_custom_polys(self, agent, cfg):
        img_path = _make_fake_image(cfg)
        ui = MockPromptInterface(responses=["myship", img_path, "16000", ""])
        _start_3d_pipeline_from_file(agent, ui, cfg)
        state = agent.get_3d_pipeline_state("u_myship")
        assert state.num_polys == 16000

    def test_rejects_duplicate_name(self, agent, cfg):
        img_path = _make_fake_image(cfg)
        ui = MockPromptInterface(responses=["myship", img_path, "", ""])
        _start_3d_pipeline_from_file(agent, ui, cfg)  # first time
        ui2 = MockPromptInterface(responses=["myship"])
        _start_3d_pipeline_from_file(agent, ui2, cfg)  # duplicate
        assert any("already exists" in m for m in ui2.messages)


# ---------------------------------------------------------------------------
# _return_to_pipeline
# ---------------------------------------------------------------------------


class TestReturnToPipeline:
    def _make_2d_review(self, agent, cfg, name="pipe1"):
        agent.start_pipeline(name, "desc")
        sp = cfg.pipelines_dir / name / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)
        return sp

    def test_no_idle_pipelines_message(self, agent, cfg):
        ui = MockPromptInterface(responses=[])
        _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())
        assert any("No idle" in m for m in ui.messages)

    def test_shows_2d_pipeline_in_list(self, agent, cfg):
        self._make_2d_review(agent, cfg)
        ui = MockPromptInterface(responses=[""])  # cancel selection
        _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())
        assert any("pipe1" in m for m in ui.messages)

    def test_cancel_on_empty_selection(self, agent, cfg):
        self._make_2d_review(agent, cfg)
        ui = MockPromptInterface(responses=[""])
        _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())
        assert any("Cancelled" in m for m in ui.messages)

    def test_returns_to_2d_pipeline_and_approves(self, agent, cfg):
        sp = self._make_2d_review(agent, cfg)
        pipeline_dir = sp.parent
        state = PipelineState.load(sp)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.save(sp)

        # Select pipeline 1, approve both images, skip 3D submission
        ui = MockPromptInterface(responses=["1", "approve 1 2", "n"])
        _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())

        state = PipelineState.load(sp)
        assert any(ca.status == ConceptArtStatus.APPROVED for ca in state.concept_arts)

    def test_shows_3d_idle_pipeline(self, agent, cfg):
        state_path = _make_3d_awaiting(agent, cfg, "mdl_1_0")
        # Move to IDLE so it shows up as "returnable"
        state = Pipeline3DState.load(state_path)
        state.status = Pipeline3DStatus.IDLE
        state.save(state_path)

        ui = MockPromptInterface(responses=[""])
        _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())
        assert any("mdl_1_0" in m for m in ui.messages)

    def test_hidden_pipeline_not_shown(self, agent, cfg):
        sp = self._make_2d_review(agent, cfg)
        state = PipelineState.load(sp)
        state.hidden = True
        state.save(sp)

        ui = MockPromptInterface(responses=[])
        _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())
        assert any("No idle" in m for m in ui.messages)

    def test_returns_false_normally(self, agent, cfg):
        ui = MockPromptInterface(responses=[])
        result = _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())
        assert result is False


# ---------------------------------------------------------------------------
# _manage_pipeline  (hide / restore / kill)
# ---------------------------------------------------------------------------


class TestManagePipeline:
    """
    Tests for the unified _manage_pipeline (pick a pipeline, then an action).
    New API: first prompt = pipeline number, second prompt = action letter.
    """

    def test_cancel_on_empty_pipeline_selection(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=[""])  # cancel at pipeline picker
        _manage_pipeline(agent, ui, cfg)  # should not raise

    def test_no_pipelines_message(self, agent, cfg):
        ui = MockPromptInterface(responses=[])
        _manage_pipeline(agent, ui, cfg)
        assert any("No pipelines" in m for m in ui.messages)

    def test_cancel_on_empty_action(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["1", ""])  # pick pipe1, cancel action
        _manage_pipeline(agent, ui, cfg)

    def test_unknown_action(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["1", "z"])
        _manage_pipeline(agent, ui, cfg)
        assert any("Unknown" in m for m in ui.messages)

    # ── hide ──────────────────────────────────────────────────────────────

    def test_hide_pipeline(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        # pick pipe1, choose hide
        ui = MockPromptInterface(responses=["1", "h"])
        _manage_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.hidden is True

    def test_hide_pipeline_informs_user(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["1", "h"])
        _manage_pipeline(agent, ui, cfg)
        assert any("pipe1" in m and "hidden" in m.lower() for m in ui.messages)

    def test_hidden_pipeline_offers_restore_not_hide(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.hidden = True
        state.save(sp)
        # pick pipe1, check action menu includes "restore" not "hide"
        ui = MockPromptInterface(responses=["1", ""])  # cancel after seeing menu
        _manage_pipeline(agent, ui, cfg)
        combined = "\n".join(ui.messages)
        assert "Restore" in combined or "restore" in combined

    # ── restore ────────────────────────────────────────────────────────────

    def test_restore_hidden_pipeline(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.hidden = True
        state.save(sp)

        ui = MockPromptInterface(responses=["1", "r"])
        _manage_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.hidden is False

    # ── kill ───────────────────────────────────────────────────────────────

    def test_kill_pipeline_confirmed(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        folder = cfg.pipelines_dir / "pipe1"
        assert folder.exists()

        ui = MockPromptInterface(responses=["1", "k", "confirm"])
        _manage_pipeline(agent, ui, cfg)
        assert not folder.exists()

    def test_kill_pipeline_aborted(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        folder = cfg.pipelines_dir / "pipe1"

        ui = MockPromptInterface(responses=["1", "k", ""])
        _manage_pipeline(agent, ui, cfg)
        assert folder.exists()

    def test_kill_pipeline_wrong_confirm(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        folder = cfg.pipelines_dir / "pipe1"

        ui = MockPromptInterface(responses=["1", "k", "yes"])
        _manage_pipeline(agent, ui, cfg)
        assert folder.exists()

    def test_kill_warns_before_deleting(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["1", "k", ""])
        _manage_pipeline(agent, ui, cfg)
        assert any("WARNING" in m or "CANNOT" in m for m in ui.messages)

    # ── return to review ───────────────────────────────────────────────────

    def test_return_to_2d_review_via_manage(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        pipeline_dir = sp.parent
        state = PipelineState.load(sp)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)

        # pick pipe1 → [a] review → approve → skip 3D
        ui = MockPromptInterface(responses=["1", "a", "approve 1 2", "n"])
        _manage_pipeline(agent, ui, cfg, MockConceptArtWorker())
        state = PipelineState.load(sp)
        assert any(ca.status == ConceptArtStatus.APPROVED for ca in state.concept_arts)

    # ── edit ──────────────────────────────────────────────────────────────

    def test_edit_pipeline_description_via_manage(self, agent, cfg):
        agent.start_pipeline("pipe1", "a sphere")
        # pick pipe1 → edit → new description → keep polys → keep sym
        ui = MockPromptInterface(responses=["1", "e", "a cube", "", ""])
        _manage_pipeline(agent, ui, cfg, MockConceptArtWorker())
        state = agent.get_pipeline_state("pipe1")
        assert state.description == "a cube"


# ---------------------------------------------------------------------------
# _unhide_pipeline
# ---------------------------------------------------------------------------


class TestUnhidePipeline:
    def test_no_hidden_pipelines_message(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=[])
        _unhide_pipeline(agent, ui, cfg)
        assert any("No hidden" in m for m in ui.messages)

    def test_lists_hidden_2d_pipeline(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.hidden = True
        state.save(sp)
        ui = MockPromptInterface(responses=[""])  # cancel
        _unhide_pipeline(agent, ui, cfg)
        assert any("pipe1" in m for m in ui.messages)

    def test_lists_hidden_3d_pipeline(self, agent, cfg):
        state_path = _make_3d_awaiting(agent, cfg, "mdl_1_0")
        state = Pipeline3DState.load(state_path)
        state.hidden = True
        state.save(state_path)
        ui = MockPromptInterface(responses=[""])  # cancel
        _unhide_pipeline(agent, ui, cfg)
        assert any("mdl_1_0" in m for m in ui.messages)

    def test_unhides_selected_pipeline(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.hidden = True
        state.save(sp)
        ui = MockPromptInterface(responses=["1"])
        _unhide_pipeline(agent, ui, cfg)
        state_after = PipelineState.load(sp)
        assert state_after.hidden is False

    def test_unhide_clears_dismissal(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.hidden = True
        state.save(sp)
        agent.dismiss_from_priority("pipe1")
        ui = MockPromptInterface(responses=["1"])
        _unhide_pipeline(agent, ui, cfg)
        assert not agent.is_dismissed_from_priority("pipe1")

    def test_idle_menu_routes_to_unhide(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.hidden = True
        state.save(sp)
        ui = MockPromptInterface(responses=["u", ""])  # choose u, cancel in unhide
        result = _idle_menu(agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())
        assert result == "unhide"


# ---------------------------------------------------------------------------
# _kill_pipeline_folder
# ---------------------------------------------------------------------------


class TestKillPipelineFolder:
    def test_deletes_folder(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        folder = cfg.pipelines_dir / "pipe1"
        assert folder.exists()
        _kill_pipeline_folder(agent, cfg, "pipe1")
        assert not folder.exists()

    def test_cancels_broker_tasks(self, agent, cfg, broker):
        agent.start_pipeline("pipe1", "desc")
        _kill_pipeline_folder(agent, cfg, "pipe1")
        tasks = broker.get_tasks(pipeline_name="pipe1")
        assert all(t.status in ("failed", "cancelled") or t.error == "cancelled" for t in tasks)

    def test_no_error_if_folder_missing(self, agent, cfg):
        # Should not raise even if folder doesn't exist
        _kill_pipeline_folder(agent, cfg, "nonexistent")


# ---------------------------------------------------------------------------
# _show_status
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

    def test_shows_no_2d_pipelines_when_empty(self, agent):
        ui = MockPromptInterface()
        _show_status(agent, ui)
        combined = "\n".join(ui.messages)
        assert "2D Pipeline" in combined
        assert "(none)" in combined

    def test_shows_pipeline_name_and_status(self, agent):
        agent.start_pipeline("pipe1", "desc1")
        ui = MockPromptInterface()
        _show_status(agent, ui)
        combined = "\n".join(ui.messages)
        assert "pipe1" in combined

    def test_shows_3d_pipeline(self, agent, cfg):
        _make_3d_awaiting(agent, cfg, "mdl_1_0")
        ui = MockPromptInterface()
        _show_status(agent, ui)
        combined = "\n".join(ui.messages)
        assert "mdl_1_0" in combined

    def test_shows_hidden_tag_for_hidden_pipeline(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.hidden = True
        state.save(sp)
        ui = MockPromptInterface()
        _show_status(agent, ui)
        combined = "\n".join(ui.messages)
        assert "hidden" in combined

    def test_returns_none(self, agent):
        ui = MockPromptInterface()
        result = _show_status(agent, ui)
        assert result is None

    def test_does_not_ask_for_approval_selection(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)
        ui = MockPromptInterface()
        _show_status(agent, ui)  # must not raise StopIteration


# ---------------------------------------------------------------------------
# _watch_mode
# ---------------------------------------------------------------------------


class TestWatchMode:
    def _run_watch(self, agent, cfg, chars, concept_worker=None, trellis_worker=None,
                   extra_responses=None):
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
        cli_mod._enter_cbreak = lambda: None
        cli_mod._exit_cbreak = lambda: None
        try:
            ui = MockPromptInterface(responses=extra_responses or [])
            _watch_mode(
                agent, ui, cfg,
                concept_worker or MockConceptArtWorker(),
                trellis_worker or MockTrellisWorker(),
                _tick=0,
                _status_interval=999,
            )
            return ui
        finally:
            cli_mod._try_read_char = original_read
            cli_mod._enter_cbreak = original_enter
            cli_mod._exit_cbreak = original_exit

    def test_exits_on_single_q(self, agent, cfg):
        ui = self._run_watch(agent, cfg, ['q'])
        assert any("menu" in m.lower() for m in ui.messages)

    def test_prints_banner_on_entry(self, agent, cfg):
        ui = self._run_watch(agent, cfg, ['q', '\n'])
        combined = "\n".join(ui.messages)
        assert "Watching" in combined

    def test_prints_initial_status(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = self._run_watch(agent, cfg, ['q', '\n'])
        combined = "\n".join(ui.messages)
        assert "pipe1" in combined

    def test_auto_surfaces_2d_approval(self, agent, cfg):
        """When a 2D pipeline enters CONCEPT_ART_REVIEW, watch mode shows the prompt."""
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        generate_concept_arts(state, MockConceptArtWorker(), sp.parent, cfg)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)

        ui = self._run_watch(
            agent, cfg, ['q'],
            extra_responses=["approve 1 2", "n"],
        )
        combined = "\n".join(ui.messages)
        assert "REVIEW NEEDED" in combined

    def test_auto_surfaces_3d_approval(self, agent, cfg):
        """When a 3D pipeline enters AWAITING_APPROVAL, watch mode shows the prompt."""
        _make_3d_awaiting(agent, cfg, "mdl_1_0")

        ui = self._run_watch(
            agent, cfg, ['q'],
            extra_responses=["approve"],
        )
        combined = "\n".join(ui.messages)
        assert "REVIEW NEEDED" in combined

    def test_prints_diff_only_on_change(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = self._run_watch(agent, cfg, ['q', '\n'])
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
        _print_watch_diffs(agent, ui, last)
        ui2 = MockPromptInterface()
        _print_watch_diffs(agent, ui2, last)
        assert not any("pipe1" in m for m in ui2.messages)

    def test_reprints_after_status_change(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        last = {}
        ui = MockPromptInterface()
        _print_watch_diffs(agent, ui, last)

        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)

        ui2 = MockPromptInterface()
        _print_watch_diffs(agent, ui2, last)
        assert any("pipe1" in m for m in ui2.messages)

    def test_prints_3d_pipeline(self, agent, cfg):
        _make_3d_awaiting(agent, cfg, "mdl_1_0")
        last = {}
        ui = MockPromptInterface()
        _print_watch_diffs(agent, ui, last)
        assert any("mdl_1_0" in m for m in ui.messages)


# ---------------------------------------------------------------------------
# _handle_priority_pipeline
# ---------------------------------------------------------------------------


class TestHandlePriorityPipeline:
    def _make_review_state(self, agent, cfg, name, status):
        agent.start_pipeline(name, "desc")
        state_path = cfg.pipelines_dir / name / "state.json"
        state = PipelineState.load(state_path)
        state.status = status
        state.save(state_path)
        return state_path

    def test_concept_art_review_calls_run_review(self, agent, cfg):
        state_path = self._make_review_state(
            agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW
        )
        pipeline_dir = state_path.parent
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.save(state_path)

        # approve + skip 3D submission
        ui = MockPromptInterface(responses=["approve 1 2", "n"])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        state = PipelineState.load(state_path)
        assert any(ca.status == ConceptArtStatus.APPROVED for ca in state.concept_arts)

    def test_approved_does_not_enqueue_mesh_when_user_declines(self, agent, cfg, broker):
        """If user declines 3D submission after approval, no mesh task is queued."""
        state_path = self._make_review_state(
            agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW
        )
        pipeline_dir = state_path.parent
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.save(state_path)
        broker.cancel_pipeline_tasks("p1")  # drain concept_art_generate

        ui = MockPromptInterface(responses=["approve 1 2", "n"])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        tasks = broker.get_tasks(task_type="mesh_generate")
        assert len(tasks) == 0

    def test_approve_then_submit_starts_3d_pipeline(self, agent, cfg):
        """If user accepts 3D submission, a 3D pipeline is started."""
        state_path = self._make_review_state(
            agent, cfg, "p1", PipelineStatus.CONCEPT_ART_REVIEW
        )
        pipeline_dir = state_path.parent
        state = PipelineState.load(state_path)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.save(state_path)

        ui = MockPromptInterface(responses=["approve 1 2", "y"])
        _handle_priority_pipeline("p1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())

        assert len(agent.list_3d_pipeline_names()) > 0

    def test_handles_3d_awaiting_approval(self, agent, cfg):
        """_handle_priority_pipeline dispatches to 3D review for AWAITING_APPROVAL."""
        state_path = _make_3d_awaiting(agent, cfg, "mdl_1_0")
        ui = MockPromptInterface(responses=["approve"])
        _handle_priority_pipeline("mdl_1_0", agent, ui, cfg, MockConceptArtWorker())
        state = Pipeline3DState.load(state_path)
        assert state.status == Pipeline3DStatus.IDLE

    def test_nonexistent_pipeline_does_not_raise(self, agent, cfg):
        ui = MockPromptInterface()
        _handle_priority_pipeline("no_such_pipe", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker())


# ---------------------------------------------------------------------------
# _handle_3d_approval
# ---------------------------------------------------------------------------


class TestHandle3DApproval:
    def test_approve_exports_mesh(self, agent, cfg):
        state_path = _make_3d_awaiting(agent, cfg)
        ui = MockPromptInterface(responses=["approve"])
        _handle_3d_approval("model_1_0", agent, ui, cfg)

        state = Pipeline3DState.load(state_path)
        assert state.status == Pipeline3DStatus.IDLE
        assert len(state.export_paths) == 1
        assert state.hidden is True  # auto-hidden after approval

    def test_approve_returns_false(self, agent, cfg):
        _make_3d_awaiting(agent, cfg)
        ui = MockPromptInterface(responses=["approve"])
        result = _handle_3d_approval("model_1_0", agent, ui, cfg)
        assert result is False

    def test_quit_returns_true(self, agent, cfg):
        _make_3d_awaiting(agent, cfg)
        ui = MockPromptInterface(responses=["quit"])
        result = _handle_3d_approval("model_1_0", agent, ui, cfg)
        assert result is True

    def test_unknown_action_loops(self, agent, cfg):
        _make_3d_awaiting(agent, cfg)
        ui = MockPromptInterface(responses=["flarp", "quit"])
        _handle_3d_approval("model_1_0", agent, ui, cfg)
        assert any("Unknown" in m for m in ui.messages)

    def test_regenerate_requeues_mesh(self, agent, cfg, broker):
        _make_3d_awaiting(agent, cfg)
        broker.cancel_pipeline_tasks("model_1_0")

        ui = MockPromptInterface(responses=["regenerate", ""])
        _handle_3d_approval("model_1_0", agent, ui, cfg)

        tasks = broker.get_tasks(pipeline_name="model_1_0")
        assert any(t.task_type == "mesh_generate" and t.status == "pending" for t in tasks)

    def test_regenerate_updates_polys(self, agent, cfg, broker):
        state_path = _make_3d_awaiting(agent, cfg)
        broker.cancel_pipeline_tasks("model_1_0")

        ui = MockPromptInterface(responses=["regenerate", "16000"])
        _handle_3d_approval("model_1_0", agent, ui, cfg)

        state = Pipeline3DState.load(state_path)
        assert state.num_polys == 16000

    def test_regenerate_returns_false(self, agent, cfg, broker):
        _make_3d_awaiting(agent, cfg)
        broker.cancel_pipeline_tasks("model_1_0")
        ui = MockPromptInterface(responses=["regenerate", ""])
        result = _handle_3d_approval("model_1_0", agent, ui, cfg)
        assert result is False

    def test_nonexistent_pipeline_returns_false(self, agent, cfg):
        ui = MockPromptInterface()
        result = _handle_3d_approval("nonexistent", agent, ui, cfg)
        assert result is False

    def test_export_version_increments_on_approve(self, agent, cfg):
        state_path = _make_3d_awaiting(agent, cfg)
        ui = MockPromptInterface(responses=["approve"])
        _handle_3d_approval("model_1_0", agent, ui, cfg)
        state = Pipeline3DState.load(state_path)
        assert state.export_version == 1  # was 0, now 1


# ---------------------------------------------------------------------------
# _submit_approved_for_3d
# ---------------------------------------------------------------------------


class TestSubmitApprovedFor3D:
    def _make_approved_2d_state(self, agent, cfg, name="mymodel"):
        """Create a 2D pipeline with approved concept art images."""
        agent.start_pipeline(name, "desc")
        sp = cfg.pipelines_dir / name / "state.json"
        pipeline_dir = sp.parent
        state = PipelineState.load(sp)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.concept_arts[0].status = ConceptArtStatus.APPROVED
        state.save(sp)
        return state, pipeline_dir

    def test_starts_3d_pipeline_for_approved(self, agent, cfg):
        state, pipeline_dir = self._make_approved_2d_state(agent, cfg)
        ui = MockPromptInterface(responses=["y"])
        _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)
        # CA index=0 (1-based: 1), version=0 → "mymodel_1_0"
        assert "mymodel_1_0" in agent.list_3d_pipeline_names()

    def test_skips_on_n_response(self, agent, cfg):
        state, pipeline_dir = self._make_approved_2d_state(agent, cfg)
        ui = MockPromptInterface(responses=["n"])
        _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)
        assert agent.list_3d_pipeline_names() == []

    def test_enter_submits_by_default(self, agent, cfg):
        state, pipeline_dir = self._make_approved_2d_state(agent, cfg)
        ui = MockPromptInterface(responses=[""])
        _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)
        assert "mymodel_1_0" in agent.list_3d_pipeline_names()

    def test_warns_on_existing_3d_pipeline(self, agent, cfg):
        state, pipeline_dir = self._make_approved_2d_state(agent, cfg)
        # Pre-create the 3D pipeline that would be derived
        img_path = str(pipeline_dir / "concept_art" / "concept_art_1_0.png")
        agent.start_3d_pipeline("mymodel_1_0", input_image_path=img_path, num_polys=8000)

        ui = MockPromptInterface(responses=["y", "n"])  # submit=yes, overwrite=no
        _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)
        assert any("already exists" in m for m in ui.messages)

    def test_overwrites_existing_3d_pipeline_when_confirmed(self, agent, cfg):
        state, pipeline_dir = self._make_approved_2d_state(agent, cfg)
        img_path = str(pipeline_dir / "concept_art" / "concept_art_1_0.png")
        agent.start_3d_pipeline("mymodel_1_0", input_image_path=img_path, num_polys=8000)

        ui = MockPromptInterface(responses=["y", "y"])  # submit=yes, overwrite=yes
        _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)

        # Pipeline should still exist (was killed and recreated)
        assert "mymodel_1_0" in agent.list_3d_pipeline_names()

    def test_no_approved_arts_does_nothing(self, agent, cfg):
        agent.start_pipeline("mymodel", "desc")
        sp = cfg.pipelines_dir / "mymodel" / "state.json"
        pipeline_dir = sp.parent
        state = PipelineState.load(sp)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        # Leave all arts in READY status (not APPROVED)
        state.save(sp)

        ui = MockPromptInterface()
        _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)
        assert agent.list_3d_pipeline_names() == []


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
        ui = MockPromptInterface(responses=["n", "mymodel", "", "", "a sphere", "", "", "q"])
        run_cli(agent, ui, cfg,
                concept_worker=MockConceptArtWorker(),
                trellis_worker=MockTrellisWorker())
        assert "mymodel" in agent.list_pipeline_names()


# ---------------------------------------------------------------------------
# _edit_pipeline
# ---------------------------------------------------------------------------


class TestEditPipeline:
    def test_no_editable_pipelines_message(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CANCELLED
        state.save(sp)

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
        ui = MockPromptInterface(responses=["1", "a cube", "", ""])
        _edit_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.description == "a cube"

    def test_edit_changes_poly_count(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["1", "", "16000", ""])
        _edit_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.num_polys == 16000

    def test_edit_enables_symmetrize(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        ui = MockPromptInterface(responses=["1", "", "", "y", ""])
        _edit_pipeline(agent, ui, cfg)
        state = agent.get_pipeline_state("pipe1")
        assert state.symmetrize is True

    def test_edit_disables_symmetrize(self, agent, cfg):
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.symmetrize = True
        state.save(sp)

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
        assert tasks
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
        tasks_after = broker.get_tasks(pipeline_name="pipe1")
        assert all(t.status == "failed" for t in tasks_after)


# ---------------------------------------------------------------------------
# Session-dismissal (priority loop) mechanism
# ---------------------------------------------------------------------------


class TestPriorityDismissal:
    """
    After the user handles (or non-fatally exits) a priority pipeline review,
    the CLI dismisses that pipeline for the session so the main loop does not
    immediately re-surface the same prompt. The user can still reach a
    dismissed pipeline via [r] Return to a pipeline.
    """

    def _make_2d_review(self, agent, cfg, name="pipe1"):
        agent.start_pipeline(name, "desc")
        sp = cfg.pipelines_dir / name / "state.json"
        pipeline_dir = sp.parent
        state = PipelineState.load(sp)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.save(sp)
        return sp

    def test_handle_priority_dismisses_2d_after_approve(self, agent, cfg):
        self._make_2d_review(agent, cfg, "pipe1")
        ui = MockPromptInterface(responses=["approve 1 2", "n"])
        _handle_priority_pipeline(
            "pipe1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker()
        )
        assert agent.is_dismissed_from_priority("pipe1")

    def test_handle_priority_does_not_dismiss_on_quit(self, agent, cfg):
        self._make_2d_review(agent, cfg, "pipe1")
        ui = MockPromptInterface(responses=["quit"])
        _handle_priority_pipeline(
            "pipe1", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker()
        )
        # Quit exits the CLI entirely — no point tracking the dismissal.
        assert not agent.is_dismissed_from_priority("pipe1")

    def test_handle_priority_dismisses_3d_after_approve(self, agent, cfg):
        # _handle_priority_pipeline no longer dismisses 3D pipelines — that is
        # now the responsibility of _manage_pipeline which can check the status
        # after the fact.  The pipeline stays undismissed so watch mode and
        # the attention note can still surface it if it ends up back in
        # AWAITING_APPROVAL (e.g. after a regenerate).
        _make_3d_awaiting(agent, cfg, "mdl_1_0")
        ui = MockPromptInterface(responses=["approve"])
        _handle_priority_pipeline(
            "mdl_1_0", agent, ui, cfg, MockConceptArtWorker(), MockTrellisWorker()
        )
        state = agent.get_3d_pipeline_state("mdl_1_0")
        # After approve the status should have moved to IDLE — it was handled.
        assert state.status == Pipeline3DStatus.IDLE

    def test_any_priority_skips_dismissed(self, agent, cfg):
        from src.agent.cli_main import _any_priority_pipeline
        self._make_2d_review(agent, cfg, "pipe1")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)

        assert _any_priority_pipeline(agent) == "pipe1"
        agent.dismiss_from_priority("pipe1")
        assert _any_priority_pipeline(agent) is None

    def test_all_needing_attention_skips_dismissed(self, agent, cfg):
        from src.agent.cli_main import _all_needing_attention
        self._make_2d_review(agent, cfg, "pipe1")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)

        assert "pipe1" in _all_needing_attention(agent)
        agent.dismiss_from_priority("pipe1")
        assert "pipe1" not in _all_needing_attention(agent)

    def test_return_to_pipeline_undismisses(self, agent, cfg):
        sp = self._make_2d_review(agent, cfg, "pipe1")
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)
        agent.dismiss_from_priority("pipe1")

        # pick pipe1 → then quit out of the review loop immediately
        ui = MockPromptInterface(responses=["1", "quit"])
        _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())
        # undismissed before the review loop ran, and since the user quit
        # (not a normal return), dismiss_from_priority is NOT called again.
        assert not agent.is_dismissed_from_priority("pipe1")

    def test_main_loop_always_shows_idle_menu_first(self, agent, cfg):
        """
        Even when a pipeline is awaiting review, the main loop shows the idle
        menu first — the user reaches pipelines via watch mode or manage, not
        by automatic interruption.
        """
        self._make_2d_review(agent, cfg, "pipe1")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.save(sp)

        # 'q' on the idle menu — the review loop must NOT appear first.
        ui = MockPromptInterface(responses=["q"])
        _main_loop(
            agent, ui, cfg,
            MockConceptArtWorker(), MockTrellisWorker(), None,
        )
        # The idle menu ask prompt should have appeared (not "Enter action").
        assert any("quickymesh" in p or "Start" in p for p in ui.asked)


# ---------------------------------------------------------------------------
# _return_to_pipeline — resets concept_art_sheet_shown
# ---------------------------------------------------------------------------


class TestReturnToPipelineResetsSheetFlag:
    def test_return_to_2d_resets_sheet_shown(self, agent, cfg):
        """
        When returning to an idle 2D pipeline, the review sheet should be
        re-shown (the user likely wants to see it again after leaving).
        """
        agent.start_pipeline("pipe1", "desc")
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        pipeline_dir = sp.parent
        state = PipelineState.load(sp)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.status = PipelineStatus.CONCEPT_ART_REVIEW
        state.concept_art_sheet_shown = True  # simulate a prior review session
        state.save(sp)

        # Select pipe1, then quit the review loop
        ui = MockPromptInterface(responses=["1", "quit"])
        _return_to_pipeline(agent, ui, cfg, MockConceptArtWorker())

        # show_image should have been called (sheet re-shown)
        assert len(ui.shown_images) >= 1


# ---------------------------------------------------------------------------
# _handle_3d_approval — show sheet once, detect silent export failure
# ---------------------------------------------------------------------------


class TestHandle3DApprovalFixes:
    def test_review_sheet_shown_only_once_across_loop(self, agent, cfg):
        """
        The review sheet should be shown once per call even when the user
        issues an unknown action and the loop iterates.
        """
        state_path = _make_3d_awaiting(agent, cfg, "mdl_1_0")
        # Attach a real-looking sheet path
        sheet = state_path.parent / "screenshots" / "reviewsheet.png"
        sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.write_bytes(b"SHEET")
        state = Pipeline3DState.load(state_path)
        state.review_sheet_path = str(sheet)
        state.save(state_path)

        # flarp → bogus → quit (3 iterations of the loop)
        ui = MockPromptInterface(responses=["flarp", "bogus", "quit"])
        _handle_3d_approval("mdl_1_0", agent, ui, cfg)
        assert len(ui.shown_images) == 1

    def test_approve_with_missing_mesh_file_does_not_lie(self, agent, cfg):
        """
        If the mesh glb file is missing, approve should surface an error
        instead of printing a misleading success message.
        """
        from pathlib import Path
        state_path = _make_3d_awaiting(agent, cfg, "mdl_1_0")
        state = Pipeline3DState.load(state_path)
        # Delete the glb file the fixture created
        Path(state.textured_mesh_path).unlink()
        state.save(state_path)

        ui = MockPromptInterface(responses=["approve", "quit"])
        _handle_3d_approval("mdl_1_0", agent, ui, cfg)
        assert any("Cannot export" in m for m in ui.messages)
        # Sanity: no export happened, no misleading success message.
        assert not any("Mesh exported" in m for m in ui.messages)


# ---------------------------------------------------------------------------
# _load_prefs — robust against malformed JSON
# ---------------------------------------------------------------------------


class TestLoadPrefs:
    def test_missing_file_returns_empty(self, cfg):
        from src.agent.cli_main import _load_prefs
        assert _load_prefs(cfg) == {}

    def test_malformed_json_returns_empty(self, cfg):
        from src.agent.cli_main import _load_prefs
        path = cfg.output_root / ".preferences.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        assert _load_prefs(cfg) == {}

    def test_valid_json_loads(self, cfg):
        from src.agent.cli_main import _load_prefs, _save_prefs
        _save_prefs(cfg, {"concept_art_backend": "flux"})
        assert _load_prefs(cfg) == {"concept_art_backend": "flux"}


# ---------------------------------------------------------------------------
# _submit_approved_for_3d — refuses to overwrite a 2D pipeline
# ---------------------------------------------------------------------------


class TestSubmitRefusesToClobber2D:
    def test_refuses_to_overwrite_2d_pipeline_with_derived_name(self, agent, cfg):
        # Create a 2D pipeline whose derived 3D name collides with an
        # unrelated 2D pipeline also happening to be called "mymodel_1_0".
        agent.start_pipeline("mymodel", "desc")
        sp = cfg.pipelines_dir / "mymodel" / "state.json"
        pipeline_dir = sp.parent
        state = PipelineState.load(sp)
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.concept_arts[0].status = ConceptArtStatus.APPROVED
        state.save(sp)

        # Pre-create a 2D pipeline at the derived name "mymodel_1_0"
        agent.start_pipeline("mymodel_1_0", "unrelated work")

        ui = MockPromptInterface(responses=["y"])  # yes, submit
        _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)

        # The pre-existing 2D pipeline must still be there and untouched.
        assert "mymodel_1_0" in agent.list_pipeline_names()
        victim = agent.get_pipeline_state("mymodel_1_0")
        assert victim is not None
        assert victim.description == "unrelated work"
        # And no 3D pipeline should have been created at that name.
        assert "mymodel_1_0" not in agent.list_3d_pipeline_names()
        assert any("Cannot create 3D pipeline" in m for m in ui.messages)


# ---------------------------------------------------------------------------
# recover_stalled_pipelines — handles INITIALIZING status
# ---------------------------------------------------------------------------


class TestRecoverStalledInitializing:
    def test_initializing_pipeline_re_enqueues_concept_art_generate(
        self, agent, cfg, broker
    ):
        agent.start_pipeline("pipe1", "desc")
        # Simulate a crash after start_pipeline but before the worker picked
        # up the task: status still INITIALIZING, broker task drained.
        sp = cfg.pipelines_dir / "pipe1" / "state.json"
        state = PipelineState.load(sp)
        state.status = PipelineStatus.INITIALIZING
        state.save(sp)
        broker.cancel_pipeline_tasks("pipe1")

        agent.recover_stalled_pipelines()
        tasks = broker.get_tasks(pipeline_name="pipe1", task_type="concept_art_generate")
        assert any(t.status == "pending" for t in tasks)
