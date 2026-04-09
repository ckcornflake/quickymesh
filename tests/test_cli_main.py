"""
Tests for src/cli/main.py — the new HTTP-client-based CLI loop.

These tests use ``unittest.mock.MagicMock(spec=QuickymeshClient)`` to fake
the API layer and :class:`MockPromptInterface` to feed keystrokes.  They
cover idle-menu dispatch, pipeline creation, management actions, concept
art review (including interruptible wait and busy-entry handling), 3D mesh
approval, and retry.

The old in-process ``cli_main`` tests in ``test_cli_main.py`` are
superseded by this file once the old CLI is deleted in Phase 4.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.cli import main as cli_main
from src.cli.client import (
    ConflictError,
    NotFoundError,
    QuickymeshAPIError,
    QuickymeshClient,
)
from src.prompt_interface.mock import MockPromptInterface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> dict:
    return {
        "output_root": "/tmp/out",
        "pipelines_dir": "/tmp/out/pipelines",
        "final_assets_dir": "/tmp/out/final",
        "background_suffix": "on a plain white background",
        "num_polys_default": 5000,
        "num_concept_arts_default": 4,
        "concept_art_image_size": 1024,
        "export_format": "glb",
        "gemini_api_key_present": True,
    }


@pytest.fixture
def client() -> MagicMock:
    c = MagicMock(spec=QuickymeshClient)
    c.list_pipelines.return_value = []
    c.list_3d_pipelines.return_value = []
    c.get_pipeline_or_none.return_value = None
    c.get_3d_pipeline_or_none.return_value = None
    c.get_pipeline_tasks.return_value = []
    c.get_3d_pipeline_tasks.return_value = []
    c.get_pipelines_with_failures.return_value = []
    return c


@pytest.fixture
def session() -> cli_main._Session:
    return cli_main._Session()


def _state_2d(
    name: str = "p",
    status: str = "concept_art_review",
    hidden: bool = False,
    description: str = "a robot",
    num_polys: int = 5000,
    concept_arts: list[dict] | None = None,
    symmetrize: bool = True,
    symmetry_axis: str = "x-",
) -> dict:
    if concept_arts is None:
        concept_arts = [
            {"index": i, "version": 0, "status": "ready"} for i in range(4)
        ]
    return {
        "name": name,
        "status": status,
        "hidden": hidden,
        "description": description,
        "num_polys": num_polys,
        "concept_arts": concept_arts,
        "symmetrize": symmetrize,
        "symmetry_axis": symmetry_axis,
    }


def _state_3d(
    name: str = "p_1_0",
    status: str = "awaiting_approval",
    hidden: bool = False,
    num_polys: int = 5000,
    source_2d_pipeline: str | None = "p",
    source_concept_art_index: int = 0,
    mesh_path: str | None = "/tmp/p_1_0/mesh.glb",
    textured_mesh_path: str | None = None,
    export_version: int = 0,
) -> dict:
    return {
        "name": name,
        "status": status,
        "hidden": hidden,
        "num_polys": num_polys,
        "source_2d_pipeline": source_2d_pipeline,
        "source_concept_art_index": source_concept_art_index,
        "mesh_path": mesh_path,
        "textured_mesh_path": textured_mesh_path,
        "export_version": export_version,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class TestSession:
    def test_dismiss_and_check(self):
        s = cli_main._Session()
        assert not s.is_dismissed("p")
        s.dismiss("p")
        assert s.is_dismissed("p")
        s.undismiss("p")
        assert not s.is_dismissed("p")


class TestConceptArtsBusy:
    def test_empty(self):
        assert cli_main._concept_arts_busy({"concept_arts": []}) is False

    def test_all_ready(self):
        state = {"concept_arts": [{"status": "ready"}, {"status": "ready"}]}
        assert cli_main._concept_arts_busy(state) is False

    def test_one_generating(self):
        state = {"concept_arts": [{"status": "ready"}, {"status": "generating"}]}
        assert cli_main._concept_arts_busy(state) is True

    def test_regenerating(self):
        state = {"concept_arts": [{"status": "regenerating"}]}
        assert cli_main._concept_arts_busy(state) is True


class TestAttention:
    def test_nothing_waiting(self, client, session):
        assert cli_main._all_needing_attention(client, session) == []

    def test_2d_in_review_surfaces(self, client, session):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d()
        assert cli_main._all_needing_attention(client, session) == [("p", "2d")]

    def test_3d_awaiting_surfaces(self, client, session):
        client.list_3d_pipelines.return_value = [{"name": "p_1_0"}]
        client.get_3d_pipeline_or_none.return_value = _state_3d()
        assert cli_main._all_needing_attention(client, session) == [("p_1_0", "3d")]

    def test_hidden_skipped(self, client, session):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d(hidden=True)
        assert cli_main._all_needing_attention(client, session) == []

    def test_dismissed_skipped(self, client, session):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d()
        session.dismiss("p")
        assert cli_main._all_needing_attention(client, session) == []


# ---------------------------------------------------------------------------
# run_cli (smoke)
# ---------------------------------------------------------------------------


class TestRunCli:
    def test_quits_cleanly(self, client, cfg):
        client.get_config.return_value = cfg
        ui = MockPromptInterface(["q"])
        cli_main.run_cli(client, ui)
        client.close.assert_called_once()
        assert any("Goodbye" in m for m in ui.messages)

    def test_exits_on_connection_error(self, client):
        from src.cli.client import ConnectionError as QmConnErr
        client.get_config.side_effect = QmConnErr("refused")
        ui = MockPromptInterface([])
        cli_main.run_cli(client, ui)
        assert any("Cannot reach" in m for m in ui.messages)


# ---------------------------------------------------------------------------
# _start_new_pipeline
# ---------------------------------------------------------------------------


class TestStartNewPipeline:
    def test_cancelled_on_empty_name(self, client, cfg):
        ui = MockPromptInterface([""])
        cli_main._start_new_pipeline(client, ui, cfg)
        client.create_pipeline.assert_not_called()
        client.create_pipeline_from_upload.assert_not_called()

    def test_happy_path_no_image(self, client, cfg, monkeypatch):
        monkeypatch.setattr(
            "src.cli.main.load_preferences",
            lambda: {"concept_art_backend": "gemini"},
        )
        monkeypatch.setattr("src.cli.main.save_preferences", lambda prefs: None)
        ui = MockPromptInterface([
            "mybot", "1", "", "a shiny robot", "3000", "y-",
        ])
        cli_main._start_new_pipeline(client, ui, cfg)
        client.create_pipeline.assert_called_once_with(
            "mybot", "a shiny robot", 3000,
            symmetrize=True, symmetry_axis="y-", concept_art_backend="gemini",
        )

    def test_happy_path_with_image(self, client, cfg, monkeypatch, tmp_path):
        monkeypatch.setattr("src.cli.main.load_preferences", lambda: {})
        monkeypatch.setattr("src.cli.main.save_preferences", lambda prefs: None)
        img = tmp_path / "base.png"
        img.write_bytes(b"\x89PNGfake")
        ui = MockPromptInterface([
            "mybot", "1", str(img), "make it red", "", "",
        ])
        cli_main._start_new_pipeline(client, ui, cfg)
        client.create_pipeline_from_upload.assert_called_once()
        call_args = client.create_pipeline_from_upload.call_args
        assert call_args.args[0] == "mybot"
        assert call_args.args[1] == "make it red"
        assert Path(call_args.args[2]) == img
        assert call_args.kwargs["symmetrize"] is True
        assert call_args.kwargs["symmetry_axis"] == "x-"

    def test_conflict_reported(self, client, cfg, monkeypatch):
        monkeypatch.setattr("src.cli.main.load_preferences", lambda: {})
        monkeypatch.setattr("src.cli.main.save_preferences", lambda prefs: None)
        client.create_pipeline.side_effect = ConflictError(409, "exists", "")
        ui = MockPromptInterface(["mybot", "1", "", "a robot", "", ""])
        cli_main._start_new_pipeline(client, ui, cfg)
        assert any("exists" in m for m in ui.messages)


class TestStart3DFromFile:
    def test_happy_path(self, client, cfg, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"png")
        client.create_3d_pipeline_from_upload.return_value = {"name": "u_mybot"}
        ui = MockPromptInterface(["mybot", str(img), "4000", ""])
        cli_main._start_3d_pipeline_from_file(client, ui, cfg)
        client.create_3d_pipeline_from_upload.assert_called_once()
        kwargs = client.create_3d_pipeline_from_upload.call_args.kwargs
        assert kwargs["num_polys"] == 4000
        assert kwargs["symmetrize"] is True
        assert kwargs["symmetry_axis"] == "x-"

    def test_cancelled_on_missing_name(self, client, cfg):
        ui = MockPromptInterface([""])
        cli_main._start_3d_pipeline_from_file(client, ui, cfg)
        client.create_3d_pipeline_from_upload.assert_not_called()


# ---------------------------------------------------------------------------
# _manage_pipeline
# ---------------------------------------------------------------------------


class TestManagePipeline:
    def test_no_pipelines(self, client, cfg, session):
        ui = MockPromptInterface([])
        cli_main._manage_pipeline(client, ui, cfg, session)
        assert any("No pipelines" in m for m in ui.messages)

    def test_hide_2d(self, client, cfg, session):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d(
            status="concept_art_generating"
        )
        ui = MockPromptInterface(["1", "h"])
        cli_main._manage_pipeline(client, ui, cfg, session)
        client.patch_pipeline.assert_called_once_with("p", hidden=True)

    def test_restore_2d(self, client, cfg, session):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d(
            status="concept_art_generating", hidden=True
        )
        ui = MockPromptInterface(["1", "r"])
        cli_main._manage_pipeline(client, ui, cfg, session)
        client.patch_pipeline.assert_called_once_with("p", hidden=False)

    def test_kill_requires_confirmation(self, client, cfg, session):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d(
            status="concept_art_generating"
        )
        ui = MockPromptInterface(["1", "k", ""])
        cli_main._manage_pipeline(client, ui, cfg, session)
        client.cancel_pipeline.assert_not_called()

    def test_kill_confirmed(self, client, cfg, session):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d(
            status="concept_art_generating"
        )
        ui = MockPromptInterface(["1", "k", "confirm"])
        cli_main._manage_pipeline(client, ui, cfg, session)
        client.cancel_pipeline.assert_called_once_with("p")
        client.patch_pipeline.assert_called_once_with("p", hidden=True)

    def test_kill_3d(self, client, cfg, session):
        client.list_3d_pipelines.return_value = [{"name": "p_1_0"}]
        client.get_3d_pipeline_or_none.return_value = _state_3d(status="queued")
        ui = MockPromptInterface(["1", "k", "confirm"])
        cli_main._manage_pipeline(client, ui, cfg, session)
        client.cancel_3d_pipeline.assert_called_once_with("p_1_0")
        client.patch_3d_pipeline.assert_called_once_with("p_1_0", hidden=True)

    def test_back_does_nothing(self, client, cfg, session):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d(
            status="concept_art_generating"
        )
        ui = MockPromptInterface(["1", "b"])
        cli_main._manage_pipeline(client, ui, cfg, session)
        client.patch_pipeline.assert_not_called()


class TestUnhide:
    def test_none_hidden(self, client):
        ui = MockPromptInterface([])
        cli_main._unhide_pipeline(client, ui)
        assert any("No hidden" in m for m in ui.messages)

    def test_unhides_selection(self, client):
        client.list_pipelines.return_value = [{"name": "p"}]
        client.get_pipeline_or_none.return_value = _state_2d(hidden=True)
        ui = MockPromptInterface(["1"])
        cli_main._unhide_pipeline(client, ui)
        client.patch_pipeline.assert_called_once_with("p", hidden=False)


class TestEditPipeline:
    def test_rejects_non_editable(self, client, cfg):
        state = _state_2d(status="cancelled")
        ui = MockPromptInterface([])
        cli_main._edit_pipeline(client, ui, cfg, "p", state)
        client.patch_pipeline.assert_not_called()
        assert any("cannot be edited" in m for m in ui.messages)

    def test_patches_description_only(self, client, cfg):
        state = _state_2d(status="concept_art_review")
        ui = MockPromptInterface(["a new description", "", ""])
        cli_main._edit_pipeline(client, ui, cfg, "p", state)
        client.patch_pipeline.assert_called_once_with(
            "p", description="a new description"
        )

    def test_patches_polys_and_symmetry(self, client, cfg):
        state = _state_2d(status="concept_art_review")
        ui = MockPromptInterface(["", "8000", "y", "z+"])
        cli_main._edit_pipeline(client, ui, cfg, "p", state)
        client.patch_pipeline.assert_called_once_with(
            "p", num_polys=8000, symmetrize=True, symmetry_axis="z+",
        )

    def test_no_changes(self, client, cfg):
        state = _state_2d(status="concept_art_review")
        ui = MockPromptInterface(["", "", ""])
        cli_main._edit_pipeline(client, ui, cfg, "p", state)
        client.patch_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# _run_concept_art_review
# ---------------------------------------------------------------------------


class TestConceptArtReview:
    def test_busy_entry_back(self, client, cfg):
        state = _state_2d(
            concept_arts=[{"index": 0, "version": 0, "status": "regenerating"}]
        )
        ui = MockPromptInterface(["b"])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "back"

    def test_busy_entry_quit(self, client, cfg):
        state = _state_2d(
            concept_arts=[{"index": 0, "version": 0, "status": "generating"}]
        )
        ui = MockPromptInterface(["q"])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "quit"

    def test_busy_entry_wait_then_ready(self, client, cfg, monkeypatch):
        state = _state_2d(
            concept_arts=[{"index": 0, "version": 0, "status": "regenerating"}]
        )
        new_state = _state_2d()
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("ready", new_state),
        )
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["w", "menu"])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "back"

    def test_menu_returns_back(self, client, cfg):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["menu"])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "back"

    def test_quit_returns_quit(self, client, cfg):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["quit"])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "quit"

    def test_approve_then_submit_yes(self, client, cfg):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        client.create_3d_pipeline_from_ref.return_value = {"name": "p_1_0"}
        ui = MockPromptInterface(["approve 1 2", ""])
        result = cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        assert result == "approved"
        assert client.create_3d_pipeline_from_ref.call_count == 2

    def test_approve_but_decline_submit(self, client, cfg):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["approve 1", "n", "menu"])
        result = cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        assert result == "back"
        client.create_3d_pipeline_from_ref.assert_not_called()

    def test_approve_invalid_indices_reprompts(self, client, cfg):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["approve 99", "menu"])
        result = cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        assert result == "back"
        client.create_3d_pipeline_from_ref.assert_not_called()

    def test_approve_without_indices_reprompts(self, client, cfg):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["approve", "menu"])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "back"

    def test_regenerate_single_happy(self, client, cfg, monkeypatch):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("ready", _state_2d()),
        )
        ui = MockPromptInterface(["regenerate", "2", "menu"])
        cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        client.regenerate_concept_art.assert_called_once_with(
            "p", indices=[1], description_override=None,
        )

    def test_regenerate_all_with_new_description(self, client, cfg, monkeypatch):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("ready", state),
        )
        ui = MockPromptInterface(["regenerate all", "a new prompt", "menu"])
        cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        client.regenerate_concept_art.assert_called_once_with(
            "p", indices=[0, 1, 2, 3], description_override="a new prompt",
        )

    def test_regenerate_interrupted_returns_back(self, client, cfg, monkeypatch):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("interrupted", None),
        )
        ui = MockPromptInterface(["regenerate", "1"])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "back"

    def test_modify_happy(self, client, cfg, monkeypatch):
        state = _state_2d()
        post_wait_state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("ready", post_wait_state),
        )
        ui = MockPromptInterface(["modify", "1", "make it blue", "menu"])
        cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        client.modify_concept_art.assert_called_once_with(
            "p", 0, "make it blue", source_version=None,
        )

    def test_modify_interrupted_returns_back(self, client, cfg, monkeypatch):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("interrupted", None),
        )
        ui = MockPromptInterface(["modify", "1", "make it blue"])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "back"

    def test_modify_unsupported_by_server(self, client, cfg):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        client.modify_concept_art.side_effect = ConflictError(
            409, "not supported", ""
        )
        ui = MockPromptInterface(["modify", "1", "make it blue", "menu"])
        cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        assert any("not supported" in m for m in ui.messages)

    def test_restyle_happy(self, client, cfg, monkeypatch):
        state = _state_2d()
        post_wait_state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("ready", post_wait_state),
        )
        ui = MockPromptInterface([
            "restyle", "2", "zerg biomechanical", "", "", "menu",
        ])
        cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        client.restyle_concept_art.assert_called_once()
        call = client.restyle_concept_art.call_args
        assert call.args[0] == "p"
        assert call.args[1] == 1
        assert call.args[2] == "zerg biomechanical"
        assert "blurry" in call.kwargs["negative"]
        assert call.kwargs["denoise"] == 0.75
        assert call.kwargs["source_version"] is None

    def test_modify_older_version(self, client, cfg, monkeypatch):
        # Slot 0 is on version 2: user should be prompted to pick a source
        # version, and picking "1" routes through as source_version=1.
        state = _state_2d(concept_arts=[
            {"index": 0, "version": 2, "status": "ready"},
            {"index": 1, "version": 0, "status": "ready"},
        ])
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("ready", state),
        )
        ui = MockPromptInterface([
            "modify", "1", "1", "redo with fur", "menu",
        ])
        cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        client.modify_concept_art.assert_called_once_with(
            "p", 0, "redo with fur", source_version=1,
        )

    def test_restyle_older_version(self, client, cfg, monkeypatch):
        state = _state_2d(concept_arts=[
            {"index": 0, "version": 3, "status": "ready"},
            {"index": 1, "version": 0, "status": "ready"},
        ])
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("ready", state),
        )
        ui = MockPromptInterface([
            "restyle", "1", "2", "cyberpunk", "", "", "menu",
        ])
        cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        call = client.restyle_concept_art.call_args
        assert call.args[0] == "p"
        assert call.args[1] == 0
        assert call.args[2] == "cyberpunk"
        assert call.kwargs["source_version"] == 2

    def test_modify_version_prompt_skipped_when_only_v0(self, client, cfg, monkeypatch):
        # All slots at v0 → no version prompt, modify goes straight through
        state = _state_2d()  # defaults to version: 0 everywhere
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("ready", state),
        )
        ui = MockPromptInterface(["modify", "1", "make it blue", "menu"])
        cli_main._run_concept_art_review(client, ui, cfg, "p", state)
        client.modify_concept_art.assert_called_once_with(
            "p", 0, "make it blue", source_version=None,
        )

    def test_restyle_interrupted_returns_back(self, client, cfg, monkeypatch):
        state = _state_2d()
        client.get_concept_art_sheet.return_value = Path("/tmp/sheet.png")
        monkeypatch.setattr(
            cli_main, "_wait_for_concept_art_ready",
            lambda *a, **kw: ("interrupted", None),
        )
        ui = MockPromptInterface([
            "restyle", "2", "zerg biomechanical", "", "",
        ])
        assert cli_main._run_concept_art_review(client, ui, cfg, "p", state) == "back"


class TestSubmitApprovedFor3D:
    def test_empty_returns_false(self, client):
        ui = MockPromptInterface([])
        assert cli_main._submit_approved_for_3d(
            client, ui, "p", _state_2d(), set()
        ) is False
        client.create_3d_pipeline_from_ref.assert_not_called()

    def test_user_declines(self, client):
        ui = MockPromptInterface(["n"])
        assert cli_main._submit_approved_for_3d(
            client, ui, "p", _state_2d(), {0, 1}
        ) is False
        client.create_3d_pipeline_from_ref.assert_not_called()

    def test_submits_all(self, client):
        client.create_3d_pipeline_from_ref.return_value = {"name": "p_x_0"}
        ui = MockPromptInterface([""])
        result = cli_main._submit_approved_for_3d(
            client, ui, "p", _state_2d(num_polys=7000, symmetry_axis="z-"), {0, 2},
        )
        assert result is True
        assert client.create_3d_pipeline_from_ref.call_count == 2
        kwargs_list = [
            c.kwargs for c in client.create_3d_pipeline_from_ref.call_args_list
        ]
        indices = sorted(k["concept_art_index"] for k in kwargs_list)
        assert indices == [0, 2]
        for kw in kwargs_list:
            assert kw["source_2d_pipeline"] == "p"
            assert kw["num_polys"] == 7000
            assert kw["symmetry_axis"] == "z-"

    def test_partial_conflict_continues(self, client):
        client.create_3d_pipeline_from_ref.side_effect = [
            ConflictError(409, "exists", ""),
            {"name": "p_3_0"},
        ]
        ui = MockPromptInterface([""])
        result = cli_main._submit_approved_for_3d(
            client, ui, "p", _state_2d(), {0, 2},
        )
        assert result is True
        assert client.create_3d_pipeline_from_ref.call_count == 2


class TestHandle3DApproval:
    def test_menu_returns_false(self, client):
        client.get_3d_pipeline.return_value = _state_3d()
        client.get_3d_review_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["menu"])
        assert cli_main._handle_3d_approval(client, ui, "p_1_0") is False

    def test_quit_returns_true(self, client):
        client.get_3d_pipeline.return_value = _state_3d()
        client.get_3d_review_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["quit"])
        assert cli_main._handle_3d_approval(client, ui, "p_1_0") is True

    def test_approve_calls_approve_and_hide(self, client):
        client.get_3d_pipeline.return_value = _state_3d()
        client.get_3d_review_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["approve"])
        cli_main._handle_3d_approval(client, ui, "p_1_0")
        client.approve_3d_mesh.assert_called_once_with("p_1_0")
        client.patch_3d_pipeline.assert_called_once_with("p_1_0", hidden=True)

    def test_regenerate_with_new_polys(self, client):
        client.get_3d_pipeline.return_value = _state_3d()
        client.get_3d_review_sheet.return_value = Path("/tmp/sheet.png")
        # Inputs: action, new polys, symmetry (blank = keep)
        ui = MockPromptInterface(["regenerate", "9000", ""])
        cli_main._handle_3d_approval(client, ui, "p_1_0")
        client.reject_3d_mesh.assert_called_once_with("p_1_0", num_polys=9000)

    def test_regenerate_keep_polys(self, client):
        client.get_3d_pipeline.return_value = _state_3d()
        client.get_3d_review_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["regenerate", "", ""])
        cli_main._handle_3d_approval(client, ui, "p_1_0")
        client.reject_3d_mesh.assert_called_once_with("p_1_0")

    def test_regenerate_change_symmetry_axis(self, client):
        client.get_3d_pipeline.return_value = _state_3d()
        client.get_3d_review_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["regenerate", "", "y+"])
        cli_main._handle_3d_approval(client, ui, "p_1_0")
        client.reject_3d_mesh.assert_called_once_with(
            "p_1_0", symmetrize=True, symmetry_axis="y+",
        )

    def test_regenerate_disable_symmetry(self, client):
        client.get_3d_pipeline.return_value = _state_3d()
        client.get_3d_review_sheet.return_value = Path("/tmp/sheet.png")
        ui = MockPromptInterface(["regenerate", "", "off"])
        cli_main._handle_3d_approval(client, ui, "p_1_0")
        client.reject_3d_mesh.assert_called_once_with("p_1_0", symmetrize=False)

    def test_not_awaiting_returns_false(self, client):
        client.get_3d_pipeline.return_value = _state_3d(status="queued")
        ui = MockPromptInterface([])
        assert cli_main._handle_3d_approval(client, ui, "p_1_0") is False
        client.approve_3d_mesh.assert_not_called()

    def test_approve_conflict_reported(self, client):
        client.get_3d_pipeline.return_value = _state_3d()
        client.get_3d_review_sheet.return_value = Path("/tmp/sheet.png")
        client.approve_3d_mesh.side_effect = ConflictError(409, "no mesh", "")
        ui = MockPromptInterface(["approve", "menu"])
        cli_main._handle_3d_approval(client, ui, "p_1_0")
        assert any("no mesh" in m for m in ui.messages)


class TestRetryFailed:
    def test_nothing_failed(self, client):
        ui = MockPromptInterface([])
        cli_main._retry_failed(client, ui)
        assert any("No pipelines" in m for m in ui.messages)

    def test_cancelled(self, client):
        client.get_pipelines_with_failures.return_value = ["p1", "p2"]
        ui = MockPromptInterface([""])
        cli_main._retry_failed(client, ui)
        client.retry_pipeline.assert_not_called()

    def test_retry_single_2d(self, client):
        client.get_pipelines_with_failures.return_value = ["p1"]
        client.retry_pipeline.return_value = 3
        ui = MockPromptInterface(["1"])
        cli_main._retry_failed(client, ui)
        client.retry_pipeline.assert_called_once_with("p1")

    def test_retry_falls_back_to_3d_on_404(self, client):
        client.get_pipelines_with_failures.return_value = ["p_1_0"]
        client.retry_pipeline.side_effect = NotFoundError(404, "not 2d", "")
        client.retry_3d_pipeline.return_value = 2
        ui = MockPromptInterface(["1"])
        cli_main._retry_failed(client, ui)
        client.retry_3d_pipeline.assert_called_once_with("p_1_0")

    def test_retry_all(self, client):
        client.get_pipelines_with_failures.return_value = ["p1", "p2"]
        client.retry_pipeline.return_value = 1
        ui = MockPromptInterface(["a"])
        cli_main._retry_failed(client, ui)
        assert client.retry_pipeline.call_count == 2


# ---------------------------------------------------------------------------
# _wait_for_concept_art_ready — interruptibility + outcomes
# ---------------------------------------------------------------------------


class TestWaitForConceptArtReady:
    def test_ready_immediately(self, client, monkeypatch):
        client.get_pipeline.return_value = _state_2d()
        monkeypatch.setattr(cli_main, "_enter_cbreak", lambda: None)
        monkeypatch.setattr(cli_main, "_exit_cbreak", lambda: None)
        ui = MockPromptInterface([])
        outcome, state = cli_main._wait_for_concept_art_ready(
            client, ui, "p", timeout=1.0, poll=0.01,
        )
        assert outcome == "ready"
        assert state is not None

    def test_interrupted_by_keypress(self, client, monkeypatch):
        busy = _state_2d(
            concept_arts=[{"index": 0, "version": 0, "status": "regenerating"}]
        )
        client.get_pipeline.return_value = busy
        monkeypatch.setattr(cli_main, "_enter_cbreak", lambda: None)
        monkeypatch.setattr(cli_main, "_exit_cbreak", lambda: None)
        monkeypatch.setattr(cli_main, "_try_read_char", lambda: "q")
        monkeypatch.setattr(cli_main.time, "sleep", lambda s: None)
        ui = MockPromptInterface([])
        outcome, state = cli_main._wait_for_concept_art_ready(
            client, ui, "p", timeout=5.0, poll=0.1,
        )
        assert outcome == "interrupted"
        assert state is None

    def test_server_error(self, client, monkeypatch):
        client.get_pipeline.side_effect = QuickymeshAPIError(500, "boom", "")
        monkeypatch.setattr(cli_main, "_enter_cbreak", lambda: None)
        monkeypatch.setattr(cli_main, "_exit_cbreak", lambda: None)
        ui = MockPromptInterface([])
        outcome, state = cli_main._wait_for_concept_art_ready(
            client, ui, "p", timeout=1.0, poll=0.01,
        )
        assert outcome == "error"
        assert state is None

    def test_timeout(self, client, monkeypatch):
        busy = _state_2d(
            concept_arts=[{"index": 0, "version": 0, "status": "regenerating"}]
        )
        client.get_pipeline.return_value = busy
        monkeypatch.setattr(cli_main, "_enter_cbreak", lambda: None)
        monkeypatch.setattr(cli_main, "_exit_cbreak", lambda: None)
        monkeypatch.setattr(cli_main, "_try_read_char", lambda: None)
        monkeypatch.setattr(cli_main.time, "sleep", lambda s: None)
        t = [1000.0]
        def fake_time():
            t[0] += 10.0
            return t[0]
        monkeypatch.setattr(cli_main.time, "time", fake_time)
        ui = MockPromptInterface([])
        outcome, state = cli_main._wait_for_concept_art_ready(
            client, ui, "p", timeout=5.0, poll=0.1,
        )
        assert outcome == "timeout"
        assert state is None
