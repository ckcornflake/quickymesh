"""
Tests for src/mesh_pipeline.py — orchestration logic.

All tests use MockTrellisWorker and MockPromptInterface.
"""

import io
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.concept_art_pipeline import generate_concept_arts
from src.config import Config
from src.mesh_pipeline import run_mesh_generation, run_mesh_texturing, run_mesh_review
from src.prompt_interface import MockPromptInterface
from src.state import (
    ConceptArtItem,
    ConceptArtStatus,
    MeshItem,
    MeshStatus,
    PipelineState,
    PipelineStatus,
)
from src.workers.concept_art import MockConceptArtWorker
from src.workers.trellis import MockTrellisWorker, _make_minimal_glb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path) -> Config:
    defaults = {
        "gemini": {"model": "m", "alternative_model": "a"},
        "generation": {
            "num_concept_arts": 2,
            "num_polys": 8000,
            "review_sheet_thumb_size": 64,
            "html_preview_size": 512,
            "export_format": "glb",
            "background_suffix": "plain white background",
        },
        "infrastructure": {
            "comfyui_url": "http://localhost:8188",
            "comfyui_install_dir": "/fake",
            "comfyui_poll_interval": 2.0,
            "comfyui_timeout": 600.0,
            "blender_path": "/fake/blender",
        },
        "output": {"root": str(tmp_path / "output")},
    }
    d = tmp_path / "defaults.yaml"
    d.write_text(yaml.dump(defaults))
    e = tmp_path / ".env"
    e.write_text("GEMINI_API_KEY=fake\n")
    return Config(defaults_path=d, env_path=e)


@pytest.fixture
def pipeline_dir(tmp_path) -> Path:
    d = tmp_path / "uncompleted_pipelines" / "test_model"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def state_with_approved_arts(pipeline_dir, cfg) -> PipelineState:
    """PipelineState with 2 approved concept arts that have image files on disk."""
    state = PipelineState(
        name="test_model",
        description="a red dragon",
        num_polys=8000,
        pipeline_dir="uncompleted_pipelines/test_model",
    )
    generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
    for ca in state.concept_arts:
        ca.status = ConceptArtStatus.APPROVED
    return state


# ---------------------------------------------------------------------------
# run_mesh_generation
# ---------------------------------------------------------------------------


class TestRunMeshGeneration:
    def test_creates_mesh_items_for_approved_arts(self, state_with_approved_arts, pipeline_dir, cfg):
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        assert len(state_with_approved_arts.meshes) == 2

    def test_mesh_items_have_correct_concept_art_index(self, state_with_approved_arts, pipeline_dir, cfg):
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        indices = {m.concept_art_index for m in state_with_approved_arts.meshes}
        assert indices == {0, 1}

    def test_mesh_paths_exist_on_disk(self, state_with_approved_arts, pipeline_dir, cfg):
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state_with_approved_arts.meshes:
            assert m.mesh_path is not None
            assert Path(m.mesh_path).exists()

    def test_mesh_files_are_valid_glb(self, state_with_approved_arts, pipeline_dir, cfg):
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state_with_approved_arts.meshes:
            data = Path(m.mesh_path).read_bytes()
            assert data[:4] == b"glTF"

    def test_mesh_status_is_mesh_done(self, state_with_approved_arts, pipeline_dir, cfg):
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state_with_approved_arts.meshes:
            assert m.status == MeshStatus.MESH_DONE

    def test_sub_name_pattern(self, state_with_approved_arts, pipeline_dir, cfg):
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        sub_names = {m.sub_name for m in state_with_approved_arts.meshes}
        assert "test_model_1" in sub_names
        assert "test_model_2" in sub_names

    def test_skips_non_approved_arts(self, pipeline_dir, cfg):
        state = PipelineState(
            name="test_model", description="x", num_polys=8000,
            pipeline_dir="uncompleted_pipelines/test_model",
        )
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        # Leave concept arts as READY (not approved)
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        assert len(state.meshes) == 0

    def test_skips_already_queued_mesh_items(self, state_with_approved_arts, pipeline_dir, cfg):
        """Calling generate twice should not duplicate mesh items."""
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        count_after_first = len(state_with_approved_arts.meshes)
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        assert len(state_with_approved_arts.meshes) == count_after_first

    def test_worker_called_with_correct_num_polys(self, state_with_approved_arts, pipeline_dir, cfg):
        state_with_approved_arts.num_polys = 4000
        worker = MockTrellisWorker()
        run_mesh_generation(state_with_approved_arts, worker, pipeline_dir, cfg)
        for call in worker.generate_calls:
            assert call["num_polys"] == 4000

    def test_generate_failure_propagates(self, state_with_approved_arts, pipeline_dir, cfg):
        worker = MockTrellisWorker(fail_on_generate=True)
        with pytest.raises(RuntimeError, match="simulated"):
            run_mesh_generation(state_with_approved_arts, worker, pipeline_dir, cfg)


# ---------------------------------------------------------------------------
# run_mesh_texturing
# ---------------------------------------------------------------------------


class TestRunMeshTexturing:
    def _generate_and_make_mesh_done(self, state, pipeline_dir, cfg):
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        # meshes are now MESH_DONE

    def test_textures_all_mesh_done_items(self, state_with_approved_arts, pipeline_dir, cfg):
        self._generate_and_make_mesh_done(state_with_approved_arts, pipeline_dir, cfg)
        run_mesh_texturing(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state_with_approved_arts.meshes:
            assert m.status == MeshStatus.TEXTURE_DONE

    def test_textured_mesh_paths_exist(self, state_with_approved_arts, pipeline_dir, cfg):
        self._generate_and_make_mesh_done(state_with_approved_arts, pipeline_dir, cfg)
        run_mesh_texturing(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state_with_approved_arts.meshes:
            assert m.textured_mesh_path is not None
            assert Path(m.textured_mesh_path).exists()

    def test_textured_glb_is_valid(self, state_with_approved_arts, pipeline_dir, cfg):
        self._generate_and_make_mesh_done(state_with_approved_arts, pipeline_dir, cfg)
        run_mesh_texturing(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state_with_approved_arts.meshes:
            data = Path(m.textured_mesh_path).read_bytes()
            assert data[:4] == b"glTF"

    def test_skips_non_mesh_done_items(self, state_with_approved_arts, pipeline_dir, cfg):
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        # Manually mark one as QUEUED to simulate partial state
        state_with_approved_arts.meshes[0].status = MeshStatus.QUEUED
        worker = MockTrellisWorker()
        run_mesh_texturing(state_with_approved_arts, worker, pipeline_dir, cfg)
        assert len(worker.texture_calls) == 1  # only 1 of 2

    def test_texture_failure_propagates(self, state_with_approved_arts, pipeline_dir, cfg):
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        worker = MockTrellisWorker(fail_on_texture=True)
        with pytest.raises(RuntimeError, match="simulated"):
            run_mesh_texturing(state_with_approved_arts, worker, pipeline_dir, cfg)


# ---------------------------------------------------------------------------
# run_mesh_review
# ---------------------------------------------------------------------------


class TestRunMeshReview:
    def _setup(self, state_with_approved_arts, pipeline_dir, cfg):
        """Full setup through texturing, return state ready for review."""
        state = state_with_approved_arts
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        run_mesh_texturing(state, MockTrellisWorker(), pipeline_dir, cfg)
        state.save(pipeline_dir / "state.json")
        return state

    def test_approve_returns_approved(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        # Prompt sequence: poly count (keep), approve name, poly count (keep), approve name
        ui = MockPromptInterface(["", "approve dragon_final", "", "approve dragon_2"])
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        assert result == "approved"

    def test_approved_mesh_has_final_name(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["", "approve my_asset", "", "approve my_asset_2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        approved = [m for m in state.meshes if m.status == MeshStatus.APPROVED]
        assert any(m.final_name == "my_asset" for m in approved)

    def test_approved_mesh_has_export_format(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["", "approve mymodel fbx", "", "approve other"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        fbx_mesh = next(m for m in state.meshes if m.final_name == "mymodel")
        assert fbx_mesh.export_format == "fbx"

    def test_cancel_returns_cancelled(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["", "cancel"])
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        assert result == "cancelled"
        assert state.status == PipelineStatus.CANCELLED

    def test_quit_returns_quit(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["", "quit"])
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        assert result == "quit"

    def test_reject_sets_mesh_back_to_queued(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["", "reject", "", "reject"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        for m in state.meshes:
            assert m.status == MeshStatus.QUEUED

    def test_poly_count_update_saved_to_state(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        # First mesh: update poly count, approve. Second: keep, approve.
        ui = MockPromptInterface(["4000", "approve asset1", "", "approve asset2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        assert state.num_polys == 4000

    def test_pipeline_status_becomes_approved(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["", "approve a1", "", "approve a2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        assert state.status == PipelineStatus.APPROVED

    def test_no_meshes_awaiting_approval_returns_approved(self, pipeline_dir, cfg):
        state = PipelineState(
            name="test_model", description="x", num_polys=8000,
            pipeline_dir="uncompleted_pipelines/test_model",
        )
        ui = MockPromptInterface()
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        assert result == "approved"

    def test_state_saved_to_disk_after_approve(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["", "approve saved_asset", "", "approve saved_2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        loaded = PipelineState.load(pipeline_dir / "state.json")
        assert any(m.final_name == "saved_asset" for m in loaded.meshes)
