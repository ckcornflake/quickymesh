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
from src.mesh_pipeline import run_mesh_generation, run_mesh_texturing, run_mesh_review, run_mesh_export
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

    def test_skips_mesh_done_items_on_second_call(self, state_with_approved_arts, pipeline_dir, cfg):
        """Calling generate twice does not duplicate items already in MESH_DONE."""
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        count_after_first = len(state_with_approved_arts.meshes)
        run_mesh_generation(state_with_approved_arts, MockTrellisWorker(), pipeline_dir, cfg)
        assert len(state_with_approved_arts.meshes) == count_after_first

    def test_regenerates_queued_items_with_no_mesh_path(self, state_with_approved_arts, pipeline_dir, cfg):
        """
        A QUEUED item with no mesh_path (crash before generation started) must
        be regenerated — not silently skipped.  This was the cause of the
        'pipeline stuck in mesh_generating forever' bug.
        """
        # Simulate a crash: mesh items were created (QUEUED) but generation never ran.
        state = state_with_approved_arts
        for ca in state.concept_arts:
            state.meshes.append(
                MeshItem(concept_art_index=ca.index,
                         sub_name=f"{state.name}_{ca.index + 1}",
                         status=MeshStatus.QUEUED,
                         mesh_path=None)
            )

        worker = MockTrellisWorker()
        run_mesh_generation(state, worker, pipeline_dir, cfg)

        assert len(worker.generate_calls) == 2         # both were regenerated
        for m in state.meshes:
            assert m.status == MeshStatus.MESH_DONE
            assert m.mesh_path is not None

    def test_regenerates_queued_items_does_not_create_duplicates(self, state_with_approved_arts, pipeline_dir, cfg):
        """Regenerating QUEUED items reuses them; count must not grow."""
        state = state_with_approved_arts
        for ca in state.concept_arts:
            state.meshes.append(
                MeshItem(concept_art_index=ca.index,
                         sub_name=f"{state.name}_{ca.index + 1}",
                         status=MeshStatus.QUEUED)
            )
        original_count = len(state.meshes)

        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)

        assert len(state.meshes) == original_count

    def test_skips_approved_and_exported_items(self, state_with_approved_arts, pipeline_dir, cfg):
        """Items in terminal statuses (APPROVED, EXPORTED) must never be restarted."""
        state = state_with_approved_arts
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        for m in state.meshes:
            m.status = MeshStatus.EXPORTED   # simulate already-exported

        worker = MockTrellisWorker()
        run_mesh_generation(state, worker, pipeline_dir, cfg)
        assert len(worker.generate_calls) == 0

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
        # Prompt sequence: approve name, approve name (no upfront poly count prompts)
        ui = MockPromptInterface(["approve dragon_final", "approve dragon_2"])
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        assert result == "approved"

    def test_approved_mesh_has_final_name(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve my_asset", "approve my_asset_2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        approved = [m for m in state.meshes if m.status == MeshStatus.APPROVED]
        assert any(m.final_name == "my_asset" for m in approved)

    def test_approved_mesh_has_export_format(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve mymodel fbx", "approve other"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        fbx_mesh = next(m for m in state.meshes if m.final_name == "mymodel")
        assert fbx_mesh.export_format == "fbx"

    def test_cancel_returns_cancelled(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["cancel"])
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        assert result == "cancelled"
        assert state.status == PipelineStatus.CANCELLED

    def test_quit_returns_quit(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["quit"])
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        assert result == "quit"

    def test_reject_sets_mesh_back_to_queued(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        # reject: poly(blank), symmetry(blank); repeat for second mesh
        ui = MockPromptInterface(["reject", "", "", "reject", "", ""])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        for m in state.meshes:
            assert m.status == MeshStatus.QUEUED

    def test_poly_count_update_saved_to_state(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        # First mesh: reject with poly count update, symmetry(blank). Second: approve.
        ui = MockPromptInterface(["reject", "4000", "", "approve asset1", "approve asset2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        assert state.num_polys == 4000

    def test_symmetrize_update_saved_to_state(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        assert state.symmetrize is False
        # reject first mesh, enable symmetrize with x- axis; approve rest
        ui = MockPromptInterface(["reject", "", "x-", "approve asset1", "approve asset2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        assert state.symmetrize is True
        assert state.symmetry_axis.value == "x-"

    def test_symmetrize_off_saved_to_state(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        state.symmetrize = True
        # reject first mesh, disable symmetrize; approve rest
        ui = MockPromptInterface(["reject", "", "off", "approve asset1", "approve asset2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        assert state.symmetrize is False

    def test_pipeline_status_becomes_approved(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve a1", "approve a2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        assert state.status == PipelineStatus.APPROVED

    def test_no_meshes_awaiting_approval_returns_no_pending(self, pipeline_dir, cfg):
        state = PipelineState(
            name="test_model", description="x", num_polys=8000,
            pipeline_dir="uncompleted_pipelines/test_model",
        )
        ui = MockPromptInterface()
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        assert result == "no_pending"

    def test_state_saved_to_disk_after_approve(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup(state_with_approved_arts, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve saved_asset", "approve saved_2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        loaded = PipelineState.load(pipeline_dir / "state.json")
        assert any(m.final_name == "saved_asset" for m in loaded.meshes)


# ---------------------------------------------------------------------------
# run_mesh_export
# ---------------------------------------------------------------------------


class TestRunMeshExport:
    def _setup_approved_meshes(self, state_with_approved_arts, pipeline_dir, cfg):
        """Setup through review with approved meshes."""
        state = state_with_approved_arts
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        run_mesh_texturing(state, MockTrellisWorker(), pipeline_dir, cfg)
        ui = MockPromptInterface(["approve asset1", "approve asset2"])
        run_mesh_review(state, pipeline_dir, ui, cfg)
        state.save(pipeline_dir / "state.json")
        return state

    def test_copies_textured_glb_to_final_assets(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup_approved_meshes(state_with_approved_arts, pipeline_dir, cfg)
        run_mesh_export(state, pipeline_dir, cfg)
        for mesh_item in state.meshes:
            if mesh_item.status == MeshStatus.EXPORTED:
                assert mesh_item.export_path is not None
                assert Path(mesh_item.export_path).exists()

    def test_export_path_in_final_assets_dir(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup_approved_meshes(state_with_approved_arts, pipeline_dir, cfg)
        run_mesh_export(state, pipeline_dir, cfg)
        for mesh_item in state.meshes:
            if mesh_item.status == MeshStatus.EXPORTED:
                export_path = Path(mesh_item.export_path)
                assert export_path.parent.parent == cfg.final_assets_dir

    def test_creates_metadata_file(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup_approved_meshes(state_with_approved_arts, pipeline_dir, cfg)
        run_mesh_export(state, pipeline_dir, cfg)
        for mesh_item in state.meshes:
            if mesh_item.status == MeshStatus.EXPORTED:
                metadata_path = Path(mesh_item.export_path).parent / "metadata.txt"
                assert metadata_path.exists()
                content = metadata_path.read_text()
                assert mesh_item.final_name in content
                assert state.name in content

    def test_mesh_status_marked_exported(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup_approved_meshes(state_with_approved_arts, pipeline_dir, cfg)
        run_mesh_export(state, pipeline_dir, cfg)
        for mesh_item in state.meshes:
            assert mesh_item.status == MeshStatus.EXPORTED

    def test_skips_non_approved_meshes(self, state_with_approved_arts, pipeline_dir, cfg):
        state = self._setup_approved_meshes(state_with_approved_arts, pipeline_dir, cfg)
        # Manually mark one mesh as queued
        state.meshes[0].status = MeshStatus.QUEUED
        run_mesh_export(state, pipeline_dir, cfg)
        # Only one mesh should be exported
        exported = [m for m in state.meshes if m.status == MeshStatus.EXPORTED]
        assert len(exported) == 1

    def test_no_approved_meshes_does_nothing(self, pipeline_dir, cfg):
        state = PipelineState(
            name="test_model", description="x", num_polys=8000,
            pipeline_dir="uncompleted_pipelines/test_model",
        )
        # Should not raise
        run_mesh_export(state, pipeline_dir, cfg)
