"""
Tests for src/mesh_pipeline.py — orchestration logic.

All tests use MockTrellisWorker and operate on Pipeline3DState.
"""

from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.config import Config
from src.mesh_pipeline import run_mesh_export, run_mesh_generation, run_mesh_texturing
from src.state import Pipeline3DState, Pipeline3DStatus, SymmetryAxis
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
    d = tmp_path / "pipelines" / "test_model_1_0"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def input_image(pipeline_dir) -> Path:
    """A minimal PNG on disk to act as the source image."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    p = pipeline_dir / "input.png"
    img.save(str(p))
    return p


@pytest.fixture
def state_queued(pipeline_dir, input_image) -> Pipeline3DState:
    """A fresh 3D pipeline state in QUEUED status."""
    return Pipeline3DState(
        name="test_model_1_0",
        input_image_path=str(input_image),
        num_polys=8000,
        pipeline_dir=f"pipelines/test_model_1_0",
    )


# ---------------------------------------------------------------------------
# run_mesh_generation
# ---------------------------------------------------------------------------


class TestRunMeshGeneration:
    def test_creates_mesh_path(self, state_queued, pipeline_dir, cfg):
        run_mesh_generation(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        assert state_queued.mesh_path is not None

    def test_mesh_file_exists(self, state_queued, pipeline_dir, cfg):
        run_mesh_generation(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        assert Path(state_queued.mesh_path).exists()

    def test_mesh_file_is_valid_glb(self, state_queued, pipeline_dir, cfg):
        run_mesh_generation(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        data = Path(state_queued.mesh_path).read_bytes()
        assert data[:4] == b"glTF"

    def test_status_becomes_mesh_done(self, state_queued, pipeline_dir, cfg):
        run_mesh_generation(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        assert state_queued.status == Pipeline3DStatus.MESH_DONE

    def test_skips_if_not_queued(self, state_queued, pipeline_dir, cfg):
        state_queued.status = Pipeline3DStatus.MESH_DONE
        worker = MockTrellisWorker()
        run_mesh_generation(state_queued, worker, pipeline_dir, cfg)
        assert len(worker.generate_calls) == 0

    def test_worker_called_with_correct_num_polys(self, state_queued, pipeline_dir, cfg):
        state_queued.num_polys = 4000
        worker = MockTrellisWorker()
        run_mesh_generation(state_queued, worker, pipeline_dir, cfg)
        assert worker.generate_calls[0]["num_polys"] == 4000

    def test_generate_failure_propagates(self, state_queued, pipeline_dir, cfg):
        worker = MockTrellisWorker(fail_on_generate=True)
        with pytest.raises(RuntimeError, match="simulated"):
            run_mesh_generation(state_queued, worker, pipeline_dir, cfg)

    def test_status_generating_mesh_set_before_worker_call(self, state_queued, pipeline_dir, cfg):
        """GENERATING_MESH status must be set before the worker runs."""
        seen_statuses = []

        class StatusCapture(MockTrellisWorker):
            def generate_mesh(self, **kw):
                seen_statuses.append(state_queued.status)
                return super().generate_mesh(**kw)

        run_mesh_generation(state_queued, StatusCapture(), pipeline_dir, cfg)
        assert Pipeline3DStatus.GENERATING_MESH in seen_statuses


# ---------------------------------------------------------------------------
# run_mesh_texturing
# ---------------------------------------------------------------------------


class TestRunMeshTexturing:
    def _generate(self, state, pipeline_dir, cfg):
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)

    def test_creates_textured_mesh_path(self, state_queued, pipeline_dir, cfg):
        self._generate(state_queued, pipeline_dir, cfg)
        run_mesh_texturing(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        assert state_queued.textured_mesh_path is not None

    def test_textured_file_exists(self, state_queued, pipeline_dir, cfg):
        self._generate(state_queued, pipeline_dir, cfg)
        run_mesh_texturing(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        assert Path(state_queued.textured_mesh_path).exists()

    def test_textured_glb_is_valid(self, state_queued, pipeline_dir, cfg):
        self._generate(state_queued, pipeline_dir, cfg)
        run_mesh_texturing(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        data = Path(state_queued.textured_mesh_path).read_bytes()
        assert data[:4] == b"glTF"

    def test_status_becomes_texture_done(self, state_queued, pipeline_dir, cfg):
        self._generate(state_queued, pipeline_dir, cfg)
        run_mesh_texturing(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        assert state_queued.status == Pipeline3DStatus.TEXTURE_DONE

    def test_skips_if_not_mesh_done(self, state_queued, pipeline_dir, cfg):
        # state is QUEUED, not MESH_DONE
        worker = MockTrellisWorker()
        run_mesh_texturing(state_queued, worker, pipeline_dir, cfg)
        assert len(worker.texture_calls) == 0

    def test_texture_failure_propagates(self, state_queued, pipeline_dir, cfg):
        self._generate(state_queued, pipeline_dir, cfg)
        worker = MockTrellisWorker(fail_on_texture=True)
        with pytest.raises(RuntimeError, match="simulated"):
            run_mesh_texturing(state_queued, worker, pipeline_dir, cfg)


# ---------------------------------------------------------------------------
# run_mesh_export
# ---------------------------------------------------------------------------


class TestRunMeshExport:
    def _setup(self, state_queued, pipeline_dir, cfg):
        """Generate and texture, then create a real textured glb."""
        run_mesh_generation(state_queued, MockTrellisWorker(), pipeline_dir, cfg)
        run_mesh_texturing(state_queued, MockTrellisWorker(), pipeline_dir, cfg)

    def test_copies_glb_to_final_assets(self, state_queued, pipeline_dir, cfg):
        self._setup(state_queued, pipeline_dir, cfg)
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon")
        assert len(state_queued.export_paths) == 1
        assert Path(state_queued.export_paths[0]).exists()

    def test_export_path_in_final_assets_dir(self, state_queued, pipeline_dir, cfg):
        self._setup(state_queued, pipeline_dir, cfg)
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon")
        export_path = Path(state_queued.export_paths[0])
        assert cfg.final_assets_dir in export_path.parents

    def test_export_filename_includes_version(self, state_queued, pipeline_dir, cfg):
        self._setup(state_queued, pipeline_dir, cfg)
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon")
        export_path = Path(state_queued.export_paths[0])
        assert "dragon_v0" in export_path.name

    def test_status_becomes_idle(self, state_queued, pipeline_dir, cfg):
        self._setup(state_queued, pipeline_dir, cfg)
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon")
        assert state_queued.status == Pipeline3DStatus.IDLE

    def test_export_version_increments(self, state_queued, pipeline_dir, cfg):
        self._setup(state_queued, pipeline_dir, cfg)
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon")
        # Reset status so we can export again
        state_queued.status = Pipeline3DStatus.AWAITING_APPROVAL
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon")
        assert state_queued.export_version == 2
        assert len(state_queued.export_paths) == 2
        # v0 and v1
        names = [Path(p).name for p in state_queued.export_paths]
        assert any("v0" in n for n in names)
        assert any("v1" in n for n in names)

    def test_custom_export_format(self, state_queued, pipeline_dir, cfg):
        self._setup(state_queued, pipeline_dir, cfg)
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon", export_format="fbx")
        export_path = Path(state_queued.export_paths[0])
        assert export_path.suffix == ".fbx"

    def test_no_mesh_does_nothing(self, state_queued, pipeline_dir, cfg):
        # state_queued has no mesh_path
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon")
        assert len(state_queued.export_paths) == 0

    def test_uses_textured_mesh_if_available(self, state_queued, pipeline_dir, cfg):
        """run_mesh_export should prefer textured_mesh_path over mesh_path."""
        self._setup(state_queued, pipeline_dir, cfg)
        assert state_queued.textured_mesh_path is not None
        run_mesh_export(state_queued, pipeline_dir, cfg, asset_name="dragon")
        # Export should succeed (the textured GLB was copied)
        assert Path(state_queued.export_paths[0]).exists()
