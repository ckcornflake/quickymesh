"""
Tests for src/workers/screenshot.py and src/screenshot_pipeline.py.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from PIL import Image

from src.concept_art_pipeline import generate_concept_arts
from src.config import Config
from src.mesh_pipeline import run_mesh_generation, run_mesh_texturing
from src.screenshot_pipeline import make_html_preview, run_cleanup, run_screenshots
from src.state import ConceptArtStatus, MeshItem, MeshStatus, PipelineState, SymmetryAxis
from src.concept_art_pipeline import generate_concept_arts
from src.mesh_pipeline import run_mesh_generation, run_mesh_texturing
from src.workers.concept_art import MockConceptArtWorker
from src.workers.screenshot import (
    DEFAULT_VIEWS,
    BlenderScreenshotWorker,
    MockScreenshotWorker,
)
from src.workers.trellis import MockTrellisWorker


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
def state_texture_done(pipeline_dir, cfg) -> PipelineState:
    """State with 2 meshes in TEXTURE_DONE status."""
    state = PipelineState(
        name="test_model",
        description="a red dragon",
        num_polys=8000,
        pipeline_dir="uncompleted_pipelines/test_model",
    )
    generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
    for ca in state.concept_arts:
        ca.status = __import__("src.state", fromlist=["ConceptArtStatus"]).ConceptArtStatus.APPROVED
    run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
    run_mesh_texturing(state, MockTrellisWorker(), pipeline_dir, cfg)
    return state


# ---------------------------------------------------------------------------
# MockScreenshotWorker
# ---------------------------------------------------------------------------


class TestMockScreenshotWorker:
    def _make_glb(self, tmp_path) -> Path:
        from src.workers.trellis import _make_minimal_glb
        p = tmp_path / "mesh.glb"
        p.write_bytes(_make_minimal_glb())
        return p

    def test_creates_png_for_each_view(self, tmp_path):
        worker = MockScreenshotWorker()
        glb = self._make_glb(tmp_path)
        out_dir = tmp_path / "shots"
        paths = worker.take_screenshots(glb, out_dir)
        assert len(paths) == len(DEFAULT_VIEWS)
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_files_are_valid_images(self, tmp_path):
        worker = MockScreenshotWorker()
        glb = self._make_glb(tmp_path)
        paths = worker.take_screenshots(glb, tmp_path / "shots")
        for p in paths:
            img = Image.open(p)
            assert img.mode == "RGB"

    def test_view_subset_honoured(self, tmp_path):
        worker = MockScreenshotWorker()
        glb = self._make_glb(tmp_path)
        paths = worker.take_screenshots(glb, tmp_path / "shots", views=["front", "back"])
        assert len(paths) == 2
        names = {p.name for p in paths}
        assert "review_front.png" in names
        assert "review_back.png" in names

    def test_records_calls(self, tmp_path):
        worker = MockScreenshotWorker()
        glb = self._make_glb(tmp_path)
        worker.take_screenshots(glb, tmp_path / "shots", use_hdri=True)
        assert len(worker.calls) == 1
        assert worker.calls[0]["use_hdri"] is True

    def test_fail_raises(self, tmp_path):
        worker = MockScreenshotWorker(fail=True)
        glb = self._make_glb(tmp_path)
        with pytest.raises(RuntimeError, match="simulated Blender failure"):
            worker.take_screenshots(glb, tmp_path / "shots")

    def test_creates_output_dir(self, tmp_path):
        worker = MockScreenshotWorker()
        glb = self._make_glb(tmp_path)
        out_dir = tmp_path / "deep" / "nested" / "shots"
        worker.take_screenshots(glb, out_dir)
        assert out_dir.exists()


# ---------------------------------------------------------------------------
# BlenderScreenshotWorker (mocked subprocess)
# ---------------------------------------------------------------------------


class TestBlenderScreenshotWorker:
    def _make_worker(self, blender_path="/fake/blender", repo_root=None):
        return BlenderScreenshotWorker(blender_path=blender_path, repo_root=repo_root)

    def _make_glb(self, tmp_path) -> Path:
        from src.workers.trellis import _make_minimal_glb
        p = tmp_path / "mesh.glb"
        p.write_bytes(_make_minimal_glb())
        return p

    def _fake_successful_run(self, output_dir: Path, views=None):
        """Side-effect: create fake PNG files, return success CompletedProcess."""
        import subprocess
        views = views or DEFAULT_VIEWS
        for view in views:
            img = Image.new("RGB", (32, 32), (100, 100, 100))
            img.save(str(output_dir / f"review_{view}.png"))
        return MagicMock(returncode=0, stderr="", stdout="Done. 6 views rendered.")

    def test_constructs_blender_command(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out_dir = tmp_path / "shots"
        worker = self._make_worker()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: self._fake_successful_run(out_dir)
            worker.take_screenshots(glb, out_dir)

        cmd = mock_run.call_args[0][0]
        assert "--background" in cmd
        assert "--python" in cmd
        assert str(glb.resolve()) in cmd
        assert str(out_dir.resolve()) in cmd

    def test_matcap_flag_when_use_hdri_false(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out_dir = tmp_path / "shots"
        worker = self._make_worker()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: self._fake_successful_run(out_dir)
            worker.take_screenshots(glb, out_dir, use_hdri=False)

        cmd = mock_run.call_args[0][0]
        assert "--matcap" in cmd
        assert "--hdri" not in cmd

    def test_hdri_flag_when_use_hdri_true(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out_dir = tmp_path / "shots"
        worker = self._make_worker()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: self._fake_successful_run(out_dir)
            worker.take_screenshots(glb, out_dir, use_hdri=True)

        cmd = mock_run.call_args[0][0]
        assert "--hdri" in cmd
        assert "--matcap" not in cmd

    def test_returns_existing_png_paths(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out_dir = tmp_path / "shots"
        worker = self._make_worker()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: self._fake_successful_run(out_dir)
            paths = worker.take_screenshots(glb, out_dir)

        assert len(paths) == len(DEFAULT_VIEWS)
        for p in paths:
            assert p.exists()

    def test_raises_on_blender_error(self, tmp_path):
        glb = self._make_glb(tmp_path)
        worker = self._make_worker()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Blender crashed")
            with pytest.raises(RuntimeError, match="Blender screenshot failed"):
                worker.take_screenshots(glb, tmp_path / "shots")

    def test_views_subset_passed_to_blender(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out_dir = tmp_path / "shots"
        worker = self._make_worker()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: self._fake_successful_run(
                out_dir, views=["front", "back"]
            )
            worker.take_screenshots(glb, out_dir, views=["front", "back"])

        cmd = mock_run.call_args[0][0]
        views_idx = cmd.index("--views") + 1
        assert cmd[views_idx] == "front,back"


# ---------------------------------------------------------------------------
# make_html_preview
# ---------------------------------------------------------------------------


class TestMakeHtmlPreview:
    def _make_glb(self, tmp_path) -> Path:
        from src.workers.trellis import _make_minimal_glb
        p = tmp_path / "mesh.glb"
        p.write_bytes(_make_minimal_glb())
        return p

    def test_creates_html_file(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out = tmp_path / "preview.html"
        result = make_html_preview(glb, out)
        assert result == out
        assert out.exists()

    def test_output_is_html(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out = tmp_path / "preview.html"
        make_html_preview(glb, out)
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content or "<html" in content

    def test_contains_threejs(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out = tmp_path / "preview.html"
        make_html_preview(glb, out)
        content = out.read_text(encoding="utf-8")
        assert "three" in content.lower() or "THREE" in content

    def test_size_injected_into_html(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out = tmp_path / "preview.html"
        make_html_preview(glb, out, size=384)
        content = out.read_text(encoding="utf-8")
        assert "384px" in content

    def test_creates_parent_dirs(self, tmp_path):
        glb = self._make_glb(tmp_path)
        out = tmp_path / "deep" / "nested" / "preview.html"
        make_html_preview(glb, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# run_screenshots (orchestration)
# ---------------------------------------------------------------------------


class TestRunScreenshots:
    def test_advances_texture_done_to_screenshot_done(self, state_texture_done, pipeline_dir, cfg):
        run_screenshots(state_texture_done, MockScreenshotWorker(), pipeline_dir, cfg)
        for m in state_texture_done.meshes:
            assert m.status == MeshStatus.SCREENSHOT_DONE

    def test_screenshot_dir_set_on_mesh_item(self, state_texture_done, pipeline_dir, cfg):
        run_screenshots(state_texture_done, MockScreenshotWorker(), pipeline_dir, cfg)
        for m in state_texture_done.meshes:
            assert m.screenshot_dir is not None
            assert Path(m.screenshot_dir).exists()

    def test_review_sheet_created(self, state_texture_done, pipeline_dir, cfg):
        run_screenshots(state_texture_done, MockScreenshotWorker(), pipeline_dir, cfg)
        for m in state_texture_done.meshes:
            assert m.review_sheet_path is not None
            assert Path(m.review_sheet_path).exists()

    def test_review_sheet_is_valid_image(self, state_texture_done, pipeline_dir, cfg):
        run_screenshots(state_texture_done, MockScreenshotWorker(), pipeline_dir, cfg)
        for m in state_texture_done.meshes:
            img = Image.open(m.review_sheet_path)
            assert img.mode == "RGB"

    def test_html_preview_created(self, state_texture_done, pipeline_dir, cfg):
        run_screenshots(state_texture_done, MockScreenshotWorker(), pipeline_dir, cfg)
        for m in state_texture_done.meshes:
            assert m.html_preview_path is not None
            assert Path(m.html_preview_path).exists()

    def test_html_preview_is_html(self, state_texture_done, pipeline_dir, cfg):
        run_screenshots(state_texture_done, MockScreenshotWorker(), pipeline_dir, cfg)
        for m in state_texture_done.meshes:
            content = Path(m.html_preview_path).read_text(encoding="utf-8")
            assert "<html" in content.lower() or "<!DOCTYPE" in content

    def test_skips_non_texture_done_meshes(self, pipeline_dir, cfg):
        state = PipelineState(
            name="test_model", description="x", num_polys=8000,
            pipeline_dir="uncompleted_pipelines/test_model",
        )
        state.meshes = [MeshItem(concept_art_index=0, sub_name="test_model_1",
                                  status=MeshStatus.MESH_DONE)]
        worker = MockScreenshotWorker()
        run_screenshots(state, worker, pipeline_dir, cfg)
        assert len(worker.calls) == 0
        assert state.meshes[0].status == MeshStatus.MESH_DONE

    def test_uses_textured_mesh_path_if_available(self, state_texture_done, pipeline_dir, cfg):
        worker = MockScreenshotWorker()
        run_screenshots(state_texture_done, worker, pipeline_dir, cfg)
        for call, mesh_item in zip(worker.calls, state_texture_done.meshes):
            assert call["mesh_path"] == Path(mesh_item.textured_mesh_path)

    def test_uses_hdri_when_textured(self, state_texture_done, pipeline_dir, cfg):
        worker = MockScreenshotWorker()
        run_screenshots(state_texture_done, worker, pipeline_dir, cfg)
        for call in worker.calls:
            assert call["use_hdri"] is True

    def test_screenshot_failure_propagates(self, state_texture_done, pipeline_dir, cfg):
        with pytest.raises(RuntimeError, match="simulated Blender failure"):
            run_screenshots(state_texture_done, MockScreenshotWorker(fail=True), pipeline_dir, cfg)

    def test_screenshot_pending_set_before_render(self, state_texture_done, pipeline_dir, cfg):
        """Verify SCREENSHOT_PENDING is set on the item before the worker is called."""
        seen_statuses = []
        original_take = MockScreenshotWorker.take_screenshots

        class StatusCapture(MockScreenshotWorker):
            def take_screenshots(self, mesh_path, output_dir, **kw):
                seen_statuses.append(state_texture_done.meshes[0].status)
                return super().take_screenshots(mesh_path, output_dir, **kw)

        run_screenshots(state_texture_done, StatusCapture(), pipeline_dir, cfg)
        assert MeshStatus.SCREENSHOT_PENDING in seen_statuses


# ---------------------------------------------------------------------------
# run_cleanup (mesh shade-smooth + optional symmetrize)
# ---------------------------------------------------------------------------


class TestRunCleanup:
    def _state_texture_done(self, pipeline_dir, cfg):
        """Fresh state with 2 meshes in TEXTURE_DONE, mirroring the fixture above."""
        state = PipelineState(
            name="test_model", description="a red dragon", num_polys=8000,
            pipeline_dir="uncompleted_pipelines/test_model",
        )
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        for ca in state.concept_arts:
            ca.status = ConceptArtStatus.APPROVED
        run_mesh_generation(state, MockTrellisWorker(), pipeline_dir, cfg)
        run_mesh_texturing(state, MockTrellisWorker(), pipeline_dir, cfg)
        return state

    def test_advances_to_cleanup_done(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        run_cleanup(state, MockScreenshotWorker(), pipeline_dir, cfg)
        for m in state.meshes:
            assert m.status == MeshStatus.CLEANUP_DONE

    def test_cleaned_file_exists_on_disk(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        run_cleanup(state, MockScreenshotWorker(), pipeline_dir, cfg)
        for m in state.meshes:
            assert m.textured_mesh_path is not None
            assert Path(m.textured_mesh_path).exists()
            assert Path(m.textured_mesh_path).name == "cleaned_mesh.glb"

    def test_updates_textured_mesh_path(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        original_paths = [m.textured_mesh_path for m in state.meshes]
        run_cleanup(state, MockScreenshotWorker(), pipeline_dir, cfg)
        for m, orig in zip(state.meshes, original_paths):
            assert m.textured_mesh_path != orig

    def test_skips_non_texture_done_meshes(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        state.meshes[0].status = MeshStatus.MESH_DONE  # not eligible
        worker = MockScreenshotWorker()
        run_cleanup(state, worker, pipeline_dir, cfg)
        cleanup_calls = [c for c in worker.calls if c["action"] == "cleanup"]
        assert len(cleanup_calls) == 1  # only the second mesh

    def test_worker_called_for_each_eligible_mesh(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        worker = MockScreenshotWorker()
        run_cleanup(state, worker, pipeline_dir, cfg)
        cleanup_calls = [c for c in worker.calls if c["action"] == "cleanup"]
        assert len(cleanup_calls) == 2

    def test_passes_symmetrize_true_to_worker(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        state.symmetrize = True
        worker = MockScreenshotWorker()
        run_cleanup(state, worker, pipeline_dir, cfg)
        for call in [c for c in worker.calls if c["action"] == "cleanup"]:
            assert call["symmetrize"] is True

    def test_passes_symmetrize_false_by_default(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        worker = MockScreenshotWorker()
        run_cleanup(state, worker, pipeline_dir, cfg)
        for call in [c for c in worker.calls if c["action"] == "cleanup"]:
            assert call["symmetrize"] is False

    def test_passes_symmetry_axis_to_worker(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        state.symmetrize = True
        state.symmetry_axis = SymmetryAxis.X_PLUS
        worker = MockScreenshotWorker()
        run_cleanup(state, worker, pipeline_dir, cfg)
        for call in [c for c in worker.calls if c["action"] == "cleanup"]:
            assert call["symmetry_axis"] == "x+"

    def test_failure_propagates(self, pipeline_dir, cfg):
        state = self._state_texture_done(pipeline_dir, cfg)
        with pytest.raises(RuntimeError, match="simulated cleanup failure"):
            run_cleanup(state, MockScreenshotWorker(fail=True), pipeline_dir, cfg)

    def test_cleanup_done_meshes_accepted_by_run_screenshots(self, pipeline_dir, cfg):
        """run_screenshots must process CLEANUP_DONE meshes (not just TEXTURE_DONE)."""
        state = self._state_texture_done(pipeline_dir, cfg)
        run_cleanup(state, MockScreenshotWorker(), pipeline_dir, cfg)
        # All meshes now CLEANUP_DONE
        worker = MockScreenshotWorker()
        run_screenshots(state, worker, pipeline_dir, cfg)
        assert len(worker.calls) == 2
        for m in state.meshes:
            assert m.status == MeshStatus.SCREENSHOT_DONE

    def test_status_set_to_cleaning_up_before_worker_called(self, pipeline_dir, cfg):
        """CLEANING_UP must be set before the worker runs (crash-consistency)."""
        seen_statuses = []

        class StatusCapture(MockScreenshotWorker):
            def cleanup_mesh(self, mesh_path, output_path, **kw):
                seen_statuses.append(state.meshes[0].status)
                return super().cleanup_mesh(mesh_path, output_path, **kw)

        state = self._state_texture_done(pipeline_dir, cfg)
        run_cleanup(state, StatusCapture(), pipeline_dir, cfg)
        assert MeshStatus.CLEANING_UP in seen_statuses
