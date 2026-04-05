"""
Tests for src/workers/trellis.py
"""

import copy
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.workers.trellis import (
    ComfyUITrellisWorker,
    MockTrellisWorker,
    _find_trellis_output,
    _inject_generate_params,
    _inject_texture_params,
    _make_minimal_glb,
)


# ---------------------------------------------------------------------------
# Minimal GLB
# ---------------------------------------------------------------------------


class TestMakeMinimalGlb:
    def test_returns_bytes(self):
        data = _make_minimal_glb()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_starts_with_gltf_magic(self):
        data = _make_minimal_glb()
        # GLB magic = 0x46546C67 little-endian = b"glTF"
        assert data[:4] == b"glTF"

    def test_trimesh_can_load_it(self):
        import io
        import trimesh
        data = _make_minimal_glb()
        scene = trimesh.load(io.BytesIO(data), file_type="glb")
        assert scene is not None


# ---------------------------------------------------------------------------
# _inject_generate_params
# ---------------------------------------------------------------------------


@pytest.fixture
def generate_workflow():
    return {
        "1": {"class_type": "Trellis2LoadModel", "inputs": {"modelname": "old", "backend": "x"}},
        "2": {"class_type": "Trellis2LoadImageWithTransparency", "inputs": {"image": "old.png"}},
        "4": {"class_type": "Trellis2MeshWithVoxelAdvancedGenerator", "inputs": {"seed": 0, "generate_texture_slat": True}},
        "6": {"class_type": "Trellis2SimplifyMesh", "inputs": {"target_face_num": 8000}},
        "10": {"class_type": "Trellis2ExportMesh", "inputs": {"filename_prefix": "old", "file_format": "obj"}},
    }


class TestInjectGenerateParams:
    def test_image_name_injected(self, generate_workflow):
        wf = _inject_generate_params(generate_workflow, "new.png", 5000, 42, "my_job")
        assert wf["2"]["inputs"]["image"] == "new.png"

    def test_num_polys_injected(self, generate_workflow):
        wf = _inject_generate_params(generate_workflow, "img.png", 5000, 42, "job")
        assert wf["6"]["inputs"]["target_face_num"] == 5000

    def test_seed_injected(self, generate_workflow):
        wf = _inject_generate_params(generate_workflow, "img.png", 8000, 999, "job")
        assert wf["4"]["inputs"]["seed"] == 999

    def test_generate_texture_slat_is_false(self, generate_workflow):
        wf = _inject_generate_params(generate_workflow, "img.png", 8000, 0, "job")
        assert wf["4"]["inputs"]["generate_texture_slat"] is False

    def test_filename_prefix_set(self, generate_workflow):
        wf = _inject_generate_params(generate_workflow, "img.png", 8000, 0, "myjob")
        assert wf["10"]["inputs"]["filename_prefix"] == "3D/myjob"

    def test_file_format_is_glb(self, generate_workflow):
        wf = _inject_generate_params(generate_workflow, "img.png", 8000, 0, "job")
        assert wf["10"]["inputs"]["file_format"] == "glb"

    def test_does_not_mutate_original(self, generate_workflow):
        original = copy.deepcopy(generate_workflow)
        _inject_generate_params(generate_workflow, "new.png", 1000, 1, "j")
        assert generate_workflow["2"]["inputs"]["image"] == original["2"]["inputs"]["image"]


# ---------------------------------------------------------------------------
# _inject_texture_params
# ---------------------------------------------------------------------------


@pytest.fixture
def texture_workflow():
    return {
        "1": {"class_type": "Trellis2LoadModel", "inputs": {}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "4": {"class_type": "Trellis2LoadMesh", "inputs": {"glb_path": ""}},
        "5": {"class_type": "Trellis2MeshTexturingMultiView", "inputs": {"seed": 0}},
        "6": {"class_type": "Trellis2ExportMesh", "inputs": {"filename_prefix": "old", "file_format": "obj"}},
    }


class TestInjectTextureParams:
    def test_image_name_injected(self, texture_workflow):
        wf = _inject_texture_params(texture_workflow, "ref.png", "/path/mesh.glb", 0, "job_tex")
        assert wf["2"]["inputs"]["image"] == "ref.png"

    def test_glb_path_injected(self, texture_workflow):
        wf = _inject_texture_params(texture_workflow, "ref.png", "/my/mesh.glb", 0, "job_tex")
        assert wf["4"]["inputs"]["glb_path"] == "/my/mesh.glb"

    def test_seed_injected(self, texture_workflow):
        wf = _inject_texture_params(texture_workflow, "ref.png", "/m.glb", 77, "job")
        assert wf["5"]["inputs"]["seed"] == 77

    def test_filename_prefix_set(self, texture_workflow):
        wf = _inject_texture_params(texture_workflow, "r.png", "/m.glb", 0, "myjob_tex")
        assert wf["6"]["inputs"]["filename_prefix"] == "3D/myjob_tex"

    def test_does_not_mutate_original(self, texture_workflow):
        original = copy.deepcopy(texture_workflow)
        _inject_texture_params(texture_workflow, "new.png", "/m.glb", 0, "j")
        assert texture_workflow["2"]["inputs"]["image"] == original["2"]["inputs"]["image"]


# ---------------------------------------------------------------------------
# _find_trellis_output
# ---------------------------------------------------------------------------


class TestFindTrellisOutput:
    def test_finds_fresh_glb(self, tmp_path):
        output_dir = tmp_path / "output"
        three_d = output_dir / "3D"
        three_d.mkdir(parents=True)
        glb = three_d / "myjob_00001_.glb"
        t0 = time.time()
        glb.write_bytes(b"fake")
        result = _find_trellis_output(output_dir, "myjob", t0 - 1)
        assert result == glb

    def test_returns_none_if_no_match(self, tmp_path):
        output_dir = tmp_path / "output"
        (output_dir / "3D").mkdir(parents=True)
        result = _find_trellis_output(output_dir, "myjob", time.time())
        assert result is None

    def test_ignores_files_with_wrong_prefix(self, tmp_path):
        output_dir = tmp_path / "output"
        three_d = output_dir / "3D"
        three_d.mkdir(parents=True)
        (three_d / "other_job.glb").write_bytes(b"x")
        result = _find_trellis_output(output_dir, "myjob", 0)
        assert result is None

    def test_returns_none_if_3d_dir_missing(self, tmp_path):
        result = _find_trellis_output(tmp_path, "myjob", 0)
        assert result is None

    def test_falls_back_to_most_recent_if_nothing_is_fresh(self, tmp_path):
        output_dir = tmp_path / "output"
        three_d = output_dir / "3D"
        three_d.mkdir(parents=True)
        glb = three_d / "myjob_00001_.glb"
        glb.write_bytes(b"old")
        # Claim the job started far in the future so nothing looks "fresh"
        result = _find_trellis_output(output_dir, "myjob", time.time() + 9999)
        assert result == glb


# ---------------------------------------------------------------------------
# MockTrellisWorker
# ---------------------------------------------------------------------------


class TestMockTrellisWorker:
    def _make_image(self, tmp_path) -> Path:
        from PIL import Image
        p = tmp_path / "img.png"
        Image.new("RGB", (64, 64), (200, 100, 50)).save(str(p))
        return p

    def test_generate_creates_glb_file(self, tmp_path):
        worker = MockTrellisWorker()
        img = self._make_image(tmp_path)
        dest_dir = tmp_path / "meshes"
        result = worker.generate_mesh(img, dest_dir, num_polys=8000, job_id="test_0")
        assert result.exists()
        assert result.suffix == ".glb"

    def test_generate_returns_path_inside_dest_dir(self, tmp_path):
        worker = MockTrellisWorker()
        img = self._make_image(tmp_path)
        dest_dir = tmp_path / "meshes"
        result = worker.generate_mesh(img, dest_dir, num_polys=8000, job_id="test_0")
        assert result.parent == dest_dir

    def test_generate_creates_valid_glb(self, tmp_path):
        import io
        import trimesh
        worker = MockTrellisWorker()
        img = self._make_image(tmp_path)
        result = worker.generate_mesh(img, tmp_path / "m", 8000, "j")
        data = result.read_bytes()
        assert data[:4] == b"glTF"

    def test_generate_records_call(self, tmp_path):
        worker = MockTrellisWorker()
        img = self._make_image(tmp_path)
        worker.generate_mesh(img, tmp_path / "m", 5000, "job_1")
        assert len(worker.generate_calls) == 1
        assert worker.generate_calls[0]["num_polys"] == 5000
        assert worker.generate_calls[0]["job_id"] == "job_1"

    def test_texture_creates_glb_file(self, tmp_path):
        worker = MockTrellisWorker()
        img = self._make_image(tmp_path)
        mesh = tmp_path / "mesh.glb"
        mesh.write_bytes(_make_minimal_glb())
        result = worker.texture_mesh(img, mesh, tmp_path / "tex", "job_1")
        assert result.exists()
        assert result.suffix == ".glb"

    def test_texture_records_call(self, tmp_path):
        worker = MockTrellisWorker()
        img = self._make_image(tmp_path)
        mesh = tmp_path / "mesh.glb"
        mesh.write_bytes(_make_minimal_glb())
        worker.texture_mesh(img, mesh, tmp_path / "tex", "job_1")
        assert len(worker.texture_calls) == 1
        assert worker.texture_calls[0]["job_id"] == "job_1"

    def test_fail_on_generate(self, tmp_path):
        worker = MockTrellisWorker(fail_on_generate=True)
        img = self._make_image(tmp_path)
        with pytest.raises(RuntimeError, match="simulated generate_mesh failure"):
            worker.generate_mesh(img, tmp_path / "m", 8000, "j")

    def test_fail_on_texture(self, tmp_path):
        worker = MockTrellisWorker(fail_on_texture=True)
        img = self._make_image(tmp_path)
        mesh = tmp_path / "mesh.glb"
        mesh.write_bytes(_make_minimal_glb())
        with pytest.raises(RuntimeError, match="simulated texture_mesh failure"):
            worker.texture_mesh(img, mesh, tmp_path / "tex", "j")


# ---------------------------------------------------------------------------
# ComfyUITrellisWorker (mocked client)
# ---------------------------------------------------------------------------


def _write_workflow(path: Path, workflow: dict) -> None:
    path.write_text(json.dumps(workflow))


def _sample_generate_workflow() -> dict:
    return {
        "2": {"class_type": "Trellis2LoadImageWithTransparency", "inputs": {"image": ""}},
        "4": {"class_type": "Trellis2MeshWithVoxelAdvancedGenerator", "inputs": {"seed": 0, "generate_texture_slat": True}},
        "6": {"class_type": "Trellis2SimplifyMesh", "inputs": {"target_face_num": 8000}},
        "10": {"class_type": "Trellis2ExportMesh", "inputs": {"filename_prefix": "", "file_format": "glb"}},
    }


def _sample_texture_workflow() -> dict:
    return {
        "2": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "4": {"class_type": "Trellis2LoadMesh", "inputs": {"glb_path": ""}},
        "5": {"class_type": "Trellis2MeshTexturingMultiView", "inputs": {"seed": 0}},
        "6": {"class_type": "Trellis2ExportMesh", "inputs": {"filename_prefix": "", "file_format": "glb"}},
    }


class TestComfyUITrellisWorker:
    def _setup(self, tmp_path):
        wf_gen = tmp_path / "gen.json"
        wf_tex = tmp_path / "tex.json"
        _write_workflow(wf_gen, _sample_generate_workflow())
        _write_workflow(wf_tex, _sample_texture_workflow())

        output_dir = tmp_path / "comfyui_output"
        three_d = output_dir / "3D"
        three_d.mkdir(parents=True)

        mock_client = MagicMock()
        mock_client.upload_image.return_value = "uploaded.png"
        mock_client.run_workflow.return_value = None

        return mock_client, output_dir, wf_gen, wf_tex, three_d

    def _place_output_glb(self, three_d: Path, prefix: str) -> Path:
        glb = three_d / f"{prefix}_00001_.glb"
        glb.write_bytes(_make_minimal_glb())
        return glb

    def test_generate_uploads_image(self, tmp_path):
        from PIL import Image
        img = tmp_path / "img.png"
        Image.new("RGB", (64, 64)).save(str(img))

        mock_client, output_dir, wf_gen, wf_tex, three_d = self._setup(tmp_path)
        self._place_output_glb(three_d, "test_job")

        worker = ComfyUITrellisWorker(mock_client, output_dir, wf_gen, wf_tex, seed=0)
        worker.generate_mesh(img, tmp_path / "dest", num_polys=5000, job_id="test_job")
        mock_client.upload_image.assert_called_once_with(img)

    def test_generate_calls_run_workflow(self, tmp_path):
        from PIL import Image
        img = tmp_path / "img.png"
        Image.new("RGB", (64, 64)).save(str(img))

        mock_client, output_dir, wf_gen, wf_tex, three_d = self._setup(tmp_path)
        self._place_output_glb(three_d, "myjob")

        worker = ComfyUITrellisWorker(mock_client, output_dir, wf_gen, wf_tex, seed=0)
        worker.generate_mesh(img, tmp_path / "dest", num_polys=8000, job_id="myjob")
        mock_client.run_workflow.assert_called_once()

    def test_generate_injects_num_polys(self, tmp_path):
        from PIL import Image
        img = tmp_path / "img.png"
        Image.new("RGB", (64, 64)).save(str(img))

        mock_client, output_dir, wf_gen, wf_tex, three_d = self._setup(tmp_path)
        self._place_output_glb(three_d, "myjob")

        worker = ComfyUITrellisWorker(mock_client, output_dir, wf_gen, wf_tex, seed=0)
        worker.generate_mesh(img, tmp_path / "dest", num_polys=3000, job_id="myjob")

        submitted = mock_client.run_workflow.call_args[0][0]
        # Find SimplifyMesh node
        simplify = next(
            v for v in submitted.values()
            if isinstance(v, dict) and v.get("class_type") == "Trellis2SimplifyMesh"
        )
        assert simplify["inputs"]["target_face_num"] == 3000

    def test_generate_saves_glb_to_dest_dir(self, tmp_path):
        from PIL import Image
        img = tmp_path / "img.png"
        Image.new("RGB", (64, 64)).save(str(img))

        mock_client, output_dir, wf_gen, wf_tex, three_d = self._setup(tmp_path)
        self._place_output_glb(three_d, "myjob")

        dest_dir = tmp_path / "dest"
        worker = ComfyUITrellisWorker(mock_client, output_dir, wf_gen, wf_tex, seed=0)
        result = worker.generate_mesh(img, dest_dir, num_polys=8000, job_id="myjob")
        assert result.parent == dest_dir
        assert result.exists()

    def test_generate_raises_after_max_retries_if_no_output(self, tmp_path):
        from PIL import Image
        img = tmp_path / "img.png"
        Image.new("RGB", (64, 64)).save(str(img))

        mock_client, output_dir, wf_gen, wf_tex, three_d = self._setup(tmp_path)
        # Do NOT place output GLB → worker should retry and fail

        worker = ComfyUITrellisWorker(mock_client, output_dir, wf_gen, wf_tex, seed=0)
        with pytest.raises(RuntimeError, match="generate_mesh failed"):
            worker.generate_mesh(img, tmp_path / "dest", num_polys=8000, job_id="myjob")

    def test_texture_injects_glb_path(self, tmp_path):
        from PIL import Image
        img = tmp_path / "img.png"
        Image.new("RGB", (64, 64)).save(str(img))
        mesh = tmp_path / "mesh.glb"
        mesh.write_bytes(_make_minimal_glb())

        mock_client, output_dir, wf_gen, wf_tex, three_d = self._setup(tmp_path)
        self._place_output_glb(three_d, "myjob_tex")

        worker = ComfyUITrellisWorker(mock_client, output_dir, wf_gen, wf_tex, seed=0)
        worker.texture_mesh(img, mesh, tmp_path / "dest", job_id="myjob")

        submitted = mock_client.run_workflow.call_args[0][0]
        load_mesh = next(
            v for v in submitted.values()
            if isinstance(v, dict) and v.get("class_type") == "Trellis2LoadMesh"
        )
        # GLB is uploaded via ComfyUI API; worker passes the returned server filename
        mock_client.upload_image.assert_called_with(mesh)
        assert load_mesh["inputs"]["glb_path"] == "uploaded.png"
