"""
Trellis workers — mesh generation and texturing via ComfyUI.

TrellisWorker           — abstract interface
ComfyUITrellisWorker    — real implementation using ComfyUIClient
MockTrellisWorker       — deterministic stub for tests (no ComfyUI needed)

Key ComfyUI behaviour to be aware of:
  Trellis2ExportMesh writes GLB files directly to disk under
  {comfyui_output_dir}/3D/ and does NOT register them in the ComfyUI
  history API.  Output discovery is therefore done by scanning the
  filesystem for files matching the job prefix written after job start.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import random
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class TrellisWorker(ABC):
    """Generate and texture 3-D meshes."""

    @abstractmethod
    def generate_mesh(
        self,
        image_path: Path,
        dest_dir: Path,
        num_polys: int,
        job_id: str,
    ) -> Path:
        """
        Run the Trellis mesh-generation workflow for `image_path`.

        Returns the path to the downloaded/copied untextured GLB inside
        `dest_dir`.  Raises RuntimeError after MAX_RETRIES failures.
        """

    @abstractmethod
    def texture_mesh(
        self,
        image_path: Path,
        mesh_path: Path,
        dest_dir: Path,
        job_id: str,
    ) -> Path:
        """
        Run the Trellis texturing workflow using `image_path` as the
        reference and `mesh_path` as the untextured input.

        Returns the path to the textured GLB inside `dest_dir`.
        """


# ---------------------------------------------------------------------------
# Real ComfyUI implementation
# ---------------------------------------------------------------------------


class ComfyUITrellisWorker(TrellisWorker):
    """
    Submits Trellis workflows to a running ComfyUI instance.

    Parameters
    ----------
    client:
        ComfyUIClient pointed at the ComfyUI instance.
    comfyui_output_dir:
        The `output/` directory of the ComfyUI installation.
        Trellis writes GLBs to `{comfyui_output_dir}/3D/`.
    workflow_generate:
        Path to trellis_generate.json.
    workflow_texture:
        Path to trellis_texture.json.
    seed:
        Fixed seed (int) or None for random per attempt.
    """

    def __init__(
        self,
        client,                   # ComfyUIClient
        comfyui_output_dir: Path,
        workflow_generate: Path,
        workflow_texture: Path,
        seed: int | None = None,
    ):
        self._client = client
        self._output_dir = Path(comfyui_output_dir)
        self._base_generate = json.loads(Path(workflow_generate).read_text(encoding="utf-8"))
        self._base_texture = json.loads(Path(workflow_texture).read_text(encoding="utf-8"))
        self._seed = seed

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_mesh(
        self,
        image_path: Path,
        dest_dir: Path,
        num_polys: int,
        job_id: str,
    ) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, MAX_RETRIES + 1):
            seed = self._seed if self._seed is not None else random.randint(0, 2**31 - 1)
            log.info(f"[generate_mesh] attempt {attempt}/{MAX_RETRIES}, seed={seed}, job={job_id}")

            server_image = self._client.upload_image(image_path)
            workflow = _inject_generate_params(
                self._base_generate,
                image_name=server_image,
                num_polys=num_polys,
                seed=seed,
                filename_prefix=job_id,
            )

            self._client.free_memory()  # evict any previously loaded model (e.g. FLUX)
            started_at = time.time()
            self._client.run_workflow(workflow)

            glb = _find_trellis_output(self._output_dir, job_id, started_at)
            if glb is None:
                log.warning(f"[generate_mesh] attempt {attempt}: no output found, retrying…")
                continue

            dest = dest_dir / "initial_mesh.glb"
            shutil.copy2(glb, dest)
            log.info(f"[generate_mesh] saved → {dest}")
            return dest

        raise RuntimeError(
            f"Trellis generate_mesh failed after {MAX_RETRIES} attempts for job '{job_id}'"
        )

    def texture_mesh(
        self,
        image_path: Path,
        mesh_path: Path,
        dest_dir: Path,
        job_id: str,
    ) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, MAX_RETRIES + 1):
            seed = self._seed if self._seed is not None else random.randint(0, 2**31 - 1)
            log.info(f"[texture_mesh] attempt {attempt}/{MAX_RETRIES}, seed={seed}, job={job_id}")

            server_image = self._client.upload_image(image_path)
            server_glb = self._client.upload_image(mesh_path)
            workflow = _inject_texture_params(
                self._base_texture,
                image_name=server_image,
                glb_path=server_glb,
                seed=seed,
                filename_prefix=f"{job_id}_tex",
            )

            self._client.free_memory()  # evict any previously loaded model (e.g. FLUX)
            started_at = time.time()
            self._client.run_workflow(workflow)

            glb = _find_trellis_output(self._output_dir, f"{job_id}_tex", started_at)
            if glb is None:
                log.warning(f"[texture_mesh] attempt {attempt}: no output found, retrying…")
                continue

            dest = dest_dir / "textured_mesh.glb"
            shutil.copy2(glb, dest)
            log.info(f"[texture_mesh] saved → {dest}")
            return dest

        raise RuntimeError(
            f"Trellis texture_mesh failed after {MAX_RETRIES} attempts for job '{job_id}'"
        )


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------


class MockTrellisWorker(TrellisWorker):
    """
    Deterministic stub that writes minimal valid GLB files instantly.

    Parameters
    ----------
    fail_on_generate:
        Raise RuntimeError from generate_mesh() (simulates Trellis failure).
    fail_on_texture:
        Raise RuntimeError from texture_mesh().
    """

    def __init__(
        self,
        fail_on_generate: bool = False,
        fail_on_texture: bool = False,
    ):
        self._fail_on_generate = fail_on_generate
        self._fail_on_texture = fail_on_texture
        self.generate_calls: list[dict] = []
        self.texture_calls: list[dict] = []

    def generate_mesh(
        self,
        image_path: Path,
        dest_dir: Path,
        num_polys: int,
        job_id: str,
    ) -> Path:
        if self._fail_on_generate:
            raise RuntimeError("MockTrellisWorker: simulated generate_mesh failure")
        self.generate_calls.append(
            {"image_path": image_path, "num_polys": num_polys, "job_id": job_id}
        )
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "initial_mesh.glb"
        dest.write_bytes(_make_minimal_glb())
        return dest

    def texture_mesh(
        self,
        image_path: Path,
        mesh_path: Path,
        dest_dir: Path,
        job_id: str,
    ) -> Path:
        if self._fail_on_texture:
            raise RuntimeError("MockTrellisWorker: simulated texture_mesh failure")
        self.texture_calls.append(
            {"image_path": image_path, "mesh_path": mesh_path, "job_id": job_id}
        )
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "textured_mesh.glb"
        dest.write_bytes(_make_minimal_glb())
        return dest


# ---------------------------------------------------------------------------
# Workflow injection helpers
# ---------------------------------------------------------------------------


def _inject_generate_params(
    workflow: dict,
    image_name: str,
    num_polys: int,
    seed: int,
    filename_prefix: str,
) -> dict:
    wf = copy.deepcopy(workflow)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})

        if ct == "Trellis2LoadImageWithTransparency":
            inp["image"] = image_name

        elif ct in ("Trellis2MeshWithVoxelGenerator", "Trellis2MeshWithVoxelAdvancedGenerator"):
            inp["seed"] = seed
            inp["generate_texture_slat"] = False   # geometry only

        elif ct in ("Trellis2SimplifyMesh", "Trellis2SimplifyTrimesh"):
            inp["target_face_num"] = num_polys

        elif ct == "Trellis2ExportMesh":
            inp["filename_prefix"] = f"3D/{filename_prefix}"
            inp["file_format"] = "glb"

    return wf


def _inject_texture_params(
    workflow: dict,
    image_name: str,
    glb_path: str,
    seed: int,
    filename_prefix: str,
) -> dict:
    wf = copy.deepcopy(workflow)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})

        if ct == "LoadImage":
            inp["image"] = image_name

        elif ct == "Trellis2LoadMesh":
            inp["glb_path"] = glb_path

        elif ct == "Trellis2MeshTexturingMultiView":
            inp["seed"] = seed

        elif ct == "Trellis2ExportMesh":
            inp["filename_prefix"] = f"3D/{filename_prefix}"
            inp["file_format"] = "glb"

    return wf


# ---------------------------------------------------------------------------
# Output discovery
# ---------------------------------------------------------------------------


def _find_trellis_output(
    comfyui_output_dir: Path,
    filename_prefix: str,
    after_time: float,
) -> Path | None:
    """
    Scan `{comfyui_output_dir}/3D/` for a GLB whose name starts with
    `filename_prefix` and whose mtime >= `after_time`.

    Falls back to the most-recently-modified matching file if nothing is
    newer than `after_time` (handles ComfyUI's cache-served responses).
    """
    search_dir = Path(comfyui_output_dir) / "3D"
    if not search_dir.exists():
        log.warning(f"Trellis output dir not found: {search_dir}")
        return None

    all_matches = sorted(
        search_dir.glob(f"{filename_prefix}*.glb"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_matches:
        return None

    fresh = [p for p in all_matches if p.stat().st_mtime >= after_time]
    result = fresh[0] if fresh else all_matches[0]
    log.info(f"Found Trellis output: {result}")
    return result


# ---------------------------------------------------------------------------
# Minimal GLB builder (for MockTrellisWorker)
# ---------------------------------------------------------------------------


def _make_minimal_glb() -> bytes:
    """Return a valid binary GLB containing a single triangle."""
    try:
        import trimesh
        mesh = trimesh.creation.box()
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb")
        return buf.getvalue()
    except ImportError:
        pass

    # Fallback: hand-craft the smallest possible valid GLB (single triangle).
    # GLB = 12-byte header + JSON chunk + BIN chunk
    import struct

    # Vertex positions for one triangle (3 × vec3 float32)
    positions = struct.pack("<9f", 0, 0, 0,  1, 0, 0,  0, 1, 0)
    # Indices (1 triangle = 3 × uint16)
    indices = struct.pack("<3H", 0, 1, 2)
    # Pad BIN chunk to 4-byte boundary
    bin_data = positions + indices
    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_data += b"\x00" * bin_pad

    byte_offset_positions = 0
    byte_length_positions = len(positions)
    byte_offset_indices = byte_length_positions
    byte_length_indices = len(indices)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 1},
                "indices": 0,
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5123,   # UNSIGNED_SHORT
                "count": 3,
                "type": "SCALAR",
            },
            {
                "bufferView": 1,
                "componentType": 5126,   # FLOAT
                "count": 3,
                "type": "VEC3",
                "max": [1, 1, 0],
                "min": [0, 0, 0],
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": byte_offset_indices, "byteLength": byte_length_indices},
            {"buffer": 0, "byteOffset": byte_offset_positions, "byteLength": byte_length_positions},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }

    import json as _json
    json_bytes = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad

    # Chunk headers: (length, type)
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes  # JSON
    bin_chunk  = struct.pack("<II", len(bin_data),   0x004E4942) + bin_data    # BIN

    total = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total)   # magic "glTF", version 2

    return header + json_chunk + bin_chunk
