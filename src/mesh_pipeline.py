"""
Mesh generation + texturing pipeline — orchestration layer for 3D pipelines.

Operates on Pipeline3DState.  Each 3D pipeline is one mesh attempt: one
source image → one mesh → one set of textures → screenshots → review → export.

Public API
----------
run_mesh_generation(state, worker, pipeline_dir, cfg) -> None
run_mesh_texturing(state, worker, pipeline_dir, cfg) -> None
run_mesh_export(state, pipeline_dir, cfg) -> None
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config, config as _default_config
from src.state import Pipeline3DState, Pipeline3DStatus
from src.workers.trellis import TrellisWorker


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


def run_mesh_generation(
    state: Pipeline3DState,
    worker: TrellisWorker,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    Generate a mesh from state.input_image_path.

    Only runs if the pipeline is in QUEUED status.
    Updates state in place.  Caller is responsible for saving state.
    """
    if state.status != Pipeline3DStatus.QUEUED:
        return

    mesh_dir = Path(pipeline_dir) / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    state.status = Pipeline3DStatus.GENERATING_MESH
    glb = worker.generate_mesh(
        image_path=Path(state.input_image_path),
        dest_dir=mesh_dir,
        num_polys=state.num_polys,
        job_id=state.name,
    )
    state.mesh_path = str(glb)
    state.status = Pipeline3DStatus.MESH_DONE


# ---------------------------------------------------------------------------
# Texturing
# ---------------------------------------------------------------------------


def run_mesh_texturing(
    state: Pipeline3DState,
    worker: TrellisWorker,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    Texture the mesh produced by run_mesh_generation.

    Only runs if the pipeline is in MESH_DONE status.
    Updates state in place.  Caller is responsible for saving state.
    """
    if state.status != Pipeline3DStatus.MESH_DONE:
        return
    if not state.mesh_path:
        return

    mesh_dir = Path(pipeline_dir) / "meshes"

    state.status = Pipeline3DStatus.GENERATING_TEXTURE
    textured_glb = worker.texture_mesh(
        image_path=Path(state.input_image_path),
        mesh_path=Path(state.mesh_path),
        dest_dir=mesh_dir,
        job_id=state.name,
    )
    state.textured_mesh_path = str(textured_glb)
    state.status = Pipeline3DStatus.TEXTURE_DONE


# ---------------------------------------------------------------------------
# Mesh export to final assets
# ---------------------------------------------------------------------------


def run_mesh_export(
    state: Pipeline3DState,
    pipeline_dir: Path,
    cfg: Config = _default_config,
    *,
    asset_name: str,
    export_format: str | None = None,
) -> None:
    """
    Export the approved mesh to the final_game_ready_assets directory.

    Writes a flat file:  final_assets_dir/{asset_name}_v{export_version}.{format}
    Increments state.export_version and appends to state.export_paths.
    Sets state.status = IDLE.

    Caller is responsible for saving state.
    """
    fmt = export_format or cfg.export_format
    glb_path = state.textured_mesh_path or state.mesh_path
    if not glb_path or not Path(glb_path).exists():
        return

    cfg.final_assets_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{asset_name}_v{state.export_version}.{fmt}"
    dst = cfg.final_assets_dir / filename
    dst.write_bytes(Path(glb_path).read_bytes())

    state.export_paths.append(str(dst))
    state.export_version += 1
    state.status = Pipeline3DStatus.IDLE
