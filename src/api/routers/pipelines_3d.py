"""
3D pipeline endpoints.

POST   /3d-pipelines/from-ref        Submit a 2D concept art image for mesh gen
POST   /3d-pipelines/from-upload     Start a 3D pipeline from an uploaded image
GET    /3d-pipelines                 List 3D pipelines
GET    /3d-pipelines/{name}          Get 3D pipeline state
DELETE /3d-pipelines/{name}          Cancel a 3D pipeline

GET    /3d-pipelines/{name}/sheet            Review sheet PNG
GET    /3d-pipelines/{name}/screenshot/{fn}  Screenshot PNG
GET    /3d-pipelines/{name}/preview          HTML preview
GET    /3d-pipelines/{name}/mesh             Textured GLB download
POST   /3d-pipelines/{name}/approve          Approve mesh and export
POST   /3d-pipelines/{name}/reject           Reject mesh and re-queue
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from src.api.auth import CurrentUser
from src.api.event_bus import event_bus
from src.api.models import (
    ApproveMeshRequest,
    Create3DPipelineFromRefRequest,
    Create3DPipelineFromUploadRequest,
    OkResponse,
    Patch3DPipelineRequest,
    RejectMeshRequest,
)
from src.state import Pipeline3DState, Pipeline3DStatus

log = logging.getLogger(__name__)
router = APIRouter(tags=["3d-pipelines"])


def _agent(request: Request):
    return request.app.state.agent


def _cfg(request: Request):
    return request.app.state.cfg


def _load_3d_state(name: str, request: Request) -> tuple[Pipeline3DState, Path]:
    """Return (state, pipeline_dir) or raise 404."""
    agent = _agent(request)
    cfg = _cfg(request)
    state = agent.get_3d_pipeline_state(name)
    if state is None:
        raise HTTPException(status_code=404, detail=f"3D pipeline '{name}' not found")
    pipeline_dir = cfg.pipelines_dir / name
    return state, pipeline_dir


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@router.post("/3d-pipelines/from-ref", status_code=status.HTTP_201_CREATED)
async def create_3d_pipeline_from_ref(
    req: Create3DPipelineFromRefRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Submit a concept art image from a 2D pipeline for mesh generation.

    Derives 3D pipeline name as "{2d_pipeline_name}_{ca_index}_{ca_version}".
    Warns (409) if that 3D pipeline already exists.
    """
    from src.concept_art_pipeline import _concept_art_path

    agent = _agent(request)
    cfg = _cfg(request)

    # Resolve 2D pipeline state
    state_2d = agent.get_pipeline_state(req.pipeline_name)
    if state_2d is None:
        raise HTTPException(status_code=404, detail=f"2D pipeline '{req.pipeline_name}' not found")

    idx = req.concept_art_index
    if idx < 0 or idx >= len(state_2d.concept_arts):
        raise HTTPException(status_code=422, detail=f"Concept art index {idx} out of range")

    ca = state_2d.concept_arts[idx]
    version = req.concept_art_version if req.concept_art_version is not None else ca.version

    # Resolve image path
    pipeline_dir_2d = cfg.pipelines_dir / req.pipeline_name
    img_path = _concept_art_path(pipeline_dir_2d, idx, version)
    if not img_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Concept art {idx} v{version} has not been generated yet",
        )

    # Derive 3D pipeline name
    name_3d = f"{req.pipeline_name}_{idx + 1}_{version}"
    if agent.pipeline_name_exists(name_3d):
        raise HTTPException(
            status_code=409,
            detail=(
                f"3D pipeline '{name_3d}' already exists. "
                "Delete it first if you want to resubmit."
            ),
        )

    # Copy the source image into the 3D pipeline's own directory so the 3D
    # pipeline is decoupled from the 2D pipeline's evolving concept art
    # folder.  This prevents any possibility of the 3D pipeline picking up
    # a different image if the 2D pipeline is regenerated, deleted, or
    # otherwise modified after the 3D pipeline is created.
    pipeline_dir_3d = cfg.pipelines_dir / name_3d
    pipeline_dir_3d.mkdir(parents=True, exist_ok=True)
    copied_img_path = pipeline_dir_3d / f"source_{img_path.name}"
    shutil.copy2(str(img_path), str(copied_img_path))

    state = agent.start_3d_pipeline(
        name=name_3d,
        input_image_path=str(copied_img_path),
        num_polys=req.num_polys,
        source_2d_pipeline=req.pipeline_name,
        source_concept_art_index=idx,
        source_concept_art_version=version,
        symmetrize=req.symmetrize,
        symmetry_axis=req.symmetry_axis,
    )
    log.info(
        "3D pipeline '%s' created from 2D ref by user '%s' "
        "(source: %s  →  %s)",
        name_3d, user.username, img_path, copied_img_path,
    )
    event_bus.publish({"event": "pipeline_3d_created", "pipeline": name_3d})
    return state.model_dump(mode="json")


@router.post("/3d-pipelines/from-upload", status_code=status.HTTP_201_CREATED)
async def create_3d_pipeline_from_upload(
    name: str,
    request: Request,
    user: CurrentUser,
    image: UploadFile = File(...),
    num_polys: int | None = None,
    symmetrize: bool = False,
    symmetry_axis: str = "x-",
):
    """
    Start a 3D pipeline from an uploaded image file.

    The pipeline folder name is "u_{name}" to distinguish it from 2D-derived names.
    """
    agent = _agent(request)
    cfg = _cfg(request)

    name_3d = f"u_{name}"
    if agent.pipeline_name_exists(name_3d):
        raise HTTPException(
            status_code=409,
            detail=f"3D pipeline '{name_3d}' already exists.",
        )

    # Save uploaded image
    pipeline_dir = cfg.pipelines_dir / name_3d
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    img_path = pipeline_dir / f"input_{image.filename}"
    contents = await image.read()
    img_path.write_bytes(contents)

    state = agent.start_3d_pipeline(
        name=name_3d,
        input_image_path=str(img_path),
        num_polys=num_polys,
        symmetrize=symmetrize,
        symmetry_axis=symmetry_axis,
    )
    log.info("3D pipeline '%s' created from upload by user '%s'", name_3d, user.username)
    event_bus.publish({"event": "pipeline_3d_created", "pipeline": name_3d})
    return state.model_dump(mode="json")


# ---------------------------------------------------------------------------
# List / Get
# ---------------------------------------------------------------------------


@router.get("/3d-pipelines")
async def list_3d_pipelines(request: Request, user: CurrentUser):
    """List all 3D pipelines."""
    agent = _agent(request)
    names = agent.list_3d_pipeline_names()
    result = []
    for name in names:
        state = agent.get_3d_pipeline_state(name)
        if state:
            result.append({
                "name": name,
                "status": state.status.value,
                "source_2d_pipeline": state.source_2d_pipeline,
                "created_at": state.created_at.isoformat(),
                "updated_at": state.updated_at.isoformat(),
            })
    return result


@router.get("/3d-pipelines/{name}")
async def get_3d_pipeline(name: str, request: Request, user: CurrentUser):
    """Return full 3D pipeline state as JSON."""
    state, _ = _load_3d_state(name, request)
    return state.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


@router.patch("/3d-pipelines/{name}")
async def patch_3d_pipeline(
    name: str,
    req: Patch3DPipelineRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Edit a 3D pipeline's settings.

    ``hidden`` may be toggled at any status.  ``num_polys`` / ``symmetrize`` /
    ``symmetry_axis`` are only accepted while the pipeline is queued or
    awaiting approval — past that point the regenerate flow (POST .../reject)
    is the supported way to change generation parameters.
    """
    from src.state import SymmetryAxis

    state, pipeline_dir = _load_3d_state(name, request)
    state_path = pipeline_dir / "state.json"

    if req.hidden is not None:
        state.hidden = req.hidden

    non_hidden_fields = (req.num_polys, req.symmetrize, req.symmetry_axis)
    if any(f is not None for f in non_hidden_fields):
        _editable = {
            Pipeline3DStatus.QUEUED,
            Pipeline3DStatus.AWAITING_APPROVAL,
        }
        if state.status not in _editable:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Cannot edit generation params for 3D pipeline in status "
                    f"'{state.status.value}' — use the reject/regenerate flow instead."
                ),
            )
        if req.num_polys is not None:
            state.num_polys = req.num_polys
        if req.symmetrize is not None:
            state.symmetrize = req.symmetrize
        if req.symmetry_axis is not None:
            state.symmetry_axis = SymmetryAxis(req.symmetry_axis)

    state.save(state_path)
    return state.model_dump(mode="json")


@router.delete("/3d-pipelines/{name}")
async def cancel_3d_pipeline(name: str, request: Request, user: CurrentUser):
    """Cancel all pending/running tasks for a 3D pipeline."""
    agent = _agent(request)
    cfg = _cfg(request)
    state, pipeline_dir = _load_3d_state(name, request)
    agent.cancel_pipeline(name)
    state.status = Pipeline3DStatus.CANCELLED
    state.save(pipeline_dir / "state.json")
    event_bus.publish({"event": "status_change", "pipeline": name, "status": "cancelled"})
    log.info("3D pipeline '%s' cancelled by user '%s'", name, user.username)
    return OkResponse()


# ---------------------------------------------------------------------------
# File downloads
# ---------------------------------------------------------------------------


@router.get("/3d-pipelines/{name}/sheet")
async def get_3d_review_sheet(name: str, request: Request, user: CurrentUser):
    """Return the mesh review sheet PNG."""
    state, _ = _load_3d_state(name, request)
    if not state.review_sheet_path or not Path(state.review_sheet_path).exists():
        raise HTTPException(status_code=404, detail="Review sheet not yet generated")
    return FileResponse(state.review_sheet_path, media_type="image/png")


@router.get("/3d-pipelines/{name}/screenshot/{filename}")
async def get_3d_screenshot(
    name: str, filename: str, request: Request, user: CurrentUser,
):
    """Return a single screenshot PNG by filename."""
    state, _ = _load_3d_state(name, request)
    if not state.screenshot_dir:
        raise HTTPException(status_code=404, detail="Screenshots not yet generated")
    img_path = Path(state.screenshot_dir) / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Screenshot '{filename}' not found")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/3d-pipelines/{name}/preview")
async def get_3d_preview(name: str, request: Request, user: CurrentUser):
    """Return the Three.js HTML preview file."""
    state, _ = _load_3d_state(name, request)
    if not state.html_preview_path or not Path(state.html_preview_path).exists():
        raise HTTPException(status_code=404, detail="HTML preview not yet generated")
    return FileResponse(state.html_preview_path, media_type="text/html")


@router.get("/3d-pipelines/{name}/mesh")
async def get_3d_mesh(name: str, request: Request, user: CurrentUser):
    """Return the textured GLB file."""
    state, _ = _load_3d_state(name, request)
    glb_path = state.textured_mesh_path or state.mesh_path
    if not glb_path or not Path(glb_path).exists():
        raise HTTPException(status_code=404, detail="Mesh file not yet generated")
    return FileResponse(
        glb_path, media_type="model/gltf-binary", filename=f"{name}.glb"
    )


# ---------------------------------------------------------------------------
# Mesh review actions
# ---------------------------------------------------------------------------


@router.post("/3d-pipelines/{name}/approve")
async def approve_3d_mesh(
    name: str,
    req: ApproveMeshRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Approve the mesh.  Exports the GLB to final_game_ready_assets.
    If the 3D pipeline is already IDLE (previously approved), creates a new
    export version.
    """
    from src.mesh_pipeline import run_mesh_export

    cfg = _cfg(request)
    state, pipeline_dir = _load_3d_state(name, request)

    if state.status not in (Pipeline3DStatus.AWAITING_APPROVAL, Pipeline3DStatus.IDLE):
        raise HTTPException(
            status_code=409,
            detail=f"3D pipeline '{name}' is not awaiting approval (status: {state.status.value})",
        )

    await asyncio.to_thread(
        run_mesh_export, state, pipeline_dir, cfg,
        asset_name=req.asset_name,
        export_format=req.export_format,
    )
    state.save(pipeline_dir / "state.json")

    event_bus.publish({
        "event": "status_change",
        "pipeline": name,
        "status": "idle",
        "message": f"Mesh approved and exported as '{req.asset_name}'.",
    })
    log.info("3D pipeline '%s' approved by user '%s'", name, user.username)
    return OkResponse()


@router.post("/3d-pipelines/{name}/reject")
async def reject_3d_mesh(
    name: str,
    req: RejectMeshRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Reject the mesh.  Optionally update poly count / symmetry settings.
    Resets status to QUEUED and re-enqueues mesh generation.
    """
    from src.state import SymmetryAxis

    agent = _agent(request)
    state, pipeline_dir = _load_3d_state(name, request)
    state_path = pipeline_dir / "state.json"

    if state.status != Pipeline3DStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"3D pipeline '{name}' is not awaiting approval (status: {state.status.value})",
        )

    if req.num_polys is not None:
        state.num_polys = req.num_polys
    if req.symmetrize is not None:
        state.symmetrize = req.symmetrize
    if req.symmetry_axis is not None:
        state.symmetry_axis = SymmetryAxis(req.symmetry_axis)

    state.status = Pipeline3DStatus.QUEUED
    state.mesh_path = None
    state.textured_mesh_path = None
    state.screenshot_dir = None
    state.review_sheet_path = None
    state.html_preview_path = None
    state.save(state_path)

    agent.enqueue_mesh_generation(name)
    event_bus.publish({
        "event": "status_change",
        "pipeline": name,
        "status": "queued",
        "message": "Mesh rejected. Re-queuing mesh generation.",
    })
    log.info("3D pipeline '%s' rejected by user '%s'", name, user.username)
    return OkResponse()


# ---------------------------------------------------------------------------
# Broker tasks (per-pipeline) — used by the CLI status / watch displays
# ---------------------------------------------------------------------------


@router.post("/3d-pipelines/{name}/retry")
async def retry_3d_pipeline(name: str, request: Request, user: CurrentUser):
    """Reset failed broker tasks for a 3D pipeline so workers will retry them."""
    _load_3d_state(name, request)  # 404 if missing
    agent = _agent(request)
    count = agent._broker.retry_failed_tasks(name)
    return {"status": "ok", "tasks_reset": count}


@router.get("/3d-pipelines/{name}/tasks")
async def get_3d_pipeline_tasks(name: str, request: Request, user: CurrentUser):
    """Return all broker tasks for a 3D pipeline.  See GET /pipelines/{name}/tasks."""
    _load_3d_state(name, request)  # 404 if missing
    agent = _agent(request)
    tasks = agent._broker.get_tasks(pipeline_name=name)
    return [
        {
            "id": t.id,
            "task_type": t.task_type,
            "status": t.status,
            "error": t.error,
        }
        for t in tasks
    ]
