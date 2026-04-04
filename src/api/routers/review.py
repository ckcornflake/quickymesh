"""
Concept art and mesh review endpoints.

Concept art
-----------
GET    /pipelines/{name}/concept_art/sheet           Download review sheet PNG
GET    /pipelines/{name}/concept_art/{idx}           Download one concept art PNG
POST   /pipelines/{name}/concept_art/approve         Approve selected images
POST   /pipelines/{name}/concept_art/regenerate      Re-queue generation for indices
POST   /pipelines/{name}/concept_art/modify          Modify one image via Gemini
POST   /pipelines/{name}/concept_art/restyle         Restyle one image via ControlNet

Mesh
----
GET    /pipelines/{name}/meshes/{mesh_name}/sheet        Download review sheet PNG
GET    /pipelines/{name}/meshes/{mesh_name}/screenshot/{filename}  Screenshot PNG
GET    /pipelines/{name}/meshes/{mesh_name}/preview      Download HTML preview
GET    /pipelines/{name}/meshes/{mesh_name}/mesh         Download textured GLB
POST   /pipelines/{name}/meshes/{mesh_name}/approve      Approve mesh
POST   /pipelines/{name}/meshes/{mesh_name}/reject       Reject mesh
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from src.api.auth import CurrentUser
from src.api.event_bus import event_bus
from src.api.models import (
    AcceptedResponse,
    ApproveMeshRequest,
    ApproveConceptArtRequest,
    ModifyConceptArtRequest,
    OkResponse,
    RegenerateConceptArtRequest,
    RejectMeshRequest,
    RestyleConceptArtRequest,
)
from src.state import (
    ConceptArtStatus,
    MeshStatus,
    PipelineState,
    PipelineStatus,
    SymmetryAxis,
)

log = logging.getLogger(__name__)
router = APIRouter(tags=["review"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(request: Request):
    return request.app.state.agent


def _cfg(request: Request):
    return request.app.state.cfg


def _load_state(name: str, request: Request) -> tuple[PipelineState, Path]:
    """Return (state, pipeline_dir) or raise 404."""
    agent = _agent(request)
    cfg = _cfg(request)
    state = agent.get_pipeline_state(name)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")
    pipeline_dir = cfg.uncompleted_pipelines_dir / name
    return state, pipeline_dir


def _get_mesh(state: PipelineState, mesh_name: str):
    for m in state.meshes:
        if m.sub_name == mesh_name:
            return m
    raise HTTPException(status_code=404, detail=f"Mesh '{mesh_name}' not found")


# ---------------------------------------------------------------------------
# Concept art — image downloads
# ---------------------------------------------------------------------------


@router.get("/pipelines/{name}/concept_art/sheet")
async def get_concept_art_sheet(name: str, request: Request, user: CurrentUser):
    """Build (or rebuild) and return the concept art review sheet PNG."""
    from src.concept_art_pipeline import build_review_sheet

    state, pipeline_dir = _load_state(name, request)
    cfg = _cfg(request)
    try:
        sheet_path = await asyncio.to_thread(build_review_sheet, state, pipeline_dir, cfg)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return FileResponse(str(sheet_path), media_type="image/png")


@router.get("/pipelines/{name}/concept_art/{idx}")
async def get_concept_art_image(name: str, idx: int, request: Request, user: CurrentUser):
    """Return a single concept art PNG by 0-based index."""
    state, _ = _load_state(name, request)
    if idx < 0 or idx >= len(state.concept_arts):
        raise HTTPException(status_code=404, detail=f"Index {idx} out of range")
    item = state.concept_arts[idx]
    if not item.image_path or not Path(item.image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image for index {idx} not yet generated")
    return FileResponse(item.image_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Concept art — review actions
# ---------------------------------------------------------------------------


@router.post("/pipelines/{name}/concept_art/approve")
async def approve_concept_art(
    name: str,
    req: ApproveConceptArtRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Approve the given concept art indices and queue mesh generation.

    Indices are 0-based.  The pipeline status transitions to MESH_GENERATING
    and a mesh_generate broker task is enqueued.
    """
    agent = _agent(request)
    state, pipeline_dir = _load_state(name, request)
    state_path = pipeline_dir / "state.json"

    for idx in req.indices:
        if idx < 0 or idx >= len(state.concept_arts):
            raise HTTPException(status_code=422, detail=f"Index {idx} out of range")
        state.concept_arts[idx].status = ConceptArtStatus.APPROVED

    state.status = PipelineStatus.MESH_GENERATING
    state.save(state_path)

    agent.enqueue_mesh_generation(name)
    event_bus.publish({
        "event": "status_change",
        "pipeline": name,
        "status": "mesh_generating",
        "message": f"Concept art approved ({len(req.indices)} image(s)). Mesh generation queued.",
    })
    log.info("Concept art approved for '%s': indices=%s", name, req.indices)
    return OkResponse()


@router.post("/pipelines/{name}/concept_art/regenerate")
async def regenerate_concept_art(
    name: str,
    req: RegenerateConceptArtRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Queue regeneration of concept art images.

    Sends the job to the concept art worker (async).  Listen to SSE or
    poll GET /pipelines/{name} to know when generation is complete.
    """
    agent = _agent(request)
    state, pipeline_dir = _load_state(name, request)
    state_path = pipeline_dir / "state.json"
    cfg = _cfg(request)

    if req.indices is None:
        indices = list(range(len(state.concept_arts)))
    else:
        for idx in req.indices:
            if idx < 0 or idx >= len(state.concept_arts):
                raise HTTPException(status_code=422, detail=f"Index {idx} out of range")
        indices = req.indices

    if req.description_override:
        state.description = req.description_override

    for idx in indices:
        state.concept_arts[idx].status = ConceptArtStatus.REGENERATING

    state.concept_art_sheet_shown = False
    state.save(state_path)

    payload = {
        "pipeline_name": name,
        "state_path": str(state_path),
        "indices": indices,
    }
    agent._broker.enqueue(name, "concept_art_generate", payload)
    event_bus.publish({
        "event": "status_change",
        "pipeline": name,
        "status": "concept_art_generating",
        "message": f"Regenerating {len(indices)} concept art image(s).",
    })
    log.info("Concept art regeneration queued for '%s': indices=%s", name, indices)
    return AcceptedResponse(message=f"Regenerating {len(indices)} image(s)")


@router.post("/pipelines/{name}/concept_art/modify")
async def modify_concept_art(
    name: str,
    req: ModifyConceptArtRequest,
    request: Request,
    user: CurrentUser,
):
    """Modify one concept art image via Gemini's edit API (runs synchronously)."""
    from src.concept_art_pipeline import modify_concept_art as _modify

    state, pipeline_dir = _load_state(name, request)
    state_path = pipeline_dir / "state.json"
    concept_worker = request.app.state.concept_worker

    if not getattr(concept_worker, "supports_modify", False):
        raise HTTPException(
            status_code=409,
            detail="Image modification is not supported by the current concept art backend",
        )

    try:
        await asyncio.to_thread(_modify, state, concept_worker, pipeline_dir, req.index, req.instruction)
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    state.concept_art_sheet_shown = False
    state.save(state_path)
    event_bus.publish({
        "event": "concept_art_updated",
        "pipeline": name,
        "index": req.index,
    })
    return OkResponse()


@router.post("/pipelines/{name}/concept_art/restyle")
async def restyle_concept_art(
    name: str,
    req: RestyleConceptArtRequest,
    request: Request,
    user: CurrentUser,
):
    """Restyle one concept art image via ControlNet Canny (runs synchronously)."""
    from src.concept_art_pipeline import restyle_concept_art as _restyle

    restyle_worker = request.app.state.restyle_worker
    if restyle_worker is None:
        raise HTTPException(
            status_code=409,
            detail="ControlNet restyle worker is not configured",
        )

    state, pipeline_dir = _load_state(name, request)
    state_path = pipeline_dir / "state.json"

    try:
        await asyncio.to_thread(
            _restyle, state, restyle_worker, pipeline_dir,
            req.index, req.positive, req.negative, req.denoise,
        )
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    state.concept_art_sheet_shown = False
    state.save(state_path)
    event_bus.publish({
        "event": "concept_art_updated",
        "pipeline": name,
        "index": req.index,
    })
    return OkResponse()


# ---------------------------------------------------------------------------
# Mesh — image / file downloads
# ---------------------------------------------------------------------------


@router.get("/pipelines/{name}/meshes/{mesh_name}/sheet")
async def get_mesh_review_sheet(
    name: str, mesh_name: str, request: Request, user: CurrentUser,
):
    """Return the mesh review sheet (screenshot grid) PNG."""
    state, _ = _load_state(name, request)
    mesh = _get_mesh(state, mesh_name)
    if not mesh.review_sheet_path or not Path(mesh.review_sheet_path).exists():
        raise HTTPException(status_code=404, detail="Review sheet not yet generated")
    return FileResponse(mesh.review_sheet_path, media_type="image/png")


@router.get("/pipelines/{name}/meshes/{mesh_name}/screenshot/{filename}")
async def get_mesh_screenshot(
    name: str, mesh_name: str, filename: str, request: Request, user: CurrentUser,
):
    """Return a single screenshot PNG by filename (e.g. 'render_front.png')."""
    state, _ = _load_state(name, request)
    mesh = _get_mesh(state, mesh_name)
    if not mesh.screenshot_dir:
        raise HTTPException(status_code=404, detail="Screenshots not yet generated")
    img_path = Path(mesh.screenshot_dir) / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Screenshot '{filename}' not found")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/pipelines/{name}/meshes/{mesh_name}/preview")
async def get_mesh_preview(
    name: str, mesh_name: str, request: Request, user: CurrentUser,
):
    """Return the Three.js HTML preview file."""
    state, _ = _load_state(name, request)
    mesh = _get_mesh(state, mesh_name)
    if not mesh.html_preview_path or not Path(mesh.html_preview_path).exists():
        raise HTTPException(status_code=404, detail="HTML preview not yet generated")
    return FileResponse(mesh.html_preview_path, media_type="text/html")


@router.get("/pipelines/{name}/meshes/{mesh_name}/mesh")
async def get_mesh_file(
    name: str, mesh_name: str, request: Request, user: CurrentUser,
):
    """Return the textured GLB file."""
    state, _ = _load_state(name, request)
    mesh = _get_mesh(state, mesh_name)
    glb_path = mesh.textured_mesh_path or mesh.mesh_path
    if not glb_path or not Path(glb_path).exists():
        raise HTTPException(status_code=404, detail="Mesh file not yet generated")
    return FileResponse(glb_path, media_type="model/gltf-binary", filename=f"{mesh_name}.glb")


# ---------------------------------------------------------------------------
# Mesh — review actions
# ---------------------------------------------------------------------------


@router.post("/pipelines/{name}/meshes/{mesh_name}/approve")
async def approve_mesh(
    name: str,
    mesh_name: str,
    req: ApproveMeshRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Approve a mesh.  If all pending meshes are now reviewed, trigger export.
    """
    from src.mesh_pipeline import run_mesh_export

    agent = _agent(request)
    cfg = _cfg(request)
    state, pipeline_dir = _load_state(name, request)
    state_path = pipeline_dir / "state.json"
    mesh = _get_mesh(state, mesh_name)

    if mesh.status != MeshStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"Mesh '{mesh_name}' is not awaiting approval (status: {mesh.status.value})",
        )

    mesh.final_name = req.asset_name
    mesh.export_format = req.export_format or cfg.export_format
    mesh.status = MeshStatus.APPROVED

    # Check if all meshes are now reviewed
    remaining = [m for m in state.meshes if m.status == MeshStatus.AWAITING_APPROVAL]
    if not remaining:
        approved = [m for m in state.meshes if m.status == MeshStatus.APPROVED]
        if approved:
            state.status = PipelineStatus.APPROVED
            state.save(state_path)
            await asyncio.to_thread(run_mesh_export, state, pipeline_dir, cfg)
            event_bus.publish({
                "event": "status_change",
                "pipeline": name,
                "status": "approved",
                "message": f"All meshes reviewed. {len(approved)} mesh(es) exported.",
            })
            log.info("Pipeline '%s' completed: %d mesh(es) exported", name, len(approved))
        else:
            # All rejected — re-queue
            state.status = PipelineStatus.MESH_GENERATING
            state.save(state_path)
            agent.enqueue_mesh_generation(name)
            event_bus.publish({
                "event": "status_change",
                "pipeline": name,
                "status": "mesh_generating",
                "message": "All meshes rejected. Re-queuing mesh generation.",
            })
    else:
        state.save(state_path)

    return OkResponse()


@router.post("/pipelines/{name}/meshes/{mesh_name}/reject")
async def reject_mesh(
    name: str,
    mesh_name: str,
    req: RejectMeshRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Reject a mesh.  Optionally update poly count and symmetry settings.
    If this was the last pending mesh, re-queue mesh generation.
    """
    agent = _agent(request)
    state, pipeline_dir = _load_state(name, request)
    state_path = pipeline_dir / "state.json"
    mesh = _get_mesh(state, mesh_name)

    if mesh.status != MeshStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"Mesh '{mesh_name}' is not awaiting approval (status: {mesh.status.value})",
        )

    # Apply optional setting updates
    if req.num_polys is not None:
        state.num_polys = req.num_polys
    if req.symmetrize is not None:
        state.symmetrize = req.symmetrize
    if req.symmetry_axis is not None:
        state.symmetry_axis = SymmetryAxis(req.symmetry_axis)

    mesh.status = MeshStatus.QUEUED
    log.info("Mesh '%s' rejected in pipeline '%s'", mesh_name, name)

    # Check if all meshes are now reviewed
    remaining = [m for m in state.meshes if m.status == MeshStatus.AWAITING_APPROVAL]
    if not remaining:
        approved = [m for m in state.meshes if m.status == MeshStatus.APPROVED]
        if approved:
            from src.mesh_pipeline import run_mesh_export
            state.status = PipelineStatus.APPROVED
            state.save(state_path)
            await asyncio.to_thread(run_mesh_export, state, pipeline_dir, _cfg(request))
            event_bus.publish({
                "event": "status_change",
                "pipeline": name,
                "status": "approved",
                "message": f"Review complete. {len(approved)} mesh(es) exported.",
            })
        else:
            state.status = PipelineStatus.MESH_GENERATING
            state.save(state_path)
            agent.enqueue_mesh_generation(name)
            event_bus.publish({
                "event": "status_change",
                "pipeline": name,
                "status": "mesh_generating",
                "message": "All meshes rejected. Re-queuing mesh generation.",
            })
    else:
        state.save(state_path)

    return OkResponse()
