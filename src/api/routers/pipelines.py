"""
Pipeline CRUD endpoints.

POST   /pipelines                   Create new pipeline (JSON)
POST   /pipelines/from-upload       Create new pipeline with an uploaded base
                                    image (multipart/form-data)
GET    /pipelines                   List pipelines (user's own; admin sees all)
GET    /pipelines/{name}            Get pipeline state
DELETE /pipelines/{name}            Cancel pipeline
PATCH  /pipelines/{name}            Edit (description, polys, symmetry, hidden)
POST   /pipelines/{name}/pause      Pause
POST   /pipelines/{name}/resume     Resume
POST   /pipelines/{name}/retry      Retry failed tasks
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from src.api.auth import CurrentUser
from src.api.event_bus import event_bus
from src.api.models import (
    AcceptedResponse,
    CreatePipelineRequest,
    OkResponse,
    PatchPipelineRequest,
)
from src.state import PipelineState, PipelineStatus, Pipeline3DStatus

log = logging.getLogger(__name__)
router = APIRouter(tags=["pipelines"])


def _agent(request: Request):
    return request.app.state.agent


def _cfg(request: Request):
    return request.app.state.cfg


def _load_state(name: str, request: Request) -> PipelineState:
    """Load pipeline state or raise 404."""
    agent = _agent(request)
    state = agent.get_pipeline_state(name)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Pipeline '{name}' not found")
    return state


def _state_path(name: str, request: Request):
    cfg = _cfg(request)
    return cfg.pipelines_dir / name / "state.json"


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@router.post("/pipelines", status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    req: CreatePipelineRequest,
    request: Request,
    user: CurrentUser,
):
    """Start a new pipeline."""
    agent = _agent(request)
    existing = agent.get_pipeline_state(req.name)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pipeline '{req.name}' already exists",
        )
    state = agent.start_pipeline(
        req.name,
        req.description,
        req.num_polys,
        input_image_path=req.input_image_path,
        symmetrize=req.symmetrize,
        symmetry_axis=req.symmetry_axis,
        concept_art_backend=req.concept_art_backend,
    )
    log.info("Pipeline '%s' created by user '%s'", req.name, user.username)
    event_bus.publish({
        "event": "pipeline_created",
        "pipeline": req.name,
        "status": state.status.value,
        "user": user.username,
    })
    return state.model_dump(mode="json")


@router.post("/pipelines/from-upload", status_code=status.HTTP_201_CREATED)
async def create_pipeline_from_upload(
    request: Request,
    user: CurrentUser,
    name: str = Form(...),
    description: str = Form(...),
    num_polys: int | None = Form(None),
    symmetrize: bool = Form(False),
    symmetry_axis: str = Form("x-"),
    concept_art_backend: str = Form("gemini"),
    image: UploadFile = File(...),
):
    """
    Start a new 2D pipeline using an uploaded base image.

    The image is saved into the pipeline directory and its server-side path
    is passed as ``input_image_path`` to the agent — the same code path as
    the JSON ``POST /pipelines`` endpoint with ``input_image_path`` set, but
    without requiring the client to know any server-side filesystem layout.
    """
    agent = _agent(request)
    cfg = _cfg(request)

    existing = agent.get_pipeline_state(name)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pipeline '{name}' already exists",
        )

    # Persist the upload into the pipeline directory before starting.
    pipeline_dir = cfg.pipelines_dir / name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    img_path = pipeline_dir / f"input_{image.filename}"
    contents = await image.read()
    img_path.write_bytes(contents)

    state = agent.start_pipeline(
        name,
        description,
        num_polys,
        input_image_path=str(img_path),
        symmetrize=symmetrize,
        symmetry_axis=symmetry_axis,
        concept_art_backend=concept_art_backend,
    )
    log.info("Pipeline '%s' created from upload by user '%s'", name, user.username)
    event_bus.publish({
        "event": "pipeline_created",
        "pipeline": name,
        "status": state.status.value,
        "user": user.username,
    })
    return state.model_dump(mode="json")


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


@router.get("/pipelines")
async def list_pipelines(request: Request, user: CurrentUser):
    """List all pipelines. Admin sees all; regular users see their own."""
    agent = _agent(request)
    names = agent.list_pipeline_names()
    result = []
    for name in names:
        state = agent.get_pipeline_state(name)
        if state:
            result.append({
                "name": name,
                "status": state.status.value,
                "description": state.description,
                "concept_art_backend": state.concept_art_backend,
                "created_at": state.created_at.isoformat(),
                "updated_at": state.updated_at.isoformat(),
            })
    return result


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


@router.get("/pipelines/{name}")
async def get_pipeline(name: str, request: Request, user: CurrentUser):
    """Return full pipeline state as JSON."""
    state = _load_state(name, request)
    return state.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


@router.delete("/pipelines/{name}")
async def cancel_pipeline(name: str, request: Request, user: CurrentUser):
    """Cancel all pending/running tasks for a pipeline and mark it cancelled."""
    agent = _agent(request)
    state = _load_state(name, request)
    agent.cancel_pipeline(name)
    state_path = _state_path(name, request)
    state.status = PipelineStatus.CANCELLED
    state.save(state_path)
    event_bus.publish({"event": "status_change", "pipeline": name, "status": "cancelled"})
    log.info("Pipeline '%s' cancelled by user '%s'", name, user.username)
    return OkResponse()


# ---------------------------------------------------------------------------
# Edit (PATCH)
# ---------------------------------------------------------------------------


@router.patch("/pipelines/{name}")
async def patch_pipeline(
    name: str,
    req: PatchPipelineRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Edit pipeline settings.

    ``hidden`` may be toggled at any status.  All other fields (description,
    num_polys, symmetrize, symmetry_axis) require the pipeline to still be in
    the editable window (before mesh generation starts) and a 409 is returned
    if any non-hidden field is supplied past that point.
    """
    from src.state import SymmetryAxis

    state = _load_state(name, request)
    state_path = _state_path(name, request)

    # ``hidden`` is always allowed.
    if req.hidden is not None:
        state.hidden = req.hidden

    non_hidden_fields = (
        req.description, req.num_polys, req.symmetrize, req.symmetry_axis,
    )
    if any(f is not None for f in non_hidden_fields):
        _editable = {
            PipelineStatus.INITIALIZING,
            PipelineStatus.CONCEPT_ART_GENERATING,
            PipelineStatus.CONCEPT_ART_REVIEW,
        }
        if state.status not in _editable:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot edit pipeline in status '{state.status.value}' (editing only allowed before mesh generation)",
            )
        if req.description is not None:
            state.description = req.description
        if req.num_polys is not None:
            state.num_polys = req.num_polys
        if req.symmetrize is not None:
            state.symmetrize = req.symmetrize
        if req.symmetry_axis is not None:
            state.symmetry_axis = SymmetryAxis(req.symmetry_axis)

    state.save(state_path)
    return state.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Pause / Resume
# ---------------------------------------------------------------------------


@router.post("/pipelines/{name}/pause")
async def pause_pipeline(name: str, request: Request, user: CurrentUser):
    """Pause a running pipeline."""
    agent = _agent(request)
    cfg = _cfg(request)
    _load_state(name, request)  # ensure exists
    agent.pause_pipeline(name, cfg)
    event_bus.publish({"event": "status_change", "pipeline": name, "status": "paused"})
    return OkResponse()


@router.post("/pipelines/{name}/resume")
async def resume_pipeline(name: str, request: Request, user: CurrentUser):
    """Resume a paused pipeline."""
    agent = _agent(request)
    state = _load_state(name, request)
    if state.status != PipelineStatus.PAUSED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pipeline '{name}' is not paused (status: {state.status.value})",
        )
    agent.resume_pipeline(name)
    event_bus.publish({"event": "status_change", "pipeline": name, "status": "resumed"})
    return OkResponse()


# ---------------------------------------------------------------------------
# Retry failed tasks
# ---------------------------------------------------------------------------


@router.post("/pipelines/{name}/retry")
async def retry_pipeline(name: str, request: Request, user: CurrentUser):
    """Reset failed broker tasks for a pipeline so workers will retry them."""
    agent = _agent(request)
    _load_state(name, request)  # ensure exists
    count = agent._broker.retry_failed_tasks(name)
    return {"status": "ok", "tasks_reset": count}


# ---------------------------------------------------------------------------
# Broker tasks (per-pipeline) — used by the CLI status / watch displays
# ---------------------------------------------------------------------------


@router.get("/pipelines/{name}/tasks")
async def get_pipeline_tasks(name: str, request: Request, user: CurrentUser):
    """
    Return all broker tasks for a 2D pipeline.

    Each task is serialised as ``{id, task_type, status, error}``.  This is the
    minimum the CLI needs to render queue depth and failure indicators in the
    status / watch views.  ``payload`` is intentionally omitted to keep the
    response small and avoid leaking server-internal fields.
    """
    _load_state(name, request)  # 404 if missing
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
