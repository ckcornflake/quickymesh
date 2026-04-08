"""
Concept art review endpoints for 2D pipelines.

GET    /pipelines/{name}/concept_art/sheet           Download review sheet PNG
GET    /pipelines/{name}/concept_art/{idx}           Download one concept art PNG
POST   /pipelines/{name}/concept_art/regenerate      Re-queue generation for indices
POST   /pipelines/{name}/concept_art/modify          Modify one image via Gemini
POST   /pipelines/{name}/concept_art/restyle         Restyle one image via ControlNet

Mesh review endpoints have moved to /3d-pipelines — see routers/pipelines_3d.py
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
    ModifyConceptArtRequest,
    OkResponse,
    RegenerateConceptArtRequest,
    RestyleConceptArtRequest,
)
from src.state import (
    ConceptArtStatus,
    PipelineState,
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
    pipeline_dir = cfg.pipelines_dir / name
    return state, pipeline_dir


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
    from src.concept_art_pipeline import _concept_art_path
    state, pipeline_dir = _load_state(name, request)
    if idx < 0 or idx >= len(state.concept_arts):
        raise HTTPException(status_code=404, detail=f"Index {idx} out of range")
    item = state.concept_arts[idx]
    img_path = _concept_art_path(pipeline_dir, idx, item.version)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image for index {idx} not yet generated")
    return FileResponse(str(img_path), media_type="image/png")


# ---------------------------------------------------------------------------
# Concept art — review actions
# ---------------------------------------------------------------------------


@router.post("/pipelines/{name}/concept_art/regenerate")
async def regenerate_concept_art(
    name: str,
    req: RegenerateConceptArtRequest,
    request: Request,
    user: CurrentUser,
):
    """
    Queue regeneration of concept art images.
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
