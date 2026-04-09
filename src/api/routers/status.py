"""
System health / status endpoints.

GET  /status                  Worker health, queue depth, active pipelines summary
GET  /config                  Read-only snapshot of server-side configuration
GET  /pipelines-with-failures Names of pipelines that have non-cancelled failed
                              broker tasks (used by the CLI [t] retry menu)
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.auth import CurrentUser

router = APIRouter(tags=["system"])


@router.get("/status")
async def get_status(request: Request, user: CurrentUser):
    """Return worker health and queue statistics."""
    agent = request.app.state.agent

    workers = [
        {"name": t.__class__.__name__, "alive": t.is_alive()}
        for t in agent._threads
    ]

    pipeline_names = agent.list_pipeline_names()
    pipelines = []
    for name in pipeline_names:
        state = agent.get_pipeline_state(name)
        tasks = agent._broker.get_tasks(pipeline_name=name)
        in_flight = [t for t in tasks if t.status in ("pending", "running")]
        failed = [t for t in tasks if t.status == "failed" and t.error != "cancelled"]
        pipelines.append({
            "name": name,
            "status": state.status.value if state else "unknown",
            "queued_tasks": len(in_flight),
            "failed_tasks": len(failed),
        })

    cfg = request.app.state.cfg
    return {
        "workers": workers,
        "all_workers_alive": all(w["alive"] for w in workers),
        "pipeline_count": len(pipeline_names),
        "pipelines": pipelines,
        "output_root": str(cfg.output_root),
    }


@router.get("/config")
async def get_config(request: Request, user: CurrentUser):
    """
    Return a read-only snapshot of server-side configuration that clients
    (CLI, web frontend) need to render menus, validate inputs, and resolve
    relative paths.

    Notes
    -----
    - Paths are server-local strings.  Clients should treat them as opaque
      identifiers — only the server reads from them — except where the user
      pastes a path the server then resolves (e.g. base-image references).
    - ``gemini_api_key_present`` is a boolean check; the key itself is never
      returned.
    """
    cfg = request.app.state.cfg

    try:
        _ = cfg.gemini_api_key
        gemini_present = True
    except EnvironmentError:
        gemini_present = False

    return {
        "output_root": str(cfg.output_root),
        "pipelines_dir": str(cfg.pipelines_dir),
        "final_assets_dir": str(cfg.final_assets_dir),
        "background_suffix": cfg.background_suffix,
        "num_polys_default": cfg.num_polys,
        "num_concept_arts_default": cfg.num_concept_arts,
        "concept_art_image_size": cfg.concept_art_image_size,
        "export_format": cfg.export_format,
        "gemini_api_key_present": gemini_present,
    }


@router.get("/pipelines-with-failures")
async def get_pipelines_with_failures(request: Request, user: CurrentUser):
    """
    List pipeline names that have at least one failed (non-cancelled) broker
    task.  This drives the CLI ``[t] Retry failed tasks`` menu so the user can
    pick which pipeline to retry without scanning every pipeline's tasks.
    """
    agent = request.app.state.agent
    return {"pipelines": agent._broker.pipelines_with_failures()}
