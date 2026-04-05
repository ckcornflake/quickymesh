"""
System health / status endpoint.

GET  /status   Worker health, queue depth, active pipelines summary
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
