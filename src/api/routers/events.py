"""
SSE (Server-Sent Events) endpoints for real-time pipeline updates.

GET  /pipelines/{name}/events   SSE stream for one pipeline
GET  /events                    SSE stream for all of the user's pipelines

Event format
------------
Each SSE message is a JSON object, e.g.:
    {"event": "status_change", "pipeline": "my_ship", "status": "mesh_review"}
    {"event": "task_complete",  "pipeline": "my_ship", "task_type": "mesh_generate"}
    {"event": "heartbeat"}

Clients reconnect automatically (EventSource behaviour); the server sends a
heartbeat every 15 seconds to keep the connection alive through proxies.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from src.api.auth import CurrentUser
from src.api.event_bus import event_bus

log = logging.getLogger(__name__)
router = APIRouter(tags=["events"])

_HEARTBEAT_INTERVAL = 15.0  # seconds


async def _event_generator(queue: asyncio.Queue, pipeline_name: str | None = None):
    """
    Yield SSE-formatted events from the queue until the client disconnects.
    Sends a heartbeat every _HEARTBEAT_INTERVAL seconds when idle.
    """
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL)
                yield {"data": json.dumps(event)}
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"event": "heartbeat"})}
    except asyncio.CancelledError:
        pass
    finally:
        event_bus.unsubscribe(queue, pipeline_name)
        log.debug("SSE client disconnected (pipeline=%s)", pipeline_name)


@router.get("/pipelines/{name}/events")
async def pipeline_events(name: str, request: Request, user: CurrentUser):
    """SSE stream for a single pipeline's status changes."""
    queue = event_bus.subscribe(pipeline_name=name)
    log.debug("SSE subscriber connected for pipeline '%s'", name)
    return EventSourceResponse(_event_generator(queue, name))


@router.get("/events")
async def all_events(request: Request, user: CurrentUser):
    """SSE stream for all pipeline events (global subscriber)."""
    queue = event_bus.subscribe(pipeline_name=None)
    log.debug("SSE global subscriber connected (user=%s)", user.username)
    return EventSourceResponse(_event_generator(queue, None))
