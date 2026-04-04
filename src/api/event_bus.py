"""
Thread-safe event bus: sync worker threads → async SSE clients.

Worker threads call event_bus.publish({...}) from any thread.
SSE endpoints subscribe() to get an asyncio.Queue that receives events.

Usage
-----
# At server startup (async context):
    event_bus.set_loop(asyncio.get_running_loop())

# In an SSE endpoint:
    q = event_bus.subscribe(pipeline_name="my_pipe")
    try:
        async for event in ...:
            event = await q.get()
    finally:
        event_bus.unsubscribe(q, pipeline_name="my_pipe")

# In any worker thread:
    event_bus.publish({"event": "status_change", "pipeline": "my_pipe",
                        "status": "mesh_review"})
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any


class EventBus:
    """Bridge between synchronous worker threads and async SSE clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        # pipeline_name → set of queues for that pipeline's subscribers
        self._pipeline_subs: dict[str, set[asyncio.Queue]] = {}
        # queues that receive every event regardless of pipeline
        self._global_subs: set[asyncio.Queue] = set()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Call once from async context at server startup."""
        self._loop = loop

    def subscribe(self, pipeline_name: str | None = None) -> "asyncio.Queue[dict]":
        """
        Return a new asyncio.Queue that receives published events.

        pipeline_name=None  → subscribe to ALL events
        pipeline_name="foo" → receive events for that pipeline only
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            if pipeline_name is None:
                self._global_subs.add(q)
            else:
                self._pipeline_subs.setdefault(pipeline_name, set()).add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[dict]", pipeline_name: str | None = None) -> None:
        """Remove a subscription. Always call from SSE finally block."""
        with self._lock:
            if pipeline_name is None:
                self._global_subs.discard(q)
            else:
                self._pipeline_subs.get(pipeline_name, set()).discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        """
        Publish an event from any thread (sync or async).

        Events with a "pipeline" key are delivered to both per-pipeline
        subscribers and global subscribers.
        """
        if self._loop is None or self._loop.is_closed():
            return
        pipeline_name: str | None = event.get("pipeline")
        with self._lock:
            targets: set[asyncio.Queue] = set(self._global_subs)
            if pipeline_name:
                targets |= self._pipeline_subs.get(pipeline_name, set())
        for q in targets:
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass  # queue full or loop closed — drop silently


# Global singleton — imported by worker threads and routers
event_bus = EventBus()
