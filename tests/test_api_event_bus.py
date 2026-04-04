"""
Tests for src/api/event_bus.py
"""
from __future__ import annotations

import asyncio

import pytest

from src.api.event_bus import EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bus_with_loop() -> tuple[EventBus, asyncio.AbstractEventLoop]:
    bus = EventBus()
    loop = asyncio.new_event_loop()
    bus.set_loop(loop)
    return bus, loop


def _run(loop: asyncio.AbstractEventLoop, coro):
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_subscribe_returns_queue(self):
        bus = EventBus()
        q = bus.subscribe()
        assert isinstance(q, asyncio.Queue)

    def test_publish_before_loop_set_does_not_raise(self):
        bus = EventBus()
        bus.subscribe()
        bus.publish({"event": "test"})  # loop is None — should be silently ignored

    def test_publish_delivers_to_global_subscriber(self):
        bus, loop = _make_bus_with_loop()
        q = bus.subscribe(pipeline_name=None)
        bus.publish({"event": "ping"})

        async def drain():
            return await asyncio.wait_for(q.get(), timeout=1.0)

        event = _run(loop, drain())
        assert event["event"] == "ping"
        loop.close()

    def test_publish_delivers_to_pipeline_subscriber(self):
        bus, loop = _make_bus_with_loop()
        q = bus.subscribe(pipeline_name="my_pipe")
        bus.publish({"event": "ping", "pipeline": "my_pipe"})

        async def drain():
            return await asyncio.wait_for(q.get(), timeout=1.0)

        event = _run(loop, drain())
        assert event["event"] == "ping"
        loop.close()

    def test_pipeline_event_also_delivered_to_global_subscriber(self):
        bus, loop = _make_bus_with_loop()
        global_q = bus.subscribe(pipeline_name=None)
        pipe_q = bus.subscribe(pipeline_name="my_pipe")
        bus.publish({"event": "status_change", "pipeline": "my_pipe"})

        async def drain_both():
            g = await asyncio.wait_for(global_q.get(), timeout=1.0)
            p = await asyncio.wait_for(pipe_q.get(), timeout=1.0)
            return g, p

        g, p = _run(loop, drain_both())
        assert g["pipeline"] == "my_pipe"
        assert p["pipeline"] == "my_pipe"
        loop.close()

    def test_event_not_delivered_to_different_pipeline_subscriber(self):
        bus, loop = _make_bus_with_loop()
        q_a = bus.subscribe(pipeline_name="pipe_a")
        q_b = bus.subscribe(pipeline_name="pipe_b")
        bus.publish({"event": "ping", "pipeline": "pipe_a"})

        async def check():
            # pipe_a gets it
            await asyncio.wait_for(q_a.get(), timeout=1.0)
            # pipe_b's queue should be empty
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q_b.get(), timeout=0.1)

        _run(loop, check())
        loop.close()

    def test_unsubscribe_stops_delivery(self):
        bus, loop = _make_bus_with_loop()
        q = bus.subscribe(pipeline_name=None)
        bus.unsubscribe(q, pipeline_name=None)
        bus.publish({"event": "ping"})

        async def check():
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), timeout=0.1)

        _run(loop, check())
        loop.close()

    def test_unsubscribe_pipeline_stops_delivery(self):
        bus, loop = _make_bus_with_loop()
        q = bus.subscribe(pipeline_name="my_pipe")
        bus.unsubscribe(q, pipeline_name="my_pipe")
        bus.publish({"event": "ping", "pipeline": "my_pipe"})

        async def check():
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), timeout=0.1)

        _run(loop, check())
        loop.close()

    def test_multiple_subscribers_all_receive(self):
        bus, loop = _make_bus_with_loop()
        queues = [bus.subscribe(pipeline_name=None) for _ in range(3)]
        bus.publish({"event": "broadcast"})

        async def drain_all():
            return [await asyncio.wait_for(q.get(), timeout=1.0) for q in queues]

        events = _run(loop, drain_all())
        assert all(e["event"] == "broadcast" for e in events)
        loop.close()

    def test_set_loop_replaces_previous_loop(self):
        bus = EventBus()
        loop1 = asyncio.new_event_loop()
        loop2 = asyncio.new_event_loop()
        bus.set_loop(loop1)
        bus.set_loop(loop2)
        q = bus.subscribe(pipeline_name=None)
        bus.publish({"event": "test"})

        async def drain():
            return await asyncio.wait_for(q.get(), timeout=1.0)

        event = loop2.run_until_complete(drain())
        assert event["event"] == "test"
        loop1.close()
        loop2.close()
