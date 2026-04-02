"""
Tests for src/broker.py — SQLite task queue.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.broker import Broker, Task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def broker():
    b = Broker()  # in-memory
    yield b
    b.close()


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------


class TestEnqueue:
    def test_returns_integer_id(self, broker):
        task_id = broker.enqueue("pipe1", "concept_art_generate")
        assert isinstance(task_id, int)
        assert task_id > 0

    def test_ids_are_unique(self, broker):
        id1 = broker.enqueue("pipe1", "concept_art_generate")
        id2 = broker.enqueue("pipe1", "concept_art_generate")
        assert id1 != id2

    def test_default_payload_is_empty_dict(self, broker):
        task_id = broker.enqueue("pipe1", "mesh_generate")
        tasks = broker.get_tasks(pipeline_name="pipe1")
        assert tasks[0].payload == {}

    def test_payload_stored_and_retrieved(self, broker):
        broker.enqueue("pipe1", "mesh_generate", payload={"index": 3, "polys": 8000})
        tasks = broker.get_tasks(pipeline_name="pipe1")
        assert tasks[0].payload == {"index": 3, "polys": 8000}

    def test_initial_status_is_pending(self, broker):
        broker.enqueue("pipe1", "concept_art_generate")
        tasks = broker.get_tasks()
        assert tasks[0].status == "pending"

    def test_pipeline_name_and_task_type_stored(self, broker):
        broker.enqueue("mypipe", "screenshot", {"mesh": "a.glb"})
        tasks = broker.get_tasks()
        assert tasks[0].pipeline_name == "mypipe"
        assert tasks[0].task_type == "screenshot"


# ---------------------------------------------------------------------------
# claim_next
# ---------------------------------------------------------------------------


class TestClaimNext:
    def test_claims_oldest_pending_task(self, broker):
        id1 = broker.enqueue("pipe1", "mesh_generate")
        id2 = broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        assert task is not None
        assert task.id == id1

    def test_returns_none_when_no_matching_task(self, broker):
        broker.enqueue("pipe1", "concept_art_generate")
        task = broker.claim_next(["mesh_generate"])
        assert task is None

    def test_sets_status_to_running(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        tasks = broker.get_tasks()
        assert tasks[0].status == "running"

    def test_skips_running_tasks(self, broker):
        id1 = broker.enqueue("pipe1", "mesh_generate")
        id2 = broker.enqueue("pipe1", "mesh_generate")
        broker.claim_next(["mesh_generate"])  # claims id1
        task = broker.claim_next(["mesh_generate"])
        assert task is not None
        assert task.id == id2

    def test_returns_task_with_correct_fields(self, broker):
        broker.enqueue("pipe1", "screenshot", {"key": "val"})
        task = broker.claim_next(["screenshot"])
        assert isinstance(task, Task)
        assert task.pipeline_name == "pipe1"
        assert task.task_type == "screenshot"
        assert task.payload == {"key": "val"}
        assert task.status == "running"

    def test_accepts_multiple_task_types(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe2", "screenshot")
        task1 = broker.claim_next(["mesh_generate", "screenshot"])
        task2 = broker.claim_next(["mesh_generate", "screenshot"])
        assert {task1.task_type, task2.task_type} == {"mesh_generate", "screenshot"}

    def test_returns_none_when_queue_empty(self, broker):
        task = broker.claim_next(["mesh_generate"])
        assert task is None

    def test_thread_safety_no_double_claim(self, broker):
        """Two threads racing to claim — each task claimed at most once."""
        for _ in range(10):
            broker.enqueue("pipe1", "mesh_generate")

        claimed = []
        lock = threading.Lock()

        def worker():
            task = broker.claim_next(["mesh_generate"])
            if task is not None:
                with lock:
                    claimed.append(task.id)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(claimed) == len(set(claimed)), "Duplicate claims detected"
        assert len(claimed) <= 10


# ---------------------------------------------------------------------------
# mark_done / mark_failed
# ---------------------------------------------------------------------------


class TestMarkDone:
    def test_sets_status_to_done(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        broker.mark_done(task.id)
        tasks = broker.get_tasks()
        assert tasks[0].status == "done"

    def test_done_tasks_not_claimable(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        broker.mark_done(task.id)
        assert broker.claim_next(["mesh_generate"]) is None


class TestMarkFailed:
    def test_sets_status_to_failed(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        broker.mark_failed(task.id, "something went wrong")
        tasks = broker.get_tasks()
        assert tasks[0].status == "failed"

    def test_stores_error_message(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        broker.mark_failed(task.id, "OOM error")
        tasks = broker.get_tasks()
        assert tasks[0].error == "OOM error"

    def test_failed_tasks_not_claimable(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        broker.mark_failed(task.id, "err")
        assert broker.claim_next(["mesh_generate"]) is None


# ---------------------------------------------------------------------------
# get_tasks
# ---------------------------------------------------------------------------


class TestGetTasks:
    def test_returns_all_tasks_unfiltered(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe2", "screenshot")
        tasks = broker.get_tasks()
        assert len(tasks) == 2

    def test_filter_by_pipeline_name(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe2", "screenshot")
        tasks = broker.get_tasks(pipeline_name="pipe1")
        assert len(tasks) == 1
        assert tasks[0].pipeline_name == "pipe1"

    def test_filter_by_status(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe1", "mesh_generate")
        broker.claim_next(["mesh_generate"])
        tasks = broker.get_tasks(status="running")
        assert len(tasks) == 1

    def test_filter_by_task_type(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe1", "screenshot")
        tasks = broker.get_tasks(task_type="screenshot")
        assert len(tasks) == 1
        assert tasks[0].task_type == "screenshot"

    def test_combined_filters(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe1", "screenshot")
        broker.enqueue("pipe2", "mesh_generate")
        tasks = broker.get_tasks(pipeline_name="pipe1", task_type="mesh_generate")
        assert len(tasks) == 1

    def test_returns_empty_list_when_no_match(self, broker):
        tasks = broker.get_tasks(pipeline_name="nonexistent")
        assert tasks == []

    def test_ordered_by_id_ascending(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe1", "screenshot")
        tasks = broker.get_tasks()
        assert tasks[0].id < tasks[1].id


# ---------------------------------------------------------------------------
# has_pending_or_running
# ---------------------------------------------------------------------------


class TestHasPendingOrRunning:
    def test_true_when_pending(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        assert broker.has_pending_or_running("pipe1", "mesh_generate") is True

    def test_true_when_running(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.claim_next(["mesh_generate"])
        assert broker.has_pending_or_running("pipe1", "mesh_generate") is True

    def test_false_when_done(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        broker.mark_done(task.id)
        assert broker.has_pending_or_running("pipe1", "mesh_generate") is False

    def test_false_when_failed(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        broker.mark_failed(task.id, "err")
        assert broker.has_pending_or_running("pipe1", "mesh_generate") is False

    def test_false_when_no_tasks(self, broker):
        assert broker.has_pending_or_running("pipe1", "mesh_generate") is False

    def test_scoped_to_pipeline(self, broker):
        broker.enqueue("pipe2", "mesh_generate")
        assert broker.has_pending_or_running("pipe1", "mesh_generate") is False


# ---------------------------------------------------------------------------
# cancel_pipeline_tasks
# ---------------------------------------------------------------------------


class TestCancelPipelineTasks:
    def test_cancels_pending_tasks(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe1", "screenshot")
        count = broker.cancel_pipeline_tasks("pipe1")
        assert count == 2
        for t in broker.get_tasks(pipeline_name="pipe1"):
            assert t.status == "failed"
            assert t.error == "cancelled"

    def test_cancels_running_tasks(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.claim_next(["mesh_generate"])
        broker.cancel_pipeline_tasks("pipe1")
        tasks = broker.get_tasks(pipeline_name="pipe1")
        assert tasks[0].status == "failed"

    def test_does_not_affect_done_tasks(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        task = broker.claim_next(["mesh_generate"])
        broker.mark_done(task.id)
        count = broker.cancel_pipeline_tasks("pipe1")
        assert count == 0
        assert broker.get_tasks()[0].status == "done"

    def test_does_not_affect_other_pipelines(self, broker):
        broker.enqueue("pipe1", "mesh_generate")
        broker.enqueue("pipe2", "screenshot")
        broker.cancel_pipeline_tasks("pipe1")
        pipe2_tasks = broker.get_tasks(pipeline_name="pipe2")
        assert pipe2_tasks[0].status == "pending"

    def test_returns_zero_when_nothing_to_cancel(self, broker):
        assert broker.cancel_pipeline_tasks("nonexistent") == 0
