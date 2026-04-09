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
        # Per-pipeline FIFO locks pipe1 until its work clears; finish it,
        # then pipe2 becomes the active pipeline.
        broker.mark_done(task1.id)
        task2 = broker.claim_next(["mesh_generate", "screenshot"])
        assert {task1.task_type, task2.task_type} == {"mesh_generate", "screenshot"}

    def test_returns_none_when_queue_empty(self, broker):
        task = broker.claim_next(["mesh_generate"])
        assert task is None

    def test_workflow_scope_locks_active_pipeline(self, broker):
        """
        Per-pipeline FIFO: once pipeA has a task in-flight within the workflow
        scope, tasks from pipeB must not be claimed — even if this worker has
        no eligible task for pipeA right now.  This mirrors the real setup
        where TrellisWorker and ScreenshotWorker share a workflow scope of
        ['mesh_generate','mesh_texture','mesh_cleanup','screenshot'].
        """
        workflow = ["mesh_generate", "mesh_texture", "mesh_cleanup", "screenshot"]

        broker.enqueue("pipeA", "mesh_generate")          # active pipeline
        broker.enqueue("pipeB", "mesh_generate")

        # TrellisWorker claims pipeA.mesh_generate.
        t1 = broker.claim_next(["mesh_texture", "mesh_generate"], workflow)
        assert t1.pipeline_name == "pipeA"
        assert t1.task_type == "mesh_generate"

        # pipeA's mesh_texture enqueues (chain).  ScreenshotWorker polls, but
        # pipeA has no screenshot/cleanup work yet.  It must NOT claim pipeB's
        # mesh_generate — it must idle.
        broker.enqueue("pipeA", "mesh_texture")
        t_ss = broker.claim_next(["mesh_cleanup", "screenshot"], workflow)
        assert t_ss is None, "ScreenshotWorker must not jump to pipeB"

        # Finish pipeA's work end-to-end.
        broker.mark_done(t1.id)
        t2 = broker.claim_next(["mesh_texture", "mesh_generate"], workflow)
        assert t2.pipeline_name == "pipeA"
        assert t2.task_type == "mesh_texture"
        broker.mark_done(t2.id)

        broker.enqueue("pipeA", "mesh_cleanup")
        t3 = broker.claim_next(["mesh_cleanup", "screenshot"], workflow)
        assert t3.pipeline_name == "pipeA"
        broker.mark_done(t3.id)

        broker.enqueue("pipeA", "screenshot")
        t4 = broker.claim_next(["mesh_cleanup", "screenshot"], workflow)
        assert t4.pipeline_name == "pipeA"
        broker.mark_done(t4.id)

        # Now pipeB becomes the active pipeline.
        t5 = broker.claim_next(["mesh_texture", "mesh_generate"], workflow)
        assert t5.pipeline_name == "pipeB"

    def test_awaiting_approval_does_not_block_other_pipelines(self, broker):
        """
        When pipeA reaches AWAITING_APPROVAL, its screenshot task is 'done'
        and no successor is enqueued (approval is a separate HTTP action).
        With zero pending/running tasks in scope, pipeA must NOT hold the
        active-pipeline slot — pipeB should immediately become active.
        """
        workflow = ["mesh_generate", "mesh_texture", "mesh_cleanup", "screenshot"]

        # pipeA runs end-to-end and reaches awaiting_approval.
        for ttype in workflow:
            broker.enqueue("pipeA", ttype)
        for _ in workflow:
            t = broker.claim_next(workflow, workflow)
            assert t.pipeline_name == "pipeA"
            broker.mark_done(t.id)

        # pipeB submitted while pipeA is awaiting user approval.
        broker.enqueue("pipeB", "mesh_generate")
        t = broker.claim_next(["mesh_texture", "mesh_generate"], workflow)
        assert t is not None
        assert t.pipeline_name == "pipeB", (
            "pipeA in AWAITING_APPROVAL must not block pipeB from running"
        )

    def test_workflow_scope_isolates_unrelated_workflows(self, broker):
        """
        A 3D workflow holding pipeA must not block a concept-art task for
        pipeB — the two workflows have disjoint scope lists.
        """
        w_3d = ["mesh_generate", "mesh_texture", "mesh_cleanup", "screenshot"]
        w_ca = ["concept_art_generate", "concept_art_modify", "concept_art_restyle"]

        broker.enqueue("pipeA", "mesh_generate")
        broker.claim_next(["mesh_texture", "mesh_generate"], w_3d)

        broker.enqueue("pipeB", "concept_art_generate")
        ca = broker.claim_next(
            ["concept_art_generate", "concept_art_modify", "concept_art_restyle"],
            w_ca,
        )
        assert ca is not None
        assert ca.pipeline_name == "pipeB"

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
