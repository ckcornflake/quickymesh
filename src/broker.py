"""
SQLite-backed task queue (broker).

All inter-component communication goes through this broker so that:
  - The pipeline survives a crash (tasks persist on disk).
  - Workers and the agent never share in-process state.
  - Multiple pipelines can be managed concurrently.

Schema
------
tasks(id, pipeline_name, task_type, payload JSON, status, error, created_at, updated_at)

Task lifecycle:  pending → running → done | failed

Task types (task_type strings)
-------------------------------
  concept_art_generate   Generate concept art images for a 2D pipeline.
                         Handled by ConceptArtWorkerThread.
  mesh_generate          Run Trellis mesh generation for a 3D pipeline.
                         Handled by TrellisWorkerThread; auto-chains to
                         mesh_texture on success.
  mesh_texture           Run Trellis texturing for a 3D pipeline's mesh.
                         Handled by TrellisWorkerThread; auto-chains to
                         mesh_cleanup on success.
  mesh_cleanup           Blender-based mesh clean-up / decimation / symmetrize.
                         Handled by ScreenshotWorkerThread; auto-chains to
                         screenshot on success.
  screenshot             Take Blender screenshots + build review sheet + HTML
                         preview, then set status AWAITING_APPROVAL.
                         Handled by ScreenshotWorkerThread.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class Task:
    id: int
    pipeline_name: str
    task_type: str
    payload: dict
    status: str          # pending | running | done | failed
    error: str | None
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_name TEXT    NOT NULL,
    task_type     TEXT    NOT NULL,
    payload       TEXT    NOT NULL DEFAULT '{}',
    status        TEXT    NOT NULL DEFAULT 'pending',
    error         TEXT,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status_type
    ON tasks (status, task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_pipeline
    ON tasks (pipeline_name);
"""


class Broker:
    """
    Thread-safe SQLite task queue.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Pass `:memory:` for in-memory (tests).
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        # check_same_thread=False is safe because we serialise via _lock
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_DDL)
            # Crash recovery: any task left as 'running' from a previous
            # session will never be re-claimed.  Reset them to 'pending' so
            # the new session's workers can pick them up.
            self._conn.execute(
                "UPDATE tasks SET status='pending', error=NULL, updated_at=? "
                "WHERE status='running'",
                (_now(),),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        pipeline_name: str,
        task_type: str,
        payload: dict | None = None,
    ) -> int:
        """Insert a new pending task and return its id."""
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tasks (pipeline_name, task_type, payload, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?)",
                (pipeline_name, task_type, json.dumps(payload or {}), now, now),
            )
            self._conn.commit()
            return cur.lastrowid

    def claim_next(
        self,
        task_types: list[str],
        workflow_types: list[str] | None = None,
    ) -> Task | None:
        """
        Atomically claim the next eligible pending task.  Returns None if no
        matching task is available.  Sets status → 'running'.

        Per-pipeline FIFO (workflow scope)
        ----------------------------------
        `workflow_types` is the full set of task_types that belong to the same
        pipeline workflow (e.g. all four 3D stages: mesh_generate, mesh_texture,
        mesh_cleanup, screenshot).  All workers that cooperate on a workflow
        should pass the same `workflow_types` list so they agree on scope.

        The broker identifies the *active pipeline* within that scope: the
        `pipeline_name` of the oldest pending/running task whose type is in
        `workflow_types`.  Only tasks belonging to the active pipeline are
        eligible for claiming.  If the active pipeline has no task of a type
        this worker handles right now, the worker idles rather than picking
        up a different pipeline — this guarantees that once a pipeline has
        work in flight, it runs end-to-end before the next pipeline starts,
        so a user who submitted first does not wait behind a large batch.

        If `workflow_types` is omitted, it defaults to `task_types` (legacy
        per-worker scope, used by tests that exercise a single worker).

        Within the active pipeline, `task_types` is still treated as a
        priority-ordered list so later stages preempt earlier ones (e.g.
        mesh_texture before a hypothetical second mesh_generate).
        """
        scope = workflow_types if workflow_types is not None else task_types
        with self._lock:
            # Find the active pipeline: among pipelines that have any
            # pending/running task in the workflow scope, pick the one whose
            # FIRST task in scope (MIN(id) including completed tasks) is
            # oldest.  This uses the original workflow entry time, not the
            # current pending task's id — otherwise a chained successor
            # task (which has a higher id) would make the pipeline look
            # "younger" than rival pipelines whose work was submitted later.
            placeholders = ",".join("?" * len(scope))
            active_row = self._conn.execute(
                f"SELECT pipeline_name FROM tasks "
                f"WHERE task_type IN ({placeholders}) "
                f"GROUP BY pipeline_name "
                f"HAVING SUM(CASE WHEN status IN ('pending','running') "
                f"                THEN 1 ELSE 0 END) > 0 "
                f"ORDER BY MIN(id) ASC LIMIT 1",
                scope,
            ).fetchone()
            if active_row is None:
                return None
            active_pipeline = active_row["pipeline_name"]

            for task_type in task_types:
                row = self._conn.execute(
                    "SELECT * FROM tasks WHERE status='pending' "
                    "AND task_type=? AND pipeline_name=? "
                    "ORDER BY id ASC LIMIT 1",
                    (task_type, active_pipeline),
                ).fetchone()
                if row is not None:
                    now = _now()
                    self._conn.execute(
                        "UPDATE tasks SET status='running', updated_at=? WHERE id=?",
                        (now, row["id"]),
                    )
                    self._conn.commit()
                    task = _row_to_task(row)
                    task.status = "running"
                    return task
            return None

    def mark_done(self, task_id: int) -> None:
        self._update_status(task_id, "done")

    def mark_failed(self, task_id: int, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status='failed', error=?, updated_at=? WHERE id=?",
                (error, _now(), task_id),
            )
            self._conn.commit()

    def get_tasks(
        self,
        pipeline_name: str | None = None,
        status: str | None = None,
        task_type: str | None = None,
    ) -> list[Task]:
        """Return tasks matching the given filters (all filters are optional)."""
        clauses, params = [], []
        if pipeline_name is not None:
            clauses.append("pipeline_name = ?")
            params.append(pipeline_name)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if task_type is not None:
            clauses.append("task_type = ?")
            params.append(task_type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM tasks {where} ORDER BY id ASC", params
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def has_pending_or_running(self, pipeline_name: str, task_type: str) -> bool:
        """True if a task of this type is already in-flight for this pipeline."""
        tasks = self.get_tasks(pipeline_name=pipeline_name, task_type=task_type)
        return any(t.status in ("pending", "running") for t in tasks)

    def cancel_pipeline_tasks(self, pipeline_name: str) -> int:
        """Mark all pending/running tasks for a pipeline as failed (cancelled)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE tasks SET status='failed', error='cancelled', updated_at=? "
                "WHERE pipeline_name=? AND status IN ('pending','running')",
                (_now(), pipeline_name),
            )
            self._conn.commit()
            return cur.rowcount

    def retry_failed_tasks(self, pipeline_name: str) -> int:
        """
        Reset all non-cancelled failed tasks for a pipeline back to pending
        so workers will re-attempt them.  Returns the number of tasks reset.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE tasks SET status='pending', error=NULL, updated_at=? "
                "WHERE pipeline_name=? AND status='failed' AND (error IS NULL OR error != 'cancelled')",
                (_now(), pipeline_name),
            )
            self._conn.commit()
            return cur.rowcount

    def pipelines_with_failures(self) -> list[str]:
        """Return distinct pipeline names that have non-cancelled failed tasks."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT pipeline_name FROM tasks "
                "WHERE status='failed' AND (error IS NULL OR error != 'cancelled') "
                "ORDER BY pipeline_name"
            ).fetchall()
        return [r["pipeline_name"] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_status(self, task_id: int, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE id=?",
                (status, _now(), task_id),
            )
            self._conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        pipeline_name=row["pipeline_name"],
        task_type=row["task_type"],
        payload=json.loads(row["payload"]),
        status=row["status"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
