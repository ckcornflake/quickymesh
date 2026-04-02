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
  concept_art_generate   Generate concept art images (may be a subset by index).
  concept_art_modify     Modify one image via the Gemini edit API.
  mesh_generate          Run Trellis mesh generation for one approved concept art.
  mesh_texture           Run Trellis texturing for one generated mesh.
  screenshot             Take Blender screenshots + build review sheet + HTML preview.
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

    def claim_next(self, task_types: list[str]) -> Task | None:
        """
        Atomically claim the oldest pending task matching any of `task_types`.
        Returns None if no matching task is available.
        Sets status → 'running'.
        """
        placeholders = ",".join("?" * len(task_types))
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM tasks WHERE status='pending' AND task_type IN ({placeholders}) "
                f"ORDER BY id ASC LIMIT 1",
                task_types,
            ).fetchone()
            if row is None:
                return None
            now = _now()
            self._conn.execute(
                "UPDATE tasks SET status='running', updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            self._conn.commit()
            task = _row_to_task(row)
            task.status = "running"
            return task

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
