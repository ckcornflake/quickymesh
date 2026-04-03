"""
Logging configuration for quickymesh.

Call configure_logging() once at process startup (in main.py / api_server.py).

Output:
  - stderr     — human-readable, WARNING and above only (keeps the terminal clean)
  - logs/quickymesh.log — structured JSON, INFO and above, rotated daily,
                          7 days retained

Set LOG_LEVEL=DEBUG in the environment to get full verbose output on both
handlers (useful when diagnosing a specific problem).

Log format (file):
  {"ts": "...", "level": "INFO", "logger": "src.workers.trellis",
   "msg": "mesh_generate task completed", "pipeline": "my_spaceship",
   "task_id": 42, "duration_ms": 18432}

Usage:
    from src.logging_config import configure_logging
    configure_logging()          # call once at startup
    # then in any module:
    import logging
    log = logging.getLogger(__name__)
    log.info("something happened", extra={"pipeline": "my_spaceship"})
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


def configure_logging(
    log_dir: Path | str | None = None,
    log_filename: str = "quickymesh.log",
) -> None:
    """
    Configure root logger with:
      - A stderr handler (WARNING+, plain text — keeps terminal uncluttered)
      - A rotating file handler (INFO+, JSON — persistent debug record)

    Parameters
    ----------
    log_dir:      Directory to write the log file.  Defaults to logs/ in the
                  repo root.  Created if it does not exist.
    log_filename: Name of the log file.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Don't add duplicate handlers if called more than once (e.g. in tests)
    if root.handlers:
        return

    # ── stderr handler ────────────────────────────────────────────────────
    # Plain text, WARNING+ by default so the terminal isn't flooded.
    # Promoting to the configured level when LOG_LEVEL=DEBUG.
    stderr_level = level if level >= logging.WARNING else logging.WARNING
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(stderr_level)
    stderr_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(stderr_handler)

    # ── file handler ─────────────────────────────────────────────────────
    if log_dir is None:
        # Repo root is two levels up from this file (src/logging_config.py)
        log_dir = Path(__file__).parent.parent / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_filename
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path,
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised — file: %s, level: %s", log_path, level_name
    )


class _JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Standard fields: ts, level, logger, msg
    Extra fields: anything passed via the `extra` kwarg to log calls,
                  e.g. log.info("done", extra={"pipeline": "x", "task_id": 1})
    """

    # Fields that live on every LogRecord but are not useful in the JSON output
    _SKIP = frozenset({
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs",
        "msg", "name", "pathname", "process", "processName", "relativeCreated",
        "stack_info", "thread", "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        record.message = record.getMessage()
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)

        out: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }

        # Include any extra fields the caller attached
        for key, val in record.__dict__.items():
            if key not in self._SKIP:
                out[key] = val

        if record.exc_text:
            out["exc"] = record.exc_text

        return json.dumps(out, default=str)
