"""
VRAM Arbiter — ensures only one GPU-heavy task runs at a time.

Trellis (mesh generate + texture) and Blender (HDRI screenshot renders)
are all VRAM-hungry.  Running them concurrently causes OOM errors.

Usage:
    arbiter = VRAMArbiter()

    # In a worker thread:
    with arbiter.acquire(timeout=600):
        run_heavy_gpu_task()

The arbiter is a thin wrapper around a threading.Lock, exposed as a context
manager so it works the same in both real and test code.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class VRAMArbiter:
    """
    Mutual-exclusion lock for GPU-bound tasks.

    Only one worker holds the lock at a time.  Others block (up to `timeout`
    seconds) before raising TimeoutError.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self, timeout: float = 0):
        """
        Context manager that acquires the VRAM lock.

        Parameters
        ----------
        timeout:
            Seconds to wait for the lock (0 = block indefinitely).

        Raises
        ------
        TimeoutError if the lock cannot be acquired within `timeout` seconds.
        """
        wait = None if timeout <= 0 else timeout
        acquired = self._lock.acquire(timeout=wait if wait is not None else -1)
        if not acquired:
            raise TimeoutError(
                f"VRAMArbiter: could not acquire GPU lock within {timeout}s. "
                "Another GPU-heavy task is still running."
            )
        try:
            yield
        finally:
            self._lock.release()

    @property
    def locked(self) -> bool:
        """True if the lock is currently held."""
        held = self._lock.acquire(blocking=False)
        if held:
            self._lock.release()
        return not held
