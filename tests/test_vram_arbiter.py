"""
Tests for src/vram_arbiter.py — GPU-lock mutual exclusion.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.vram_arbiter import VRAMArbiter


# ---------------------------------------------------------------------------
# Basic acquire/release
# ---------------------------------------------------------------------------


class TestAcquire:
    def test_context_manager_enters_and_exits(self):
        arb = VRAMArbiter()
        with arb.acquire():
            pass  # should not raise

    def test_lock_is_held_inside_context(self):
        arb = VRAMArbiter()
        with arb.acquire():
            assert arb.locked is True

    def test_lock_is_released_after_context(self):
        arb = VRAMArbiter()
        with arb.acquire():
            pass
        assert arb.locked is False

    def test_lock_released_on_exception(self):
        arb = VRAMArbiter()
        try:
            with arb.acquire():
                raise ValueError("boom")
        except ValueError:
            pass
        assert arb.locked is False

    def test_can_re_acquire_after_release(self):
        arb = VRAMArbiter()
        with arb.acquire():
            pass
        with arb.acquire():  # should not raise or deadlock
            pass


# ---------------------------------------------------------------------------
# Timeout behaviour
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_raises_when_lock_held(self):
        arb = VRAMArbiter()
        acquired = threading.Event()
        hold = threading.Event()

        def holder():
            with arb.acquire():
                acquired.set()
                hold.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        acquired.wait(timeout=2)

        try:
            with pytest.raises(TimeoutError):
                with arb.acquire(timeout=0.05):
                    pass
        finally:
            hold.set()
            t.join()

    def test_timeout_error_message_mentions_timeout(self):
        arb = VRAMArbiter()
        acquired = threading.Event()
        hold = threading.Event()

        def holder():
            with arb.acquire():
                acquired.set()
                hold.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        acquired.wait(timeout=2)

        try:
            with pytest.raises(TimeoutError, match="0.05"):
                with arb.acquire(timeout=0.05):
                    pass
        finally:
            hold.set()
            t.join()

    def test_zero_timeout_blocks_indefinitely(self):
        """
        With timeout=0, a second thread eventually gets the lock after
        the first releases it (not a hard guarantee on exact timing,
        but the acquire must NOT raise TimeoutError).
        """
        arb = VRAMArbiter()
        results = []

        def first():
            with arb.acquire(timeout=0):
                time.sleep(0.05)
                results.append("first")

        def second():
            time.sleep(0.01)  # start after first
            with arb.acquire(timeout=0):
                results.append("second")

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)

        assert results == ["first", "second"]


# ---------------------------------------------------------------------------
# locked property
# ---------------------------------------------------------------------------


class TestLockedProperty:
    def test_false_when_not_held(self):
        arb = VRAMArbiter()
        assert arb.locked is False

    def test_true_when_held(self):
        arb = VRAMArbiter()
        with arb.acquire():
            assert arb.locked is True

    def test_false_after_release(self):
        arb = VRAMArbiter()
        with arb.acquire():
            pass
        assert arb.locked is False

    def test_checking_locked_does_not_deadlock_on_second_check(self):
        arb = VRAMArbiter()
        _ = arb.locked
        _ = arb.locked  # must not deadlock


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_only_one_thread_in_critical_section_at_a_time(self):
        arb = VRAMArbiter()
        concurrent_count = []
        active = [0]
        lock = threading.Lock()
        errors = []

        def worker():
            with arb.acquire(timeout=5):
                with lock:
                    active[0] += 1
                    if active[0] > 1:
                        errors.append("overlap")
                time.sleep(0.005)
                with lock:
                    active[0] -= 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent access detected: {errors}"


# ---------------------------------------------------------------------------
# holder_changed
# ---------------------------------------------------------------------------


class TestHolderChanged:
    def test_first_call_returns_true(self):
        arb = VRAMArbiter()
        obj = object()
        assert arb.holder_changed(obj) is True

    def test_same_holder_twice_returns_false(self):
        arb = VRAMArbiter()
        obj = object()
        arb.holder_changed(obj)
        assert arb.holder_changed(obj) is False

    def test_different_holder_returns_true(self):
        arb = VRAMArbiter()
        a, b = object(), object()
        arb.holder_changed(a)
        assert arb.holder_changed(b) is True

    def test_alternating_holders_always_returns_true(self):
        arb = VRAMArbiter()
        a, b = object(), object()
        arb.holder_changed(a)
        assert arb.holder_changed(b) is True
        assert arb.holder_changed(a) is True

    def test_records_new_holder_after_change(self):
        arb = VRAMArbiter()
        a, b = object(), object()
        arb.holder_changed(a)
        arb.holder_changed(b)
        assert arb.holder_changed(b) is False
