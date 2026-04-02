"""
CLI main loop for quickymesh.

The loop alternates between two modes:

Priority mode
    When a pipeline is waiting for user input (concept-art review or mesh
    review), the CLI immediately surfaces that pipeline's review prompt.

Idle mode
    When no pipeline needs attention, the CLI displays a menu:
      [n] Start new pipeline
      [s] Status (workers + pipelines)
      [w] Watch for approvals (live feed + auto-prompt)
      [q] Quit

Watch mode
    Prints a timestamped status line whenever pipeline state changes.
    Automatically surfaces approval prompts the moment they arrive.
    Type 'q' + Enter to return to the idle menu.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.pipeline_agent import PipelineAgent
    from src.config import Config
    from src.prompt_interface.base import PromptInterface

from src.concept_art_pipeline import run_concept_art_review
from src.mesh_pipeline import run_mesh_review
from src.state import PipelineState, PipelineStatus

log = logging.getLogger(__name__)

_IDLE_MENU = """\
--- quickymesh ---
[n] Start a new pipeline
[s] Status (workers + pipelines)
[w] Watch for approvals
[q] Quit
"""

_WATCH_EXIT_CMD = "q"
_WATCH_TICK = 0.1        # seconds between keypress/approval checks
_WATCH_STATUS_INTERVAL = 3.0  # seconds between status-diff prints


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_cli(
    agent: "PipelineAgent",
    ui: "PromptInterface",
    cfg: "Config",
    *,
    concept_worker=None,
    trellis_worker=None,
    screenshot_worker=None,
) -> None:
    """
    Main CLI loop.  Runs until the user quits.

    Parameters
    ----------
    agent:              The PipelineAgent managing workers and state.
    ui:                 PromptInterface for user interaction.
    cfg:                Application configuration.
    concept_worker:     Passed through to run_concept_art_review (real or mock).
    trellis_worker:     Passed through to run_mesh_review.
    screenshot_worker:  Not currently used in reviews but available.
    """
    agent.start_workers()
    ui.inform("quickymesh started.  Workers are running in the background.")

    try:
        _main_loop(agent, ui, cfg, concept_worker, trellis_worker)
    finally:
        agent.stop_workers()
        ui.inform("Workers stopped.  Goodbye.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _main_loop(agent, ui, cfg, concept_worker, trellis_worker) -> None:
    while True:
        priority = agent.highest_priority_pipeline()
        if priority is not None:
            _handle_priority_pipeline(priority, agent, ui, cfg, concept_worker, trellis_worker)
        else:
            action = _idle_menu(agent, ui, cfg, concept_worker, trellis_worker)
            if action == "quit":
                break


# ---------------------------------------------------------------------------
# Idle menu
# ---------------------------------------------------------------------------


def _idle_menu(agent, ui, cfg, concept_worker, trellis_worker) -> str:
    """Display idle menu. Returns 'new' | 'status' | 'watch' | 'quit'."""
    choice = ui.ask(_IDLE_MENU, options=["n", "s", "w", "q"])
    if choice == "q":
        return "quit"
    if choice == "n":
        _start_new_pipeline(agent, ui)
        return "new"
    if choice == "s":
        _show_status(agent, ui)
        return "status"
    if choice == "w":
        _watch_mode(agent, ui, cfg, concept_worker, trellis_worker)
        return "watch"
    return "status"


def _start_new_pipeline(agent, ui) -> None:
    name = ui.ask("Pipeline name (no spaces):").strip()
    if not name:
        ui.inform("Cancelled.")
        return
    description = ui.ask("Describe the 3-D object to generate:").strip()
    if not description:
        ui.inform("Cancelled.")
        return
    polys_str = ui.ask(f"Target polygon count (Enter for default {agent._cfg.num_polys}):").strip()
    num_polys = int(polys_str) if polys_str.isdigit() else None
    agent.start_pipeline(name, description, num_polys)
    ui.inform(f"Pipeline '{name}' started.  Concept art generation queued.")


# ---------------------------------------------------------------------------
# Status snapshot (no approval prompting)
# ---------------------------------------------------------------------------


def _show_status(agent, ui) -> None:
    """Print a one-shot snapshot of worker threads and all pipeline statuses."""
    lines = ["=== Worker threads ==="]
    for t in agent._threads:
        alive = "running" if t.is_alive() else "stopped"
        lines.append(f"  {t.__class__.__name__}: {alive}")

    lines.append("")
    lines.append("=== Pipelines ===")
    names = agent.list_pipeline_names()
    if not names:
        lines.append("  (none)")
    else:
        for name in names:
            state = agent.get_pipeline_state(name)
            status = state.status.value if state else "unknown"
            tasks = agent._broker.get_tasks(pipeline_name=name)
            in_flight = [t for t in tasks if t.status in ("pending", "running")]
            task_info = f"  [{len(in_flight)} task(s) queued]" if in_flight else ""
            lines.append(f"  {name}: {status}{task_info}")

    ui.inform("\n".join(lines))


# ---------------------------------------------------------------------------
# Watch mode — live feed + auto-surface approvals
# ---------------------------------------------------------------------------


def _watch_mode(
    agent,
    ui,
    cfg,
    concept_worker,
    trellis_worker,
    *,
    _tick: float = _WATCH_TICK,
    _status_interval: float = _WATCH_STATUS_INTERVAL,
) -> None:
    """
    Live watch loop.

    - Prints a timestamped line whenever a pipeline's status or task list
      changes (at most every `_status_interval` seconds).
    - Automatically surfaces approval prompts the moment they arrive.
    - Flushes buffered stdin before every approval prompt so stray keypresses
      from watch navigation don't bleed into review answers.
    - Exits when the user types 'q' + Enter.

    `_tick` and `_status_interval` are exposed for testing (inject small
    values to make tests fast without real sleeping).
    """
    ui.inform(
        "Watch mode active.  Updates will appear below as pipelines progress.\n"
        f"Type '{_WATCH_EXIT_CMD}' + Enter at any time to return to the menu.\n"
    )

    last_states: dict = {}
    input_buf: list[str] = []
    last_status_print = 0.0

    # Print current state immediately on entry
    _print_watch_diffs(agent, ui, last_states)
    last_status_print = time.time()

    while True:
        time.sleep(_tick)

        # ── Check for keypress (non-blocking, cross-platform) ─────────────
        ch = _try_read_char()
        if ch is not None:
            if ch in ('\r', '\n'):
                line = ''.join(input_buf).strip().lower()
                input_buf.clear()
                if line == _WATCH_EXIT_CMD:
                    ui.inform("Returning to menu.")
                    return
            else:
                input_buf.append(ch)

        # ── Check for pending approvals ────────────────────────────────────
        waiting = agent.pipelines_needing_attention()
        for name in waiting:
            ts = time.strftime("%H:%M:%S")
            ui.inform(f"\n[{ts}] !!! APPROVAL NEEDED: {name} !!!")
            _flush_stdin()
            input_buf.clear()
            _handle_priority_pipeline(name, agent, ui, cfg, concept_worker, trellis_worker)
            last_states.pop(name, None)  # force reprint of new status after review
            _print_watch_diffs(agent, ui, last_states)
            last_status_print = time.time()

        # ── Periodic status diff print ─────────────────────────────────────
        now = time.time()
        if now - last_status_print >= _status_interval:
            last_status_print = now
            _print_watch_diffs(agent, ui, last_states)


def _print_watch_diffs(agent, ui, last_states: dict) -> None:
    """
    For each pipeline whose (status, in-flight tasks) has changed since the
    last call, print a timestamped status line.
    """
    for name in agent.list_pipeline_names():
        state = agent.get_pipeline_state(name)
        if state is None:
            continue
        tasks = agent._broker.get_tasks(pipeline_name=name)
        in_flight = [t for t in tasks if t.status in ("pending", "running")]
        sig = (state.status.value, tuple(sorted(t.task_type for t in in_flight)))
        if last_states.get(name) != sig:
            task_detail = (
                f" [{', '.join(t.task_type for t in in_flight)}]" if in_flight else ""
            )
            ts = time.strftime("%H:%M:%S")
            ui.inform(f"[{ts}] {name}: {state.status.value}{task_detail}")
            last_states[name] = sig


# ---------------------------------------------------------------------------
# Cross-platform stdin helpers
# ---------------------------------------------------------------------------


def _try_read_char() -> str | None:
    """
    Non-blocking read of one character from stdin.

    Windows: uses msvcrt.kbhit() + msvcrt.getwche() (character-by-character).
    POSIX:   uses select.select with timeout=0; readline returns after Enter
             because stdin is line-buffered in a terminal.  The user therefore
             types 'q' + Enter, which is detected on the next tick.
    """
    if os.name == "nt":
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwche()
        return None
    else:
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def _flush_stdin() -> None:
    """
    Discard any buffered stdin input so stray keypresses don't contaminate
    the next review prompt.
    """
    if os.name == "nt":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    else:
        import termios
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:
            pass  # not a tty (e.g. piped input in tests) — safe to ignore


# ---------------------------------------------------------------------------
# Review dispatcher
# ---------------------------------------------------------------------------


def _handle_priority_pipeline(name, agent, ui, cfg, concept_worker, trellis_worker) -> None:
    state = agent.get_pipeline_state(name)
    if state is None:
        return

    pipeline_dir = cfg.uncompleted_pipelines_dir / name
    state_path = pipeline_dir / "state.json"

    if state.status == PipelineStatus.CONCEPT_ART_REVIEW:
        result = run_concept_art_review(state, concept_worker, pipeline_dir, ui, cfg)
        state.save(state_path)
        if result == "approved":
            agent.enqueue_mesh_generation(name)
            ui.inform(f"[{name}] Concept art approved.  Mesh generation queued.")
        elif result == "cancelled":
            agent.cancel_pipeline(name)
            ui.inform(f"[{name}] Pipeline cancelled.")

    elif state.status == PipelineStatus.MESH_REVIEW:
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        state.save(state_path)
        if result == "approved":
            ui.inform(f"[{name}] Mesh approved.")
        elif result == "cancelled":
            agent.cancel_pipeline(name)
            ui.inform(f"[{name}] Pipeline cancelled.")
