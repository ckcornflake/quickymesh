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
from src.mesh_pipeline import run_mesh_review, run_mesh_export
from src.state import PipelineState, PipelineStatus

log = logging.getLogger(__name__)

_IDLE_MENU = """\
--- quickymesh ---
[n] Start a new pipeline
[e] Edit a pipeline
[p] Pause / Resume / Cancel a pipeline
[s] Status (workers + pipelines)
[w] Watch for approvals
[r] Retry failed tasks
[q] Quit"""

# Numeric shortcut → option letter (matches _IDLE_MENU order)
_IDLE_OPT_MAP = {"1": "n", "2": "e", "3": "p", "4": "s", "5": "w", "6": "r", "7": "q"}

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
    restyle_worker=None,
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
    restyle_worker:     Optional ControlNetRestyleWorker for concept art restyling.
    """
    agent.start_workers()
    ui.inform("quickymesh started.  Workers are running in the background.")

    try:
        _main_loop(agent, ui, cfg, concept_worker, trellis_worker, restyle_worker)
    finally:
        agent.stop_workers()
        ui.inform("Workers stopped.  Goodbye.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _main_loop(agent, ui, cfg, concept_worker, trellis_worker, restyle_worker=None) -> None:
    while True:
        priority = agent.highest_priority_pipeline()
        if priority is not None:
            quit_requested = _handle_priority_pipeline(priority, agent, ui, cfg, concept_worker, trellis_worker, restyle_worker)
            if quit_requested:
                break
        else:
            action = _idle_menu(agent, ui, cfg, concept_worker, trellis_worker, restyle_worker)
            if action == "quit":
                break


# ---------------------------------------------------------------------------
# Idle menu
# ---------------------------------------------------------------------------


def _idle_menu(agent, ui, cfg, concept_worker, trellis_worker, restyle_worker=None) -> str:
    """Display idle menu, with a live notification when approvals are pending.

    Pressing Enter (empty input) immediately surfaces any waiting approval.
    Numeric shortcuts 1–7 map to the same options as the letter keys.
    """
    waiting = agent.pipelines_needing_attention()
    if waiting:
        names = ", ".join(f"'{n}'" for n in waiting)
        noun = "Pipeline" if len(waiting) == 1 else "Pipelines"
        verb = "is" if len(waiting) == 1 else "are"
        approval_note = f"\n[!] {noun} {names} {verb} waiting for approval — press Enter to review.\n"
    else:
        approval_note = ""

    choice = ui.ask(_IDLE_MENU + approval_note).strip().lower()
    choice = _IDLE_OPT_MAP.get(choice, choice)   # "1" → "n", "2" → "e", …

    if not choice:          # bare Enter — fall through to priority check
        return "check"
    if choice == "q":
        return "quit"
    if choice == "n":
        _start_new_pipeline(agent, ui)
        return "new"
    if choice == "e":
        _edit_pipeline(agent, ui, cfg)
        return "edit"
    if choice == "p":
        _manage_pipeline(agent, ui, cfg)
        return "manage"
    if choice == "s":
        _show_status(agent, ui)
        return "status"
    if choice == "w":
        _watch_mode(agent, ui, cfg, concept_worker, trellis_worker, restyle_worker)
        return "watch"
    if choice == "r":
        _retry_failed(agent, ui)
        return "retry"
    return "check"  # unrecognised input → loop back, same as bare Enter


def _manage_pipeline(agent, ui, cfg) -> None:
    """Pause, resume, or cancel a pipeline from the idle menu."""
    from src.state import PipelineStatus

    names = agent.list_pipeline_names()
    if not names:
        ui.inform("No pipelines.")
        return

    lines = ["Pipelines:"]
    for i, name in enumerate(names, 1):
        state = agent.get_pipeline_state(name)
        status = state.status.value if state else "unknown"
        lines.append(f"  {i}. {name}  [{status}]")
    ui.inform("\n".join(lines))
    choice = ui.ask("Enter pipeline number (Enter to cancel):").strip()
    if not choice or not choice.isdigit() or not (1 <= int(choice) <= len(names)):
        ui.inform("Cancelled.")
        return

    name = names[int(choice) - 1]
    state = agent.get_pipeline_state(name)
    is_paused = state and state.status == PipelineStatus.PAUSED

    if is_paused:
        action = ui.ask(f"'{name}' is paused. [r]esume or [c]ancel?", options=["r", "c"]).strip().lower()
        if action == "r":
            agent.resume_pipeline(name)
            ui.inform(f"Pipeline '{name}' resumed.")
        elif action == "c":
            agent.cancel_pipeline(name)
            _mark_cancelled(agent, cfg, name)
            ui.inform(f"Pipeline '{name}' cancelled.")
    else:
        action = ui.ask(
            f"'{name}' is {state.status.value if state else 'unknown'}. [p]ause or [c]ancel?",
            options=["p", "c"],
        ).strip().lower()
        if action == "p":
            agent.pause_pipeline(name, cfg)
            ui.inform(f"Pipeline '{name}' paused.")
        elif action == "c":
            agent.cancel_pipeline(name)
            _mark_cancelled(agent, cfg, name)
            ui.inform(f"Pipeline '{name}' cancelled.")


def _mark_cancelled(agent, cfg, name: str) -> None:
    """Set PipelineStatus.CANCELLED in the state file."""
    from src.state import PipelineState, PipelineStatus
    state_path = cfg.uncompleted_pipelines_dir / name / "state.json"
    if state_path.exists():
        state = PipelineState.load(state_path)
        state.status = PipelineStatus.CANCELLED
        state.save(state_path)


def _retry_failed(agent, ui) -> None:
    """Reset failed tasks for a pipeline back to pending so workers retry them."""
    pipelines = agent._broker.pipelines_with_failures()
    if not pipelines:
        ui.inform("No pipelines have failed tasks.")
        return
    lines = ["Pipelines with failed tasks:"]
    for i, name in enumerate(pipelines, 1):
        lines.append(f"  {i}. {name}")
    lines.append("  a. All pipelines")
    ui.inform("\n".join(lines))
    choice = ui.ask("Enter pipeline number (or 'a' for all, Enter to cancel):").strip().lower()
    if not choice:
        return
    if choice == "a":
        total = sum(agent._broker.retry_failed_tasks(n) for n in pipelines)
        ui.inform(f"Reset {total} failed task(s) across all pipelines.")
    elif choice.isdigit() and 1 <= int(choice) <= len(pipelines):
        name = pipelines[int(choice) - 1]
        count = agent._broker.retry_failed_tasks(name)
        ui.inform(f"Reset {count} failed task(s) for '{name}'.")
    else:
        ui.inform("Cancelled.")


def _edit_pipeline(agent, ui, cfg) -> None:
    """
    Interactively edit a pipeline's settings (description, poly count, symmetry).
    Only available while the pipeline is still in the concept-art or pre-mesh phase.
    """
    from src.state import PipelineStatus, SymmetryAxis

    names = agent.list_pipeline_names()
    if not names:
        ui.inform("No pipelines to edit.")
        return

    # List editable pipelines (concept art phase only — can't change after mesh starts)
    editable_statuses = {
        PipelineStatus.INITIALIZING,
        PipelineStatus.CONCEPT_ART_GENERATING,
        PipelineStatus.CONCEPT_ART_REVIEW,
    }
    editable = [
        n for n in names
        if (s := agent.get_pipeline_state(n)) and s.status in editable_statuses
    ]
    if not editable:
        ui.inform("No editable pipelines (editing is only available before mesh generation).")
        return

    lines = ["Editable pipelines:"]
    for i, name in enumerate(editable, 1):
        state = agent.get_pipeline_state(name)
        lines.append(f"  {i}. {name}  [{state.status.value}]")
    ui.inform("\n".join(lines))
    choice = ui.ask("Enter pipeline number (Enter to cancel):").strip()
    if not choice or not choice.isdigit() or not (1 <= int(choice) <= len(editable)):
        ui.inform("Cancelled.")
        return

    name = editable[int(choice) - 1]
    state_path = cfg.uncompleted_pipelines_dir / name / "state.json"
    state = agent.get_pipeline_state(name)

    # Description
    suffix = cfg.background_suffix
    ui.inform(f'Current description: "{state.description}"')
    ui.inform(f'(Suffix appended automatically: "{suffix}")')
    new_desc = ui.ask("New description (Enter to keep current):").strip()
    if new_desc:
        state.description = new_desc

    # Polygon count
    ui.inform(f"Current polygon target: {state.num_polys}")
    poly_raw = ui.ask("New polygon count (Enter to keep current):").strip()
    if poly_raw.isdigit() and int(poly_raw) > 0:
        state.num_polys = int(poly_raw)

    # Symmetrize
    axis_options = [a.value for a in SymmetryAxis]
    sym_display = f"{'on' if state.symmetrize else 'off'}, axis={state.symmetry_axis.value}"
    ui.inform(f"Current symmetry: {sym_display}")
    sym_raw = ui.ask("Enable symmetrize? (y/n, Enter to keep current):").strip().lower()
    if sym_raw == "y":
        state.symmetrize = True
        axis_raw = ui.ask(
            f"Symmetry axis — options: {', '.join(axis_options)}. Enter to keep '{state.symmetry_axis.value}':"
        ).strip()
        if axis_raw in axis_options:
            state.symmetry_axis = SymmetryAxis(axis_raw)
    elif sym_raw == "n":
        state.symmetrize = False

    state.save(state_path)
    ui.inform(f"Pipeline '{name}' updated.")


def _load_prefs(cfg) -> dict:
    """Load user preferences from {output_root}/.preferences.json."""
    import json
    path = cfg.output_root / ".preferences.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_prefs(cfg, prefs: dict) -> None:
    """Persist user preferences to {output_root}/.preferences.json."""
    import json
    path = cfg.output_root / ".preferences.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


def _start_new_pipeline(agent, ui) -> None:
    name = ui.ask("Pipeline name (no spaces):").strip()
    if not name:
        ui.inform("Cancelled.")
        return

    # ── Concept art backend ────────────────────────────────────────────────
    prefs = _load_prefs(agent._cfg)
    default_backend = prefs.get("concept_art_backend", "gemini")
    ui.inform(
        "Choose concept art generator:\n"
        "  1. Gemini Flash  — requires API key, small cost per image, "
        "very accurate results, can use an existing image as a base\n"
        "  2. FLUX.1 [dev]  — runs locally via ComfyUI, ~25 GB models, "
        "~16 GB VRAM, less accurate\n"
        f"Default: {default_backend}"
    )
    choice = ui.ask("Enter 1 or 2 (Enter to keep default):").strip()
    if choice == "1":
        backend = "gemini"
    elif choice == "2":
        backend = "flux"
    else:
        backend = default_backend
    _save_prefs(agent._cfg, {**prefs, "concept_art_backend": backend})

    # Check/prompt for Gemini API key before proceeding
    if backend == "gemini":
        try:
            agent._cfg.gemini_api_key  # raises EnvironmentError if missing
        except EnvironmentError:
            key = ui.ask(
                "No Gemini API key found. Enter your GEMINI_API_KEY (or leave blank to cancel):"
            ).strip()
            if not key:
                ui.inform("Gemini API key required. Cancelled.")
                return
            os.environ["GEMINI_API_KEY"] = key

    # ── Optional input image (Gemini only — FLUX generates from prompt) ────
    input_image_path: str | None = None
    if backend == "gemini":
        while True:
            img_raw = ui.ask(
                "Base concept art on an existing image? "
                "Enter a path (absolute or relative to pipeline root), or leave blank:"
            ).strip()
            if not img_raw:
                break

            from pathlib import Path as _Path
            p = _Path(img_raw)
            candidate = p if p.is_absolute() else agent._cfg.output_root / img_raw
            if candidate.exists():
                input_image_path = str(candidate)
                ui.inform(f"Image found: {candidate}")
                break
            ui.inform(f"Image not found: {candidate}\nPlease try again, or press Enter to skip.")

    # ── Description ────────────────────────────────────────────────────────
    suffix = agent._cfg.background_suffix
    if input_image_path:
        description = ui.ask("How do you want to change/adapt this image?").strip()
    else:
        ui.inform(
            f"Note: the following suffix will be appended to your description:\n"
            f'  "{suffix}"\n'
            f"This is required so Trellis can correctly reconstruct the 3-D geometry."
        )
        description = ui.ask("Describe the 3-D object to generate:").strip()

    if not description:
        ui.inform("Cancelled.")
        return

    # ── Polygon count ──────────────────────────────────────────────────────
    polys_str = ui.ask(f"Target polygon count (Enter for default {agent._cfg.num_polys}):").strip()
    num_polys = int(polys_str) if polys_str.isdigit() else None

    # ── Symmetry ───────────────────────────────────────────────────────────
    _SYM_OPTIONS = ["x-", "x+", "y-", "y+", "z-", "z+"]
    sym_raw = ui.ask(
        "Symmetrize mesh after generation?\n"
        f"  Options: {', '.join(_SYM_OPTIONS)}\n"
        "  Enter an axis to enable (default: x-), or leave blank to skip:"
    ).strip().lower()
    if sym_raw in _SYM_OPTIONS:
        symmetrize, symmetry_axis = True, sym_raw
    elif not sym_raw:
        symmetrize, symmetry_axis = True, "x-"   # bare Enter → x-
    else:
        symmetrize, symmetry_axis = False, "x-"  # anything else → off

    agent.start_pipeline(
        name, description, num_polys,
        input_image_path=input_image_path,
        symmetrize=symmetrize,
        symmetry_axis=symmetry_axis,
        concept_art_backend=backend,
    )
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
            failed   = [t for t in tasks if t.status == "failed" and t.error != "cancelled"]
            task_info = f"  [{len(in_flight)} queued]" if in_flight else ""
            fail_info = f"  [{len(failed)} FAILED]" if failed else ""
            lines.append(f"  {name}: {status}{task_info}{fail_info}")
            for ft in failed:
                lines.append(f"    ! {ft.task_type}: {ft.error}")

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
    restyle_worker=None,
    *,
    _tick: float = _WATCH_TICK,
    _status_interval: float = _WATCH_STATUS_INTERVAL,
) -> None:
    """
    Live watch loop.

    - Prints a timestamped line whenever a pipeline's status or task list
      changes (at most every `_status_interval` seconds).
    - Automatically surfaces approval prompts the moment they arrive.
    - Exits immediately on a single 'q' or Escape keypress (no Enter needed).

    On Windows single-char input works natively via msvcrt.
    On POSIX stdin is put into cbreak mode for the duration of the watch loop
    so characters are delivered without waiting for Enter.  cbreak is restored
    before every review prompt (which needs normal line-buffered input) and
    re-entered after.

    `_tick` and `_status_interval` are exposed for testing.
    """
    ui.inform(
        "Watch mode active.  Updates will appear below as pipelines progress.\n"
        f"Press '{_WATCH_EXIT_CMD}' to return to the menu.\n"
    )

    last_states: dict = {}
    last_status_print = 0.0

    _enter_cbreak()
    try:
        # Print current state immediately on entry
        _print_watch_diffs(agent, ui, last_states)
        last_status_print = time.time()

        while True:
            time.sleep(_tick)

            # ── Pending approvals (checked before keypress so approvals ────
            # ── are never silently skipped if the user presses q) ──────────
            waiting = agent.pipelines_needing_attention()
            if waiting:
                _exit_cbreak()
                for name in waiting:
                    ts = time.strftime("%H:%M:%S")
                    ui.inform(f"\n[{ts}] !!! APPROVAL NEEDED: {name} !!!")
                    _flush_stdin()
                    _handle_priority_pipeline(name, agent, ui, cfg, concept_worker, trellis_worker, restyle_worker)
                ui.inform("\nApprovals complete.  Returning to menu.")
                return

            # ── Single-key exit check ──────────────────────────────────────
            ch = _try_read_char()
            if ch is not None and ch.lower() in (_WATCH_EXIT_CMD, '\x1b', '\x03'):
                ui.inform("\nReturning to menu.")
                return

            # ── Periodic status diff ───────────────────────────────────────
            now = time.time()
            if now - last_status_print >= _status_interval:
                last_status_print = now
                _print_watch_diffs(agent, ui, last_states)
    finally:
        _exit_cbreak()


def _print_watch_diffs(agent, ui, last_states: dict) -> None:
    """
    For each pipeline whose (status, in-flight tasks, failed tasks) has
    changed since the last call, print a timestamped status line.
    Failed tasks include their error message so you can see what went wrong.
    """
    for name in agent.list_pipeline_names():
        state = agent.get_pipeline_state(name)
        if state is None:
            continue
        tasks = agent._broker.get_tasks(pipeline_name=name)
        in_flight = [t for t in tasks if t.status in ("pending", "running")]
        failed    = [t for t in tasks if t.status == "failed" and t.error != "cancelled"]
        sig = (
            state.status.value,
            tuple(sorted(t.task_type for t in in_flight)),
            tuple(sorted(t.id for t in failed)),
        )
        if last_states.get(name) != sig:
            ts = time.strftime("%H:%M:%S")
            if in_flight:
                detail = f" [{', '.join(t.task_type for t in in_flight)}]"
            elif failed:
                detail = f" [FAILED: {', '.join(t.task_type for t in failed)}]"
            else:
                detail = ""
            ui.inform(f"[{ts}] {name}: {state.status.value}{detail}")
            for ft in failed:
                ui.inform(f"[{ts}]   ! {ft.task_type} error: {ft.error}")
            last_states[name] = sig


# ---------------------------------------------------------------------------
# Cross-platform stdin helpers
# ---------------------------------------------------------------------------

# Saved POSIX terminal settings while in cbreak mode (None = not in cbreak).
_saved_term_settings = None


def _enter_cbreak() -> None:
    """
    Switch stdin to cbreak mode on POSIX so single keypresses are delivered
    without waiting for Enter.  No-op on Windows (msvcrt already works that
    way) and no-op if stdin is not a tty (e.g. test/piped input).
    """
    global _saved_term_settings
    if os.name == "nt" or _saved_term_settings is not None:
        return
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        _saved_term_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        pass  # not a tty — safe to ignore


def _exit_cbreak() -> None:
    """Restore the terminal settings saved by _enter_cbreak."""
    global _saved_term_settings
    if os.name == "nt" or _saved_term_settings is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _saved_term_settings)
    except Exception:
        pass
    _saved_term_settings = None


def _try_read_char() -> str | None:
    """
    Non-blocking read of one character from stdin.

    In watch mode the terminal is in cbreak (POSIX) or msvcrt is used
    (Windows), so characters are available immediately on keypress with no
    Enter required.
    """
    if os.name == "nt":
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch()   # silent — no echo in the status feed
        return None
    else:
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def _flush_stdin() -> None:
    """
    Discard any buffered stdin input so stray keypresses from watch mode
    don't contaminate the next review prompt.
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
            pass  # not a tty — safe to ignore


# ---------------------------------------------------------------------------
# Review dispatcher
# ---------------------------------------------------------------------------


def _handle_priority_pipeline(name, agent, ui, cfg, concept_worker, trellis_worker, restyle_worker=None) -> bool:
    """
    Handle one priority pipeline review.  Returns True if the user asked to quit.
    """
    pipeline_dir = cfg.uncompleted_pipelines_dir / name
    if not pipeline_dir.exists():
        return False  # already moved to completed_pipelines

    state = agent.get_pipeline_state(name)
    if state is None:
        return False

    if state.status == PipelineStatus.CONCEPT_ART_REVIEW:
        # run_concept_art_review saves state internally on every mutation
        result = run_concept_art_review(state, concept_worker, pipeline_dir, ui, cfg, restyle_worker=restyle_worker)
        if result == "approved":
            agent.enqueue_mesh_generation(name)
            ui.inform(f"[{name}] Concept art approved.  Mesh generation queued.")
        elif result == "cancelled":
            agent.cancel_pipeline(name)
            ui.inform(f"[{name}] Pipeline cancelled.")
        elif result == "quit":
            return True

    elif state.status == PipelineStatus.MESH_REVIEW:
        # run_mesh_review saves state internally on every mutation — no extra save needed
        result = run_mesh_review(state, pipeline_dir, ui, cfg)
        if result == "approved":
            run_mesh_export(state, pipeline_dir, cfg)
            ui.inform(f"[{name}] Meshes exported to {cfg.final_assets_dir}.")
        elif result in ("all_rejected", "no_pending"):
            # "no_pending" can occur on restart when all meshes were already rejected
            # in a prior session — the pipeline status is stuck at MESH_REVIEW with no
            # reviewable meshes, causing an infinite spin.  Treat it the same as
            # all_rejected: update status and re-queue mesh generation.
            #
            # IMPORTANT: reload state from disk — run_mesh_review already persisted
            # user changes (e.g. symmetrize=False).  Saving the old in-memory state
            # object here would silently overwrite those changes.
            state_path = pipeline_dir / "state.json"
            state = PipelineState.load(state_path)
            from src.state import PipelineStatus as _PS
            state.status = _PS.MESH_GENERATING
            state.save(state_path)
            agent.enqueue_mesh_generation(name)
            ui.inform(f"[{name}] All meshes rejected — mesh generation re-queued.")
        elif result == "quit":
            return True
        elif result == "cancelled":
            agent.cancel_pipeline(name)
            ui.inform(f"[{name}] Pipeline cancelled.")

    return False
