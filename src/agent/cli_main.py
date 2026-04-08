"""
CLI main loop for quickymesh.

The loop alternates between two modes:

Priority mode
    When a pipeline is waiting for user input (concept-art review or 3D mesh
    approval), the CLI immediately surfaces that pipeline's prompt.

Idle mode
    When no pipeline needs attention, the CLI displays a menu:
      [n] Start a new 2D pipeline
      [3] Start a 3D pipeline from a local image
      [r] Return to a pipeline
      [e] Edit a 2D pipeline
      [m] Manage pipelines  (hide / restore / kill)
      [s] Status (workers + pipelines)
      [w] Watch for approvals
      [t] Retry failed tasks
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
from src.state import Pipeline3DStatus, PipelineState, PipelineStatus

log = logging.getLogger(__name__)

_IDLE_MENU = """\
--- quickymesh ---
[n] Start a new 2D pipeline
[3] Start a 3D pipeline from a local image
[m] Manage a pipeline  ( [re]submit / edit / hide / kill )
[w] Watch for status updates
[t] Retry failed tasks
[q] Quit"""

# Numeric shortcut → option letter (matches _IDLE_MENU order)
_IDLE_OPT_MAP = {
    "1": "n", "2": "3", "3": "m", "4": "w", "5": "t", "6": "q",
}

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
    trellis_worker:     Available for future review integrations.
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
        action = _idle_menu(agent, ui, cfg, concept_worker, trellis_worker, restyle_worker)
        if action == "quit":
            break


# ---------------------------------------------------------------------------
# Attention helpers  (2D + 3D)
# ---------------------------------------------------------------------------


def _any_priority_pipeline(agent) -> str | None:
    """
    Return the name of any pipeline waiting for user attention (2D or 3D),
    or None if nothing needs attention.  2D concept-art review takes priority.
    Hidden and session-dismissed pipelines are skipped.
    """
    # 2D: CONCEPT_ART_REVIEW
    for name in agent.list_pipeline_names():
        if agent.is_dismissed_from_priority(name):
            continue
        s = agent.get_pipeline_state(name)
        if s and not s.hidden and s.status == PipelineStatus.CONCEPT_ART_REVIEW:
            return name
    # 3D: AWAITING_APPROVAL
    for name in agent.list_3d_pipeline_names():
        if agent.is_dismissed_from_priority(name):
            continue
        s = agent.get_3d_pipeline_state(name)
        if s and not s.hidden and s.status == Pipeline3DStatus.AWAITING_APPROVAL:
            return name
    return None


def _all_needing_attention(agent) -> list[str]:
    """
    Return names of all pipelines (2D or 3D) waiting for user input,
    excluding hidden and session-dismissed ones.
    """
    names: list[str] = []
    for name in agent.list_pipeline_names():
        if agent.is_dismissed_from_priority(name):
            continue
        s = agent.get_pipeline_state(name)
        if s and not s.hidden and s.status == PipelineStatus.CONCEPT_ART_REVIEW:
            names.append(name)
    for name in agent.list_3d_pipeline_names():
        if agent.is_dismissed_from_priority(name):
            continue
        s = agent.get_3d_pipeline_state(name)
        if s and not s.hidden and s.status == Pipeline3DStatus.AWAITING_APPROVAL:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Idle menu
# ---------------------------------------------------------------------------


def _idle_menu(agent, ui, cfg, concept_worker, trellis_worker, restyle_worker=None) -> str:
    """
    Display idle menu, with a live notification when approvals are pending.
    Pressing Enter (empty input) immediately surfaces any waiting approval.
    """
    waiting = _all_needing_attention(agent)
    if waiting:
        names = ", ".join(f"'{n}'" for n in waiting)
        noun = "Pipeline" if len(waiting) == 1 else "Pipelines"
        verb = "is" if len(waiting) == 1 else "are"
        approval_note = f"\n[!] {noun} {names} {verb} waiting for review — press Enter to open.\n"
    else:
        approval_note = ""

    choice = ui.ask(_IDLE_MENU + approval_note).strip().lower()
    # Apply numeric shortcut only when the input is not already a recognised option.
    # "3" is both the menu letter for the 3D pipeline option and a digit, so it
    # must not be overwritten by the numeric map.
    _VALID = {"n", "3", "m", "w", "t", "q", ""}
    if choice not in _VALID:
        choice = _IDLE_OPT_MAP.get(choice, choice)

    if not choice:
        # Bare Enter — open the first needing-attention pipeline directly,
        # or fall back to the manage menu if nothing is pending.
        if waiting:
            result = _manage_pipeline(
                agent, ui, cfg, concept_worker, restyle_worker,
                preselect=waiting[0],
            )
            if result == "quit":
                return "quit"
            return "manage"
        return "check"
    if choice == "q":
        return "quit"
    if choice == "n":
        _start_new_pipeline(agent, ui)
        return "new"
    if choice == "3":
        _start_3d_pipeline_from_file(agent, ui, cfg)
        return "3d"
    if choice == "m":
        result = _manage_pipeline(agent, ui, cfg, concept_worker, restyle_worker)
        if result == "quit":
            return "quit"
        return "manage"
    if choice == "w":
        _watch_mode(agent, ui, cfg, concept_worker, trellis_worker, restyle_worker)
        return "watch"
    if choice == "t":
        _retry_failed(agent, ui)
        return "retry"
    return "check"  # unrecognised input → loop back


# ---------------------------------------------------------------------------
# Start new 2D pipeline
# ---------------------------------------------------------------------------


def _load_prefs(cfg) -> dict:
    """Load user preferences from {output_root}/.preferences.json."""
    import json
    path = cfg.output_root / ".preferences.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
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
# Start new 3D pipeline from a local image file
# ---------------------------------------------------------------------------


def _start_3d_pipeline_from_file(agent, ui, cfg) -> None:
    """
    Start a 3D pipeline by submitting a local image file.
    The pipeline name will be prefixed with 'u_' to distinguish it from
    pipelines derived from 2D concept art.
    """
    short_name = ui.ask(
        "Pipeline name (no spaces — will be prefixed with 'u_'):"
    ).strip()
    if not short_name:
        ui.inform("Cancelled.")
        return

    name = f"u_{short_name}"
    if agent.pipeline_name_exists(name):
        ui.inform(f"Pipeline '{name}' already exists. Choose a different name.")
        return

    # ── Image path ─────────────────────────────────────────────────────────
    while True:
        img_raw = ui.ask(
            "Path to the image file (absolute or relative to output root):"
        ).strip()
        if not img_raw:
            ui.inform("Cancelled.")
            return

        from pathlib import Path as _Path
        p = _Path(img_raw)
        candidate = p if p.is_absolute() else cfg.output_root / img_raw
        if candidate.exists():
            image_path = str(candidate)
            break
        ui.inform(f"File not found: {candidate}\nPlease try again, or leave blank to cancel.")

    # ── Polygon count ──────────────────────────────────────────────────────
    polys_str = ui.ask(
        f"Target polygon count (Enter for default {cfg.num_polys}):"
    ).strip()
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
        symmetrize, symmetry_axis = True, "x-"
    else:
        symmetrize, symmetry_axis = False, "x-"

    agent.start_3d_pipeline(
        name=name,
        input_image_path=image_path,
        num_polys=num_polys,
        symmetrize=symmetrize,
        symmetry_axis=symmetry_axis,
    )
    ui.inform(f"3D pipeline '{name}' started.  Mesh generation queued.")


# ---------------------------------------------------------------------------
# Return to an idle pipeline
# ---------------------------------------------------------------------------


def _return_to_pipeline(agent, ui, cfg, concept_worker, restyle_worker=None) -> bool:
    """
    List all idle (non-hidden) pipelines and let the user pick one to work with.
    Returns True if the user asked to quit.
    """
    # Collect idle candidates
    options: list[tuple[str, str]] = []  # (name, "2d" | "3d")
    for name in agent.list_pipeline_names():
        s = agent.get_pipeline_state(name)
        if s and not s.hidden and s.status == PipelineStatus.CONCEPT_ART_REVIEW:
            options.append((name, "2d"))
    for name in agent.list_3d_pipeline_names():
        s = agent.get_3d_pipeline_state(name)
        if s and not s.hidden and s.status in (
            Pipeline3DStatus.AWAITING_APPROVAL, Pipeline3DStatus.IDLE
        ):
            options.append((name, "3d"))

    if not options:
        ui.inform("No idle pipelines to return to.")
        return False

    lines = ["Idle pipelines:"]
    for i, (name, ptype) in enumerate(options, 1):
        if ptype == "2d":
            s = agent.get_pipeline_state(name)
            lines.append(f"  {i}. [2D] {name}  [{s.status.value}]")
        else:
            s = agent.get_3d_pipeline_state(name)
            lines.append(f"  {i}. [3D] {name}  [{s.status.value}]")
    ui.inform("\n".join(lines))

    choice = ui.ask("Enter number (Enter to cancel):").strip()
    if not choice or not choice.isdigit() or not (1 <= int(choice) <= len(options)):
        ui.inform("Cancelled.")
        return False

    name, ptype = options[int(choice) - 1]
    # User is explicitly re-engaging — clear any session dismissal so future
    # priority passes can re-surface the pipeline if it ends up needing
    # attention again.
    agent.undismiss_from_priority(name)
    if ptype == "2d":
        pipeline_dir = cfg.pipelines_dir / name
        state = agent.get_pipeline_state(name)
        # Force the review sheet to be re-shown on this re-entry. The flag
        # stays True between review sessions so that successive regenerate
        # actions inside one session don't repeatedly blast the sheet.
        state.concept_art_sheet_shown = False
        result = run_concept_art_review(
            state, concept_worker, pipeline_dir, ui, cfg,
            restyle_worker=restyle_worker,
        )
        if result == "approved":
            _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)
        elif result == "cancelled":
            agent.cancel_pipeline(name)
            ui.inform(f"[{name}] Pipeline cancelled.")
        elif result == "quit":
            return True
        # Re-dismiss after the user exits so the priority loop doesn't
        # immediately pull them back in.
        agent.dismiss_from_priority(name)
    else:
        quit_requested = _handle_3d_approval(name, agent, ui, cfg)
        if quit_requested:
            return True
        # Only dismiss if the pipeline is no longer awaiting approval
        # (i.e. the user approved or regenerated, not just returned to menu).
        state_after = agent.get_3d_pipeline_state(name)
        if state_after is None or state_after.status != Pipeline3DStatus.AWAITING_APPROVAL:
            agent.dismiss_from_priority(name)

    return False


# ---------------------------------------------------------------------------
# Unified pipeline management
# ---------------------------------------------------------------------------


def _manage_pipeline(
    agent, ui, cfg, concept_worker=None, restyle_worker=None, *, preselect: str | None = None
):
    """
    Unified pipeline manager.

    Lists all pipelines, lets the user pick one (or pre-selects one via
    `preselect`), then shows rich pipeline info and a context-sensitive
    action menu.  The review/resubmit action is offered directly without
    an extra [v] keystroke.

    Returns "quit" if the user asked to quit, else None.
    """
    from src.state import Pipeline3DState

    options = _list_all_pipelines(agent, cfg)
    if not options:
        ui.inform("No pipelines found.")
        return

    # ── Step 1: pick a pipeline ────────────────────────────────────────────
    if preselect is not None:
        # Find preselect in the options list; fall through to interactive pick
        # if not found (e.g. already hidden).
        for i, (n, pt) in enumerate(options):
            if n == preselect:
                name, ptype = n, pt
                break
        else:
            preselect = None  # not found, show picker

    if preselect is None:
        lines = ["Pipelines:"]
        for i, (n, pt) in enumerate(options, 1):
            s = agent.get_pipeline_state(n) if pt == "2d" else agent.get_3d_pipeline_state(n)
            status = s.status.value if s else "unknown"
            hidden_tag = "  [hidden]" if (s and s.hidden) else ""
            lines.append(f"  {i}. [{pt.upper()}] {n}  {status}{hidden_tag}")
        ui.inform("\n".join(lines))

        choice_raw = ui.ask("Enter number (Enter to cancel):").strip()
        if not choice_raw or not choice_raw.isdigit() or not (1 <= int(choice_raw) <= len(options)):
            ui.inform("Cancelled.")
            return
        name, ptype = options[int(choice_raw) - 1]

    state = (
        agent.get_pipeline_state(name) if ptype == "2d"
        else agent.get_3d_pipeline_state(name)
    )
    if state is None:
        ui.inform(f"Could not load state for '{name}'.")
        return

    # ── Step 2: rich pipeline info ─────────────────────────────────────────
    info_lines = [f"\nPipeline '{name}' [{ptype.upper()}]"]
    info_lines.append(f"  Status : {state.status.value}")

    if ptype == "2d":
        info_lines.append(f"  Description: {state.description}")
        if state.concept_arts:
            from src.state import ConceptArtStatus
            counts: dict[str, int] = {}
            for ca in state.concept_arts:
                counts[ca.status.value] = counts.get(ca.status.value, 0) + 1
            summary = ", ".join(f"{v} {k}" for k, v in counts.items())
            info_lines.append(f"  Concept arts: {summary}")
        # Spawned 3D pipelines
        spawned = [
            n for n in agent.list_3d_pipeline_names()
            if (s3 := agent.get_3d_pipeline_state(n))
            and s3.source_2d_pipeline == name
        ]
        if spawned:
            info_lines.append("  Spawned 3D pipelines:")
            for sp_name in spawned:
                sp = agent.get_3d_pipeline_state(sp_name)
                sp_status = sp.status.value if sp else "unknown"
                exports = f", {sp.export_version} export(s)" if sp else ""
                info_lines.append(f"    • {sp_name}  [{sp_status}{exports}]")
    else:
        if state.source_2d_pipeline is not None:
            info_lines.append(
                f"  Source: {state.source_2d_pipeline} "
                f"(concept art {state.source_concept_art_index + 1})"
            )
        mesh = state.textured_mesh_path or state.mesh_path or "(none)"
        info_lines.append(f"  Mesh  : {mesh}")
        info_lines.append(f"  Exports: {state.export_version}")

    ui.inform("\n".join(info_lines))

    # ── Step 3: context-sensitive action menu ──────────────────────────────
    actions: list[tuple[str, str]] = []

    can_review = (
        (ptype == "2d" and state.status == PipelineStatus.CONCEPT_ART_REVIEW)
        or (ptype == "3d" and state.status in (
            Pipeline3DStatus.AWAITING_APPROVAL, Pipeline3DStatus.IDLE
        ))
    )
    # 2D pipelines can always (re)submit approved concept arts
    can_submit = ptype == "2d"

    if can_review and ptype == "2d":
        actions.append(("[a]", "Review concept art / resubmit for 3D"))
    elif can_submit:
        actions.append(("[a]", "Resubmit approved concept art(s) for 3D"))
    if ptype == "3d" and state.status in (
        Pipeline3DStatus.AWAITING_APPROVAL, Pipeline3DStatus.IDLE
    ):
        actions.append(("[a]", "Review mesh / approve"))
    if ptype == "2d":
        actions.append(("[e]", "Edit settings (description, poly count, symmetry)"))
    if state.hidden:
        actions.append(("[r]", "Restore (make visible again)"))
    else:
        actions.append(("[h]", "Hide (keep folder, remove from active view)"))
    actions.append(("[k]", "Kill — permanently delete all content"))

    action_lines = ["\nActions:"]
    for letter, label in actions:
        action_lines.append(f"  {letter} {label}")
    ui.inform("\n".join(action_lines))

    action_choice = ui.ask("Enter action (Enter to cancel):").strip().lower()
    if not action_choice:
        return

    valid_letters = {ltr.strip("[]") for ltr, _ in actions}
    if action_choice not in valid_letters:
        ui.inform(f"Unknown action '{action_choice}'.")
        return

    # ── Dispatch ───────────────────────────────────────────────────────────
    if action_choice == "a":
        if ptype == "2d":
            pipeline_dir = cfg.pipelines_dir / name
            state.concept_art_sheet_shown = False
            result = run_concept_art_review(
                state, concept_worker, pipeline_dir, ui, cfg,
                restyle_worker=restyle_worker,
            )
            if result == "approved":
                _submit_approved_for_3d(state, agent, ui, cfg, pipeline_dir)
                agent.dismiss_from_priority(name)
            elif result == "cancelled":
                agent.cancel_pipeline(name)
                ui.inform(f"[{name}] Pipeline cancelled.")
                agent.dismiss_from_priority(name)
            elif result == "quit":
                return "quit"
            # "back" → leave undismissed so watch/attention note keeps showing it
        else:
            quit_requested = _handle_3d_approval(name, agent, ui, cfg)
            if quit_requested:
                return "quit"
            # Dismiss 3D on any non-quit exit (approve or regenerate both close
            # the loop cleanly; menu/back also returns False but the pipeline
            # status will have changed away from AWAITING_APPROVAL anyway).
            state_after = agent.get_3d_pipeline_state(name)
            if state_after and state_after.status not in (
                Pipeline3DStatus.AWAITING_APPROVAL,
            ):
                agent.dismiss_from_priority(name)

    elif action_choice == "e":
        _edit_pipeline_for(agent, ui, cfg, name)

    elif action_choice == "h":
        _set_hidden(cfg, name, ptype, hidden=True)
        ui.inform(f"Pipeline '{name}' hidden.")

    elif action_choice == "r":
        _set_hidden(cfg, name, ptype, hidden=False)
        ui.inform(f"Pipeline '{name}' restored.")

    elif action_choice == "k":
        ui.inform(
            f"\nWARNING: This will permanently delete pipeline '{name}' "
            "and ALL its generated content.\n"
            "This action CANNOT be undone."
        )
        confirm = ui.ask("Type 'confirm' to proceed, or Enter to cancel:").strip().lower()
        if confirm != "confirm":
            ui.inform("Cancelled.")
            return
        _kill_pipeline_folder(agent, cfg, name)
        ui.inform(f"Pipeline '{name}' permanently deleted.")


def _kill_pipeline_folder(agent, cfg, name: str) -> None:
    """Cancel broker tasks and delete the pipeline folder entirely."""
    import shutil
    agent.cancel_pipeline(name)
    folder = cfg.pipelines_dir / name
    if folder.exists():
        shutil.rmtree(folder)




def _list_all_pipelines(
    agent, cfg, *, visible_only: bool = False, hidden_only: bool = False
) -> list[tuple[str, str]]:
    """Return [(name, '2d'|'3d'), ...] filtered by visibility."""
    options: list[tuple[str, str]] = []
    for name in agent.list_pipeline_names():
        s = agent.get_pipeline_state(name)
        if s is None:
            continue
        if visible_only and s.hidden:
            continue
        if hidden_only and not s.hidden:
            continue
        options.append((name, "2d"))
    for name in agent.list_3d_pipeline_names():
        s = agent.get_3d_pipeline_state(name)
        if s is None:
            continue
        if visible_only and s.hidden:
            continue
        if hidden_only and not s.hidden:
            continue
        options.append((name, "3d"))
    return options


def _print_pipeline_list(ui, options: list[tuple[str, str]]) -> None:
    lines = ["Pipelines:"]
    for i, (name, ptype) in enumerate(options, 1):
        lines.append(f"  {i}. [{ptype.upper()}] {name}")
    ui.inform("\n".join(lines))


def _set_hidden(cfg, name: str, ptype: str, *, hidden: bool) -> None:
    """Load the pipeline state, flip the hidden flag, and save."""
    from src.state import Pipeline3DState
    state_path = cfg.pipelines_dir / name / "state.json"
    if ptype == "2d":
        state = PipelineState.load(state_path)
    else:
        state = Pipeline3DState.load(state_path)
    state.hidden = hidden
    state.save(state_path)


# ---------------------------------------------------------------------------
# Edit pipeline (2D only — description / poly count / symmetry)
# ---------------------------------------------------------------------------


def _edit_pipeline_for(agent, ui, cfg, name: str) -> None:
    """
    Edit settings for a specific 2D pipeline by name.
    Called from _manage_pipeline after the user has already picked a pipeline.
    """
    from src.state import SymmetryAxis

    editable_statuses = {
        PipelineStatus.INITIALIZING,
        PipelineStatus.CONCEPT_ART_GENERATING,
        PipelineStatus.CONCEPT_ART_REVIEW,
    }
    state = agent.get_pipeline_state(name)
    if state is None or state.status not in editable_statuses:
        ui.inform(
            f"Pipeline '{name}' cannot be edited "
            "(editing is only available before mesh generation)."
        )
        return

    state_path = cfg.pipelines_dir / name / "state.json"

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
            f"Symmetry axis — options: {', '.join(axis_options)}. "
            f"Enter to keep '{state.symmetry_axis.value}':"
        ).strip()
        if axis_raw in axis_options:
            state.symmetry_axis = SymmetryAxis(axis_raw)
    elif sym_raw == "n":
        state.symmetrize = False

    state.save(state_path)
    ui.inform(f"Pipeline '{name}' updated.")


def _edit_pipeline(agent, ui, cfg) -> None:
    """
    Legacy entry point: lists editable pipelines and lets the user pick one.
    Kept so test code that calls _edit_pipeline directly still works.
    """
    names = agent.list_pipeline_names()
    if not names:
        ui.inform("No pipelines to edit.")
        return

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
    _edit_pipeline_for(agent, ui, cfg, name)


# ---------------------------------------------------------------------------
# Status snapshot (2D + 3D) — shared helper, called from watch mode
# ---------------------------------------------------------------------------


def _print_full_status(agent, ui) -> None:
    """Print a full snapshot of worker threads and all pipeline statuses."""
    lines = ["=== Worker threads ==="]
    for t in agent._threads:
        alive = "running" if t.is_alive() else "stopped"
        lines.append(f"  {t.__class__.__name__}: {alive}")

    lines.append("")
    lines.append("=== 2D Pipelines ===")
    names_2d = agent.list_pipeline_names()
    if not names_2d:
        lines.append("  (none)")
    else:
        for name in names_2d:
            state = agent.get_pipeline_state(name)
            _append_pipeline_status_line(agent, lines, name, state)

    lines.append("")
    lines.append("=== 3D Pipelines ===")
    names_3d = agent.list_3d_pipeline_names()
    if not names_3d:
        lines.append("  (none)")
    else:
        for name in names_3d:
            state = agent.get_3d_pipeline_state(name)
            _append_pipeline_status_line(agent, lines, name, state)

    ui.inform("\n".join(lines))


# Keep _show_status as an alias so existing tests continue to pass.
_show_status = _print_full_status


def _append_pipeline_status_line(agent, lines: list[str], name: str, state) -> None:
    status = state.status.value if state else "unknown"
    hidden_tag = "  [hidden]" if (state and getattr(state, "hidden", False)) else ""
    tasks = agent._broker.get_tasks(pipeline_name=name)
    in_flight = [t for t in tasks if t.status in ("pending", "running")]
    failed = [t for t in tasks if t.status == "failed" and t.error != "cancelled"]
    task_info = f"  [{len(in_flight)} queued]" if in_flight else ""
    fail_info = f"  [{len(failed)} FAILED]" if failed else ""
    lines.append(f"  {name}: {status}{hidden_tag}{task_info}{fail_info}")
    for ft in failed:
        lines.append(f"    ! {ft.task_type}: {ft.error}")


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
    - Automatically surfaces approval prompts (both 2D and 3D) the moment
      they arrive.
    - Exits immediately on a single 'q' or Escape keypress (no Enter needed).

    On Windows single-char input works natively via msvcrt.
    On POSIX stdin is put into cbreak mode for the duration of the watch loop
    so characters are delivered without waiting for Enter.  cbreak is restored
    before every review prompt (which needs normal line-buffered input) and
    re-entered after.

    `_tick` and `_status_interval` are exposed for testing.
    """
    ui.inform(
        "Watching for status updates.  "
        f"Press '{_WATCH_EXIT_CMD}' to return to the menu.\n"
    )
    _print_full_status(agent, ui)

    last_states: dict = {}
    last_status_print = 0.0

    _enter_cbreak()
    try:
        # Seed last_states so the diff loop only prints genuine changes
        # after the initial full snapshot, not a duplicate of everything.
        _seed_watch_states(agent, last_states)
        last_status_print = time.time()

        while True:
            time.sleep(_tick)

            # ── Pending approvals ──────────────────────────────────────────
            waiting = _all_needing_attention(agent)
            if waiting:
                _exit_cbreak()
                quit_requested = False
                for name in waiting:
                    ts = time.strftime("%H:%M:%S")
                    ui.inform(f"\n[{ts}] !!! REVIEW NEEDED: {name} !!!")
                    _flush_stdin()
                    if _handle_priority_pipeline(
                        name, agent, ui, cfg, concept_worker, trellis_worker, restyle_worker
                    ):
                        quit_requested = True
                        break
                if quit_requested:
                    return
                # Stay in watch mode after reviews — pipelines that were
                # regenerated will surface again when they finish processing.
                ui.inform(
                    "\nReviews handled.  Continuing to watch "
                    f"(press '{_WATCH_EXIT_CMD}' to return to menu).\n"
                )
                # Re-seed so status diffs reflect the post-review state.
                _seed_watch_states(agent, last_states)
                last_status_print = time.time()
                _enter_cbreak()
                continue

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


def _seed_watch_states(agent, last_states: dict) -> None:
    """
    Populate last_states with the current signature for every pipeline so
    that _print_watch_diffs only emits lines for genuine changes that occur
    *after* the initial full-status snapshot.
    """
    all_names: list[tuple[str, str]] = (
        [(n, "2d") for n in agent.list_pipeline_names()]
        + [(n, "3d") for n in agent.list_3d_pipeline_names()]
    )
    for name, ptype in all_names:
        state = (
            agent.get_pipeline_state(name)
            if ptype == "2d"
            else agent.get_3d_pipeline_state(name)
        )
        if state is None:
            continue
        tasks = agent._broker.get_tasks(pipeline_name=name)
        in_flight = [t for t in tasks if t.status in ("pending", "running")]
        failed = [t for t in tasks if t.status == "failed" and t.error != "cancelled"]
        last_states[name] = (
            state.status.value,
            tuple(sorted(t.task_type for t in in_flight)),
            tuple(sorted(t.id for t in failed)),
        )


def _print_watch_diffs(agent, ui, last_states: dict) -> None:
    """
    For each pipeline (2D and 3D) whose (status, in-flight tasks, failed tasks)
    has changed since the last call, print a timestamped status line.
    """
    all_names: list[tuple[str, str]] = (
        [(n, "2d") for n in agent.list_pipeline_names()]
        + [(n, "3d") for n in agent.list_3d_pipeline_names()]
    )
    for name, ptype in all_names:
        state = (
            agent.get_pipeline_state(name)
            if ptype == "2d"
            else agent.get_3d_pipeline_state(name)
        )
        if state is None:
            continue
        tasks = agent._broker.get_tasks(pipeline_name=name)
        in_flight = [t for t in tasks if t.status in ("pending", "running")]
        failed = [t for t in tasks if t.status == "failed" and t.error != "cancelled"]
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

_saved_term_settings = None


def _enter_cbreak() -> None:
    global _saved_term_settings
    if os.name == "nt" or _saved_term_settings is not None:
        return
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        _saved_term_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        pass


def _exit_cbreak() -> None:
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
    if os.name == "nt":
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch()
        return None
    else:
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def _flush_stdin() -> None:
    if os.name == "nt":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    else:
        import termios
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Review dispatcher — 2D concept art or 3D mesh approval
# ---------------------------------------------------------------------------


def _handle_priority_pipeline(
    name, agent, ui, cfg, concept_worker, trellis_worker=None, restyle_worker=None
) -> bool:
    """
    Handle one priority pipeline review.
    Dispatches to concept-art review (2D) or mesh approval (3D).
    Returns True if the user asked to quit.
    """
    pipeline_dir = cfg.pipelines_dir / name
    if not pipeline_dir.exists():
        return False

    # 2D pipeline: concept art review
    state_2d = agent.get_pipeline_state(name)
    if state_2d is not None and state_2d.status == PipelineStatus.CONCEPT_ART_REVIEW:
        # Always re-show the review sheet and menu when entering a fresh
        # review session (flag may be True from a previous run).
        state_2d.concept_art_sheet_shown = False
        result = run_concept_art_review(
            state_2d, concept_worker, pipeline_dir, ui, cfg,
            restyle_worker=restyle_worker,
        )
        if result == "approved":
            _submit_approved_for_3d(state_2d, agent, ui, cfg, pipeline_dir)
            agent.dismiss_from_priority(name)
        elif result == "cancelled":
            agent.cancel_pipeline(name)
            ui.inform(f"[{name}] Pipeline cancelled.")
            agent.dismiss_from_priority(name)
        elif result == "quit":
            return True
        # "back" → user chose menu; leave undismissed so watch mode and the
        # attention note still surface this pipeline.
        return False

    # 3D pipeline: mesh approval
    state_3d = agent.get_3d_pipeline_state(name)
    if state_3d is not None:
        if state_3d.status == Pipeline3DStatus.AWAITING_APPROVAL:
            quit_requested = _handle_3d_approval(name, agent, ui, cfg)
            # Only dismiss on a real action (approve/regenerate), not on menu/back.
            # _handle_3d_approval returns False for both; we can't tell here, so
            # we don't dismiss at all from this path — manage pipeline handles it.
            return quit_requested
        else:
            # Status changed between _all_needing_attention and now (race condition).
            ui.inform(
                f"[!] Pipeline '{name}' status changed to '{state_3d.status.value}' "
                "before review could start — skipping."
            )
            return False

    # state_3d is None: state file is missing or could not be parsed.
    ui.inform(
        f"[!] Could not load state for pipeline '{name}'. "
        "The state file may be missing or corrupt — skipping."
    )
    log.warning("_handle_priority_pipeline: get_3d_pipeline_state('%s') returned None", name)
    return False


# ---------------------------------------------------------------------------
# 3D mesh review and approval
# ---------------------------------------------------------------------------


def _handle_3d_approval(name: str, agent, ui, cfg) -> bool:
    """
    Interactive 3D mesh review loop.

    Shows the mesh review sheet and lets the user approve (export to final
    assets), regenerate (re-queue with optionally different poly count), or
    quit.

    Returns True if the user asked to quit.
    """
    from pathlib import Path as _Path
    from src.mesh_pipeline import run_mesh_export

    state_path = cfg.pipelines_dir / name / "state.json"
    pipeline_dir = cfg.pipelines_dir / name
    sheet_shown = False

    while True:
        state = agent.get_3d_pipeline_state(name)
        if state is None:
            ui.inform(
                f"[!] Could not reload state for '{name}' — "
                "the state file may have been removed. Returning to menu."
            )
            log.warning("_handle_3d_approval: get_3d_pipeline_state('%s') returned None", name)
            return False

        if state.status not in (
            Pipeline3DStatus.AWAITING_APPROVAL, Pipeline3DStatus.IDLE
        ):
            ui.inform(
                f"Pipeline '{name}' is not awaiting approval "
                f"(status: {state.status.value})."
            )
            return False

        # Show review sheet once per call (further prompts in the loop don't
        # need to re-blast the image).
        if not sheet_shown and state.review_sheet_path:
            review_sheet = _Path(state.review_sheet_path)
            if review_sheet.exists():
                ui.show_image(review_sheet)
        sheet_shown = True

        mesh_display = state.textured_mesh_path or state.mesh_path or "(none)"
        source_display = (
            f"  Source 2D pipeline: {state.source_2d_pipeline} "
            f"(CA {state.source_concept_art_index})\n"
            if state.source_2d_pipeline else ""
        )
        ui.inform(
            f"\n{'─' * 50}\n"
            f"3D mesh review for '{name}'\n"
            f"  Status: {state.status.value}\n"
            + source_display +
            f"  Mesh: {mesh_display}\n"
            f"  Exports so far: {state.export_version}\n"
            "Actions:\n"
            "  approve       — export mesh to final assets\n"
            "  regenerate    — re-queue mesh generation (optionally with different poly count)\n"
            "  menu          — return to main menu\n"
            "  quit          — exit the program\n"
        )

        raw = ui.ask("Enter action").lower().strip()
        tokens = raw.split()
        if not tokens:
            continue
        action = tokens[0]

        if action == "approve":
            # run_mesh_export silently returns if the glb file is missing,
            # so check up-front and surface a clear error to the user
            # rather than printing a misleading "exported" message.
            glb_candidate = state.textured_mesh_path or state.mesh_path
            if not glb_candidate or not _Path(glb_candidate).exists():
                ui.inform(
                    f"Cannot export: no mesh file found for '{name}'. "
                    "Try 'regenerate' to rebuild the mesh."
                )
                continue
            version_before = state.export_version
            run_mesh_export(state, pipeline_dir, cfg, asset_name=name)
            state.save(state_path)
            if state.export_version == version_before:
                ui.inform(
                    f"Export failed for '{name}' — mesh export did not produce a new version."
                )
                continue
            version_exported = state.export_version - 1
            ui.inform(
                f"Mesh exported (version {version_exported}).  "
                f"Pipeline is now idle — you can approve again for additional export versions."
            )
            return False

        elif action == "regenerate":
            new_polys_raw = ui.ask(
                f"New polygon count (current: {state.num_polys}, Enter to keep):"
            ).strip()
            if new_polys_raw.isdigit() and int(new_polys_raw) > 0:
                state.num_polys = int(new_polys_raw)
            agent.cancel_pipeline(name)
            state.status = Pipeline3DStatus.QUEUED
            state.save(state_path)
            agent.enqueue_mesh_generation(name)
            ui.inform(
                f"Mesh generation re-queued for '{name}' "
                f"with {state.num_polys} polygons."
            )
            return False

        elif action == "menu":
            return False

        elif action == "quit":
            return True

        else:
            ui.inform("Unknown action.  Use: approve, regenerate, menu, quit")


# ---------------------------------------------------------------------------
# Submit approved 2D concept arts for 3D pipeline generation
# ---------------------------------------------------------------------------


def _submit_approved_for_3d(
    state: PipelineState,
    agent,
    ui,
    cfg,
    pipeline_dir,
) -> None:
    """
    Called after a 2D concept-art review returns 'approved'.

    Lists approved concept arts, asks the user if they want to submit them
    for 3D mesh generation, and starts a 3D pipeline for each one selected.

    Derived 3D pipeline name: "{pipeline_name}_{ca_index+1}_{ca_version}"
    Existing 3D pipelines with the same name prompt the user to overwrite.
    """
    from pathlib import Path as _Path
    approved = state.approved_concept_arts()
    if not approved:
        return

    count = len(approved)
    noun = "image" if count == 1 else "images"
    indices_str = ", ".join(str(ca.index + 1) for ca in approved)
    choice = ui.ask(
        f"Submit {count} approved {noun} (index {indices_str}) for 3D mesh generation? "
        "(y/n, Enter=yes):"
    ).strip().lower()
    if choice == "n":
        ui.inform("Skipped 3D mesh generation.")
        return

    for ca in approved:
        derived_name = f"{state.name}_{ca.index + 1}_{ca.version}"
        image_path = str(
            _Path(pipeline_dir) / "concept_art"
            / f"concept_art_{ca.index + 1}_{ca.version}.png"
        )

        # Check for existing pipeline with this derived name
        if agent.pipeline_name_exists(derived_name):
            existing_3d = agent.get_3d_pipeline_state(derived_name)
            existing_2d = agent.get_pipeline_state(derived_name)
            # If a 2D pipeline happens to share the derived name, refuse to
            # touch it — overwriting would destroy unrelated work.
            if existing_2d is not None and existing_3d is None:
                ui.inform(
                    f"  Cannot create 3D pipeline '{derived_name}': "
                    f"a 2D pipeline with that name already exists. "
                    f"Rename the 2D pipeline or kill it first."
                )
                ui.inform(f"  Skipping image {ca.index + 1}.")
                continue
            if existing_3d is not None:
                ui.inform(
                    f"  3D pipeline '{derived_name}' already exists "
                    f"(status: {existing_3d.status.value})."
                )
                overwrite = ui.ask(
                    "  Cancel and overwrite? (y/N):"
                ).strip().lower()
                if overwrite != "y":
                    ui.inform(f"  Skipping image {ca.index + 1}.")
                    continue
                _kill_pipeline_folder(agent, cfg, derived_name)

        agent.start_3d_pipeline(
            name=derived_name,
            input_image_path=image_path,
            num_polys=state.num_polys,
            source_2d_pipeline=state.name,
            source_concept_art_index=ca.index,
            source_concept_art_version=ca.version,
            symmetrize=state.symmetrize,
            symmetry_axis=state.symmetry_axis.value,
        )
        ui.inform(f"  3D pipeline '{derived_name}' queued for mesh generation.")


# ---------------------------------------------------------------------------
# Retry failed tasks
# ---------------------------------------------------------------------------


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
