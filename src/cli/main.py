"""
quickymesh CLI — HTTP client edition.

This is the user-facing terminal interface.  It talks to a running
``api_server`` over HTTP via :class:`src.cli.client.QuickymeshClient`.

The menu structure intentionally mirrors the old in-process CLI so users
see no behavioural change — only the plumbing differs.  All state lives on
the server; the client is stateless apart from a per-session set of
priority-dismissed pipeline names and a per-session cache of user-approved
concept art indices (which exist only until they are submitted for 3D mesh
generation).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.cli.client import (
    AuthError,
    ConflictError,
    ConnectionError as QmConnectionError,
    NotFoundError,
    QuickymeshAPIError,
    QuickymeshClient,
    load_preferences,
    save_preferences,
)

if TYPE_CHECKING:
    from src.prompt_interface.base import PromptInterface

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IDLE_MENU = """\
--- quickymesh ---
[n] Start a new 2D pipeline
[3] Start a 3D pipeline from a local image
[m] Manage a pipeline  ( [re]submit / edit / hide / kill )
[u] Unhide a pipeline
[w] Watch for status updates
[t] Retry failed tasks
[q] Quit"""

_IDLE_OPT_MAP = {
    "1": "n", "2": "3", "3": "m", "4": "u", "5": "w", "6": "t", "7": "q",
}
_IDLE_VALID = {"n", "3", "m", "u", "w", "t", "q", ""}

_WATCH_EXIT_CMD = "q"
_WATCH_TICK = 0.1
_WATCH_STATUS_INTERVAL = 3.0

_SYM_OPTIONS = ["x-", "x+", "y-", "y+", "z-", "z+"]

# 2D statuses (strings — we compare against JSON payloads, not enums)
_S2D_REVIEW = "concept_art_review"
_S2D_EDITABLE = {"initializing", "concept_art_generating", "concept_art_review"}

# 3D statuses
_S3D_AWAITING = "awaiting_approval"
_S3D_IDLE = "idle"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_cli(
    client: QuickymeshClient,
    ui: "PromptInterface",
) -> None:
    """
    Main CLI entry point.  Runs the idle-menu loop until the user quits.

    Parameters
    ----------
    client:  A connected :class:`QuickymeshClient`.
    ui:      A :class:`PromptInterface` for user I/O.
    """
    # Probe the server up front so we fail fast with a useful message.
    try:
        cfg = client.get_config()
    except QmConnectionError as e:
        ui.inform(f"Cannot reach quickymesh server: {e}")
        return
    except AuthError:
        ui.inform(
            "Server requires authentication, but no valid API key was provided.\n"
            "Run with --api-key KEY or set QUICKYMESH_API_KEY in the environment."
        )
        return

    ui.inform(f"Connected to quickymesh server.  Output root: {cfg.get('output_root')}")

    session = _Session()
    try:
        while True:
            action = _idle_menu(client, ui, cfg, session)
            if action == "quit":
                break
    finally:
        client.close()
        ui.inform("Goodbye.")


# ---------------------------------------------------------------------------
# Session state (priority dismissals live here, not on the server)
# ---------------------------------------------------------------------------


class _Session:
    def __init__(self) -> None:
        self._dismissed: set[str] = set()

    def dismiss(self, name: str) -> None:
        self._dismissed.add(name)

    def undismiss(self, name: str) -> None:
        self._dismissed.discard(name)

    def is_dismissed(self, name: str) -> bool:
        return name in self._dismissed


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------


def _pipeline_lists(client: QuickymeshClient) -> tuple[list[dict], list[dict]]:
    """Return (2D list, 3D list) or ([], []) on server errors."""
    try:
        p2 = client.list_pipelines()
    except QmConnectionError as e:
        log.warning("list_pipelines failed: %s", e)
        p2 = []
    try:
        p3 = client.list_3d_pipelines()
    except QmConnectionError as e:
        log.warning("list_3d_pipelines failed: %s", e)
        p3 = []
    return p2, p3


def _all_needing_attention(client: QuickymeshClient, session: _Session) -> list[tuple[str, str]]:
    """Return [(name, '2d'|'3d')] for pipelines waiting on the user."""
    waiting: list[tuple[str, str]] = []
    p2, p3 = _pipeline_lists(client)
    for item in p2:
        name = item["name"]
        if session.is_dismissed(name):
            continue
        full = client.get_pipeline_or_none(name)
        if full and not full.get("hidden") and full.get("status") == _S2D_REVIEW:
            waiting.append((name, "2d"))
    for item in p3:
        name = item["name"]
        if session.is_dismissed(name):
            continue
        full = client.get_3d_pipeline_or_none(name)
        if full and not full.get("hidden") and full.get("status") == _S3D_AWAITING:
            waiting.append((name, "3d"))
    return waiting


# ---------------------------------------------------------------------------
# Idle menu
# ---------------------------------------------------------------------------


def _idle_menu(client, ui, cfg, session) -> str:
    waiting = _all_needing_attention(client, session)
    if waiting:
        names = ", ".join(f"'{n}'" for n, _ in waiting)
        noun = "Pipeline" if len(waiting) == 1 else "Pipelines"
        verb = "is" if len(waiting) == 1 else "are"
        approval_note = f"\n[!] {noun} {names} {verb} waiting for review — press Enter to open.\n"
    else:
        approval_note = ""

    choice = ui.ask(_IDLE_MENU + approval_note).strip().lower()
    if choice not in _IDLE_VALID:
        choice = _IDLE_OPT_MAP.get(choice, choice)

    if not choice:
        if waiting:
            result = _manage_pipeline(client, ui, cfg, session, preselect=waiting[0][0])
            return "quit" if result == "quit" else "manage"
        return "check"
    if choice == "q":
        return "quit"
    if choice == "n":
        _start_new_pipeline(client, ui, cfg)
        return "new"
    if choice == "3":
        _start_3d_pipeline_from_file(client, ui, cfg)
        return "3d"
    if choice == "m":
        result = _manage_pipeline(client, ui, cfg, session)
        return "quit" if result == "quit" else "manage"
    if choice == "u":
        _unhide_pipeline(client, ui)
        return "unhide"
    if choice == "w":
        _watch_mode(client, ui, cfg, session)
        return "watch"
    if choice == "t":
        _retry_failed(client, ui)
        return "retry"
    return "check"


# ---------------------------------------------------------------------------
# Start new 2D pipeline
# ---------------------------------------------------------------------------


def _start_new_pipeline(client, ui, cfg) -> None:
    name = ui.ask("Pipeline name (no spaces):").strip()
    if not name:
        ui.inform("Cancelled.")
        return

    # Concept art backend selection
    prefs = load_preferences()
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
    save_preferences({**prefs, "concept_art_backend": backend})

    # Gemini API key check (server reports whether it has one configured)
    if backend == "gemini" and not cfg.get("gemini_api_key_present"):
        ui.inform(
            "Server reports no Gemini API key configured.  "
            "Set GEMINI_API_KEY in the server's environment and restart it before using Gemini."
        )
        # Don't hard-fail — the user may still want to queue the pipeline.

    # Optional input image (Gemini only)
    input_image_path: Path | None = None
    if backend == "gemini":
        while True:
            img_raw = ui.ask(
                "Base concept art on an existing image? "
                "Enter a local path (absolute), or leave blank:"
            ).strip()
            if not img_raw:
                break
            p = Path(img_raw)
            if p.exists():
                input_image_path = p
                ui.inform(f"Image found: {p}")
                break
            ui.inform(f"Image not found: {p}\nPlease try again, or press Enter to skip.")

    # Description
    suffix = cfg.get("background_suffix", "")
    if input_image_path is not None:
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

    # Polygon count
    default_polys = cfg.get("num_polys_default")
    polys_str = ui.ask(f"Target polygon count (Enter for default {default_polys}):").strip()
    num_polys = int(polys_str) if polys_str.isdigit() else None

    # Symmetry
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

    # Create via appropriate endpoint
    try:
        if input_image_path is not None:
            client.create_pipeline_from_upload(
                name, description, input_image_path,
                num_polys=num_polys, symmetrize=symmetrize,
                symmetry_axis=symmetry_axis, concept_art_backend=backend,
            )
        else:
            client.create_pipeline(
                name, description, num_polys,
                symmetrize=symmetrize, symmetry_axis=symmetry_axis,
                concept_art_backend=backend,
            )
    except ConflictError as e:
        ui.inform(f"Could not create pipeline: {e.detail}")
        return
    except QuickymeshAPIError as e:
        ui.inform(f"Server error creating pipeline: {e.detail}")
        return
    ui.inform(f"Pipeline '{name}' started.  Concept art generation queued.")


# ---------------------------------------------------------------------------
# Start 3D pipeline from a local image
# ---------------------------------------------------------------------------


def _start_3d_pipeline_from_file(client, ui, cfg) -> None:
    short_name = ui.ask(
        "Pipeline name (no spaces — will be prefixed with 'u_'):"
    ).strip()
    if not short_name:
        ui.inform("Cancelled.")
        return

    while True:
        img_raw = ui.ask("Path to the image file (absolute):").strip()
        if not img_raw:
            ui.inform("Cancelled.")
            return
        p = Path(img_raw)
        if p.exists():
            image_path = p
            break
        ui.inform(f"File not found: {p}\nPlease try again, or leave blank to cancel.")

    default_polys = cfg.get("num_polys_default")
    polys_str = ui.ask(f"Target polygon count (Enter for default {default_polys}):").strip()
    num_polys = int(polys_str) if polys_str.isdigit() else None

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

    try:
        result = client.create_3d_pipeline_from_upload(
            short_name, image_path, num_polys=num_polys,
            symmetrize=symmetrize, symmetry_axis=symmetry_axis,
        )
    except ConflictError as e:
        ui.inform(f"Could not create 3D pipeline: {e.detail}")
        return
    except QuickymeshAPIError as e:
        ui.inform(f"Server error: {e.detail}")
        return
    ui.inform(
        f"3D pipeline '{result.get('name', 'u_' + short_name)}' started.  "
        "Mesh generation queued."
    )


# ---------------------------------------------------------------------------
# Manage a pipeline (pick, show info, act)
# ---------------------------------------------------------------------------


def _list_all_pipelines(
    client, *, visible_only: bool = False, hidden_only: bool = False,
) -> list[tuple[str, str, dict]]:
    """Return [(name, '2d'|'3d', state_dict)] filtered by visibility."""
    out: list[tuple[str, str, dict]] = []
    p2, p3 = _pipeline_lists(client)
    for item in p2:
        name = item["name"]
        s = client.get_pipeline_or_none(name)
        if s is None:
            continue
        if visible_only and s.get("hidden"):
            continue
        if hidden_only and not s.get("hidden"):
            continue
        out.append((name, "2d", s))
    for item in p3:
        name = item["name"]
        s = client.get_3d_pipeline_or_none(name)
        if s is None:
            continue
        if visible_only and s.get("hidden"):
            continue
        if hidden_only and not s.get("hidden"):
            continue
        out.append((name, "3d", s))
    return out


def _manage_pipeline(client, ui, cfg, session, *, preselect: str | None = None):
    options = _list_all_pipelines(client)
    if not options:
        ui.inform("No pipelines found.")
        return

    name: str | None = None
    ptype: str | None = None
    state: dict | None = None

    if preselect is not None:
        for n, pt, s in options:
            if n == preselect:
                name, ptype, state = n, pt, s
                break

    if name is None:
        lines = ["Pipelines:"]
        for i, (n, pt, s) in enumerate(options, 1):
            status = s.get("status", "unknown")
            hidden_tag = "  [hidden]" if s.get("hidden") else ""
            lines.append(f"  {i}. [{pt.upper()}] {n}  {status}{hidden_tag}")
        ui.inform("\n".join(lines))
        choice_raw = ui.ask("Enter number (Enter to cancel):").strip()
        if not choice_raw or not choice_raw.isdigit() or not (1 <= int(choice_raw) <= len(options)):
            ui.inform("Cancelled.")
            return
        name, ptype, state = options[int(choice_raw) - 1]

    # Rich info
    info_lines = [f"\nPipeline '{name}' [{ptype.upper()}]"]
    info_lines.append(f"  Status : {state.get('status', 'unknown')}")
    if ptype == "2d":
        info_lines.append(f"  Description: {state.get('description', '')}")
        cas = state.get("concept_arts") or []
        if cas:
            counts: dict[str, int] = {}
            for ca in cas:
                counts[ca["status"]] = counts.get(ca["status"], 0) + 1
            info_lines.append("  Concept arts: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
        # Spawned 3D pipelines
        _, p3 = _pipeline_lists(client)
        spawned: list[tuple[str, dict]] = []
        for item in p3:
            s3 = client.get_3d_pipeline_or_none(item["name"])
            if s3 and s3.get("source_2d_pipeline") == name:
                spawned.append((item["name"], s3))
        if spawned:
            info_lines.append("  Spawned 3D pipelines:")
            for sp_name, sp in spawned:
                info_lines.append(
                    f"    • {sp_name}  [{sp.get('status', 'unknown')}, "
                    f"{sp.get('export_version', 0)} export(s)]"
                )
    else:
        src = state.get("source_2d_pipeline")
        if src:
            info_lines.append(
                f"  Source: {src} (concept art {int(state.get('source_concept_art_index', 0)) + 1})"
            )
        mesh = state.get("textured_mesh_path") or state.get("mesh_path") or "(none)"
        info_lines.append(f"  Mesh  : {mesh}")
        info_lines.append(f"  Exports: {state.get('export_version', 0)}")
    ui.inform("\n".join(info_lines))

    # Build actions
    actions: list[tuple[str, str]] = []
    status = state.get("status", "")
    can_review_2d = ptype == "2d" and status == _S2D_REVIEW
    can_review_3d = ptype == "3d" and status in (_S3D_AWAITING, _S3D_IDLE)
    can_submit_2d = ptype == "2d"

    if can_review_2d:
        actions.append(("[a]", "Review concept art / resubmit for 3D"))
    elif can_submit_2d:
        actions.append(("[a]", "Resubmit approved concept art(s) for 3D"))
    if can_review_3d:
        actions.append(("[a]", "Review mesh / approve"))
    if ptype == "2d":
        actions.append(("[e]", "Edit settings (description, poly count, symmetry)"))
    if state.get("hidden"):
        actions.append(("[r]", "Restore (make visible again)"))
    else:
        actions.append(("[h]", "Hide (keep folder, remove from active view)"))
    actions.append(("[k]", "Kill — cancel and remove from active view"))
    actions.append(("[b]", "Back to main menu"))

    action_lines = ["\nActions:"]
    for letter, label in actions:
        action_lines.append(f"  {letter} {label}")
    ui.inform("\n".join(action_lines))

    action_choice = ui.ask("Enter action (Enter to go back):").strip().lower()
    if not action_choice or action_choice == "b":
        return

    valid_letters = {lt.strip("[]") for lt, _ in actions}
    if action_choice not in valid_letters:
        ui.inform(f"Unknown action '{action_choice}'.")
        return

    # Dispatch
    if action_choice == "a":
        if ptype == "2d":
            result = _run_concept_art_review(client, ui, cfg, name, state)
            if result == "quit":
                return "quit"
            if result == "approved":
                session.dismiss(name)
        else:
            if _handle_3d_approval(client, ui, name):
                return "quit"
            after = client.get_3d_pipeline_or_none(name)
            if after is None or after.get("status") != _S3D_AWAITING:
                session.dismiss(name)
    elif action_choice == "e":
        _edit_pipeline(client, ui, cfg, name, state)
    elif action_choice == "h":
        _set_hidden(client, ui, name, ptype, hidden=True)
    elif action_choice == "r":
        _set_hidden(client, ui, name, ptype, hidden=False)
        session.undismiss(name)
    elif action_choice == "k":
        ui.inform(
            f"\nWARNING: This will cancel pipeline '{name}' and remove it from the active view."
        )
        confirm = ui.ask("Type 'confirm' to proceed, or Enter to cancel:").strip().lower()
        if confirm != "confirm":
            ui.inform("Cancelled.")
            return
        try:
            if ptype == "2d":
                client.cancel_pipeline(name)
            else:
                client.cancel_3d_pipeline(name)
            _set_hidden(client, ui, name, ptype, hidden=True, quiet=True)
        except QuickymeshAPIError as e:
            ui.inform(f"Server error: {e.detail}")
            return
        ui.inform(f"Pipeline '{name}' cancelled and hidden.")


def _set_hidden(client, ui, name: str, ptype: str, *, hidden: bool, quiet: bool = False) -> None:
    try:
        if ptype == "2d":
            client.patch_pipeline(name, hidden=hidden)
        else:
            client.patch_3d_pipeline(name, hidden=hidden)
    except QuickymeshAPIError as e:
        ui.inform(f"Server error: {e.detail}")
        return
    if not quiet:
        ui.inform(f"Pipeline '{name}' {'hidden' if hidden else 'restored'}.")


def _unhide_pipeline(client, ui) -> None:
    hidden = _list_all_pipelines(client, hidden_only=True)
    if not hidden:
        ui.inform("No hidden pipelines.")
        return
    lines = ["Hidden pipelines:"]
    for i, (n, pt, s) in enumerate(hidden, 1):
        lines.append(f"  {i}. [{pt.upper()}] {n}  [{s.get('status', 'unknown')}]")
    ui.inform("\n".join(lines))
    choice = ui.ask("Enter number to unhide (Enter to cancel):").strip()
    if not choice or not choice.isdigit():
        return
    idx = int(choice) - 1
    if not (0 <= idx < len(hidden)):
        ui.inform("Invalid selection.")
        return
    n, pt, _ = hidden[idx]
    _set_hidden(client, ui, n, pt, hidden=False)


# ---------------------------------------------------------------------------
# Edit 2D pipeline
# ---------------------------------------------------------------------------


def _edit_pipeline(client, ui, cfg, name: str, state: dict) -> None:
    if state.get("status") not in _S2D_EDITABLE:
        ui.inform(
            f"Pipeline '{name}' cannot be edited "
            "(editing is only available before mesh generation)."
        )
        return

    patch: dict[str, Any] = {}
    suffix = cfg.get("background_suffix", "")
    ui.inform(f'Current description: "{state.get("description", "")}"')
    ui.inform(f'(Suffix appended automatically: "{suffix}")')
    new_desc = ui.ask("New description (Enter to keep current):").strip()
    if new_desc:
        patch["description"] = new_desc

    ui.inform(f"Current polygon target: {state.get('num_polys')}")
    poly_raw = ui.ask("New polygon count (Enter to keep current):").strip()
    if poly_raw.isdigit() and int(poly_raw) > 0:
        patch["num_polys"] = int(poly_raw)

    cur_sym = state.get("symmetrize", False)
    cur_axis = state.get("symmetry_axis", "x-")
    ui.inform(f"Current symmetry: {'on' if cur_sym else 'off'}, axis={cur_axis}")
    sym_raw = ui.ask("Enable symmetrize? (y/n, Enter to keep current):").strip().lower()
    if sym_raw == "y":
        patch["symmetrize"] = True
        axis_raw = ui.ask(
            f"Symmetry axis — options: {', '.join(_SYM_OPTIONS)}. "
            f"Enter to keep '{cur_axis}':"
        ).strip()
        if axis_raw in _SYM_OPTIONS:
            patch["symmetry_axis"] = axis_raw
    elif sym_raw == "n":
        patch["symmetrize"] = False

    if not patch:
        ui.inform("No changes.")
        return
    try:
        client.patch_pipeline(name, **patch)
    except ConflictError as e:
        ui.inform(f"Edit rejected: {e.detail}")
        return
    except QuickymeshAPIError as e:
        ui.inform(f"Server error: {e.detail}")
        return
    ui.inform(f"Pipeline '{name}' updated.")


# ---------------------------------------------------------------------------
# 2D concept art review (client-side loop)
# ---------------------------------------------------------------------------


def _parse_indices(tokens: list[str], max_count: int) -> list[int] | None:
    result = []
    for t in tokens:
        if not t.isdigit():
            return None
        idx = int(t) - 1
        if idx < 0 or idx >= max_count:
            return None
        result.append(idx)
    return result


_BUSY_CA_STATUSES = ("generating", "regenerating", "modified")


def _ask_source_version(ui, ca: dict) -> int | None | str:
    """
    If the concept art slot has more than one version (i.e. version > 0),
    let the user pick which version to modify/restyle from.  Returns:
      * an int version number
      * None to mean "latest" (Enter)
      * the string "cancel" if the user typed garbage
    """
    latest = int(ca.get("version", 0))
    if latest <= 0:
        return None
    raw = ui.ask(
        f"Slot {int(ca.get('index', 0)) + 1} has versions 0–{latest} (latest={latest}).\n"
        f"Which version to use as the source? (Enter=latest):"
    ).strip()
    if not raw:
        return None
    if not raw.isdigit():
        return "cancel"
    v = int(raw)
    if v < 0 or v > latest:
        ui.inform(f"Invalid version {v}; cancelling.")
        return "cancel"
    return v


def _concept_arts_busy(state: dict) -> bool:
    return any(
        ca.get("status") in _BUSY_CA_STATUSES
        for ca in state.get("concept_arts") or []
    )


def _wait_for_concept_art_ready(
    client, ui, name: str, *, timeout: float = 600.0, poll: float = 1.0,
) -> tuple[str, dict | None]:
    """
    Poll until no concept art is in 'generating'/'regenerating'.

    Interruptible: press ``q``, ESC, or Ctrl-C to abandon the wait and return
    to the caller without cancelling the server-side task.

    Returns ``(outcome, state)`` where outcome is one of:
      * ``"ready"``       — all concept arts settled, ``state`` is the new state
      * ``"interrupted"`` — user pressed q/ESC/Ctrl-C, ``state`` is None
      * ``"timeout"``     — deadline exceeded, ``state`` is None
      * ``"error"``       — server error already reported, ``state`` is None
    """
    ui.inform(
        "Waiting for concept art generation to finish "
        "(press 'q' to return to the menu; the task keeps running on the server)..."
    )
    # Flush any residual input from the previous ui.ask() prompts before
    # entering cbreak — otherwise a stray Enter / whitespace from the prior
    # line-input can land in the cbreak read buffer and look like an exit
    # keypress, causing the wait to immediately "abandon" itself.
    _flush_stdin()
    deadline = time.time() + timeout
    _enter_cbreak()
    try:
        while time.time() < deadline:
            try:
                state = client.get_pipeline(name)
            except QuickymeshAPIError as e:
                ui.inform(f"Server error while polling: {e.detail}")
                return "error", None
            if not _concept_arts_busy(state):
                ui.inform("Concept art regeneration complete.")
                return "ready", state

            # Interruptible sleep: check for a keypress every ~100 ms so the
            # user can bail without waiting for the full poll interval.
            slept = 0.0
            while slept < poll:
                ch = _try_read_char()
                if ch is not None and ch.lower() in ("q", "\x1b", "\x03"):
                    ui.inform("\nAbandoning wait — regeneration is still running on the server.")
                    return "interrupted", None
                time.sleep(0.1)
                slept += 0.1
    finally:
        _exit_cbreak()
    ui.inform("Timed out waiting for concept art generation.")
    return "timeout", None


def _run_concept_art_review(client, ui, cfg, name: str, state: dict) -> str:
    """
    Interactive review loop for 2D concept art.

    Returns one of:
      "approved" — user approved at least one image and submitted them for 3D
      "back"     — user returned to the manage menu
      "quit"     — user asked to quit the CLI
    """
    approved_indices: set[int] = set()
    sheet_shown = False

    # If the pipeline is mid-generation on entry, give the user a clear
    # choice rather than silently trying to fetch a sheet that will 409.
    if _concept_arts_busy(state):
        ui.inform(
            f"\nPipeline '{name}' is still generating concept art.\n"
            "  [w] Wait for it to finish\n"
            "  [b] Back to main menu\n"
            "  [q] Quit"
        )
        choice = ui.ask("Choice:").strip().lower()
        if choice == "q":
            return "quit"
        if choice != "w":
            return "back"
        outcome, new_state = _wait_for_concept_art_ready(client, ui, name)
        if outcome == "ready" and new_state is not None:
            state = new_state
        else:
            return "back"

    while True:
        cas = state.get("concept_arts") or []
        if not cas:
            ui.inform("No concept arts yet.")
            return "back"

        if not sheet_shown:
            try:
                sheet_path = client.get_concept_art_sheet(name)
                ui.show_image(sheet_path)
            except (NotFoundError, ConflictError) as e:
                ui.inform(f"Could not fetch review sheet: {e.detail}")
            except QmConnectionError as e:
                ui.inform(f"Connection error: {e}")
                return "back"
            supports_modify = cfg.get("concept_art_supports_modify", True)
            modify_line = (
                "  modify              — modify one image via Gemini\n"
                if supports_modify else ""
            )
            restyle_line = (
                "  restyle             — restyle image shape/silhouette via ControlNet\n"
                if cfg.get("restyle_worker_available", True) else ""
            )
            ui.inform(
                f"\nConcept art review for '{name}'\n"
                "Actions:\n"
                "  approve <indices>   — e.g. 'approve 1 3' to send to mesh gen\n"
                "  regenerate          — pick an image (or type 'regenerate all')\n"
                + modify_line
                + restyle_line +
                "  menu                — return to main menu\n"
                "  quit                — exit the program\n"
            )
            for ca in cas:
                idx = ca.get("index", 0)
                ver = ca.get("version", 0)
                ver_tag = f"  v{ver}" if ver > 0 else ""
                approved_tag = "  *APPROVED*" if idx in approved_indices else ""
                ui.inform(f"  [{idx + 1}] {ca.get('status', 'unknown')}{ver_tag}{approved_tag}")
            sheet_shown = True

        raw = ui.ask("Enter action").lower().strip()
        tokens = raw.split()
        if not tokens:
            continue
        action = tokens[0]

        if action == "approve":
            if len(tokens) < 2:
                ui.inform("Please specify which images to approve, e.g. 'approve 1 2'")
                continue
            picked = _parse_indices(tokens[1:], len(cas))
            if picked is None:
                ui.inform("Invalid indices — use 1-based numbers matching the review sheet.")
                continue
            approved_indices.update(picked)
            ui.inform(
                f"Approved {len(picked)} image(s).  "
                f"Total approved this session: {len(approved_indices)}."
            )
            if _submit_approved_for_3d(client, ui, name, state, approved_indices):
                return "approved"
            # User declined to submit — drop back to the menu so they can
            # regenerate/modify or approve more.
            sheet_shown = False

        elif action == "regenerate":
            if len(tokens) >= 2 and tokens[1] == "all":
                regen_indices = list(range(len(cas)))
                current_desc = state.get("description", "")
                ui.inform(
                    f"Current description: {current_desc}\n"
                    f'(Suffix "{cfg.get("background_suffix", "")}" appended automatically)'
                )
                new_desc = ui.ask("New description (Enter to keep current):").strip()
                desc_override = new_desc if new_desc else None
            else:
                idx_raw = ui.ask(f"Which image to regenerate? (1–{len(cas)}):").strip()
                if not idx_raw.isdigit():
                    ui.inform("Cancelled.")
                    continue
                idx = int(idx_raw) - 1
                if not (0 <= idx < len(cas)):
                    ui.inform("Invalid index.")
                    continue
                regen_indices = [idx]
                desc_override = None
            try:
                client.regenerate_concept_art(
                    name, indices=regen_indices, description_override=desc_override,
                )
            except QuickymeshAPIError as e:
                ui.inform(f"Regenerate failed: {e.detail}")
                continue
            outcome, new_state = _wait_for_concept_art_ready(client, ui, name)
            if outcome == "ready" and new_state is not None:
                state = new_state
                sheet_shown = False
            elif outcome == "interrupted":
                # User bailed — drop back to the manage menu.  The regen
                # keeps running on the server and will surface via watch mode
                # / attention-note when it finishes.
                return "back"
            else:
                return "back"

        elif action == "modify":
            idx_raw = ui.ask(f"Which image to modify? (1–{len(cas)}):").strip()
            if not idx_raw.isdigit():
                ui.inform("Cancelled.")
                continue
            idx = int(idx_raw) - 1
            if not (0 <= idx < len(cas)):
                ui.inform("Invalid index.")
                continue
            source_version = _ask_source_version(ui, cas[idx])
            if source_version == "cancel":
                ui.inform("Cancelled.")
                continue
            instruction = ui.ask(f"Describe the change to make to image {idx + 1}:")
            if not instruction.strip():
                ui.inform("Cancelled.")
                continue
            try:
                client.modify_concept_art(
                    name, idx, instruction, source_version=source_version,
                )
            except ConflictError as e:
                ui.inform(f"Modify unavailable: {e.detail}")
                continue
            except QuickymeshAPIError as e:
                ui.inform(f"Modify failed: {e.detail}")
                continue
            outcome, new_state = _wait_for_concept_art_ready(client, ui, name)
            if outcome == "ready" and new_state is not None:
                state = new_state
                sheet_shown = False
            else:
                # interrupted / timeout / error — bail to manage menu; the work
                # keeps running on the server and will surface via watch mode.
                return "back"

        elif action == "restyle":
            idx_raw = ui.ask(f"Which image to restyle? (1–{len(cas)}):").strip()
            if not idx_raw.isdigit():
                ui.inform("Cancelled.")
                continue
            idx = int(idx_raw) - 1
            if not (0 <= idx < len(cas)):
                ui.inform("Invalid index.")
                continue
            source_version = _ask_source_version(ui, cas[idx])
            if source_version == "cancel":
                ui.inform("Cancelled.")
                continue
            positive = ui.ask(
                f"Positive prompt for image {idx + 1}\n"
                "  Comma-separated words/phrases describing the style you want:"
            ).strip()
            if not positive:
                ui.inform("Cancelled.")
                continue
            negative = ui.ask(
                "Negative prompt (Enter for defaults):"
            ).strip()
            denoise_raw = ui.ask(
                "Denoise strength 0.1–1.0 (Enter for default 0.75):"
            ).strip()
            try:
                denoise = float(denoise_raw) if denoise_raw else 0.75
                denoise = max(0.1, min(1.0, denoise))
            except ValueError:
                denoise = 0.75
            try:
                client.restyle_concept_art(
                    name, idx, positive,
                    negative=negative or "blurry, low quality, text, watermark, deformed",
                    denoise=denoise,
                    source_version=source_version,
                )
            except ConflictError as e:
                ui.inform(f"Restyle unavailable: {e.detail}")
                continue
            except QuickymeshAPIError as e:
                ui.inform(f"Restyle failed: {e.detail}")
                continue
            outcome, new_state = _wait_for_concept_art_ready(client, ui, name)
            if outcome == "ready" and new_state is not None:
                state = new_state
                sheet_shown = False
            else:
                return "back"

        elif action == "menu":
            return "back"
        elif action == "quit":
            return "quit"
        else:
            ui.inform(f"Unknown action '{action}'.")


def _submit_approved_for_3d(
    client, ui, name: str, state: dict, approved: set[int],
) -> bool:
    """
    Ask the user whether to submit the currently-approved concept arts for
    3D mesh generation.  Returns True if anything was submitted.
    """
    if not approved:
        return False
    indices_sorted = sorted(approved)
    indices_str = ", ".join(str(i + 1) for i in indices_sorted)
    count = len(indices_sorted)
    noun = "image" if count == 1 else "images"
    choice = ui.ask(
        f"Submit {count} approved {noun} (index {indices_str}) for 3D mesh generation? "
        "(y/n, Enter=yes):"
    ).strip().lower()
    if choice == "n":
        ui.inform("Skipped 3D mesh generation.")
        return False

    num_polys = state.get("num_polys")
    symmetrize = bool(state.get("symmetrize", False))
    symmetry_axis = state.get("symmetry_axis", "x-")

    any_submitted = False
    for idx in indices_sorted:
        try:
            result = client.create_3d_pipeline_from_ref(
                source_2d_pipeline=name,
                concept_art_index=idx,
                num_polys=num_polys,
                symmetrize=symmetrize,
                symmetry_axis=symmetry_axis,
            )
        except ConflictError as e:
            ui.inform(f"  Skipping image {idx + 1}: {e.detail}")
            continue
        except QuickymeshAPIError as e:
            ui.inform(f"  Skipping image {idx + 1}: {e.detail}")
            continue
        any_submitted = True
        ui.inform(f"  3D pipeline '{result.get('name')}' queued for mesh generation.")
    return any_submitted


# ---------------------------------------------------------------------------
# 3D mesh approval
# ---------------------------------------------------------------------------


def _handle_3d_approval(client, ui, name: str) -> bool:
    """
    Interactive 3D mesh review loop.  Returns True if the user asked to quit.
    """
    sheet_shown = False
    while True:
        try:
            state = client.get_3d_pipeline(name)
        except NotFoundError:
            ui.inform(f"Pipeline '{name}' no longer exists.")
            return False
        except QuickymeshAPIError as e:
            ui.inform(f"Server error: {e.detail}")
            return False

        status = state.get("status", "")
        if status not in (_S3D_AWAITING, _S3D_IDLE):
            ui.inform(f"Pipeline '{name}' is not awaiting approval (status: {status}).")
            return False

        if not sheet_shown:
            try:
                sheet_path = client.get_3d_review_sheet(name)
                ui.show_image(sheet_path)
            except NotFoundError:
                ui.inform("(no review sheet yet)")
            except QmConnectionError as e:
                ui.inform(f"Connection error: {e}")
                return False
            sheet_shown = True

        mesh = state.get("textured_mesh_path") or state.get("mesh_path") or "(none)"
        src = state.get("source_2d_pipeline")
        source_display = (
            f"  Source 2D pipeline: {src} (CA {state.get('source_concept_art_index')})\n"
            if src else ""
        )
        ui.inform(
            f"\n{'─' * 50}\n"
            f"3D mesh review for '{name}'\n"
            f"  Status: {status}\n"
            + source_display +
            f"  Mesh: {mesh}\n"
            f"  Exports so far: {state.get('export_version', 0)}\n"
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
            try:
                client.approve_3d_mesh(name)
            except ConflictError as e:
                ui.inform(f"Approve rejected: {e.detail}")
                continue
            except QuickymeshAPIError as e:
                ui.inform(f"Server error: {e.detail}")
                continue
            # Mirror old behaviour: hide after successful export
            try:
                client.patch_3d_pipeline(name, hidden=True)
            except QuickymeshAPIError:
                pass
            ui.inform(
                f"Mesh exported.  Pipeline '{name}' hidden — "
                "use 'Unhide' from the main menu to access it again."
            )
            return False

        elif action == "regenerate":
            cur_polys = state.get("num_polys")
            new_polys_raw = ui.ask(
                f"New polygon count (current: {cur_polys}, Enter to keep):"
            ).strip()
            kwargs: dict[str, Any] = {}
            if new_polys_raw.isdigit() and int(new_polys_raw) > 0:
                kwargs["num_polys"] = int(new_polys_raw)

            cur_sym = bool(state.get("symmetrize", False))
            cur_axis = state.get("symmetry_axis") or "x-"
            cur_desc = f"{cur_axis}" if cur_sym else "off"
            sym_raw = ui.ask(
                f"Symmetrize (current: {cur_desc})?\n"
                f"  Options: {', '.join(_SYM_OPTIONS)}, 'off', or Enter to keep:"
            ).strip().lower()
            if sym_raw == "off":
                kwargs["symmetrize"] = False
            elif sym_raw in _SYM_OPTIONS:
                kwargs["symmetrize"] = True
                kwargs["symmetry_axis"] = sym_raw
            # else: blank → keep current (omit from kwargs)
            try:
                client.reject_3d_mesh(name, **kwargs)
            except ConflictError as e:
                ui.inform(f"Regenerate rejected: {e.detail}")
                continue
            except QuickymeshAPIError as e:
                ui.inform(f"Server error: {e.detail}")
                continue
            ui.inform(f"Mesh generation re-queued for '{name}'.")
            return False

        elif action == "menu":
            return False
        elif action == "quit":
            return True
        else:
            ui.inform("Unknown action.  Use: approve, regenerate, menu, quit")


# ---------------------------------------------------------------------------
# Status snapshot + watch mode
# ---------------------------------------------------------------------------


def _pipeline_sig(client, name: str, ptype: str) -> tuple | None:
    """
    Return (status, in_flight_task_types, failed_task_ids) for one pipeline
    — used by watch mode to detect genuine changes.
    """
    try:
        state = (
            client.get_pipeline_or_none(name) if ptype == "2d"
            else client.get_3d_pipeline_or_none(name)
        )
        if state is None:
            return None
        tasks = (
            client.get_pipeline_tasks(name) if ptype == "2d"
            else client.get_3d_pipeline_tasks(name)
        )
    except QuickymeshAPIError:
        return None
    in_flight = [t for t in tasks if t.get("status") in ("pending", "running")]
    failed = [t for t in tasks if t.get("status") == "failed" and t.get("error") != "cancelled"]
    return (
        state.get("status", "unknown"),
        tuple(sorted(t["task_type"] for t in in_flight)),
        tuple(sorted(str(t["id"]) for t in failed)),
        tuple((t["task_type"], t.get("error") or "") for t in failed),
    )


def _print_full_status(client, ui) -> None:
    try:
        status = client.get_status()
    except QuickymeshAPIError as e:
        ui.inform(f"Could not fetch status: {e.detail}")
        return

    lines = ["=== Workers ==="]
    workers = status.get("workers") or []
    if not workers:
        lines.append("  (none reported)")
    else:
        for w in workers:
            state_str = "running" if w.get("alive") else "stopped"
            lines.append(f"  {w.get('name', '?')}: {state_str}")

    p2, p3 = _pipeline_lists(client)
    lines.append("")
    lines.append("=== 2D Pipelines ===")
    if not p2:
        lines.append("  (none)")
    else:
        for item in p2:
            _append_status_line(client, lines, item["name"], "2d")
    lines.append("")
    lines.append("=== 3D Pipelines ===")
    if not p3:
        lines.append("  (none)")
    else:
        for item in p3:
            _append_status_line(client, lines, item["name"], "3d")
    ui.inform("\n".join(lines))


def _append_status_line(client, lines: list[str], name: str, ptype: str) -> None:
    try:
        state = (
            client.get_pipeline_or_none(name) if ptype == "2d"
            else client.get_3d_pipeline_or_none(name)
        )
        tasks = (
            client.get_pipeline_tasks(name) if ptype == "2d"
            else client.get_3d_pipeline_tasks(name)
        )
    except QuickymeshAPIError:
        lines.append(f"  {name}: (unavailable)")
        return
    if state is None:
        lines.append(f"  {name}: unknown")
        return
    in_flight = [t for t in tasks if t.get("status") in ("pending", "running")]
    failed = [t for t in tasks if t.get("status") == "failed" and t.get("error") != "cancelled"]
    hidden_tag = "  [hidden]" if state.get("hidden") else ""
    task_info = f"  [{len(in_flight)} queued]" if in_flight else ""
    fail_info = f"  [{len(failed)} FAILED]" if failed else ""
    lines.append(f"  {name}: {state.get('status')}{hidden_tag}{task_info}{fail_info}")
    for ft in failed:
        lines.append(f"    ! {ft.get('task_type')}: {ft.get('error')}")


def _watch_mode(
    client, ui, cfg, session,
    *, _tick: float = _WATCH_TICK, _status_interval: float = _WATCH_STATUS_INTERVAL,
) -> None:
    ui.inform(
        "Watching for status updates.  "
        f"Press '{_WATCH_EXIT_CMD}' to return to the menu.\n"
    )
    _print_full_status(client, ui)

    last_sigs: dict[tuple[str, str], tuple | None] = {}

    def _seed() -> None:
        last_sigs.clear()
        p2, p3 = _pipeline_lists(client)
        for item in p2:
            last_sigs[(item["name"], "2d")] = _pipeline_sig(client, item["name"], "2d")
        for item in p3:
            last_sigs[(item["name"], "3d")] = _pipeline_sig(client, item["name"], "3d")

    _seed()
    last_status_print = time.time()

    # Flush residual stdin from the menu prompt before entering cbreak so
    # stale keystrokes don't get misread as the exit key on the first tick.
    _flush_stdin()
    _enter_cbreak()
    try:
        while True:
            time.sleep(_tick)

            # Pending approvals
            waiting = _all_needing_attention(client, session)
            if waiting:
                _exit_cbreak()
                quit_requested = False
                for name, ptype in waiting:
                    ts = time.strftime("%H:%M:%S")
                    ui.inform(f"\n[{ts}] !!! REVIEW NEEDED: {name} !!!")
                    _flush_stdin()
                    if ptype == "2d":
                        state_2d = client.get_pipeline_or_none(name)
                        if state_2d is None:
                            continue
                        result = _run_concept_art_review(client, ui, cfg, name, state_2d)
                        if result == "quit":
                            quit_requested = True
                            break
                        # Dismiss in all non-quit cases (approved, back,
                        # interrupted wait, etc.) — the user has seen the
                        # pipeline this session and chose their action.
                        # Re-entry is available via manage pipeline; without
                        # this the busy-state prompt would loop every tick.
                        session.dismiss(name)
                    else:
                        if _handle_3d_approval(client, ui, name):
                            quit_requested = True
                            break
                        # Dismiss in all non-quit cases.  If the user approved
                        # or rejected, the state will have moved out of
                        # AWAITING_APPROVAL anyway; if they went back to menu,
                        # dismissing prevents the watch loop from re-triggering
                        # the same review prompt every tick.
                        session.dismiss(name)
                if quit_requested:
                    return
                ui.inform(
                    f"\nReviews handled.  Continuing to watch "
                    f"(press '{_WATCH_EXIT_CMD}' to return to menu).\n"
                )
                _seed()
                last_status_print = time.time()
                _enter_cbreak()
                continue

            # Exit key
            ch = _try_read_char()
            if ch is not None and ch.lower() in (_WATCH_EXIT_CMD, '\x1b', '\x03'):
                ui.inform("\nReturning to menu.")
                return

            # Periodic diff
            now = time.time()
            if now - last_status_print >= _status_interval:
                last_status_print = now
                _print_watch_diffs(client, ui, last_sigs)
    finally:
        _exit_cbreak()


def _print_watch_diffs(client, ui, last_sigs: dict) -> None:
    p2, p3 = _pipeline_lists(client)
    current_keys: set[tuple[str, str]] = set()
    for item in p2:
        current_keys.add((item["name"], "2d"))
    for item in p3:
        current_keys.add((item["name"], "3d"))
    for key in list(last_sigs.keys()):
        if key not in current_keys:
            last_sigs.pop(key, None)

    for name, ptype in current_keys:
        sig = _pipeline_sig(client, name, ptype)
        if sig is None:
            continue
        prev = last_sigs.get((name, ptype))
        if prev == sig:
            continue
        last_sigs[(name, ptype)] = sig
        status, in_flight, _failed_ids, failed_info = sig
        ts = time.strftime("%H:%M:%S")
        if in_flight:
            detail = f" [{', '.join(in_flight)}]"
        elif failed_info:
            detail = f" [FAILED: {', '.join(t for t, _ in failed_info)}]"
        else:
            detail = ""
        ui.inform(f"[{ts}] {name}: {status}{detail}")
        for task_type, err in failed_info:
            ui.inform(f"[{ts}]   ! {task_type} error: {err}")


# ---------------------------------------------------------------------------
# Retry failed tasks
# ---------------------------------------------------------------------------


def _retry_failed(client, ui) -> None:
    try:
        pipelines = client.get_pipelines_with_failures()
    except QuickymeshAPIError as e:
        ui.inform(f"Server error: {e.detail}")
        return
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

    def _retry_one(name: str) -> int:
        # Try 2D first, then 3D.  Both endpoints 404 if the name doesn't
        # belong to their namespace.
        try:
            return client.retry_pipeline(name)
        except NotFoundError:
            pass
        try:
            return client.retry_3d_pipeline(name)
        except NotFoundError:
            return 0

    if choice == "a":
        total = 0
        for name in pipelines:
            try:
                total += _retry_one(name)
            except QuickymeshAPIError as e:
                ui.inform(f"  {name}: {e.detail}")
        ui.inform(f"Reset {total} failed task(s) across all pipelines.")
    elif choice.isdigit() and 1 <= int(choice) <= len(pipelines):
        name = pipelines[int(choice) - 1]
        try:
            count = _retry_one(name)
        except QuickymeshAPIError as e:
            ui.inform(f"Server error: {e.detail}")
            return
        ui.inform(f"Reset {count} failed task(s) for '{name}'.")
    else:
        ui.inform("Cancelled.")


# ---------------------------------------------------------------------------
# Cross-platform single-key stdin helpers (copied from old cli_main.py)
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
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _flush_stdin() -> None:
    if os.name == "nt":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
        return
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass
