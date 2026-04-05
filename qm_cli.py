"""
quickymesh CLI client (Phase 1).

Talks to the quickymesh API server over HTTP instead of calling pipeline
functions directly.  The user experience is identical to the original CLI —
same prompts, same review flows — but all state reads/writes go through
the REST API.

Usage
-----
    python qm_cli.py                           # connects to localhost:8000
    python qm_cli.py --server http://10.0.0.2:8000 --key my-api-key

Configuration (in ~/.qm_config or env vars)
-------------------------------------------
    QM_SERVER   default server URL
    QM_API_KEY  API key
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

log = logging.getLogger(__name__)

_CONFIG_FILE = Path.home() / ".qm_config"
_DEFAULT_SERVER = "http://localhost:8000"
_API_PREFIX = "/api/v1"

_IDLE_MENU = """\
--- quickymesh ---
[n] Start a new pipeline
[e] Edit a pipeline
[p] Pause / Resume / Cancel a pipeline
[s] Status (workers + pipelines)
[w] Watch for approvals
[r] Retry failed tasks
[q] Quit"""

_IDLE_OPT_MAP = {"1": "n", "2": "e", "3": "p", "4": "s", "5": "w", "6": "r", "7": "q"}

_WATCH_TICK = 3.0


# ---------------------------------------------------------------------------
# HTTP client wrapper
# ---------------------------------------------------------------------------


class QMClient:
    """Thin wrapper around httpx.Client for the quickymesh API."""

    def __init__(self, server: str, api_key: str) -> None:
        self._base = server.rstrip("/") + _API_PREFIX
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300.0,  # generation calls can take a while
        )

    def get(self, path: str, **kwargs) -> dict | list | bytes:
        resp = self._client.get(self._base + path, **kwargs)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "json" in ct:
            return resp.json()
        return resp.content

    def post(self, path: str, json_body: Any = None, **kwargs) -> dict:
        resp = self._client.post(self._base + path, json=json_body, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def patch(self, path: str, json_body: Any = None) -> dict:
        resp = self._client.patch(self._base + path, json=json_body)
        resp.raise_for_status()
        return resp.json()

    def delete(self, path: str) -> dict:
        resp = self._client.delete(self._base + path)
        resp.raise_for_status()
        return resp.json()

    def download(self, path: str) -> bytes:
        resp = self._client.get(self._base + path)
        resp.raise_for_status()
        return resp.content

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _ask(prompt: str) -> str:
    try:
        return input(f"{prompt}\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _inform(msg: str) -> None:
    print(msg, flush=True)


def _show_image(image_bytes: bytes) -> None:
    """Save bytes to a temp file and open with the OS default viewer."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            img.show(title="quickymesh review")
    except Exception as exc:
        log.warning("Could not display image: %s", exc)



_WATCH_GEN_TICK = 2.0   # seconds between polls while watching generation


def _watch_generating(
    client: QMClient,
    name: str,
    ready_status: str = "concept_art_review",
    timeout: float = 600.0,
) -> str:
    """
    Live watch loop used while waiting for async generation to complete.

    Prints a timestamped status line whenever per-image statuses change.
    Exits when the pipeline reaches `ready_status` (returns "ready"),
    the user presses 'q' (returns "quit"), or timeout elapses (returns
    "timeout").

    Works like the idle watch mode: non-blocking keypress check each tick.
    """
    _inform(
        "\nWatching generation progress... "
        f"(press 'q' + Enter to return to menu)\n"
    )

    # Enter cbreak so 'q' can be detected without Enter on POSIX
    _enter_cbreak()
    last_sig: str | None = None
    deadline = time.time() + timeout

    try:
        while time.time() < deadline:
            # Non-blocking keypress check
            ch = _try_read_char()
            if ch is not None and ch.lower() in ("q", "\x1b", "\x03"):
                _inform("\nReturning to menu.")
                return "quit"

            state = client.get(f"/pipelines/{name}")

            # Build a compact status signature from per-item statuses
            pipeline_status = state["status"]
            arts = state.get("concept_arts", [])
            art_sig = "|".join(f"{a['index']+1}:{a['status']}" for a in arts)
            mesh_sig = "|".join(f"{m['sub_name']}:{m['status']}" for m in state.get("meshes", []))
            sig = f"{pipeline_status}:{art_sig}:{mesh_sig}"

            if sig != last_sig:
                ts = time.strftime("%H:%M:%S")
                # Show per-item breakdown when available
                if arts:
                    counts: dict[str, int] = {}
                    for a in arts:
                        counts[a["status"]] = counts.get(a["status"], 0) + 1
                    counts_str = "  ".join(f"{s}×{n}" for s, n in counts.items())
                    _inform(f"[{ts}] {name}: {pipeline_status}  ({counts_str})")
                else:
                    meshes = state.get("meshes", [])
                    if meshes:
                        mcounts: dict[str, int] = {}
                        for m in meshes:
                            mcounts[m["status"]] = mcounts.get(m["status"], 0) + 1
                        mcounts_str = "  ".join(f"{s}×{n}" for s, n in mcounts.items())
                        _inform(f"[{ts}] {name}: {pipeline_status}  ({mcounts_str})")
                    else:
                        _inform(f"[{ts}] {name}: {pipeline_status}")
                last_sig = sig

            if pipeline_status == ready_status:
                _inform(f"\n[{time.strftime('%H:%M:%S')}] Ready for review!")
                return "ready"

            time.sleep(_WATCH_GEN_TICK)
    finally:
        _exit_cbreak()

    _inform("Timed out waiting for generation.")
    return "timeout"


# ---------------------------------------------------------------------------
# Cross-platform single-keypress helpers (mirrors the standalone CLI)
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="quickymesh CLI client")
    parser.add_argument("--server", default=None, help="API server URL")
    parser.add_argument("--key", default=None, help="API key")
    args = parser.parse_args()

    server, api_key = _resolve_config(args.server, args.key)
    if not api_key:
        _inform("No API key.  Set QM_API_KEY, pass --key, or add to ~/.qm_config")
        sys.exit(1)

    client = QMClient(server, api_key)
    try:
        _inform(f"Connected to {server}")
        _main_loop(client)
    except KeyboardInterrupt:
        _inform("\nInterrupted.")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            _inform(
                "\nError: 401 Unauthorized — API key rejected by the server.\n"
                "Make sure the server was started with the same API_KEY, or that\n"
                "your key matches an entry in the server's users.yaml."
            )
        else:
            _inform(f"\nHTTP error {exc.response.status_code}: {exc.response.text[:200]}")
        log.error("Unhandled HTTP error: %s", exc)
        sys.exit(1)
    except httpx.ConnectError:
        _inform(f"\nError: Could not connect to {server}\nIs the server running?")
        sys.exit(1)
    finally:
        client.close()


def _resolve_config(server_arg: str | None, key_arg: str | None) -> tuple[str, str]:
    """Resolve server URL and API key from args → env → config file."""
    cfg: dict = {}
    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text())
        except Exception:
            pass

    server = server_arg or os.environ.get("QM_SERVER") or cfg.get("server") or _DEFAULT_SERVER
    api_key = key_arg or os.environ.get("QM_API_KEY") or cfg.get("api_key") or ""
    return server, api_key


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _main_loop(client: QMClient) -> None:
    _inform("quickymesh started.")
    while True:
        priority = _highest_priority_pipeline(client)
        if priority:
            quit_req = _handle_priority_pipeline(client, priority)
            if quit_req:
                break
        else:
            action = _idle_menu(client)
            if action == "quit":
                break


def _highest_priority_pipeline(client: QMClient) -> str | None:
    pipelines = client.get("/pipelines")
    for target_status in ("concept_art_review", "mesh_review"):
        for p in pipelines:
            if p["status"] == target_status:
                return p["name"]
    return None


def _pipelines_needing_attention(client: QMClient) -> list[str]:
    pipelines = client.get("/pipelines")
    review_statuses = {"concept_art_review", "mesh_review"}
    return [p["name"] for p in pipelines if p["status"] in review_statuses]


# ---------------------------------------------------------------------------
# Idle menu
# ---------------------------------------------------------------------------


def _idle_menu(client: QMClient) -> str:
    waiting = _pipelines_needing_attention(client)
    approval_note = ""
    if waiting:
        names = ", ".join(f"'{n}'" for n in waiting)
        noun = "Pipeline" if len(waiting) == 1 else "Pipelines"
        verb = "is" if len(waiting) == 1 else "are"
        approval_note = f"\n[!] {noun} {names} {verb} waiting for approval — press Enter to review.\n"

    choice = _ask(_IDLE_MENU + approval_note).lower()
    choice = _IDLE_OPT_MAP.get(choice, choice)

    if not choice:
        return "check"
    if choice == "q":
        return "quit"
    if choice == "n":
        _start_new_pipeline(client)
    elif choice == "e":
        _edit_pipeline(client)
    elif choice == "p":
        _manage_pipeline(client)
    elif choice == "s":
        _show_status(client)
    elif choice == "w":
        if _watch_mode(client):
            return "quit"
    elif choice == "r":
        _retry_failed(client)
    return "check"


# ---------------------------------------------------------------------------
# Start new pipeline
# ---------------------------------------------------------------------------


def _start_new_pipeline(client: QMClient) -> None:
    name = _ask("Pipeline name (no spaces):")
    if not name:
        _inform("Cancelled.")
        return

    # Backend choice (no server-side prefs API yet — local only)
    _inform(
        "Choose concept art generator:\n"
        "  1. Gemini Flash  — requires API key, small cost per image, "
        "very accurate results, can use an existing image as a base\n"
        "  2. FLUX.1 [dev]  — runs locally via ComfyUI, ~25 GB models, "
        "~16 GB VRAM, less accurate"
    )
    choice = _ask("Enter 1 or 2 (Enter for Gemini):").strip()
    backend = "flux" if choice == "2" else "gemini"

    # Input image (Gemini only)
    input_image_path: str | None = None
    if backend == "gemini":
        while True:
            img_raw = _ask(
                "Base concept art on an existing image? "
                "Enter a local path, or leave blank:"
            )
            if not img_raw:
                break
            p = Path(img_raw)
            if p.exists():
                input_image_path = str(p.resolve())
                _inform(f"Image found: {input_image_path}")
                break
            _inform(f"Image not found: {img_raw}\nPlease try again, or press Enter to skip.")

    # Description
    _BACKGROUND_SUFFIX = (
        "3/4 isometric view, three quarters lighting, "
        "plain contrasting background, 1:1 ratio"
    )
    if input_image_path:
        description = _ask("How do you want to change/adapt this image?")
    else:
        description = _ask(
            "Describe the 3-D object to generate:\n"
            f'  (The suffix "{_BACKGROUND_SUFFIX}" will be appended automatically.\n'
            "   Trellis needs these qualities to correctly reconstruct 3-D geometry.)"
        )
    if not description:
        _inform("Cancelled.")
        return

    # Poly count
    _DEFAULT_POLYS = 8000
    polys_str = _ask(f"Target polygon count (Enter for default: {_DEFAULT_POLYS}):")
    num_polys = int(polys_str) if polys_str.isdigit() else None

    # Symmetry
    _SYM_OPTIONS = ["x-", "x+", "y-", "y+", "z-", "z+"]
    sym_raw = _ask(
        f"Symmetrize mesh? Options: {', '.join(_SYM_OPTIONS)}\n"
        "Enter an axis to enable, or leave blank to skip (Enter defaults to x-):"
    ).lower()
    if sym_raw in _SYM_OPTIONS:
        symmetrize, symmetry_axis = True, sym_raw
    elif not sym_raw:
        symmetrize, symmetry_axis = True, "x-"
    else:
        symmetrize, symmetry_axis = False, "x-"

    client.post("/pipelines", {
        "name": name,
        "description": description,
        "num_polys": num_polys,
        "input_image_path": input_image_path,
        "symmetrize": symmetrize,
        "symmetry_axis": symmetry_axis,
        "concept_art_backend": backend,
    })
    _inform(f"Pipeline '{name}' started.  Concept art generation queued.")


# ---------------------------------------------------------------------------
# Edit pipeline
# ---------------------------------------------------------------------------


def _edit_pipeline(client: QMClient) -> None:
    pipelines = client.get("/pipelines")
    editable_statuses = {"initializing", "concept_art_generating", "concept_art_review"}
    editable = [p for p in pipelines if p["status"] in editable_statuses]
    if not editable:
        _inform("No editable pipelines (editing only allowed before mesh generation).")
        return

    _inform("Editable pipelines:")
    for i, p in enumerate(editable, 1):
        _inform(f"  {i}. {p['name']}  [{p['status']}]")
    choice = _ask("Enter pipeline number (Enter to cancel):")
    if not choice or not choice.isdigit() or not (1 <= int(choice) <= len(editable)):
        _inform("Cancelled.")
        return

    p = editable[int(choice) - 1]
    name = p["name"]
    state = client.get(f"/pipelines/{name}")

    _inform(f'Current description: "{state["description"]}"')
    new_desc = _ask("New description (Enter to keep):")

    _inform(f"Current polygon target: {state['num_polys']}")
    poly_raw = _ask("New polygon count (Enter to keep):")

    sym_display = f"{'on' if state['symmetrize'] else 'off'}, axis={state['symmetry_axis']}"
    _inform(f"Current symmetry: {sym_display}")
    sym_raw = _ask("Enable symmetrize? (y/n, Enter to keep):").lower()

    patch: dict = {}
    if new_desc:
        patch["description"] = new_desc
    if poly_raw.isdigit() and int(poly_raw) > 0:
        patch["num_polys"] = int(poly_raw)
    if sym_raw == "y":
        patch["symmetrize"] = True
        axis_options = ["x-", "x+", "y-", "y+", "z-", "z+"]
        axis_raw = _ask(f"Axis ({', '.join(axis_options)}, Enter to keep):").strip()
        if axis_raw in axis_options:
            patch["symmetry_axis"] = axis_raw
    elif sym_raw == "n":
        patch["symmetrize"] = False

    if patch:
        client.patch(f"/pipelines/{name}", patch)
        _inform(f"Pipeline '{name}' updated.")
    else:
        _inform("No changes.")


# ---------------------------------------------------------------------------
# Manage (pause / resume / cancel)
# ---------------------------------------------------------------------------


def _manage_pipeline(client: QMClient) -> None:
    pipelines = client.get("/pipelines")
    if not pipelines:
        _inform("No pipelines.")
        return

    _inform("Pipelines:")
    for i, p in enumerate(pipelines, 1):
        _inform(f"  {i}. {p['name']}  [{p['status']}]")
    choice = _ask("Enter pipeline number (Enter to cancel):")
    if not choice or not choice.isdigit() or not (1 <= int(choice) <= len(pipelines)):
        _inform("Cancelled.")
        return

    p = pipelines[int(choice) - 1]
    name = p["name"]

    if p["status"] == "paused":
        action = _ask(f"'{name}' is paused. [r]esume or [c]ancel?").lower()
        if action == "r":
            client.post(f"/pipelines/{name}/resume")
            _inform(f"Pipeline '{name}' resumed.")
        elif action == "c":
            client.delete(f"/pipelines/{name}")
            _inform(f"Pipeline '{name}' cancelled.")
    else:
        action = _ask(
            f"'{name}' is {p['status']}. [p]ause or [c]ancel?"
        ).lower()
        if action == "p":
            client.post(f"/pipelines/{name}/pause")
            _inform(f"Pipeline '{name}' paused.")
        elif action == "c":
            client.delete(f"/pipelines/{name}")
            _inform(f"Pipeline '{name}' cancelled.")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _get_output_root(client: QMClient) -> str:
    """Return the server's output_root path (container path if using Docker)."""
    try:
        return client.get("/status").get("output_root", "pipeline_root")
    except Exception:
        return "pipeline_root"


def _show_status(client: QMClient) -> None:
    status = client.get("/status")
    lines = ["=== Workers ==="]
    for w in status["workers"]:
        lines.append(f"  {w['name']}: {'running' if w['alive'] else 'STOPPED'}")
    lines.append("\n=== Pipelines ===")
    if not status["pipelines"]:
        lines.append("  (none)")
    for p in status["pipelines"]:
        queued = f"  [{p['queued_tasks']} queued]" if p["queued_tasks"] else ""
        failed = f"  [{p['failed_tasks']} FAILED]" if p["failed_tasks"] else ""
        lines.append(f"  {p['name']}: {p['status']}{queued}{failed}")
    output_root = status.get("output_root", "")
    if output_root:
        lines.append(f"\n=== Files ===")
        lines.append(f"  Pipeline root (container) : {output_root}")
        lines.append(f"  Final assets  (container) : {output_root}/final_game_ready_assets/")
        lines.append(
            "  (If running via Docker, host path = QUICKYMESH_PIPELINE_ROOT in docker/.env,\n"
            "   defaulting to pipeline_root/ in the repo root.)"
        )
    _inform("\n".join(lines))


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


def _watch_mode(client: QMClient) -> bool:
    """Watch for approvals.  Returns True if the user asked to quit."""
    _inform(
        "Watch mode active.  Polling for updates every "
        f"{int(_WATCH_TICK)}s.\nPress Ctrl-C to return to the menu.\n"
    )
    last_sigs: dict = {}
    try:
        while True:
            waiting = _pipelines_needing_attention(client)
            if waiting:
                _inform(f"\n[{time.strftime('%H:%M:%S')}] !!! APPROVAL NEEDED: {', '.join(waiting)} !!!")
                for name in waiting:
                    quit_req = _handle_priority_pipeline(client, name)
                    if quit_req:
                        return True
                _inform("\nApprovals complete.  Returning to menu.")
                return False

            # Print status diffs
            pipelines = client.get("/pipelines")
            for p in pipelines:
                sig = p["status"]
                if last_sigs.get(p["name"]) != sig:
                    ts = time.strftime("%H:%M:%S")
                    _inform(f"[{ts}] {p['name']}: {sig}")
                    last_sigs[p["name"]] = sig

            time.sleep(_WATCH_TICK)
    except KeyboardInterrupt:
        _inform("\nReturning to menu.")
    return False


# ---------------------------------------------------------------------------
# Retry failed
# ---------------------------------------------------------------------------


def _retry_failed(client: QMClient) -> None:
    status = client.get("/status")
    failed_pipes = [p["name"] for p in status["pipelines"] if p["failed_tasks"] > 0]
    if not failed_pipes:
        _inform("No pipelines have failed tasks.")
        return
    _inform("Pipelines with failed tasks:")
    for i, name in enumerate(failed_pipes, 1):
        _inform(f"  {i}. {name}")
    _inform("  a. All pipelines")
    choice = _ask("Enter pipeline number (or 'a', Enter to cancel):").lower()
    if not choice:
        return
    if choice == "a":
        total = 0
        for name in failed_pipes:
            result = client.post(f"/pipelines/{name}/retry")
            total += result.get("tasks_reset", 0)
        _inform(f"Reset {total} failed task(s) across all pipelines.")
    elif choice.isdigit() and 1 <= int(choice) <= len(failed_pipes):
        name = failed_pipes[int(choice) - 1]
        result = client.post(f"/pipelines/{name}/retry")
        _inform(f"Reset {result.get('tasks_reset', 0)} failed task(s) for '{name}'.")
    else:
        _inform("Cancelled.")


# ---------------------------------------------------------------------------
# Review dispatcher
# ---------------------------------------------------------------------------


def _handle_priority_pipeline(client: QMClient, name: str) -> bool:
    """Handle one priority pipeline review. Returns True if user asked to quit."""
    state = client.get(f"/pipelines/{name}")
    if state["status"] == "concept_art_review":
        result = _review_concept_art(client, name, state)
        if result == "approved":
            _inform(f"[{name}] Concept art approved.  Mesh generation queued.")
        elif result == "cancelled":
            _inform(f"[{name}] Pipeline cancelled.")
        elif result == "quit":
            return True
    elif state["status"] == "mesh_review":
        result = _review_meshes(client, name, state)
        if result == "approved":
            output_root = _get_output_root(client)
            _inform(
                f"[{name}] Meshes approved and exported.\n"
                f"  Final assets: {output_root}/final_game_ready_assets/\n"
                f"  (Docker: host path = pipeline_root/final_game_ready_assets/ in repo root)"
            )
        elif result == "all_rejected":
            _inform(f"[{name}] All meshes rejected — mesh generation re-queued.")
        elif result == "cancelled":
            _inform(f"[{name}] Pipeline cancelled.")
        elif result == "quit":
            return True
    return False


# ---------------------------------------------------------------------------
# Concept art review
# ---------------------------------------------------------------------------


def _review_concept_art(client: QMClient, name: str, state: dict) -> str:
    """
    Interactive concept art review loop.
    Returns: "approved" | "cancelled" | "quit"
    """
    sheet_shown = state.get("concept_art_sheet_shown", False)

    while True:
        # Refresh state at top of each loop iteration
        state = client.get(f"/pipelines/{name}")

        # If images are still generating, show a live watch instead of blocking
        if state["status"] == "concept_art_generating":
            result = _watch_generating(client, name, ready_status="concept_art_review")
            if result == "quit":
                return "quit"
            if result == "timeout":
                return "cancelled"
            state = client.get(f"/pipelines/{name}")

        # Show review sheet once per batch of images
        if not sheet_shown:
            sheet_shown = True
            output_root = _get_output_root(client)
            _inform(
                f"\nConcept art images are in:\n"
                f"  {output_root}/uncompleted_pipelines/{name}/concept_arts/\n"
                f"  (If using Docker, host path = pipeline_root/uncompleted_pipelines/{name}/concept_arts/ in repo root)"
            )
            try:
                sheet_bytes = client.download(f"/pipelines/{name}/concept_art/sheet")
                _show_image(sheet_bytes)
            except Exception as exc:
                _inform(f"(Could not display review sheet: {exc})")

            supports_modify = any(
                ca.get("status") == "ready"
                for ca in state.get("concept_arts", [])
            )
            has_restyle = True  # API exposes it; server returns 409 if not configured
            modify_line = "  modify <idx>           — modify one image via Gemini\n"
            restyle_line = "  restyle <idx>          — restyle image shape/silhouette via ControlNet\n"
            _inform(
                f"\nConcept art review for '{name}'\n"
                "Actions:\n"
                "  approve <indices>      — e.g. 'approve 1 3' (1-based)\n"
                "  regenerate <idx|all>   — e.g. 'regenerate 2 4' or 'regenerate all'\n"
                + modify_line
                + restyle_line +
                "  cancel                 — cancel this pipeline\n"
                "  quit                   — exit the program\n"
            )
        for item in state.get("concept_arts", []):
            _inform(f"  [{item['index'] + 1}] {item['status']}")

        raw = _ask("Enter action").lower()
        tokens = raw.split()
        if not tokens:
            continue
        action = tokens[0]

        if action == "approve":
            if len(tokens) < 2:
                _inform("Specify which images: e.g. 'approve 1 3'")
                continue
            try:
                indices = [int(t) - 1 for t in tokens[1:]]
            except ValueError:
                _inform("Invalid indices.")
                continue
            client.post(f"/pipelines/{name}/concept_art/approve", {"indices": indices})
            return "approved"

        elif action == "regenerate":
            if len(tokens) < 2:
                _inform("Specify: 'regenerate 2' or 'regenerate all'")
                continue
            if tokens[1] == "all":
                current_desc = state.get("description", "")
                _inform(f"Current description: {current_desc}")
                new_desc = _ask("New description (Enter to keep):") or None
                client.post(f"/pipelines/{name}/concept_art/regenerate", {
                    "description_override": new_desc,
                })
            else:
                try:
                    indices = [int(t) - 1 for t in tokens[1:]]
                except ValueError:
                    _inform("Invalid indices.")
                    continue
                client.post(f"/pipelines/{name}/concept_art/regenerate", {"indices": indices})
            _inform("Regeneration queued.")
            sheet_shown = False
            result = _watch_generating(client, name, ready_status="concept_art_review")
            if result == "quit":
                return "quit"
            if result == "timeout":
                _inform("Timed out waiting for regeneration.")
            state = client.get(f"/pipelines/{name}")

        elif action == "modify":
            if len(tokens) < 2 or not tokens[1].isdigit():
                _inform("Usage: modify <index>  e.g. 'modify 3'")
                continue
            idx = int(tokens[1]) - 1
            instruction = _ask(f"Describe the change to make to image {idx + 1}:")
            if not instruction:
                _inform("Cancelled.")
                continue
            try:
                client.post(f"/pipelines/{name}/concept_art/modify", {
                    "index": idx, "instruction": instruction,
                })
                sheet_shown = False
            except httpx.HTTPStatusError as exc:
                _inform(f"Modify failed: {exc.response.json().get('detail', str(exc))}")

        elif action == "restyle":
            if len(tokens) < 2 or not tokens[1].isdigit():
                _inform("Usage: restyle <index>  e.g. 'restyle 2'")
                continue
            idx = int(tokens[1]) - 1
            positive = _ask(
                f"Positive prompt for image {idx + 1}\n"
                "  e.g. 'zerg, biomechanical, dark chitin, alien, glossy':"
            )
            if not positive:
                _inform("Cancelled.")
                continue
            negative = _ask(
                "Negative prompt (Enter for defaults):"
            ) or "blurry, low quality, text, watermark, deformed"
            denoise_raw = _ask("Denoise strength 0.1–1.0 (Enter for 0.75):")
            try:
                denoise = float(denoise_raw) if denoise_raw else 0.75
                denoise = max(0.1, min(1.0, denoise))
            except ValueError:
                denoise = 0.75
            try:
                client.post(f"/pipelines/{name}/concept_art/restyle", {
                    "index": idx,
                    "positive": positive,
                    "negative": negative,
                    "denoise": denoise,
                })
                sheet_shown = False
            except httpx.HTTPStatusError as exc:
                _inform(f"Restyle failed: {exc.response.json().get('detail', str(exc))}")

        elif action == "cancel":
            client.delete(f"/pipelines/{name}")
            return "cancelled"

        elif action == "quit":
            return "quit"

        else:
            _inform("Unknown action. Valid: approve, regenerate, modify, restyle, cancel, quit")


# ---------------------------------------------------------------------------
# Mesh review
# ---------------------------------------------------------------------------


def _review_meshes(client: QMClient, name: str, state: dict) -> str:
    """
    Interactive mesh review loop.
    Returns: "approved" | "all_rejected" | "cancelled" | "quit"
    """
    while True:
        try:
            state = client.get(f"/pipelines/{name}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # Pipeline was moved to completed_pipelines after final export
                return "approved"
            raise
        pending_meshes = [
            m for m in state.get("meshes", [])
            if m["status"] == "awaiting_approval"
        ]

        if not pending_meshes:
            # Check if more meshes are still generating/texturing/screenshotting
            _IN_PROGRESS = {"queued", "generating", "texturing", "screenshotting"}
            in_progress = [m for m in state.get("meshes", []) if m["status"] in _IN_PROGRESS]
            if in_progress and state.get("status") == "mesh_review":
                names_str = ", ".join(m["sub_name"] for m in in_progress)
                _inform(
                    f"\nWaiting for {len(in_progress)} more mesh(es) to finish: {names_str}\n"
                    "  (press Ctrl-C to return to menu — generation continues in background)"
                )
                try:
                    while True:
                        time.sleep(5)
                        try:
                            state = client.get(f"/pipelines/{name}")
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 404:
                                return "approved"
                            raise
                        pending_meshes = [
                            m for m in state.get("meshes", [])
                            if m["status"] == "awaiting_approval"
                        ]
                        if pending_meshes:
                            _inform(f"\n[{time.strftime('%H:%M:%S')}] New mesh ready for review!")
                            break
                        in_progress = [m for m in state.get("meshes", []) if m["status"] in _IN_PROGRESS]
                        if not in_progress:
                            break
                except KeyboardInterrupt:
                    _inform("\nReturning to menu.")
                    return "approved"
                if not pending_meshes:
                    # All done — nothing came through
                    has_approved = any(m["status"] == "approved" for m in state.get("meshes", []))
                    return "approved" if has_approved else "all_rejected"
            else:
                has_approved = any(m["status"] == "approved" for m in state.get("meshes", []))
                return "approved" if has_approved else "all_rejected"

        output_root = _get_output_root(client)
        for mesh in pending_meshes:
            mesh_name = mesh["sub_name"]

            # Show review sheet
            try:
                sheet_bytes = client.download(f"/pipelines/{name}/meshes/{mesh_name}/sheet")
                _show_image(sheet_bytes)
            except Exception as exc:
                _inform(f"(No review sheet for '{mesh_name}': {exc})")

            mesh_dir = f"{output_root}/uncompleted_pipelines/{name}/meshes/{mesh_name}"
            _inform(
                f"\nMesh '{mesh_name}' is ready for review.\n"
                f"  GLB + screenshots: {mesh_dir}/\n"
                f"  3-D preview: {mesh_dir}/preview.html\n"
                f"  (Docker: host path = pipeline_root/uncompleted_pipelines/{name}/meshes/{mesh_name}/ in repo root)\n"
                "\nActions:\n"
                "  approve <asset_name> [format]  — approve and name the asset\n"
                "  reject                          — reject this mesh\n"
                "  cancel                          — cancel the pipeline\n"
                "  quit                            — exit the program"
            )

            while True:
                raw = _ask("Enter action").strip()
                tokens = raw.split()
                if not tokens:
                    continue
                action = tokens[0].lower()

                if action == "approve":
                    if len(tokens) < 2:
                        _inform("Usage: approve <asset_name> [format]  e.g. 'approve my_ship glb'")
                        continue
                    asset_name = tokens[1]
                    export_fmt = tokens[2] if len(tokens) > 2 else None
                    try:
                        client.post(
                            f"/pipelines/{name}/meshes/{mesh_name}/approve",
                            {"asset_name": asset_name, "export_format": export_fmt},
                        )
                        _inform(f"Approved '{asset_name}'.")
                    except httpx.HTTPStatusError as exc:
                        _inform(f"Approve failed: {exc.response.json().get('detail', str(exc))}")
                        continue
                    break

                elif action == "reject":
                    # Optional settings updates
                    poly_raw = _ask(
                        f"Current polygon target: {state['num_polys']}. "
                        "Enter new value or press Enter to keep:"
                    )
                    num_polys = int(poly_raw) if poly_raw.isdigit() else None

                    sym_options = ["x-", "x+", "y-", "y+", "z-", "z+"]
                    sym_display = f"{'on' if state['symmetrize'] else 'off'}, axis={state['symmetry_axis']}"
                    sym_raw = _ask(
                        f"Current symmetry: {sym_display}. "
                        f"Enter an axis ({', '.join(sym_options)}) to enable, 'off' to disable, or Enter to keep:"
                    ).lower()

                    reject_body: dict = {}
                    if num_polys:
                        reject_body["num_polys"] = num_polys
                    if sym_raw in sym_options:
                        reject_body["symmetrize"] = True
                        reject_body["symmetry_axis"] = sym_raw
                    elif sym_raw == "off":
                        reject_body["symmetrize"] = False

                    try:
                        client.post(
                            f"/pipelines/{name}/meshes/{mesh_name}/reject",
                            reject_body,
                        )
                        _inform(f"Mesh '{mesh_name}' rejected.")
                    except httpx.HTTPStatusError as exc:
                        _inform(f"Reject failed: {exc.response.json().get('detail', str(exc))}")
                        continue
                    break

                elif action == "cancel":
                    client.delete(f"/pipelines/{name}")
                    return "cancelled"

                elif action == "quit":
                    return "quit"

                else:
                    _inform("Unknown action. Valid: approve <name>, reject, cancel, quit")

        # Refresh state and check if review is complete
        try:
            state = client.get(f"/pipelines/{name}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # Pipeline was moved to completed_pipelines after final export
                return "approved"
            raise
        if state["status"] == "mesh_generating":
            # All meshes were rejected and generation was re-queued —
            # watch progress and loop back to review when ready.
            result = _watch_generating(client, name, ready_status="mesh_review")
            if result == "quit":
                return "quit"
            if result == "timeout":
                return "all_rejected"
            state = client.get(f"/pipelines/{name}")
        elif state["status"] not in ("mesh_review",):
            break

    # Final outcome
    has_approved = any(m["status"] == "approved" for m in state.get("meshes", []))
    return "approved" if has_approved else "all_rejected"


if __name__ == "__main__":
    # File log for the CLI — separate from the server's log
    _cli_log_dir = Path.home() / ".quickymesh"
    _cli_log_dir.mkdir(exist_ok=True)
    from src.logging_config import configure_logging
    configure_logging(log_dir=_cli_log_dir, log_filename="cli.log")
    main()
