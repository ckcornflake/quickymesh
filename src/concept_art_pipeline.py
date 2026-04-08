"""
Concept art pipeline — orchestration layer.

This module owns all decisions about *where* files go and *how* the pipeline
state changes during the concept art phase.  Workers are injected so tests
can use MockConceptArtWorker without hitting any external API.

Public API
----------
build_prompt(description, background_suffix) -> str
generate_concept_arts(state, worker, pipeline_dir, config) -> None
regenerate_concept_arts(state, worker, pipeline_dir, indices, config) -> None
modify_concept_art(state, worker, pipeline_dir, index, instruction) -> None
build_review_sheet(state, pipeline_dir, config) -> Path
run_concept_art_review(state, worker, pipeline_dir, ui, config) -> None
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from src.config import Config, config as _default_config
from src.image_utils import make_review_sheet, pad_to_square
from src.prompt_interface import PromptInterface
from src.state import ConceptArtItem, ConceptArtStatus, PipelineState, PipelineStatus
from src.workers.concept_art import ConceptArtWorker

_CONCEPT_ART_DIR = "concept_art"
_REVIEW_SHEET_NAME = "reviewsheet.png"
_IMAGE_SIZE = 1024  # target size for Trellis


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(description: str, background_suffix: str) -> str:
    """
    Append the background suffix to the user's description.
    Returns the full prompt string submitted to Gemini.
    """
    return f"{description.rstrip(', ')}. {background_suffix}"


# ---------------------------------------------------------------------------
# Image saving helper
# ---------------------------------------------------------------------------


def _save_concept_art(image_bytes: bytes, dest_path: Path) -> None:
    """
    Save raw image bytes to `dest_path`, padding to 1024×1024 on a white
    background.  Creates parent directories as needed.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    padded = pad_to_square(img, size=_IMAGE_SIZE)
    padded.save(str(dest_path))


def _concept_art_path(pipeline_dir: Path, index: int, version: int) -> Path:
    return pipeline_dir / _CONCEPT_ART_DIR / f"concept_art_{index + 1}_{version}.png"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_concept_arts(
    state: PipelineState,
    worker: ConceptArtWorker,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    Generate all concept arts for `state` from scratch.

    Initialises `state.concept_arts` with `cfg.num_concept_arts` items,
    submits each to the worker, saves the result, and updates state.
    Caller should save state to disk after this returns.
    """
    state.status = PipelineStatus.CONCEPT_ART_GENERATING
    prompt = build_prompt(state.description, cfg.background_suffix)
    state.all_prompts.append(prompt)

    count = cfg.num_concept_arts
    # Initialise concept_arts list if needed
    if not state.concept_arts:
        state.concept_arts = [ConceptArtItem(index=i) for i in range(count)]

    # Load input image bytes once if image-based generation was requested
    input_image_bytes: bytes | None = None
    if state.input_image_path:
        input_path = Path(state.input_image_path)
        if input_path.exists():
            input_image_bytes = input_path.read_bytes()

    for item in state.concept_arts:
        if item.status not in (ConceptArtStatus.PENDING, ConceptArtStatus.REGENERATING):
            continue
        item.status = ConceptArtStatus.GENERATING
        item.prompts.append(prompt)

        dest = _concept_art_path(pipeline_dir, item.index, item.version)
        if input_image_bytes is not None:
            # Image-based: treat description as the modification instruction
            image_bytes = worker.modify_image(input_image_bytes, state.description)
        else:
            image_bytes = worker.generate_image(prompt)
        _save_concept_art(image_bytes, dest)

        item.status = ConceptArtStatus.READY

    state.status = PipelineStatus.CONCEPT_ART_REVIEW


def regenerate_concept_arts(
    state: PipelineState,
    worker: ConceptArtWorker,
    pipeline_dir: Path,
    indices: list[int],
    cfg: Config = _default_config,
    *,
    description_override: str | None = None,
) -> None:
    """
    Re-generate specific concept arts by their 0-based index.

    Uses the original description by default. Pass `description_override` to
    regenerate with a different description (updates state.description too).
    Never picks up modify instructions from all_prompts.
    """
    if description_override is not None:
        state.description = description_override
    prompt = build_prompt(state.description, cfg.background_suffix)
    if not state.all_prompts or state.all_prompts[0] != prompt:
        state.all_prompts.insert(0, prompt)

    for idx in indices:
        if idx < 0 or idx >= len(state.concept_arts):
            raise IndexError(f"Concept art index {idx} out of range")
        item = state.concept_arts[idx]
        item.status = ConceptArtStatus.REGENERATING
        item.prompts.append(prompt)

        item.version += 1
        dest = _concept_art_path(pipeline_dir, idx, item.version)
        image_bytes = worker.generate_image(prompt)
        _save_concept_art(image_bytes, dest)

        item.status = ConceptArtStatus.READY


def modify_concept_art(
    state: PipelineState,
    worker: ConceptArtWorker,
    pipeline_dir: Path,
    index: int,
    instruction: str,
    *,
    source_version: int | None = None,
) -> None:
    """
    Modify an existing concept art image using Gemini's edit capability.
    Writes the result as a new version of the same slot.

    source_version: which saved version to edit (default: current latest).
    """
    if index < 0 or index >= len(state.concept_arts):
        raise IndexError(f"Concept art index {index} out of range")

    item = state.concept_arts[index]
    sv = source_version if source_version is not None else item.version
    current_path = _concept_art_path(pipeline_dir, index, sv)
    if not current_path.exists():
        raise ValueError(f"Concept art {index} version {sv} has no image to modify")

    item.status = ConceptArtStatus.MODIFIED
    item.prompts.append(f"[modify] {instruction}")
    state.all_prompts.append(f"[modify:{index}] {instruction}")

    existing_bytes = current_path.read_bytes()
    new_bytes = worker.modify_image(existing_bytes, instruction)

    item.version += 1
    dest = _concept_art_path(pipeline_dir, index, item.version)
    _save_concept_art(new_bytes, dest)

    item.status = ConceptArtStatus.READY


# ---------------------------------------------------------------------------
# Review sheet
# ---------------------------------------------------------------------------


def build_review_sheet(
    state: PipelineState,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> Path:
    """
    Build a review sheet from all READY/APPROVED concept arts in state.
    Returns the path to the saved review sheet PNG.
    """
    image_paths = [
        _concept_art_path(pipeline_dir, item.index, item.version)
        for item in state.concept_arts
        if item.status in (ConceptArtStatus.READY, ConceptArtStatus.APPROVED)
        and _concept_art_path(pipeline_dir, item.index, item.version).exists()
    ]
    if not image_paths:
        raise ValueError("No ready concept arts to build a review sheet from")

    dest = pipeline_dir / _CONCEPT_ART_DIR / _REVIEW_SHEET_NAME
    return make_review_sheet(
        image_paths,
        output_path=dest,
        thumb_size=cfg.review_sheet_thumb_size,
    )


# ---------------------------------------------------------------------------
# Interactive concept art review (CLI or any PromptInterface)
# ---------------------------------------------------------------------------

_ACTION_APPROVE = "approve"
_ACTION_REGENERATE = "regenerate"
_ACTION_MODIFY = "modify"
_ACTION_RESTYLE = "restyle"
_ACTION_CANCEL = "cancel"
_ACTION_QUIT = "quit"
_ACTION_MENU = "menu"


def restyle_concept_art(
    state: PipelineState,
    restyle_worker,
    pipeline_dir: Path,
    index: int,
    positive: str,
    negative: str,
    denoise: float,
    *,
    source_version: int | None = None,
) -> None:
    """
    Restyle an existing concept art image using ControlNet Canny.
    Writes the result as a new version of the same slot.

    source_version: which saved version to restyle from (default: current latest).
    """
    if index < 0 or index >= len(state.concept_arts):
        raise IndexError(f"Concept art index {index} out of range")

    item = state.concept_arts[index]
    sv = source_version if source_version is not None else item.version
    current_path = _concept_art_path(pipeline_dir, index, sv)
    if not current_path.exists():
        raise ValueError(f"Concept art {index} version {sv} has no image to restyle")

    item.status = ConceptArtStatus.MODIFIED
    item.prompts.append(f"[restyle] {positive}")
    state.all_prompts.append(f"[restyle:{index}] {positive}")

    existing_bytes = current_path.read_bytes()
    new_bytes = restyle_worker.restyle_image(existing_bytes, positive, negative, denoise)

    item.version += 1
    dest = _concept_art_path(pipeline_dir, index, item.version)
    _save_concept_art(new_bytes, dest)

    item.status = ConceptArtStatus.READY


def _ask_concept_art_index(
    ui: PromptInterface,
    state: PipelineState,
    action_name: str,
) -> int | None:
    """
    Interactively ask the user which concept art slot to act on.
    Returns a 0-based index, or None if the user cancelled / gave invalid input.
    """
    n = len(state.concept_arts)
    raw = ui.ask(f"Which image to {action_name}? (1–{n}):").strip()
    if not raw or not raw.isdigit():
        ui.inform("Cancelled.")
        return None
    idx = int(raw) - 1
    if idx < 0 or idx >= n:
        ui.inform(f"Invalid — enter a number between 1 and {n}.")
        return None
    return idx


def _ask_concept_art_version(
    ui: PromptInterface,
    item,  # ConceptArtItem
    idx: int,
) -> int:
    """
    If the concept art slot has more than one saved version, let the user
    choose which version to work from.  Defaults to the current (latest) one.
    Returns the chosen version number.
    """
    if item.version == 0:
        return 0
    versions_str = ", ".join(
        f"{v}{'=current' if v == item.version else ''}"
        for v in range(item.version + 1)
    )
    raw = ui.ask(
        f"Image {idx + 1} has versions: {versions_str}.\n"
        f"Which version to edit? (Enter for current):"
    ).strip()
    if not raw or not raw.isdigit():
        return item.version
    v = int(raw)
    if v < 0 or v > item.version:
        ui.inform(f"Invalid — using current (version {item.version}).")
        return item.version
    return v


def run_concept_art_review(
    state: PipelineState,
    worker: ConceptArtWorker,
    pipeline_dir: Path,
    ui: PromptInterface,
    cfg: Config = _default_config,
    *,
    restyle_worker=None,
) -> str:
    """
    Drive the interactive concept art review loop.

    Returns one of: "approved", "cancelled", "quit".

    The caller is responsible for saving state between iterations if desired;
    this function saves state after every mutating action.
    """
    state_path = pipeline_dir / "state.json"

    while True:
        if not state.concept_art_sheet_shown:
            # Try to show the review sheet — tolerate failures (e.g. no READY
            # images yet) so the text menu always appears.
            try:
                sheet = build_review_sheet(state, pipeline_dir, cfg)
                ui.show_image(sheet)
            except Exception:
                pass
            modify_line = (
                "  modify              — modify one image via Gemini\n"
                if worker.supports_modify else ""
            )
            restyle_line = (
                "  restyle             — restyle image shape/silhouette via ControlNet\n"
                if restyle_worker is not None else ""
            )
            ui.inform(
                f"\nConcept art review for '{state.name}'\n"
                "Actions:\n"
                "  approve <indices>  — e.g. 'approve 1 3' to send to mesh gen\n"
                "  regenerate         — pick an image (or type 'regenerate all')\n"
                + modify_line
                + restyle_line +
                "  menu               — return to main menu\n"
                "  cancel             — cancel this pipeline\n"
                "  quit               — exit the program\n"
            )
            for item in state.concept_arts:
                ver_tag = f"  v{item.version}" if item.version > 0 else ""
                ui.inform(f"  [{item.index + 1}] {item.status.value}{ver_tag}")
            # Mark shown only after the menu has been printed successfully.
            state.concept_art_sheet_shown = True
            state.save(state_path)

        raw = ui.ask("Enter action").lower().strip()
        tokens = raw.split()
        if not tokens:
            continue

        action = tokens[0]

        # ── approve ────────────────────────────────────────────────────────
        if action == _ACTION_APPROVE:
            if len(tokens) < 2:
                ui.inform("Please specify which images to approve, e.g. 'approve 1 2'")
                continue
            approved_indices = _parse_indices(tokens[1:], len(state.concept_arts))
            if approved_indices is None:
                ui.inform("Invalid indices — use 1-based numbers matching the review sheet.")
                continue
            for idx in approved_indices:
                state.concept_arts[idx].status = ConceptArtStatus.APPROVED
            state.save(state_path)
            ui.inform(
                f"Approved {len(approved_indices)} image(s)."
            )
            return "approved"

        # ── regenerate ─────────────────────────────────────────────────────
        elif action == _ACTION_REGENERATE:
            if len(tokens) >= 2 and tokens[1] == "all":
                # 'regenerate all' path: optionally change the description
                regen_indices = list(range(len(state.concept_arts)))
                current_prompt = build_prompt(state.description, cfg.background_suffix)
                ui.inform(
                    f"Current prompt: {current_prompt}\n"
                    f'(Suffix "{cfg.background_suffix}" will be appended automatically)'
                )
                new_desc = ui.ask("New description (Enter to keep current):").strip()
                description_override = new_desc if new_desc else None
                regenerate_concept_arts(
                    state, worker, pipeline_dir, regen_indices, cfg,
                    description_override=description_override,
                )
            else:
                # Interactive: ask which image to regenerate
                idx = _ask_concept_art_index(ui, state, "regenerate")
                if idx is None:
                    continue
                regenerate_concept_arts(state, worker, pipeline_dir, [idx], cfg)
            state.concept_art_sheet_shown = False
            state.save(state_path)

        # ── modify ─────────────────────────────────────────────────────────
        elif action == _ACTION_MODIFY:
            if not worker.supports_modify:
                ui.inform(
                    "Image modification is not supported by the current concept art "
                    "generator (FLUX). Use 'regenerate' instead."
                )
                continue
            idx = _ask_concept_art_index(ui, state, "modify")
            if idx is None:
                continue
            item = state.concept_arts[idx]
            source_version = _ask_concept_art_version(ui, item, idx)
            ui.inform(
                f"Modifying image {idx + 1} (version {source_version}) — "
                "result will be saved as the next version."
            )
            instruction = ui.ask(f"Describe the change to make to image {idx + 1}:")
            modify_concept_art(
                state, worker, pipeline_dir, idx, instruction,
                source_version=source_version,
            )
            state.concept_art_sheet_shown = False
            state.save(state_path)

        # ── restyle ────────────────────────────────────────────────────────
        elif action == _ACTION_RESTYLE:
            if restyle_worker is None:
                ui.inform(
                    "ControlNet Restyle is not available. "
                    "Ensure ComfyUI is running and the worker is configured."
                )
                continue
            idx = _ask_concept_art_index(ui, state, "restyle")
            if idx is None:
                continue
            item = state.concept_arts[idx]
            source_version = _ask_concept_art_version(ui, item, idx)
            ui.inform(
                "ControlNet Restyle: preserves the shape/silhouette, applies a new visual style.\n"
                f"Working from image {idx + 1} version {source_version} — result saved as next version."
            )
            positive = ui.ask(
                f"Positive prompt for image {idx + 1}\n"
                "  Comma-separated words/phrases describing the style you want\n"
                "  e.g. 'zerg, biomechanical, dark chitin, alien, glossy':"
            ).strip()
            if not positive:
                ui.inform("Cancelled.")
                continue
            negative = ui.ask(
                "Negative prompt (comma-separated elements to avoid)\n"
                "  e.g. 'weapons, people, blurry, low quality, text'\n"
                "  (Enter to use defaults):"
            ).strip()
            if not negative:
                negative = "blurry, low quality, text, watermark, deformed"
            denoise_raw = ui.ask(
                "Denoise strength (0.1–1.0):\n"
                "  Lower = more faithful to original shape\n"
                "  Higher = more creative freedom to match style\n"
                "  (Enter for default 0.75):"
            ).strip()
            try:
                denoise = float(denoise_raw) if denoise_raw else 0.75
                denoise = max(0.1, min(1.0, denoise))
            except ValueError:
                ui.inform("Invalid value — using 0.75.")
                denoise = 0.75
            restyle_concept_art(
                state, restyle_worker, pipeline_dir, idx, positive, negative, denoise,
                source_version=source_version,
            )
            state.concept_art_sheet_shown = False
            state.save(state_path)

        # ── menu / cancel / quit ───────────────────────────────────────────
        elif action == _ACTION_MENU:
            state.save(state_path)
            return "back"

        elif action == _ACTION_CANCEL:
            state.status = PipelineStatus.CANCELLED
            state.save(state_path)
            ui.inform("Pipeline cancelled.")
            return "cancelled"

        elif action == _ACTION_QUIT:
            state.save(state_path)
            ui.inform("Quitting.")
            return "quit"

        else:
            valid_actions = ["approve", "regenerate"]
            if worker.supports_modify:
                valid_actions.append("modify")
            if restyle_worker is not None:
                valid_actions.append("restyle")
            valid_actions += ["menu", "cancel", "quit"]
            ui.inform(f"Unknown action '{action}'. Valid actions: {', '.join(valid_actions)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_indices(tokens: list[str], max_count: int) -> list[int] | None:
    """
    Convert 1-based string tokens to 0-based integer indices.
    Returns None if any token is invalid.
    """
    result = []
    for t in tokens:
        if not t.isdigit():
            return None
        idx = int(t) - 1
        if idx < 0 or idx >= max_count:
            return None
        result.append(idx)
    return result
