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


def _concept_art_path(pipeline_dir: Path, index: int) -> Path:
    return pipeline_dir / _CONCEPT_ART_DIR / f"concept_art_{index + 1}.png"


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

        dest = _concept_art_path(pipeline_dir, item.index)
        if input_image_bytes is not None:
            # Image-based: treat description as the modification instruction
            image_bytes = worker.modify_image(input_image_bytes, state.description)
        else:
            image_bytes = worker.generate_image(prompt)
        _save_concept_art(image_bytes, dest)

        item.image_path = str(dest)
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

        dest = _concept_art_path(pipeline_dir, idx)
        image_bytes = worker.generate_image(prompt)
        _save_concept_art(image_bytes, dest)

        item.image_path = str(dest)
        item.status = ConceptArtStatus.READY


def modify_concept_art(
    state: PipelineState,
    worker: ConceptArtWorker,
    pipeline_dir: Path,
    index: int,
    instruction: str,
) -> None:
    """
    Modify an existing concept art image using Gemini's edit capability.
    Overwrites the existing image file.
    """
    if index < 0 or index >= len(state.concept_arts):
        raise IndexError(f"Concept art index {index} out of range")

    item = state.concept_arts[index]
    if not item.image_path:
        raise ValueError(f"Concept art {index} has no image to modify")

    item.status = ConceptArtStatus.MODIFIED
    item.prompts.append(f"[modify] {instruction}")
    state.all_prompts.append(f"[modify:{index}] {instruction}")

    existing_bytes = Path(item.image_path).read_bytes()
    new_bytes = worker.modify_image(existing_bytes, instruction)

    dest = _concept_art_path(pipeline_dir, index)
    _save_concept_art(new_bytes, dest)

    item.image_path = str(dest)
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
        Path(item.image_path)
        for item in state.concept_arts
        if item.image_path and item.status in (ConceptArtStatus.READY, ConceptArtStatus.APPROVED)
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


def restyle_concept_art(
    state: PipelineState,
    restyle_worker,
    pipeline_dir: Path,
    index: int,
    positive: str,
    negative: str,
    denoise: float,
) -> None:
    """
    Restyle an existing concept art image using ControlNet Canny.

    Overwrites the existing image file and updates the item's status/prompts.
    """
    if index < 0 or index >= len(state.concept_arts):
        raise IndexError(f"Concept art index {index} out of range")

    item = state.concept_arts[index]
    if not item.image_path:
        raise ValueError(f"Concept art {index} has no image to restyle")

    item.status = ConceptArtStatus.MODIFIED
    item.prompts.append(f"[restyle] {positive}")
    state.all_prompts.append(f"[restyle:{index}] {positive}")

    existing_bytes = Path(item.image_path).read_bytes()
    new_bytes = restyle_worker.restyle_image(existing_bytes, positive, negative, denoise)

    dest = _concept_art_path(pipeline_dir, index)
    _save_concept_art(new_bytes, dest)

    item.image_path = str(dest)
    item.status = ConceptArtStatus.READY


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
            state.concept_art_sheet_shown = True
            state.save(state_path)
            sheet = build_review_sheet(state, pipeline_dir, cfg)
            ui.show_image(sheet)
            modify_line = (
                "  modify <idx>           — modify one image via Gemini\n"
                if worker.supports_modify else ""
            )
            restyle_line = (
                "  restyle <idx>          — restyle image shape/silhouette via ControlNet\n"
                if restyle_worker is not None else ""
            )
            ui.inform(
                f"\nConcept art review for '{state.name}'\n"
                "Actions:\n"
                "  approve <indices>      — e.g. 'approve 1 3' to send to mesh gen\n"
                "  regenerate <idx|all>   — e.g. 'regenerate 2 4' or 'regenerate all'\n"
                + modify_line
                + restyle_line +
                "  cancel                 — cancel this pipeline\n"
                "  quit                   — exit the program\n"
            )
            for item in state.concept_arts:
                ui.inform(f"  [{item.index + 1}] {item.status.value}")

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
            state.status = PipelineStatus.MESH_GENERATING
            state.save(state_path)
            ui.inform(
                f"Approved {len(approved_indices)} image(s). "
                "Moving on to mesh generation automatically."
            )
            return "approved"

        # ── regenerate ─────────────────────────────────────────────────────
        elif action == _ACTION_REGENERATE:
            if len(tokens) < 2:
                ui.inform("Please specify indices to regenerate, e.g. 'regenerate 2' or 'regenerate all'")
                continue
            if tokens[1] == "all":
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
                regen_indices = _parse_indices(tokens[1:], len(state.concept_arts))
                if regen_indices is None:
                    ui.inform("Invalid indices.")
                    continue
                regenerate_concept_arts(state, worker, pipeline_dir, regen_indices, cfg)
            state.concept_art_sheet_shown = False
            state.save(state_path)

        # ── modify ─────────────────────────────────────────────────────────
        elif action == _ACTION_MODIFY:
            if not worker.supports_modify:
                ui.inform("Image modification is not supported by the current concept art generator (FLUX). Use 'regenerate' instead.")
                continue
            if len(tokens) < 2 or not tokens[1].isdigit():
                ui.inform("Usage: modify <index>  e.g. 'modify 3'")
                continue
            idx = int(tokens[1]) - 1
            if idx < 0 or idx >= len(state.concept_arts):
                ui.inform("Index out of range.")
                continue
            ui.inform("WARNING: This will replace the original concept art image.")
            instruction = ui.ask(f"Describe the change to make to image {idx + 1}")
            modify_concept_art(state, worker, pipeline_dir, idx, instruction)
            state.concept_art_sheet_shown = False
            state.save(state_path)

        # ── restyle ────────────────────────────────────────────────────────
        elif action == _ACTION_RESTYLE:
            if restyle_worker is None:
                ui.inform("ControlNet Restyle is not available. Ensure ComfyUI is running and the worker is configured.")
                continue
            if len(tokens) < 2 or not tokens[1].isdigit():
                ui.inform("Usage: restyle <index>  e.g. 'restyle 2'")
                continue
            idx = int(tokens[1]) - 1
            if idx < 0 or idx >= len(state.concept_arts):
                ui.inform("Index out of range.")
                continue
            ui.inform(
                "ControlNet Restyle: preserves the shape/silhouette, applies a new visual style.\n"
                "This will replace the image."
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
            restyle_concept_art(state, restyle_worker, pipeline_dir, idx, positive, negative, denoise)
            state.concept_art_sheet_shown = False
            state.save(state_path)

        # ── cancel / quit ──────────────────────────────────────────────────
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
            valid_actions += ["cancel", "quit"]
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
