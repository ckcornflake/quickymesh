"""
Mesh generation + texturing pipeline — orchestration layer.

Owns all decisions about where files go and how state changes during the
mesh phase.  TrellisWorker is injected so tests can use MockTrellisWorker
without a running ComfyUI instance.

Public API
----------
run_mesh_generation(state, worker, pipeline_dir, cfg) -> None
run_mesh_texturing(state, worker, pipeline_dir, cfg) -> None
run_mesh_review(state, pipeline_dir, ui, cfg) -> str
run_mesh_export(state, pipeline_dir, cfg) -> None
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config, config as _default_config
from src.prompt_interface import PromptInterface
from src.state import (
    ConceptArtStatus,
    MeshItem,
    MeshStatus,
    PipelineState,
    PipelineStatus,
    SymmetryAxis,
)
from src.workers.trellis import TrellisWorker


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _mesh_sub_dir(pipeline_dir: Path, sub_name: str) -> Path:
    return pipeline_dir / sub_name


def _job_id(state_name: str, ca_index: int) -> str:
    """Filesystem-safe ComfyUI output prefix for this concept art slot."""
    safe = state_name.replace(" ", "_")
    return f"{safe}_{ca_index}"


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


def run_mesh_generation(
    state: PipelineState,
    worker: TrellisWorker,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    For every approved concept art that does not yet have a generated mesh,
    run Trellis mesh generation.

    QUEUED items (initial state or post-rejection) are regenerated — they
    represent work that either never started or was explicitly re-queued by
    the user after rejection.  Items in any other status (in-progress,
    generated, or further along) are skipped to prevent duplicate work.

    Updates state in place.  Caller is responsible for saving state.
    """
    # Statuses that mean a mesh is already being worked on or is done —
    # do NOT restart generation for these.
    _skip_statuses = {
        MeshStatus.GENERATING_MESH,
        MeshStatus.MESH_DONE,
        MeshStatus.GENERATING_TEXTURE,
        MeshStatus.TEXTURE_DONE,
        MeshStatus.CLEANING_UP,
        MeshStatus.CLEANUP_DONE,
        MeshStatus.SCREENSHOT_PENDING,
        MeshStatus.SCREENSHOT_DONE,
        MeshStatus.AWAITING_APPROVAL,
        MeshStatus.APPROVED,
        MeshStatus.EXPORTED,
    }
    skip_ca_indices = {m.concept_art_index for m in state.meshes if m.status in _skip_statuses}

    for ca in state.concept_arts:
        if ca.status != ConceptArtStatus.APPROVED:
            continue
        if ca.index in skip_ca_indices:
            continue
        if not ca.image_path:
            continue

        # Reuse an existing QUEUED item rather than creating a duplicate.
        existing_queued = next(
            (m for m in state.meshes
             if m.concept_art_index == ca.index and m.status == MeshStatus.QUEUED),
            None,
        )
        if existing_queued:
            mesh_item = existing_queued
        else:
            mesh_item = MeshItem(
                concept_art_index=ca.index,
                sub_name=f"{state.name}_{ca.index + 1}",
            )
            state.meshes.append(mesh_item)

        mesh_dir = _mesh_sub_dir(pipeline_dir, mesh_item.sub_name) / "meshes"
        job = _job_id(state.name, ca.index)

        mesh_item.status = MeshStatus.GENERATING_MESH
        glb = worker.generate_mesh(
            image_path=Path(ca.image_path),
            dest_dir=mesh_dir,
            num_polys=state.num_polys,
            job_id=job,
        )
        mesh_item.mesh_path = str(glb)
        mesh_item.status = MeshStatus.MESH_DONE


# ---------------------------------------------------------------------------
# Texturing
# ---------------------------------------------------------------------------


def run_mesh_texturing(
    state: PipelineState,
    worker: TrellisWorker,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    For every MeshItem in MESH_DONE status, run Trellis texturing.

    Updates state in place.  Caller is responsible for saving state.
    """
    for mesh_item in state.meshes:
        if mesh_item.status != MeshStatus.MESH_DONE:
            continue

        ca = state.concept_arts[mesh_item.concept_art_index]
        if not ca.image_path or not mesh_item.mesh_path:
            continue

        texture_dir = _mesh_sub_dir(pipeline_dir, mesh_item.sub_name) / "meshes"
        job = _job_id(state.name, mesh_item.concept_art_index)

        mesh_item.status = MeshStatus.GENERATING_TEXTURE
        textured_glb = worker.texture_mesh(
            image_path=Path(ca.image_path),
            mesh_path=Path(mesh_item.mesh_path),
            dest_dir=texture_dir,
            job_id=job,
        )
        mesh_item.textured_mesh_path = str(textured_glb)
        mesh_item.status = MeshStatus.TEXTURE_DONE


# ---------------------------------------------------------------------------
# Mesh review (interactive)
# ---------------------------------------------------------------------------

_ACTION_APPROVE = "approve"
_ACTION_CANCEL = "cancel"
_ACTION_QUIT = "quit"


def run_mesh_review(
    state: PipelineState,
    pipeline_dir: Path,
    ui: PromptInterface,
    cfg: Config = _default_config,
) -> str:
    """
    Interactive review of generated meshes.

    Returns one of: "approved", "cancelled", "quit".

    For each mesh awaiting approval:
      - Show the review sheet (screenshots, built in Phase 4)
      - Prompt: approve <name> [format] | reject | cancel | quit
        - approve requires a name for the final asset
        - format defaults to cfg.export_format
      - On approve: mark APPROVED, record final_name + export_format,
        prompt for symmetrize confirmation if state.symmetrize is True
    """
    state_path = pipeline_dir / "state.json"

    # Mark meshes ready for review as AWAITING_APPROVAL.
    # SCREENSHOT_DONE is the normal path; CLEANUP_DONE/TEXTURE_DONE are
    # fallbacks when the screenshot step was skipped or not yet run.
    _ready = {MeshStatus.SCREENSHOT_DONE, MeshStatus.CLEANUP_DONE, MeshStatus.TEXTURE_DONE}
    for m in state.meshes:
        if m.status in _ready:
            m.status = MeshStatus.AWAITING_APPROVAL

    pending = [m for m in state.meshes if m.status == MeshStatus.AWAITING_APPROVAL]
    if not pending:
        return "no_pending"

    for mesh_item in pending:
        first_prompt = True
        while True:
            # Show the review sheet and full help text only on the first prompt
            # for this mesh — never on re-prompts caused by invalid input.
            if first_prompt:
                first_prompt = False
                if mesh_item.review_sheet_path and Path(mesh_item.review_sheet_path).exists():
                    ui.show_image(Path(mesh_item.review_sheet_path))
                else:
                    ui.inform(f"\n[Mesh {mesh_item.sub_name}] (no screenshots — review the GLB manually)")
                    if mesh_item.textured_mesh_path:
                        ui.inform(f"  Textured GLB: {mesh_item.textured_mesh_path}")
                ui.inform(
                    f"\nMesh '{mesh_item.sub_name}' is ready for review.\n"
                    "Actions:\n"
                    "  approve <asset_name> [format]  — approve and name the asset\n"
                    "  reject                          — reject this mesh (pipeline continues)\n"
                    "  cancel                          — cancel the pipeline\n"
                    "  quit                            — exit the program"
                )

            raw = ui.ask("Enter action").strip()
            tokens = raw.split()
            if not tokens:
                continue
            action = tokens[0].lower()

            if action == _ACTION_APPROVE:
                if len(tokens) < 2:
                    ui.inform("Please provide a name: approve <asset_name> [format]")
                    continue
                asset_name = tokens[1]
                export_fmt = tokens[2] if len(tokens) > 2 else cfg.export_format
                mesh_item.final_name = asset_name
                mesh_item.export_format = export_fmt
                mesh_item.status = MeshStatus.APPROVED

                # Confirm symmetry axis if symmetrize is on
                if state.symmetrize:
                    axis_options = [a.value for a in SymmetryAxis]
                    axis_raw = ui.ask(
                        f"Symmetry axis (current: {state.symmetry_axis.value}). "
                        f"Options: {', '.join(axis_options)}. Press Enter to keep:",
                    ).strip()
                    if axis_raw in axis_options:
                        state.symmetry_axis = SymmetryAxis(axis_raw)

                state.save(state_path)
                ui.inform(f"Approved '{asset_name}' ({export_fmt}). Will be exported after all reviews.")
                break

            elif action == "reject":
                # Offer a poly-count update before re-queuing
                poly_raw = ui.ask(
                    f"Current polygon target: {state.num_polys}. "
                    "Enter a new value to change it, or press Enter to keep:"
                ).strip().lower()
                if poly_raw == "quit":
                    state.save(state_path)
                    ui.inform("Quitting.")
                    return "quit"
                if poly_raw.isdigit() and int(poly_raw) > 0:
                    state.num_polys = int(poly_raw)
                mesh_item.status = MeshStatus.QUEUED  # could be re-queued later
                state.save(state_path)
                ui.inform(f"Mesh '{mesh_item.sub_name}' rejected.")
                break

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
                ui.inform("Unknown action. Valid: approve <name>, reject, cancel, quit")

    approved = [m for m in state.meshes if m.status == MeshStatus.APPROVED]
    if approved:
        state.status = PipelineStatus.APPROVED
        state.save(state_path)
        return "approved"

    # All meshes were rejected — move back to MESH_GENERATING so workers can
    # re-run mesh generation with the (potentially updated) poly count.
    state.status = PipelineStatus.MESH_GENERATING
    state.save(state_path)
    return "all_rejected"


# ---------------------------------------------------------------------------
# Mesh export to final assets
# ---------------------------------------------------------------------------


def run_mesh_export(
    state: PipelineState,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    Export all APPROVED meshes to the final_game_ready_assets directory.

    For each approved mesh:
      - Copy textured GLB to final_assets_dir/{final_name}/mesh.{export_format}
      - Create metadata.txt with pipeline info
      - Update export_path in state
    
    Also moves the entire pipeline from uncompleted_pipelines to completed_pipelines.

    Updates state in place.  Caller is responsible for saving state.
    """
    state_path = pipeline_dir / "state.json"

    # Export each approved mesh
    for mesh_item in state.meshes:
        if mesh_item.status != MeshStatus.APPROVED:
            continue
        if not mesh_item.final_name:
            continue
        if not mesh_item.textured_mesh_path:
            continue

        # Create asset directory
        asset_dir = cfg.final_assets_dir / mesh_item.final_name
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Copy textured GLB
        src_glb = Path(mesh_item.textured_mesh_path)
        dst_glb = asset_dir / f"mesh.{mesh_item.export_format}"
        if src_glb.exists():
            dst_glb.write_bytes(src_glb.read_bytes())
            mesh_item.export_path = str(dst_glb)

        # Write metadata
        metadata_path = asset_dir / "metadata.txt"
        ca_index = mesh_item.concept_art_index
        ca = state.concept_arts[ca_index]
        metadata_lines = [
            f"Pipeline: {state.name}",
            f"Asset name: {mesh_item.final_name}",
            f"Export format: {mesh_item.export_format}",
            f"Concept art index: {ca_index}",
            f"Concept art image: {ca.image_path}",
            f"Pipeline folder: {pipeline_dir}",
            f"Created: {state.created_at.isoformat()}",
        ]
        metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

        mesh_item.status = MeshStatus.EXPORTED

    state.save(state_path)

    # Move pipeline from uncompleted to completed
    import logging, shutil
    _log = logging.getLogger(__name__)
    completed_pipeline_dir = cfg.completed_pipelines_dir / state.name
    if any(m.status == MeshStatus.EXPORTED for m in state.meshes):
        completed_pipeline_dir.parent.mkdir(parents=True, exist_ok=True)
        if pipeline_dir.exists() and not completed_pipeline_dir.exists():
            try:
                shutil.move(str(pipeline_dir), str(completed_pipeline_dir))
                _log.info("Pipeline '%s' moved to completed_pipelines", state.name)
            except Exception as exc:
                _log.warning("Could not move pipeline dir to completed: %s", exc)
