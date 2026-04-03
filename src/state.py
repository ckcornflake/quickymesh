"""
Pydantic models for pipeline state.

The entire state of one pipeline run is captured in PipelineState and can
be round-tripped to/from a JSON file on disk.  Workers update sub-models
(ConceptArtItem, MeshItem) and the top-level agent calls state.save() after
each mutation.

File location convention:
    <output_root>/uncompleted_pipelines/<name>/state.json   (in progress)
    <output_root>/completed_pipelines/<name>/state.json     (done / approved)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PipelineStatus(str, Enum):
    INITIALIZING = "initializing"
    CONCEPT_ART_GENERATING = "concept_art_generating"
    CONCEPT_ART_REVIEW = "concept_art_review"
    MESH_GENERATING = "mesh_generating"
    MESH_REVIEW = "mesh_review"
    APPROVED = "approved"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ConceptArtStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    APPROVED = "approved"       # approved for mesh generation
    REGENERATING = "regenerating"
    MODIFIED = "modified"       # being modified via Gemini edit API
    REJECTED = "rejected"


class MeshStatus(str, Enum):
    QUEUED = "queued"
    GENERATING_MESH = "generating_mesh"
    MESH_DONE = "mesh_done"
    GENERATING_TEXTURE = "generating_texture"
    TEXTURE_DONE = "texture_done"
    CLEANING_UP = "cleaning_up"
    CLEANUP_DONE = "cleanup_done"
    SCREENSHOT_PENDING = "screenshot_pending"
    SCREENSHOT_DONE = "screenshot_done"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXPORTED = "exported"


class SymmetryAxis(str, Enum):
    AUTO = "auto"
    X_MINUS = "x-"
    X_PLUS = "x+"
    Y_MINUS = "y-"
    Y_PLUS = "y+"
    Z_MINUS = "z-"
    Z_PLUS = "z+"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ConceptArtItem(BaseModel):
    """State for a single generated concept art image."""

    index: int
    image_path: Optional[str] = None   # str so JSON serialises cleanly
    status: ConceptArtStatus = ConceptArtStatus.PENDING
    prompts: list[str] = Field(default_factory=list)


class MeshItem(BaseModel):
    """State for one mesh derived from an approved concept art."""

    concept_art_index: int
    sub_name: str                       # e.g. "dragon_01_1"
    status: MeshStatus = MeshStatus.QUEUED

    # Paths (all optional until the relevant step completes)
    mesh_path: Optional[str] = None
    textured_mesh_path: Optional[str] = None
    screenshot_dir: Optional[str] = None
    review_sheet_path: Optional[str] = None
    html_preview_path: Optional[str] = None

    # Final export
    final_name: Optional[str] = None
    export_format: str = "glb"
    export_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level pipeline state
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    """Complete, serialisable state for one pipeline run."""

    name: str
    description: str
    input_image_path: Optional[str] = None   # user-supplied reference image

    # Per-pipeline mesh settings (can be updated any time before mesh gen)
    num_polys: int
    symmetrize: bool = False
    symmetry_axis: SymmetryAxis = SymmetryAxis.AUTO

    status: PipelineStatus = PipelineStatus.INITIALIZING

    concept_arts: list[ConceptArtItem] = Field(default_factory=list)
    meshes: list[MeshItem] = Field(default_factory=list)

    # Which generator was used for concept art: "gemini" or "flux"
    concept_art_backend: str = "gemini"

    # Set to True after showing the concept-art review sheet; reset to False
    # whenever images change (regenerate / modify) so the sheet re-opens.
    concept_art_sheet_shown: bool = False

    # Every prompt ever submitted for this pipeline (generation + modifications)
    all_prompts: list[str] = Field(default_factory=list)

    # Relative path from output_root to this pipeline's folder
    pipeline_dir: str

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Update the `updated_at` timestamp.  Call before save()."""
        self.updated_at = datetime.now(timezone.utc)

    def save(self, path: Path) -> None:
        """Serialise to JSON and write atomically."""
        self.touch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first, then rename for atomicity
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        """Deserialise from a JSON file."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def approved_concept_arts(self) -> list[ConceptArtItem]:
        return [ca for ca in self.concept_arts if ca.status == ConceptArtStatus.APPROVED]

    def ready_concept_arts(self) -> list[ConceptArtItem]:
        return [ca for ca in self.concept_arts if ca.status == ConceptArtStatus.READY]

    def pending_mesh_approvals(self) -> list[MeshItem]:
        return [m for m in self.meshes if m.status == MeshStatus.AWAITING_APPROVAL]


# ---------------------------------------------------------------------------
# Utility: square-ish grid dimensions for review sheets
# ---------------------------------------------------------------------------


def review_sheet_dims(n: int) -> tuple[int, int]:
    """
    Return (cols, rows) that are as close to square as possible while
    cols * rows >= n.

    Examples:
        n=1  → (1, 1)
        n=4  → (2, 2)
        n=5  → (3, 2)
        n=6  → (3, 2)
        n=7  → (3, 3)
        n=9  → (3, 3)
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows
