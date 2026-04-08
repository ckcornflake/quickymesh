"""
Pydantic models for pipeline state.

Two top-level state models exist:

PipelineState       — 2D pipeline (concept art generation / review).
                      Saved at: <output_root>/pipelines/<name>/state.json

Pipeline3DState     — 3D pipeline (mesh gen → texture → cleanup → screenshots
                      → review → export).  May be linked to a 2D pipeline or
                      submitted independently from an uploaded image.
                      Saved at: <output_root>/pipelines/<name>/state.json
                      where name is either "{2d_name}_{ca_idx}_{ca_ver}" (derived)
                      or "u_{user_name}" (standalone upload).
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
    """Status of a 2D (concept art) pipeline."""
    INITIALIZING = "initializing"
    CONCEPT_ART_GENERATING = "concept_art_generating"
    CONCEPT_ART_REVIEW = "concept_art_review"   # also the idle state
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ConceptArtStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    APPROVED = "approved"       # selected for mesh generation submission
    REGENERATING = "regenerating"
    MODIFIED = "modified"       # being modified via Gemini edit API
    REJECTED = "rejected"


class Pipeline3DStatus(str, Enum):
    """Status of a 3D (mesh generation) pipeline."""
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
    IDLE = "idle"               # exported; user can approve again for new version
    CANCELLED = "cancelled"


class SymmetryAxis(str, Enum):
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
    version: int = 0                   # increments on each regen/modify/restyle
    status: ConceptArtStatus = ConceptArtStatus.PENDING
    prompts: list[str] = Field(default_factory=list)

    def image_filename(self) -> str:
        """Derive the expected filename for the current version."""
        return f"concept_art_{self.index + 1}_{self.version}.png"


# ---------------------------------------------------------------------------
# Top-level pipeline state
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    """Complete, serialisable state for a 2D (concept art) pipeline."""

    name: str
    description: str
    input_image_path: Optional[str] = None   # user-supplied reference image

    # Default mesh settings passed to any 3D pipeline spawned from this one
    num_polys: int
    symmetrize: bool = False
    symmetry_axis: SymmetryAxis = SymmetryAxis.X_MINUS

    status: PipelineStatus = PipelineStatus.INITIALIZING
    hidden: bool = False

    concept_arts: list[ConceptArtItem] = Field(default_factory=list)

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
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Migrate legacy "auto" symmetry_axis (removed — default to x-)
        if data.get("symmetry_axis") == "auto":
            data["symmetry_axis"] = "x-"
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def approved_concept_arts(self) -> list[ConceptArtItem]:
        return [ca for ca in self.concept_arts if ca.status == ConceptArtStatus.APPROVED]

    def ready_concept_arts(self) -> list[ConceptArtItem]:
        return [ca for ca in self.concept_arts if ca.status == ConceptArtStatus.READY]


# ---------------------------------------------------------------------------
# 3D pipeline state
# ---------------------------------------------------------------------------


class Pipeline3DState(BaseModel):
    """
    Complete, serialisable state for a 3D (mesh generation) pipeline.

    A 3D pipeline may originate from:
      - A 2D pipeline concept art image (source_2d_pipeline is set).
      - A user-uploaded image (source_2d_pipeline is None, name starts with "u_").

    Folder name conventions:
      Derived:   {2d_pipeline_name}_{ca_index}_{ca_version}   e.g. "hypership_1_0"
      Uploaded:  u_{user_name}                                 e.g. "u_myship"
    """

    name: str
    source_2d_pipeline: Optional[str] = None
    source_concept_art_index: Optional[int] = None
    source_concept_art_version: Optional[int] = None

    # Absolute path to the source image on the server
    input_image_path: str

    num_polys: int
    symmetrize: bool = False
    symmetry_axis: SymmetryAxis = SymmetryAxis.X_MINUS
    hidden: bool = False

    status: Pipeline3DStatus = Pipeline3DStatus.QUEUED

    # Paths set progressively as each step completes
    mesh_path: Optional[str] = None
    textured_mesh_path: Optional[str] = None
    screenshot_dir: Optional[str] = None
    review_sheet_path: Optional[str] = None
    html_preview_path: Optional[str] = None

    # Export tracking — export_version increments each time the user approves
    export_version: int = 0
    export_paths: list[str] = Field(default_factory=list)

    # Relative path from output_root to this pipeline's folder
    pipeline_dir: str

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def save(self, path: Path) -> None:
        self.touch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Pipeline3DState":
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


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
