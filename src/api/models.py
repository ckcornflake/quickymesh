"""
Pydantic request/response models for the quickymesh API.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline requests
# ---------------------------------------------------------------------------


class CreatePipelineRequest(BaseModel):
    name: str
    description: str
    num_polys: Optional[int] = None
    input_image_path: Optional[str] = None
    symmetrize: bool = False
    symmetry_axis: str = "x-"
    concept_art_backend: str = "gemini"


class PatchPipelineRequest(BaseModel):
    description: Optional[str] = None
    num_polys: Optional[int] = None
    symmetrize: Optional[bool] = None
    symmetry_axis: Optional[str] = None


# ---------------------------------------------------------------------------
# Concept art review requests
# ---------------------------------------------------------------------------


class ApproveConceptArtRequest(BaseModel):
    """0-based indices of concept arts to approve."""
    indices: list[int]


class RegenerateConceptArtRequest(BaseModel):
    """
    0-based indices to regenerate, or omit/None to regenerate all.
    description_override replaces state.description for this regeneration.
    """
    indices: Optional[list[int]] = None
    description_override: Optional[str] = None


class ModifyConceptArtRequest(BaseModel):
    """Modify one concept art image via Gemini's edit API."""
    index: int  # 0-based
    instruction: str


class RestyleConceptArtRequest(BaseModel):
    """Restyle one concept art image via ControlNet Canny."""
    index: int  # 0-based
    positive: str
    negative: str = "blurry, low quality, text, watermark, deformed"
    denoise: float = Field(default=0.75, ge=0.1, le=1.0)


# ---------------------------------------------------------------------------
# Mesh review requests
# ---------------------------------------------------------------------------


class ApproveMeshRequest(BaseModel):
    """Approve a mesh and give it a final asset name."""
    asset_name: str
    export_format: Optional[str] = None  # defaults to cfg.export_format


class RejectMeshRequest(BaseModel):
    """Reject a mesh, optionally updating generation settings."""
    num_polys: Optional[int] = None
    symmetrize: Optional[bool] = None
    symmetry_axis: Optional[str] = None


# ---------------------------------------------------------------------------
# 3D pipeline requests
# ---------------------------------------------------------------------------


class Create3DPipelineFromRefRequest(BaseModel):
    """Submit a concept art image from a 2D pipeline for mesh generation."""
    pipeline_name: str              # name of the source 2D pipeline
    concept_art_index: int          # 0-based CA slot index
    concept_art_version: Optional[int] = None  # defaults to current version
    num_polys: Optional[int] = None
    symmetrize: bool = False
    symmetry_axis: str = "x-"


class Create3DPipelineFromUploadRequest(BaseModel):
    """Start a 3D pipeline from a user-provided name + uploaded image."""
    name: str
    num_polys: Optional[int] = None
    symmetrize: bool = False
    symmetry_axis: str = "x-"


# ---------------------------------------------------------------------------
# Generic responses
# ---------------------------------------------------------------------------


class OkResponse(BaseModel):
    status: str = "ok"


class AcceptedResponse(BaseModel):
    status: str = "accepted"
    message: str = ""
