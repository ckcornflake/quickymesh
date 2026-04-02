"""
Tests for src/state.py — Pydantic pipeline state models.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from src.state import (
    ConceptArtItem,
    ConceptArtStatus,
    MeshItem,
    MeshStatus,
    PipelineState,
    PipelineStatus,
    SymmetryAxis,
    review_sheet_dims,
)


# ---------------------------------------------------------------------------
# review_sheet_dims
# ---------------------------------------------------------------------------


class TestReviewSheetDims:
    @pytest.mark.parametrize("n, expected_cols, expected_rows", [
        (1, 1, 1),
        (2, 2, 1),
        (3, 2, 2),
        (4, 2, 2),
        (5, 3, 2),
        (6, 3, 2),
        (7, 3, 3),
        (8, 3, 3),
        (9, 3, 3),
        (12, 4, 3),
        (16, 4, 4),
    ])
    def test_dims(self, n, expected_cols, expected_rows):
        cols, rows = review_sheet_dims(n)
        assert cols == expected_cols
        assert rows == expected_rows
        assert cols * rows >= n

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            review_sheet_dims(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            review_sheet_dims(-1)


# ---------------------------------------------------------------------------
# ConceptArtItem
# ---------------------------------------------------------------------------


class TestConceptArtItem:
    def test_default_status_is_pending(self):
        item = ConceptArtItem(index=0)
        assert item.status == ConceptArtStatus.PENDING

    def test_prompts_default_empty(self):
        item = ConceptArtItem(index=1)
        assert item.prompts == []

    def test_set_image_path(self):
        item = ConceptArtItem(index=0, image_path="/some/path.png")
        assert item.image_path == "/some/path.png"

    def test_status_transition(self):
        item = ConceptArtItem(index=0, status=ConceptArtStatus.GENERATING)
        item.status = ConceptArtStatus.READY
        assert item.status == ConceptArtStatus.READY


# ---------------------------------------------------------------------------
# MeshItem
# ---------------------------------------------------------------------------


class TestMeshItem:
    def test_default_status_is_queued(self):
        item = MeshItem(concept_art_index=0, sub_name="test_1")
        assert item.status == MeshStatus.QUEUED

    def test_default_export_format(self):
        item = MeshItem(concept_art_index=0, sub_name="test_1")
        assert item.export_format == "glb"

    def test_optional_paths_none_by_default(self):
        item = MeshItem(concept_art_index=0, sub_name="test_1")
        assert item.mesh_path is None
        assert item.textured_mesh_path is None
        assert item.export_path is None


# ---------------------------------------------------------------------------
# PipelineState creation
# ---------------------------------------------------------------------------


class TestPipelineStateCreation:
    def test_required_fields(self):
        state = PipelineState(
            name="my_model",
            description="a red car",
            num_polys=8000,
            pipeline_dir="uncompleted_pipelines/my_model",
        )
        assert state.name == "my_model"
        assert state.num_polys == 8000
        assert state.status == PipelineStatus.INITIALIZING

    def test_default_symmetry(self):
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p"
        )
        assert state.symmetrize is False
        assert state.symmetry_axis == SymmetryAxis.AUTO

    def test_concept_arts_and_meshes_empty_by_default(self):
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p"
        )
        assert state.concept_arts == []
        assert state.meshes == []

    def test_created_at_is_utc(self):
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p"
        )
        assert state.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# PipelineState persistence (save / load)
# ---------------------------------------------------------------------------


class TestPipelineStatePersistence:
    def test_save_creates_file(self, tmp_path, minimal_pipeline):
        p = tmp_path / "state.json"
        minimal_pipeline.save(p)
        assert p.exists()

    def test_saved_file_is_valid_json(self, tmp_path, minimal_pipeline):
        p = tmp_path / "state.json"
        minimal_pipeline.save(p)
        data = json.loads(p.read_text())
        assert data["name"] == "test_dragon"

    def test_round_trip(self, tmp_path, minimal_pipeline):
        p = tmp_path / "state.json"
        minimal_pipeline.save(p)
        loaded = PipelineState.load(p)
        assert loaded.name == minimal_pipeline.name
        assert loaded.num_polys == minimal_pipeline.num_polys
        assert loaded.status == minimal_pipeline.status

    def test_save_creates_parent_dirs(self, tmp_path, minimal_pipeline):
        p = tmp_path / "deep" / "nested" / "state.json"
        minimal_pipeline.save(p)
        assert p.exists()

    def test_save_updates_updated_at(self, tmp_path, minimal_pipeline):
        before = minimal_pipeline.updated_at
        p = tmp_path / "state.json"
        minimal_pipeline.save(p)
        assert minimal_pipeline.updated_at >= before

    def test_load_preserves_concept_arts(self, tmp_path):
        state = PipelineState(
            name="n",
            description="d",
            num_polys=1000,
            pipeline_dir="p",
        )
        state.concept_arts = [
            ConceptArtItem(index=0, status=ConceptArtStatus.READY, image_path="/img.png"),
            ConceptArtItem(index=1, status=ConceptArtStatus.APPROVED),
        ]
        p = tmp_path / "state.json"
        state.save(p)
        loaded = PipelineState.load(p)
        assert len(loaded.concept_arts) == 2
        assert loaded.concept_arts[0].image_path == "/img.png"
        assert loaded.concept_arts[1].status == ConceptArtStatus.APPROVED

    def test_load_preserves_meshes(self, tmp_path):
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p"
        )
        state.meshes = [
            MeshItem(
                concept_art_index=0,
                sub_name="n_1",
                status=MeshStatus.GENERATING_MESH,
                mesh_path="/mesh.glb",
            )
        ]
        p = tmp_path / "state.json"
        state.save(p)
        loaded = PipelineState.load(p)
        assert loaded.meshes[0].mesh_path == "/mesh.glb"
        assert loaded.meshes[0].status == MeshStatus.GENERATING_MESH


# ---------------------------------------------------------------------------
# PipelineState convenience queries
# ---------------------------------------------------------------------------


class TestPipelineStateQueries:
    def _state_with_arts(self, statuses: list[ConceptArtStatus]) -> PipelineState:
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p"
        )
        state.concept_arts = [
            ConceptArtItem(index=i, status=s) for i, s in enumerate(statuses)
        ]
        return state

    def test_approved_concept_arts(self):
        state = self._state_with_arts(
            [ConceptArtStatus.READY, ConceptArtStatus.APPROVED, ConceptArtStatus.APPROVED]
        )
        assert len(state.approved_concept_arts()) == 2

    def test_ready_concept_arts(self):
        state = self._state_with_arts(
            [ConceptArtStatus.READY, ConceptArtStatus.APPROVED, ConceptArtStatus.PENDING]
        )
        assert len(state.ready_concept_arts()) == 1

    def test_pending_mesh_approvals(self):
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p"
        )
        state.meshes = [
            MeshItem(concept_art_index=0, sub_name="n_1", status=MeshStatus.AWAITING_APPROVAL),
            MeshItem(concept_art_index=1, sub_name="n_2", status=MeshStatus.GENERATING_MESH),
        ]
        assert len(state.pending_mesh_approvals()) == 1


# ---------------------------------------------------------------------------
# Status enum round-trips
# ---------------------------------------------------------------------------


class TestEnumRoundTrips:
    def test_pipeline_status_serialises_as_string(self, tmp_path, minimal_pipeline):
        minimal_pipeline.status = PipelineStatus.CONCEPT_ART_REVIEW
        p = tmp_path / "s.json"
        minimal_pipeline.save(p)
        data = json.loads(p.read_text())
        assert data["status"] == "concept_art_review"

    def test_symmetry_axis_serialises_as_string(self, tmp_path):
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p",
            symmetry_axis=SymmetryAxis.X_MINUS,
        )
        p = tmp_path / "s.json"
        state.save(p)
        data = json.loads(p.read_text())
        assert data["symmetry_axis"] == "x-"
