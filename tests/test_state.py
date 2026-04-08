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
    Pipeline3DState,
    Pipeline3DStatus,
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

    def test_default_version_is_zero(self):
        item = ConceptArtItem(index=0)
        assert item.version == 0

    def test_image_filename_reflects_index_and_version(self):
        item = ConceptArtItem(index=0, version=0)
        assert item.image_filename() == "concept_art_1_0.png"
        item.version = 3
        assert item.image_filename() == "concept_art_1_3.png"

    def test_image_filename_second_slot(self):
        item = ConceptArtItem(index=1, version=2)
        assert item.image_filename() == "concept_art_2_2.png"

    def test_status_transition(self):
        item = ConceptArtItem(index=0, status=ConceptArtStatus.GENERATING)
        item.status = ConceptArtStatus.READY
        assert item.status == ConceptArtStatus.READY


# ---------------------------------------------------------------------------
# PipelineState creation
# ---------------------------------------------------------------------------


class TestPipelineStateCreation:
    def test_required_fields(self):
        state = PipelineState(
            name="my_model",
            description="a red car",
            num_polys=8000,
            pipeline_dir="pipelines/my_model",
        )
        assert state.name == "my_model"
        assert state.num_polys == 8000
        assert state.status == PipelineStatus.INITIALIZING

    def test_default_symmetry(self):
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p"
        )
        assert state.symmetrize is False
        assert state.symmetry_axis == SymmetryAxis.X_MINUS

    def test_concept_arts_empty_by_default(self):
        state = PipelineState(
            name="n", description="d", num_polys=1000, pipeline_dir="p"
        )
        assert state.concept_arts == []

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
            ConceptArtItem(index=0, status=ConceptArtStatus.READY, version=2),
            ConceptArtItem(index=1, status=ConceptArtStatus.APPROVED),
        ]
        p = tmp_path / "state.json"
        state.save(p)
        loaded = PipelineState.load(p)
        assert len(loaded.concept_arts) == 2
        assert loaded.concept_arts[0].version == 2
        assert loaded.concept_arts[1].status == ConceptArtStatus.APPROVED


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


# ---------------------------------------------------------------------------
# Pipeline3DState
# ---------------------------------------------------------------------------


class TestPipeline3DState:
    def _make_state(self, tmp_path, **kwargs) -> Pipeline3DState:
        defaults = dict(
            name="ship_1_0",
            input_image_path="/tmp/input.png",
            num_polys=8000,
            pipeline_dir="pipelines/ship_1_0",
        )
        defaults.update(kwargs)
        return Pipeline3DState(**defaults)

    def test_default_status_is_queued(self, tmp_path):
        state = self._make_state(tmp_path)
        assert state.status == Pipeline3DStatus.QUEUED

    def test_default_symmetry(self, tmp_path):
        state = self._make_state(tmp_path)
        assert state.symmetrize is False
        assert state.symmetry_axis == SymmetryAxis.X_MINUS

    def test_export_fields_default(self, tmp_path):
        state = self._make_state(tmp_path)
        assert state.export_version == 0
        assert state.export_paths == []

    def test_optional_paths_none(self, tmp_path):
        state = self._make_state(tmp_path)
        assert state.mesh_path is None
        assert state.textured_mesh_path is None
        assert state.screenshot_dir is None
        assert state.review_sheet_path is None
        assert state.html_preview_path is None

    def test_source_fields_none_for_upload(self, tmp_path):
        state = self._make_state(tmp_path, name="u_myship", pipeline_dir="pipelines/u_myship")
        assert state.source_2d_pipeline is None
        assert state.source_concept_art_index is None
        assert state.source_concept_art_version is None

    def test_source_fields_set_for_derived(self, tmp_path):
        state = self._make_state(
            tmp_path,
            source_2d_pipeline="ship",
            source_concept_art_index=0,
            source_concept_art_version=2,
        )
        assert state.source_2d_pipeline == "ship"
        assert state.source_concept_art_index == 0
        assert state.source_concept_art_version == 2

    def test_save_and_load_round_trip(self, tmp_path):
        state = self._make_state(tmp_path, status=Pipeline3DStatus.MESH_DONE, mesh_path="/tmp/mesh.glb")
        p = tmp_path / "state.json"
        state.save(p)
        loaded = Pipeline3DState.load(p)
        assert loaded.name == state.name
        assert loaded.status == Pipeline3DStatus.MESH_DONE
        assert loaded.mesh_path == "/tmp/mesh.glb"

    def test_save_creates_parent_dirs(self, tmp_path):
        state = self._make_state(tmp_path)
        p = tmp_path / "deep" / "nested" / "state.json"
        state.save(p)
        assert p.exists()

    def test_serialises_status_as_string(self, tmp_path):
        state = self._make_state(tmp_path, status=Pipeline3DStatus.AWAITING_APPROVAL)
        p = tmp_path / "state.json"
        state.save(p)
        data = json.loads(p.read_text())
        assert data["status"] == "awaiting_approval"

    def test_touch_updates_updated_at(self, tmp_path):
        state = self._make_state(tmp_path)
        before = state.updated_at
        state.touch()
        assert state.updated_at >= before


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
