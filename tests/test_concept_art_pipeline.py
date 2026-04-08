"""
Tests for src/concept_art_pipeline.py — orchestration logic.

All tests use MockConceptArtWorker and MockPromptInterface so no external
APIs or human interaction is required.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from src.concept_art_pipeline import (
    build_prompt,
    build_review_sheet,
    generate_concept_arts,
    modify_concept_art,
    regenerate_concept_arts,
    run_concept_art_review,
    _concept_art_path,
)
from src.config import Config
from src.prompt_interface import MockPromptInterface
from src.state import (
    ConceptArtItem,
    ConceptArtStatus,
    PipelineState,
    PipelineStatus,
)
from src.workers.concept_art import MockConceptArtWorker

import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Minimal config pointing output to tmp_path."""
    defaults = {
        "gemini": {"model": "m", "alternative_model": "a"},
        "generation": {
            "num_concept_arts": 3,
            "num_polys": 8000,
            "review_sheet_thumb_size": 64,
            "html_preview_size": 512,
            "export_format": "glb",
            "background_suffix": "plain white background",
        },
        "infrastructure": {
            "comfyui_url": "http://localhost:8188",
            "comfyui_install_dir": "/fake",
            "comfyui_poll_interval": 2.0,
            "comfyui_timeout": 600.0,
            "blender_path": "/fake/blender",
        },
        "output": {"root": str(tmp_path / "output")},
    }
    d = tmp_path / "defaults.yaml"
    d.write_text(yaml.dump(defaults))
    e = tmp_path / ".env"
    e.write_text("GEMINI_API_KEY=fake\n")
    return Config(defaults_path=d, env_path=e)


@pytest.fixture
def state() -> PipelineState:
    return PipelineState(
        name="test_model",
        description="a red dragon",
        num_polys=8000,
        pipeline_dir="uncompleted_pipelines/test_model",
    )


@pytest.fixture
def pipeline_dir(tmp_path) -> Path:
    d = tmp_path / "uncompleted_pipelines" / "test_model"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_appends_background_suffix(self):
        prompt = build_prompt("a red dragon", "plain white background")
        assert "a red dragon" in prompt
        assert "plain white background" in prompt

    def test_strips_trailing_comma_from_description(self):
        prompt = build_prompt("a car,", "white background")
        assert not prompt.startswith("a car,.")

    def test_returns_string(self):
        assert isinstance(build_prompt("x", "y"), str)


# ---------------------------------------------------------------------------
# generate_concept_arts
# ---------------------------------------------------------------------------


class TestGenerateConceptArts:
    def test_generates_correct_number_of_images(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        assert len(state.concept_arts) == cfg.num_concept_arts

    def test_all_concept_arts_are_ready(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        for item in state.concept_arts:
            assert item.status == ConceptArtStatus.READY

    def test_image_files_exist_on_disk(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        for item in state.concept_arts:
            assert _concept_art_path(pipeline_dir, item.index, item.version).exists()

    def test_images_are_1024x1024(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        for item in state.concept_arts:
            img = Image.open(_concept_art_path(pipeline_dir, item.index, item.version))
            assert img.size == (1024, 1024)

    def test_images_have_white_background(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker(image_size=32)
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        for item in state.concept_arts:
            img = Image.open(_concept_art_path(pipeline_dir, item.index, item.version))
            assert img.getpixel((0, 0)) == (255, 255, 255)

    def test_prompt_saved_to_all_prompts(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        assert len(state.all_prompts) == 1
        assert "plain white background" in state.all_prompts[0]
        assert "a red dragon" in state.all_prompts[0]

    def test_state_status_becomes_concept_art_review(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        assert state.status == PipelineStatus.CONCEPT_ART_REVIEW

    def test_worker_receives_correct_prompt(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        assert len(worker.generate_prompts) == cfg.num_concept_arts
        for p in worker.generate_prompts:
            assert "plain white background" in p

    def test_skips_already_ready_items(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        state.concept_arts = [
            ConceptArtItem(index=0, status=ConceptArtStatus.READY),
            ConceptArtItem(index=1, status=ConceptArtStatus.PENDING),
            ConceptArtItem(index=2, status=ConceptArtStatus.PENDING),
        ]
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        # Only 2 pending items should be generated
        assert worker._generate_call_count == 2


# ---------------------------------------------------------------------------
# regenerate_concept_arts
# ---------------------------------------------------------------------------


class TestRegenerateConceptArts:
    def _setup(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        return worker

    def test_regenerates_only_specified_indices(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        worker2 = MockConceptArtWorker(colors=[(10, 10, 10)])
        regenerate_concept_arts(state, worker2, pipeline_dir, [0], cfg)
        assert worker2._generate_call_count == 1

    def test_regenerated_image_is_ready(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        worker2 = MockConceptArtWorker()
        regenerate_concept_arts(state, worker2, pipeline_dir, [1], cfg)
        assert state.concept_arts[1].status == ConceptArtStatus.READY

    def test_regenerated_file_exists(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        worker2 = MockConceptArtWorker()
        regenerate_concept_arts(state, worker2, pipeline_dir, [0], cfg)
        item = state.concept_arts[0]
        assert _concept_art_path(pipeline_dir, item.index, item.version).exists()

    def test_regenerate_increments_version(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        original_version = state.concept_arts[0].version
        regenerate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, [0], cfg)
        assert state.concept_arts[0].version == original_version + 1

    def test_invalid_index_raises(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        with pytest.raises(IndexError):
            regenerate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, [99], cfg)

    def test_prompt_appended_to_item_prompts(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        regenerate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, [0], cfg)
        assert len(state.concept_arts[0].prompts) == 2  # original + regen


# ---------------------------------------------------------------------------
# modify_concept_art
# ---------------------------------------------------------------------------


class TestModifyConceptArt:
    def _setup(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)

    def test_modify_updates_image_file(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        worker = MockConceptArtWorker(modify_color=(1, 2, 3))
        modify_concept_art(state, worker, pipeline_dir, 0, "make it blue")
        item = state.concept_arts[0]
        img = Image.open(_concept_art_path(pipeline_dir, item.index, item.version))
        assert img.size == (1024, 1024)

    def test_modify_increments_version(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        original_version = state.concept_arts[0].version
        modify_concept_art(state, MockConceptArtWorker(), pipeline_dir, 0, "make it blue")
        assert state.concept_arts[0].version == original_version + 1

    def test_modify_records_instruction(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        worker = MockConceptArtWorker()
        modify_concept_art(state, worker, pipeline_dir, 0, "remove wings")
        assert "remove wings" in worker.modify_instructions

    def test_modify_appends_to_all_prompts(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        worker = MockConceptArtWorker()
        modify_concept_art(state, worker, pipeline_dir, 0, "remove wings")
        assert any("remove wings" in p for p in state.all_prompts)

    def test_modify_status_is_ready_after(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        modify_concept_art(state, MockConceptArtWorker(), pipeline_dir, 1, "add fire")
        assert state.concept_arts[1].status == ConceptArtStatus.READY

    def test_modify_invalid_index_raises(self, state, pipeline_dir, cfg):
        self._setup(state, pipeline_dir, cfg)
        with pytest.raises(IndexError):
            modify_concept_art(state, MockConceptArtWorker(), pipeline_dir, 99, "x")

    def test_modify_without_image_raises(self, state, pipeline_dir, cfg):
        # Item at version 0 with no file on disk
        state.concept_arts = [ConceptArtItem(index=0)]
        with pytest.raises(ValueError, match="no image"):
            modify_concept_art(state, MockConceptArtWorker(), pipeline_dir, 0, "x")


# ---------------------------------------------------------------------------
# build_review_sheet
# ---------------------------------------------------------------------------


class TestBuildReviewSheet:
    def test_creates_review_sheet_file(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        sheet = build_review_sheet(state, pipeline_dir, cfg)
        assert sheet.exists()
        assert sheet.suffix == ".png"

    def test_review_sheet_is_valid_image(self, state, pipeline_dir, cfg):
        worker = MockConceptArtWorker()
        generate_concept_arts(state, worker, pipeline_dir, cfg)
        sheet = build_review_sheet(state, pipeline_dir, cfg)
        img = Image.open(sheet)
        assert img.mode == "RGB"

    def test_no_ready_arts_raises(self, state, pipeline_dir, cfg):
        state.concept_arts = [ConceptArtItem(index=0, status=ConceptArtStatus.PENDING)]
        with pytest.raises(ValueError, match="No ready"):
            build_review_sheet(state, pipeline_dir, cfg)


# ---------------------------------------------------------------------------
# run_concept_art_review (interactive loop)
# ---------------------------------------------------------------------------


class TestRunConceptArtReview:
    def _generate(self, state, pipeline_dir, cfg):
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        state.save(pipeline_dir / "state.json")

    def test_approve_returns_approved(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve 1 2"])
        result = run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert result == "approved"

    def test_approved_items_have_approved_status(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve 1"])
        run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert state.concept_arts[0].status == ConceptArtStatus.APPROVED

    def test_quit_returns_quit(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        ui = MockPromptInterface(["quit"])
        result = run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert result == "quit"

    def test_regenerate_then_approve(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        # Interactive: "regenerate" triggers "Which image?" prompt → pick 1
        ui = MockPromptInterface(["regenerate", "1", "approve 1"])
        result = run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert result == "approved"
        assert state.concept_arts[0].status == ConceptArtStatus.APPROVED

    def test_modify_then_approve(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        # "modify" → which image? → version? (only v0 exists, skipped) → instruction
        ui = MockPromptInterface(["modify", "1", "make it blue", "approve 1"])
        result = run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert result == "approved"

    def test_unknown_action_loops(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        # Unknown action → loop again → then approve
        ui = MockPromptInterface(["foobar", "approve 1"])
        result = run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert result == "approved"

    def test_state_saved_to_disk_after_approve(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve 1"])
        run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        loaded = PipelineState.load(pipeline_dir / "state.json")
        assert any(a.status == ConceptArtStatus.APPROVED for a in loaded.concept_arts)

    def test_review_sheet_shown_to_user(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve 1"])
        run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert len(ui.shown_images) >= 1

    def test_approve_keeps_pipeline_in_concept_art_review(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        ui = MockPromptInterface(["approve 1"])
        run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert state.status == PipelineStatus.CONCEPT_ART_REVIEW

    def test_regenerate_all_regenerates_every_image(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)  # initial generation via separate worker
        worker = MockConceptArtWorker()
        # Enter to keep the same prompt, then approve all
        ui = MockPromptInterface(["regenerate all", "", "approve 1 2 3"])
        run_concept_art_review(state, worker, pipeline_dir, ui, cfg)
        assert worker._generate_call_count == cfg.num_concept_arts  # only the regen

    def test_regenerate_all_with_new_description(self, state, pipeline_dir, cfg):
        self._generate(state, pipeline_dir, cfg)
        ui = MockPromptInterface(["regenerate all", "a blue robot", "approve 1 2 3"])
        run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert state.description == "a blue robot"

    def test_regenerate_all_keeps_description_on_empty_input(self, state, pipeline_dir, cfg):
        original = state.description
        self._generate(state, pipeline_dir, cfg)
        ui = MockPromptInterface(["regenerate all", "", "approve 1 2 3"])
        run_concept_art_review(state, MockConceptArtWorker(), pipeline_dir, ui, cfg)
        assert state.description == original

    def test_regenerate_does_not_use_modify_instruction_as_prompt(self, state, pipeline_dir, cfg):
        """Bug fix: regen after modify must NOT use the modify instruction as the generation prompt."""
        self._generate(state, pipeline_dir, cfg)
        # Modify image 1
        modify_concept_art(state, MockConceptArtWorker(), pipeline_dir, 0, "make it purple")
        # Now regenerate image 2 — should use original description, NOT "make it purple"
        regen_worker = MockConceptArtWorker()
        regenerate_concept_arts(state, regen_worker, pipeline_dir, [1], cfg)
        prompt_used = regen_worker.generate_prompts[-1]
        assert "make it purple" not in prompt_used
        assert state.description in prompt_used or cfg.background_suffix in prompt_used


# ---------------------------------------------------------------------------
# Regression — regen all chaining
# ---------------------------------------------------------------------------


class TestRegenerateAllChaining:
    def test_regen_all_indices_correct_count(self, state, pipeline_dir, cfg):
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        worker = MockConceptArtWorker()
        regenerate_concept_arts(state, worker, pipeline_dir, list(range(cfg.num_concept_arts)), cfg)
        assert worker._generate_call_count == cfg.num_concept_arts

    def test_description_override_stored(self, state, pipeline_dir, cfg):
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        regenerate_concept_arts(
            state, MockConceptArtWorker(), pipeline_dir, [0], cfg,
            description_override="a purple sphere"
        )
        assert state.description == "a purple sphere"

    def test_no_override_preserves_description(self, state, pipeline_dir, cfg):
        original = state.description
        generate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, cfg)
        regenerate_concept_arts(state, MockConceptArtWorker(), pipeline_dir, [0], cfg)
        assert state.description == original
