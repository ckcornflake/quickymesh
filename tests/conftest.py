"""
Shared pytest fixtures.
"""

import pytest
from pathlib import Path

from src.state import PipelineState, PipelineStatus


@pytest.fixture
def minimal_pipeline(tmp_path) -> PipelineState:
    """A freshly-created pipeline state with required fields only."""
    return PipelineState(
        name="test_dragon",
        description="a dragon",
        num_polys=8000,
        pipeline_dir="uncompleted_pipelines/test_dragon",
    )


@pytest.fixture
def state_file(tmp_path, minimal_pipeline) -> tuple[PipelineState, Path]:
    """A pipeline state saved to a temp file; returns (state, path)."""
    p = tmp_path / "state.json"
    minimal_pipeline.save(p)
    return minimal_pipeline, p
