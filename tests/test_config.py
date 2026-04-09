"""
Tests for src/config.py — configuration loading and precedence.
"""

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from src.config import Config, _REPO_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_defaults(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Write a minimal defaults.yaml to tmp_path and return its path."""
    base = {
        "gemini": {
            "model": "gemini-test-model",
            "alternative_model": "gemini-alt-model",
        },
        "generation": {
            "num_concept_arts": 4,
            "num_polys": 8000,
            "review_sheet_thumb_size": 256,
            "html_preview_size": 512,
            "export_format": "glb",
            "background_suffix": "plain white background",
        },
        "infrastructure": {
            "comfyui_url": "http://localhost:8188",
            "comfyui_install_dir": "/fake/comfyui",
            "comfyui_poll_interval": 2.0,
            "comfyui_timeout": 600.0,
            "blender_path": "/fake/blender",
        },
        "output": {"root": "pipeline_root"},
    }
    if overrides:
        base.update(overrides)
    p = tmp_path / "defaults.yaml"
    p.write_text(yaml.dump(base), encoding="utf-8")
    return p


def _make_env(tmp_path: Path, contents: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(textwrap.dedent(contents), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_loads_gemini_model_from_defaults(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=fake_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.gemini_model == "gemini-test-model"

    def test_loads_num_concept_arts_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NUM_CONCEPT_ARTS", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=fake_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.num_concept_arts == 4

    def test_loads_num_polys_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NUM_POLYS", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=fake_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.num_polys == 8000

    def test_review_sheet_thumb_size_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("REVIEW_SHEET_THUMB_SIZE", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=fake_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.review_sheet_thumb_size == 256

    def test_export_format_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EXPORT_FORMAT", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=fake_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.export_format == "glb"

    def test_background_suffix_present(self, tmp_path):
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=fake_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert "white background" in cfg.background_suffix


class TestEnvOverrides:
    def test_gemini_model_overridden_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-override-model")
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=fake_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.gemini_model == "gemini-override-model"
        monkeypatch.delenv("GEMINI_MODEL")

    def test_num_polys_overridden_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NUM_POLYS", "4000")
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=fake_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.num_polys == 4000
        monkeypatch.delenv("NUM_POLYS")


class TestApiKeyHandling:
    def test_api_key_loaded_from_env_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=test_secret_key\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.gemini_api_key == "test_secret_key"

    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "# no key here\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            _ = cfg.gemini_api_key


class TestOutputPaths:
    def test_output_root_is_absolute(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OUTPUT_ROOT", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=k\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.output_root.is_absolute()

    def test_output_root_env_override_absolute(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "custom_root"))
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=k\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.output_root == tmp_path / "custom_root"
        monkeypatch.delenv("OUTPUT_ROOT")

    def test_subdirectory_paths_are_under_output_root(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OUTPUT_ROOT", raising=False)
        defaults = _write_defaults(tmp_path)
        env = _make_env(tmp_path, "GEMINI_API_KEY=k\n")
        cfg = Config(defaults_path=defaults, env_path=env)
        assert cfg.final_assets_dir == cfg.output_root / "final_game_ready_assets"
        assert cfg.pipelines_dir == cfg.output_root / "pipelines"

    def test_workflow_paths_exist(self):
        # Actual workflow files should be present in the repo
        from src.config import config
        assert config.workflow_generate.exists(), f"Missing: {config.workflow_generate}"
        assert config.workflow_texture.exists(), f"Missing: {config.workflow_texture}"


class TestMissingDefaultsFile:
    def test_raises_if_defaults_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="defaults.yaml"):
            Config(defaults_path=tmp_path / "nonexistent.yaml")
