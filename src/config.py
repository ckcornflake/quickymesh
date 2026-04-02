"""
Configuration loader for quickymesh.

Loads defaults from defaults.yaml and secrets from .env.
Environment variables take precedence over defaults.yaml for all settings.

Usage:
    from src.config import config
    print(config.gemini_model)
    print(config.output_root)
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repo root is one level up from this file (src/config.py → repo root)
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULTS_PATH = _REPO_ROOT / "defaults.yaml"
_ENV_PATH = _REPO_ROOT / ".env"


def _load_defaults(path: Path = _DEFAULTS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"defaults.yaml not found at {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Config:
    """
    Merged configuration.  Attribute access is the public API — do not access
    the raw _data dict directly in application code.
    """

    def __init__(self, defaults_path: Path = _DEFAULTS_PATH, env_path: Path = _ENV_PATH):
        load_dotenv(env_path)
        self._data = _load_defaults(defaults_path)

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    @property
    def gemini_api_key(self) -> str:
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to .env or export it as an environment variable."
            )
        return key

    @property
    def gemini_model(self) -> str:
        return os.getenv("GEMINI_MODEL", self._data["gemini"]["model"])

    @property
    def gemini_alternative_model(self) -> str:
        return self._data["gemini"]["alternative_model"]

    # ------------------------------------------------------------------
    # Generation defaults
    # ------------------------------------------------------------------

    @property
    def num_concept_arts(self) -> int:
        return int(os.getenv("NUM_CONCEPT_ARTS", self._data["generation"]["num_concept_arts"]))

    @property
    def num_polys(self) -> int:
        return int(os.getenv("NUM_POLYS", self._data["generation"]["num_polys"]))

    @property
    def review_sheet_thumb_size(self) -> int:
        return int(
            os.getenv(
                "REVIEW_SHEET_THUMB_SIZE",
                self._data["generation"]["review_sheet_thumb_size"],
            )
        )

    @property
    def html_preview_size(self) -> int:
        return int(
            os.getenv("HTML_PREVIEW_SIZE", self._data["generation"]["html_preview_size"])
        )

    @property
    def export_format(self) -> str:
        return os.getenv("EXPORT_FORMAT", self._data["generation"]["export_format"])

    @property
    def background_suffix(self) -> str:
        return self._data["generation"]["background_suffix"]

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    @property
    def comfyui_url(self) -> str:
        return os.getenv("COMFYUI_URL", self._data["infrastructure"]["comfyui_url"])

    @property
    def comfyui_install_dir(self) -> Path:
        raw = os.getenv(
            "COMFYUI_INSTALL_DIR", self._data["infrastructure"]["comfyui_install_dir"]
        )
        return Path(raw)

    @property
    def comfyui_output_dir(self) -> Path:
        raw = os.getenv(
            "COMFYUI_OUTPUT_DIR", self._data["infrastructure"]["comfyui_output_dir"]
        )
        return Path(raw)

    @property
    def comfyui_poll_interval(self) -> float:
        return float(
            os.getenv(
                "COMFYUI_POLL_INTERVAL",
                self._data["infrastructure"]["comfyui_poll_interval"],
            )
        )

    @property
    def comfyui_timeout(self) -> float:
        return float(
            os.getenv(
                "COMFYUI_TIMEOUT", self._data["infrastructure"]["comfyui_timeout"]
            )
        )

    @property
    def blender_path(self) -> Path:
        raw = os.getenv("BLENDER_PATH", self._data["infrastructure"]["blender_path"])
        return Path(raw)

    @property
    def vram_lock_timeout(self) -> float:
        return float(
            os.getenv(
                "VRAM_LOCK_TIMEOUT",
                self._data["infrastructure"].get("vram_lock_timeout", 1800.0),
            )
        )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    @property
    def output_root(self) -> Path:
        raw = os.getenv("OUTPUT_ROOT", self._data["output"]["root"])
        p = Path(raw)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        return p

    @property
    def final_assets_dir(self) -> Path:
        return self.output_root / "final_game_ready_assets"

    @property
    def completed_pipelines_dir(self) -> Path:
        return self.output_root / "completed_pipelines"

    @property
    def uncompleted_pipelines_dir(self) -> Path:
        return self.output_root / "uncompleted_pipelines"

    # ------------------------------------------------------------------
    # ComfyUI workflow paths
    # ------------------------------------------------------------------

    @property
    def workflow_generate(self) -> Path:
        return _REPO_ROOT / "comfyui_workflows" / "trellis_generate.json"

    @property
    def workflow_texture(self) -> Path:
        return _REPO_ROOT / "comfyui_workflows" / "trellis_texture.json"


# Module-level singleton — import this in application code
config = Config()
