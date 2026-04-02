"""
Screenshot workers.

ScreenshotWorker        — abstract interface
BlenderScreenshotWorker — shells out to Blender headlessly
MockScreenshotWorker    — instant solid-colour PNGs for tests

The worker is responsible only for rendering images.
Building the review sheet and HTML preview is handled by
screenshot_pipeline.py.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

# The 6 canonical views, in the order they appear on the review sheet
DEFAULT_VIEWS = ["front", "back", "top", "bottom", "perspective", "perspective2"]

# Blender script, relative to repo root
_SCRIPT_REL = "blender_scripts/screenshot.py"


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class ScreenshotWorker(ABC):

    @abstractmethod
    def take_screenshots(
        self,
        mesh_path: Path,
        output_dir: Path,
        use_hdri: bool = False,
        views: list[str] | None = None,
        resolution: int = 1024,
    ) -> list[Path]:
        """
        Render turntable screenshots of `mesh_path` into `output_dir`.

        Parameters
        ----------
        mesh_path:   Path to the .glb (or .obj/.blend) to render.
        output_dir:  Directory to save rendered PNGs.
        use_hdri:    True for PBR/textured review; False for clay/matcap.
        views:       Subset of DEFAULT_VIEWS to render.  Defaults to all 6.
        resolution:  Square render resolution in pixels.

        Returns list of Paths to the rendered PNG files (in view order).
        Raises RuntimeError if rendering fails.
        """


# ---------------------------------------------------------------------------
# Blender implementation
# ---------------------------------------------------------------------------


class BlenderScreenshotWorker(ScreenshotWorker):
    """Runs Blender in --background mode to render the screenshots."""

    def __init__(self, blender_path: Path, repo_root: Path | None = None):
        self._blender = Path(blender_path)
        # If repo_root not supplied, infer from this file's location (src/workers/ → repo root)
        self._script = (
            Path(repo_root) if repo_root else Path(__file__).parent.parent.parent
        ) / _SCRIPT_REL

    def take_screenshots(
        self,
        mesh_path: Path,
        output_dir: Path,
        use_hdri: bool = False,
        views: list[str] | None = None,
        resolution: int = 1024,
    ) -> list[Path]:
        views = views or DEFAULT_VIEWS
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self._blender),
            "--background",
            "--python", str(self._script),
            "--",
            "--input",      str(Path(mesh_path).resolve()),
            "--output_dir", str(output_dir.resolve()),
            "--prefix",     "review",
            "--views",      ",".join(views),
            "--resolution", str(resolution),
            "--hdri" if use_hdri else "--matcap",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender screenshot failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        # Return only the files that actually exist (Blender skips unknown views)
        return [
            output_dir / f"review_{v}.png"
            for v in views
            if (output_dir / f"review_{v}.png").exists()
        ]


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------


class MockScreenshotWorker(ScreenshotWorker):
    """
    Deterministic stub that writes solid-colour PNGs instantly.

    Parameters
    ----------
    fail:        Raise RuntimeError on every call (simulates Blender crash).
    image_size:  Pixel dimensions of the generated square images.
    """

    _COLORS = [
        (60,  80, 120),   # front  — blue-grey
        (80,  60, 120),   # back   — purple-grey
        (60, 120,  80),   # top    — green-grey
        (120, 80,  60),   # bottom — brown-grey
        (120, 100, 60),   # perspective
        (60, 120, 120),   # perspective2
    ]

    def __init__(self, fail: bool = False, image_size: int = 64):
        self._fail = fail
        self._image_size = image_size
        self.calls: list[dict] = []

    def take_screenshots(
        self,
        mesh_path: Path,
        output_dir: Path,
        use_hdri: bool = False,
        views: list[str] | None = None,
        resolution: int = 1024,
    ) -> list[Path]:
        if self._fail:
            raise RuntimeError("MockScreenshotWorker: simulated Blender failure")

        views = views or DEFAULT_VIEWS
        self.calls.append({
            "mesh_path": mesh_path,
            "use_hdri": use_hdri,
            "views": views,
        })

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, view in enumerate(views):
            color = self._COLORS[i % len(self._COLORS)]
            img = Image.new("RGB", (self._image_size, self._image_size), color)
            p = output_dir / f"review_{view}.png"
            img.save(str(p))
            paths.append(p)
        return paths
