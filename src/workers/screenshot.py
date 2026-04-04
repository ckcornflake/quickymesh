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
    def cleanup_mesh(
        self,
        mesh_path: Path,
        output_path: Path,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
    ) -> Path:
        """
        Apply shade-smooth and optional symmetrize to `mesh_path`, write the
        result to `output_path`, and return `output_path`.
        Raises RuntimeError if Blender fails.
        """

    @abstractmethod
    def take_screenshots(
        self,
        mesh_path: Path,
        output_dir: Path,
        use_hdri: bool = False,
        views: list[str] | None = None,
        resolution: int = 1024,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
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

    # The cleanup script, relative to repo root
    _CLEANUP_SCRIPT_REL = "blender_scripts/cleanup.py"

    def cleanup_mesh(
        self,
        mesh_path: Path,
        output_path: Path,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
    ) -> Path:
        cleanup_script = (
            Path(__file__).parent.parent.parent / self._CLEANUP_SCRIPT_REL
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self._blender),
            "--background",
            "--python", str(cleanup_script),
            "--",
            "--input",  str(Path(mesh_path).resolve()),
            "--output", str(output_path.resolve()),
        ]
        if symmetrize:
            cmd += ["--symmetrize", "--symmetry_axis", symmetry_axis]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender cleanup failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )
        # Blender scripts can fail internally yet still exit 0 — verify the
        # output was actually written.
        if not output_path.exists():
            raise RuntimeError(
                f"Blender cleanup exited 0 but did not produce: {output_path}\n"
                f"Blender stderr: {result.stderr[-2000:]}"
            )
        return output_path

    def take_screenshots(
        self,
        mesh_path: Path,
        output_dir: Path,
        use_hdri: bool = False,
        views: list[str] | None = None,
        resolution: int = 1024,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
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
        found = [
            output_dir / f"review_{v}.png"
            for v in views
            if (output_dir / f"review_{v}.png").exists()
        ]
        if not found:
            # Blender exited 0 but produced nothing — surface the stderr so the
            # issue can be diagnosed without digging into Blender output manually.
            import logging as _log
            _log.getLogger(__name__).warning(
                "Blender screenshot produced no PNG files for %s.\nBlender stderr: %s",
                mesh_path, result.stderr[-2000:],
            )
        return found


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

    def cleanup_mesh(
        self,
        mesh_path: Path,
        output_path: Path,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
    ) -> Path:
        if self._fail:
            raise RuntimeError("MockScreenshotWorker: simulated cleanup failure")
        # Just copy the file — mock doesn't actually modify geometry
        import shutil
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(mesh_path), str(output_path))
        self.calls.append({
            "action": "cleanup",
            "mesh_path": mesh_path,
            "output_path": output_path,
            "symmetrize": symmetrize,
            "symmetry_axis": symmetry_axis,
        })
        return output_path

    def take_screenshots(
        self,
        mesh_path: Path,
        output_dir: Path,
        use_hdri: bool = False,
        views: list[str] | None = None,
        resolution: int = 1024,
        symmetrize: bool = False,
        symmetry_axis: str = "x-",
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
