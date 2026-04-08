"""
Screenshot + HTML preview pipeline — orchestration layer for 3D pipelines.

Operates on Pipeline3DState.

Public API
----------
run_cleanup(state, worker, pipeline_dir, cfg) -> None
run_screenshots(state, worker, pipeline_dir, cfg) -> None
make_html_preview(mesh_path, output_path, size) -> Path
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config, config as _default_config
from src.image_utils import make_review_sheet
from src.state import Pipeline3DState, Pipeline3DStatus
from src.workers.screenshot import ScreenshotWorker


# ---------------------------------------------------------------------------
# HTML preview
# ---------------------------------------------------------------------------


def make_html_preview(
    mesh_path: Path,
    output_path: Path,
    size: int = 512,
) -> Path:
    """
    Generate a standalone Three.js HTML viewer for `mesh_path` using trimesh.

    The HTML file is self-contained and can be opened directly in a browser
    to rotate the model interactively.

    Returns `output_path`.
    Raises ImportError if trimesh is not installed.
    Raises RuntimeError if the mesh cannot be loaded.
    """
    import trimesh
    from trimesh.viewer.notebook import scene_to_html

    mesh = trimesh.load(str(mesh_path))

    # trimesh.load returns a Trimesh for single meshes, Scene for multi-mesh GLBs
    if isinstance(mesh, trimesh.Trimesh):
        scene = mesh.scene()
    elif isinstance(mesh, trimesh.Scene):
        scene = mesh
    else:
        raise RuntimeError(f"Unexpected trimesh type: {type(mesh)}")

    html = scene_to_html(scene)

    # Inject styles: fixed canvas size, light-gray background, centered layout
    injected_style = (
        f"<style>\n"
        f"  body {{ margin: 0; display: flex; justify-content: center; "
        f"align-items: center; min-height: 100vh; background: #2a2a2a; }}\n"
        f"  canvas {{ width: {size}px !important; height: {size}px !important; "
        f"background: #3c3c3c !important; display: block; }}\n"
        f"</style>"
    )
    html = html.replace("</head>", f"{injected_style}\n</head>", 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Cleanup orchestration
# ---------------------------------------------------------------------------


def run_cleanup(
    state: Pipeline3DState,
    worker: ScreenshotWorker,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    Run mesh cleanup (shade-smooth + symmetrize) via Blender.

    Only runs if the pipeline is in TEXTURE_DONE status.
    Updates state in place.  Caller is responsible for saving state.
    """
    if state.status != Pipeline3DStatus.TEXTURE_DONE:
        return
    if not state.textured_mesh_path:
        return

    pipeline_dir = Path(pipeline_dir)
    mesh_dir = pipeline_dir / "meshes"
    cleaned_path = mesh_dir / "cleaned_mesh.glb"

    state.status = Pipeline3DStatus.CLEANING_UP

    worker.cleanup_mesh(
        mesh_path=Path(state.textured_mesh_path),
        output_path=cleaned_path,
        symmetrize=state.symmetrize,
        symmetry_axis=state.symmetry_axis.value,
    )

    state.textured_mesh_path = str(cleaned_path)
    state.status = Pipeline3DStatus.CLEANUP_DONE


# ---------------------------------------------------------------------------
# Screenshot orchestration
# ---------------------------------------------------------------------------


def run_screenshots(
    state: Pipeline3DState,
    worker: ScreenshotWorker,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    Take screenshots and build a review sheet + HTML preview.

    Accepts CLEANUP_DONE or TEXTURE_DONE (fallback when cleanup was skipped).
    Updates state in place.  Caller is responsible for saving state.
    """
    _ready = {Pipeline3DStatus.CLEANUP_DONE, Pipeline3DStatus.TEXTURE_DONE}
    if state.status not in _ready:
        return

    pipeline_dir = Path(pipeline_dir)
    screenshot_dir = pipeline_dir / "screenshots"

    render_path = Path(
        state.textured_mesh_path if state.textured_mesh_path else state.mesh_path
    )
    use_hdri = state.textured_mesh_path is not None

    state.status = Pipeline3DStatus.SCREENSHOT_PENDING

    # 1. Render views
    screenshots = worker.take_screenshots(
        mesh_path=render_path,
        output_dir=screenshot_dir,
        use_hdri=use_hdri,
        resolution=1024,
    )

    # 2. Review sheet
    if screenshots:
        review_sheet = make_review_sheet(
            screenshots,
            output_path=screenshot_dir / "reviewsheet.png",
            thumb_size=cfg.review_sheet_thumb_size,
        )
        state.review_sheet_path = str(review_sheet)

    # 3. HTML preview
    try:
        html_path = make_html_preview(
            render_path,
            output_path=pipeline_dir / "preview.html",
            size=cfg.html_preview_size,
        )
        state.html_preview_path = str(html_path)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"HTML preview failed: {e}")

    state.screenshot_dir = str(screenshot_dir)
    state.status = Pipeline3DStatus.SCREENSHOT_DONE
