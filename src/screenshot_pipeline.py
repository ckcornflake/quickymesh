"""
Screenshot + HTML preview pipeline — orchestration layer.

Public API
----------
run_screenshots(state, worker, pipeline_dir, cfg) -> None
make_html_preview(mesh_path, output_path, size) -> Path
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config, config as _default_config
from src.image_utils import make_review_sheet
from src.state import MeshItem, MeshStatus, PipelineState
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

    # Inject a <style> block to set the canvas to the requested size
    size_style = (
        f"<style>canvas {{ width: {size}px !important; height: {size}px !important; }}</style>"
    )
    html = html.replace("</head>", f"{size_style}\n</head>", 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Screenshot orchestration
# ---------------------------------------------------------------------------


def run_screenshots(
    state: PipelineState,
    worker: ScreenshotWorker,
    pipeline_dir: Path,
    cfg: Config = _default_config,
) -> None:
    """
    Take screenshots for every mesh in TEXTURE_DONE status.

    For each such mesh:
      1. Render 6 turntable views via `worker`.
      2. Concatenate them into a review sheet.
      3. Generate a Three.js HTML preview.
      4. Advance mesh status to SCREENSHOT_DONE.

    Uses HDRI lighting when a textured mesh is available; clay/matcap otherwise.
    Updates state in place.  Caller is responsible for saving state to disk.
    """
    pipeline_dir = Path(pipeline_dir)

    for mesh_item in state.meshes:
        if mesh_item.status != MeshStatus.TEXTURE_DONE:
            continue

        sub_dir = pipeline_dir / mesh_item.sub_name
        screenshot_dir = sub_dir / "screenshots"

        # Prefer the textured GLB for rendering; fall back to raw mesh
        render_path = Path(
            mesh_item.textured_mesh_path
            if mesh_item.textured_mesh_path
            else mesh_item.mesh_path
        )
        use_hdri = mesh_item.textured_mesh_path is not None

        mesh_item.status = MeshStatus.SCREENSHOT_PENDING

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
            mesh_item.review_sheet_path = str(review_sheet)

        # 3. HTML preview
        try:
            html_path = make_html_preview(
                render_path,
                output_path=sub_dir / "preview.html",
                size=cfg.html_preview_size,
            )
            mesh_item.html_preview_path = str(html_path)
        except Exception as e:
            # HTML preview is nice-to-have; don't block the pipeline on failure
            import logging
            logging.getLogger(__name__).warning(f"HTML preview failed: {e}")

        mesh_item.screenshot_dir = str(screenshot_dir)
        mesh_item.status = MeshStatus.SCREENSHOT_DONE
