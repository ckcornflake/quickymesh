"""
Smoke test for the Blender screenshot worker.

Run from the repo root (Blender must be installed at the path in defaults.yaml):
    python scripts/smoke_test_screenshots.py --mesh path/to/mesh.glb

Optional flags:
    --mesh     path/to/mesh.glb     Mesh to render (required)
    --blender  "C:/path/blender.exe" Override Blender path
    --hdri                           Use HDRI/PBR lighting (for textured mesh)
    --out      smoke_screenshots/    Output directory

The script renders all 6 views, builds a review sheet, and generates an HTML
preview.  Open the HTML in a browser to verify the 3-D viewer works.

Exit code 0 = success, 1 = failure.
"""

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from src.config import config
from src.image_utils import make_review_sheet
from src.screenshot_pipeline import make_html_preview
from src.workers.screenshot import BlenderScreenshotWorker


def parse_args():
    p = argparse.ArgumentParser(description="Smoke-test Blender screenshot rendering.")
    p.add_argument("--mesh",    required=True,  help="Path to .glb or .obj mesh file")
    p.add_argument("--blender", default=None,   help="Override Blender executable path")
    p.add_argument("--hdri",    action="store_true", help="Use HDRI lighting (textured mesh)")
    p.add_argument("--out",     default="smoke_screenshots", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"ERROR: Mesh not found: {mesh_path}")
        sys.exit(1)

    blender_path = Path(args.blender) if args.blender else config.blender_path
    if not blender_path.exists():
        print(f"ERROR: Blender not found at: {blender_path}")
        print("  Check 'blender_path' in defaults.yaml or pass --blender.")
        sys.exit(1)

    out_dir = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Blender  : {blender_path}")
    print(f"Mesh     : {mesh_path}")
    print(f"Lighting : {'HDRI (textured)' if args.hdri else 'Matcap (clay)'}")
    print(f"Out dir  : {out_dir}")
    print()

    worker = BlenderScreenshotWorker(blender_path=blender_path, repo_root=_REPO)

    print("Rendering 6 views (this takes 30–90 s)...", flush=True)
    try:
        screenshots = worker.take_screenshots(
            mesh_path=mesh_path,
            output_dir=out_dir / "views",
            use_hdri=args.hdri,
            resolution=1024,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Rendered {len(screenshots)} views:")
    for p in screenshots:
        print(f"    {p}")

    print("\nBuilding review sheet...", flush=True)
    sheet = make_review_sheet(screenshots, out_dir / "reviewsheet.png", thumb_size=256)
    print(f"  Review sheet: {sheet}")

    print("\nBuilding HTML preview...", flush=True)
    try:
        html = make_html_preview(mesh_path, out_dir / "preview.html", size=512)
        print(f"  HTML preview: {html}")
        print(f"  Open in browser: file:///{html.as_posix()}")
    except Exception as e:
        print(f"  WARNING: HTML preview failed: {e}")

    print("\nSmoke test PASSED.")
    print(f"\nOpen {out_dir / 'reviewsheet.png'} to check render quality.")


if __name__ == "__main__":
    main()
