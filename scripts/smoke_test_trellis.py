"""
Smoke test for the Trellis ComfyUI mesh generation + texturing workflow.

Run from the repo root (ComfyUI must be running on port 8188):
    python scripts/smoke_test_trellis.py --image path/to/concept_art.png

Optional flags:
    --image   path/to/image.png   Concept art to use (required)
    --polys   8000                Target polygon count
    --url     http://localhost:8188
    --out     smoke_trellis/      Output directory for results
    --skip-texture                Only run mesh generation, skip texturing

Exit code 0 = success, 1 = failure.
"""

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from src.config import config
from src.workers.comfyui_client import ComfyUIClient
from src.workers.trellis import ComfyUITrellisWorker


def parse_args():
    p = argparse.ArgumentParser(description="Smoke-test the Trellis ComfyUI workflows.")
    p.add_argument("--image", required=True, help="Path to a 1024x1024 concept art PNG")
    p.add_argument("--polys", type=int, default=config.num_polys, help="Target polygon count")
    p.add_argument("--url", default=config.comfyui_url, help="ComfyUI base URL")
    p.add_argument("--out", default="smoke_trellis", help="Output directory")
    p.add_argument("--skip-texture", action="store_true", help="Skip texturing step")
    return p.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    out_dir = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ComfyUI URL      : {args.url}")
    print(f"ComfyUI output   : {config.comfyui_output_dir}")
    print(f"Image            : {image_path}")
    print(f"Target polys     : {args.polys}")
    print(f"Smoke output dir : {out_dir}")
    print()

    client = ComfyUIClient(
        base_url=args.url,
        poll_interval=config.comfyui_poll_interval,
        timeout=config.comfyui_timeout,
    )

    print("Checking ComfyUI is alive...", flush=True)
    if not client.is_alive():
        print(f"ERROR: ComfyUI not reachable at {args.url}")
        print("  Start ComfyUI and try again.")
        sys.exit(1)
    print("  ComfyUI is up.\n")

    comfyui_output_dir = config.comfyui_output_dir
    worker = ComfyUITrellisWorker(
        client=client,
        comfyui_output_dir=comfyui_output_dir,
        workflow_generate=config.workflow_generate,
        workflow_texture=config.workflow_texture,
    )

    # --- Mesh generation ---
    print("Step 1: Mesh generation (this may take several minutes)...", flush=True)
    t0 = time.perf_counter()
    try:
        mesh_path = worker.generate_mesh(
            image_path=image_path,
            dest_dir=out_dir / "meshes",
            num_polys=args.polys,
            job_id="smoke_test",
        )
    except Exception as e:
        print(f"ERROR during mesh generation: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Mesh saved: {mesh_path}")

    # Validate with trimesh
    try:
        import trimesh
        scene = trimesh.load(str(mesh_path))
        if hasattr(scene, "faces"):
            face_count = len(scene.faces)
        else:
            geoms = list(scene.geometry.values()) if hasattr(scene, "geometry") else []
            face_count = sum(len(g.faces) for g in geoms)
        print(f"  Face count: {face_count:,}")
        if face_count == 0:
            print("  WARNING: mesh has 0 faces — Trellis may have failed silently")
    except Exception as e:
        print(f"  (Could not validate with trimesh: {e})")

    if args.skip_texture:
        print("\nSkipping texturing (--skip-texture).")
        print("\nSmoke test PASSED (mesh generation only).")
        return

    # --- Texturing ---
    print(f"\nStep 2: Texturing (this may take several minutes)...", flush=True)
    t0 = time.perf_counter()
    try:
        textured_path = worker.texture_mesh(
            image_path=image_path,
            mesh_path=mesh_path,
            dest_dir=out_dir / "meshes",
            job_id="smoke_test",
        )
    except Exception as e:
        print(f"ERROR during texturing: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Textured mesh saved: {textured_path}")

    print("\nSmoke test PASSED.")
    print(f"\nOutputs are in: {out_dir}")
    print("Open the .glb files in a 3D viewer (e.g. https://gltf-viewer.donmccurdy.com) to verify quality.")


if __name__ == "__main__":
    main()
