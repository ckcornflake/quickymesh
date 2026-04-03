"""
Headless Blender script — mesh cleanup phase.

Applies post-processing to a Trellis-generated mesh and re-exports it as a
cleaned GLB.  This step runs between texturing and screenshotting so both the
final game asset and the review screenshots reflect the cleaned mesh.

Operations (applied in order):
  1. Shade smooth  — replaces flat per-face normals with smooth per-vertex
                     normals, eliminating the faceted look.
  2. Symmetrize    — optional; mirrors one half of the mesh across an axis
                     to produce a perfectly symmetrical result.
  3. Export        — writes the modified mesh to --output as a GLB.

Run via BlenderScreenshotWorker.cleanup_mesh() (never call directly):

    blender --background --python blender_scripts/cleanup.py -- \\
        --input   textured_mesh.glb \\
        --output  cleaned_mesh.glb \\
        [--symmetrize] [--symmetry_axis auto|x-|x+|y-|y+|z-|z+]

Exit code 0 = success.
"""

import bpy
import sys
import os
import math
import argparse
from mathutils import Vector


# ---------------------------------------------------------------------------
# Argument parsing  (everything after "--")
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Path to input .glb")
    parser.add_argument("--output",  required=True, help="Path to write cleaned .glb")
    parser.add_argument("--symmetrize",    action="store_true")
    parser.add_argument("--symmetry_axis", default="auto",
                        choices=["auto", "x-", "x+", "y-", "y+", "z-", "z+"])
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def load_mesh(filepath: str) -> list:
    """Import the GLB and return all mesh objects."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=filepath)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects found in: {filepath}")
    return meshes


# ---------------------------------------------------------------------------
# Shade smooth
# ---------------------------------------------------------------------------

def apply_shade_smooth(meshes: list) -> None:
    """
    Apply smooth shading to all mesh objects.

    Uses shade_smooth_by_angle (Blender 4.1+) so edges sharper than 30° stay
    hard — preserving intentional creases while eliminating polygon faceting.
    Falls back to shade_smooth() on older builds.
    """
    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            # Blender 4.1+ preferred path
            bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
        except AttributeError:
            # Fallback for older Blender builds
            bpy.ops.object.shade_smooth()
        obj.select_set(False)
    print(f"  Shade smooth applied to {len(meshes)} mesh(es)")


# ---------------------------------------------------------------------------
# Symmetrize
# ---------------------------------------------------------------------------

_DIR_MAP = {
    "x-": "NEGATIVE_X",
    "x+": "POSITIVE_X",
    "y-": "NEGATIVE_Y",
    "y+": "POSITIVE_Y",
    "z-": "NEGATIVE_Z",
    "z+": "POSITIVE_Z",
}


def apply_symmetrize(meshes: list, axis_spec: str) -> None:
    """
    Mirror the mesh across the specified axis to produce a symmetric result.

    axis_spec:
      "auto"  — symmetrize across the axis with the widest extent.
      "x-"/"x+" — keep the negative/positive X half and mirror it.
      (same for y and z)
    """
    if axis_spec == "auto":
        extents = {"x": 0.0, "y": 0.0, "z": 0.0}
        for obj in meshes:
            d = obj.dimensions
            extents["x"] = max(extents["x"], d.x)
            extents["y"] = max(extents["y"], d.y)
            extents["z"] = max(extents["z"], d.z)
        largest = max(extents, key=extents.get)
        direction = f"NEGATIVE_{largest.upper()}"
    else:
        direction = _DIR_MAP.get(axis_spec, "NEGATIVE_X")

    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.symmetrize(direction=direction)
        bpy.ops.object.mode_set(mode="OBJECT")

    print(f"  Symmetrized {len(meshes)} mesh(es) (direction={direction})")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_glb(output_path: str) -> None:
    """Export the entire scene as a GLB file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        use_selection=False,
        export_apply=True,          # apply modifiers
        export_materials="EXPORT",
        export_texcoords=True,
    )
    print(f"  Exported: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"[cleanup] Input:  {args.input}")
    print(f"[cleanup] Output: {args.output}")

    meshes = load_mesh(args.input)
    print(f"[cleanup] Loaded {len(meshes)} mesh object(s)")

    apply_shade_smooth(meshes)

    if args.symmetrize:
        apply_symmetrize(meshes, args.symmetry_axis)

    export_glb(args.output)
    print("[cleanup] Done.")


try:
    main()
except Exception as exc:
    import traceback
    traceback.print_exc()
    sys.exit(1)
