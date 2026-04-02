"""
Headless Blender script — render turntable screenshots of a mesh.

Run via BlenderScreenshotWorker (never directly):
    blender --background --python blender_scripts/screenshot.py -- \
        --input  model.glb \
        --output_dir ./screenshots/ \
        --prefix  review \
        --views   front,back,top,bottom,perspective,perspective2 \
        --resolution 1024 \
        --matcap          (clay shading, for untextured mesh)
        --hdri            (PBR env lighting, for textured mesh)

Exit code 0 = success.  Each rendered view is saved as:
    {output_dir}/{prefix}_{view_name}.png
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
    parser.add_argument("--input",      required=True,  help="Path to .glb, .obj, or .blend")
    parser.add_argument("--output_dir", required=True,  help="Directory for rendered PNGs")
    parser.add_argument("--prefix",     default="review")
    parser.add_argument("--views",      default="front,back,top,bottom,perspective,perspective2")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--matcap",     action="store_true", help="Clay/matcap shading")
    parser.add_argument("--hdri",       action="store_true", help="PBR env lighting")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# View definitions: (azimuth_deg, elevation_deg)
# ---------------------------------------------------------------------------

VIEW_ANGLES = {
    "front":        (0,    15),
    "back":         (180,  15),
    "top":          (0,    88),
    "bottom":       (0,   -88),
    "perspective":  (35,   30),
    "perspective2": (-145, 25),
}


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def setup_scene(filepath: str):
    """Clear default scene, import mesh, centre and normalise to unit size."""
    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=filepath)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects found in scene.")

    # Centre all meshes at origin
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    for obj in meshes:
        obj.location = (0, 0, 0)

    # Scale to fit inside a 2-unit cube
    max_dim = max((max(obj.dimensions) for obj in meshes if max(obj.dimensions) > 0), default=1.0)
    if max_dim > 0:
        scale = 2.0 / max_dim
        for obj in meshes:
            obj.scale = (obj.scale[0] * scale, obj.scale[1] * scale, obj.scale[2] * scale)
    bpy.ops.object.transform_apply(scale=True)

    return meshes


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def add_camera(name: str, distance: float, azimuth_deg: float,
               elevation_deg: float) -> bpy.types.Object:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x =  distance * math.cos(el) * math.sin(az)
    y = -distance * math.cos(el) * math.cos(az)
    z =  distance * math.sin(el)

    cam_data = bpy.data.cameras.new(name)
    cam_obj  = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = (x, y, z)

    direction = Vector((0, 0, 0)) - Vector((x, y, z))
    rot_quat  = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()
    return cam_obj


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def setup_matcap_lighting(meshes):
    """Neutral clay shading — good for untextured / initial mesh review."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.film_transparent = False

    world = bpy.data.worlds.new("MatcapWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value    = (0.25, 0.25, 0.25, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world

    sun = bpy.data.lights.new("KeySun", type="SUN")
    sun.energy = 3.0
    sun_obj = bpy.data.objects.new("KeySun", sun)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(45), 0, math.radians(45))

    fill = bpy.data.lights.new("FillSun", type="SUN")
    fill.energy = 1.0
    fill_obj = bpy.data.objects.new("FillSun", fill)
    scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(-30), 0, math.radians(-120))

    clay = bpy.data.materials.new("Clay")
    clay.use_nodes = True
    bsdf = clay.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.72, 1.0)
        bsdf.inputs["Roughness"].default_value  = 0.8
        bsdf.inputs["Metallic"].default_value   = 0.0

    for obj in meshes:
        if not obj.data.materials:
            obj.data.materials.append(clay)


def setup_hdri_lighting():
    """Simple area-light rig — good for PBR textured mesh review."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"

    world = bpy.data.worlds.new("HDRIWorld")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg  = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.5
    bg.inputs["Color"].default_value    = (0.05, 0.05, 0.1, 1.0)
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    scene.world = world

    key = bpy.data.lights.new("KeyArea", type="AREA")
    key.energy = 500
    key.size   = 2.0
    key_obj = bpy.data.objects.new("KeyArea", key)
    scene.collection.objects.link(key_obj)
    key_obj.location       = (3, -3, 4)
    key_obj.rotation_euler = (math.radians(45), 0, math.radians(45))


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_view(cam_obj, output_path: str, resolution: int):
    scene = bpy.context.scene
    scene.camera = cam_obj
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    meshes = setup_scene(args.input)

    if args.hdri:
        setup_hdri_lighting()
    else:
        setup_matcap_lighting(meshes)

    views = [v.strip() for v in args.views.split(",")]
    rendered = []
    for view_name in views:
        if view_name not in VIEW_ANGLES:
            print(f"WARNING: Unknown view '{view_name}', skipping.")
            continue
        az, el = VIEW_ANGLES[view_name]
        cam = add_camera(f"Cam_{view_name}", distance=4.0, azimuth_deg=az, elevation_deg=el)
        out = os.path.join(args.output_dir, f"{args.prefix}_{view_name}.png")
        render_view(cam, out, args.resolution)
        rendered.append(out)
        print(f"  Rendered: {out}")

    print(f"Done. {len(rendered)} views rendered.")


main()
