#!/usr/bin/env python3
"""Render a calibrated camera array with Blender."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def material(name, color, metallic=0.0, roughness=0.4, emission=0.0):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1.0)
    result.use_nodes = True
    shader = result.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Emission Color"].default_value = (*color, 1.0)
    shader.inputs["Emission Strength"].default_value = emission
    return result


def cube(name, size, location, mat, parent=None, bevel=0.012):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.object
    obj.name, obj.dimensions = name, size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.parent, obj.location = parent, location
    obj.data.materials.append(mat)
    if bevel:
        modifier = obj.modifiers.new("Edge finish", "BEVEL")
        modifier.width, modifier.segments = bevel, 3
        obj.modifiers.new("Surface normals", "WEIGHTED_NORMAL")
    return obj


def cylinder(name, radius, depth, location, mat, parent=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=radius, depth=depth)
    obj = bpy.context.object
    obj.name, obj.parent, obj.location = name, parent, location
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def curve(name, points, mat, radius=0.002, parent=None, closed=False):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions, data.bevel_depth, data.bevel_resolution = "3D", radius, 2
    spline = data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for target, point in zip(spline.points, points):
        target.co = (*point, 1.0)
    spline.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    data.materials.append(mat)
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def camera_body(root, dynamic, graphite, accent, glass):
    cube(
        "Sensor housing",
        (0.18, 0.12, 0.12) if dynamic else (0.145, 0.10, 0.095),
        (0, 0, 0.02),
        graphite,
        root,
    )
    cube(
        "Camera top plate",
        (0.176, 0.009, 0.107) if dynamic else (0.143, 0.008, 0.09),
        (0, 0.064 if dynamic else 0.054, 0.02),
        accent,
        root,
        0.002,
    )
    depth = 0.14 if dynamic else 0.07
    cylinder(
        "Lens barrel",
        0.039 if dynamic else 0.033,
        depth,
        (0, 0, -0.04 - depth / 2),
        graphite,
        root,
    )
    cylinder(
        "Focus ring",
        0.043 if dynamic else 0.037,
        0.016,
        (0, 0, -0.045 - depth * 0.7),
        accent,
        root,
    )
    cylinder(
        "Front element", 0.035 if dynamic else 0.029, 0.007, (0, 0, -0.045 - depth), glass, root
    )


def build_scene(packet):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 32
    scene.render.resolution_x, scene.render.resolution_y = packet["render_size"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 12
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.25
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.008,
        0.016,
        0.028,
        1.0,
    )
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    graphite = material("Graphite housings", (0.028, 0.045, 0.070), 0.68, 0.24)
    cyan = material("Static cyan", (0.035, 0.61, 0.80), 0.58, 0.26, 0.65)
    amber = material("Dynamic amber", (1.0, 0.49, 0.12), 0.5, 0.24, 0.7)
    glass = material("Optical glass", (0.016, 0.12, 0.19), 0.88, 0.13, 0.3)
    guide = material("Survey guides", (0.018, 0.055, 0.078), 0.25, 0.6, 0.35)
    beam = material("Dynamic field of view", (0.45, 0.22, 0.06), 0.1, 0.55, 0.45)
    trail = material("Motion paths", (0.70, 0.29, 0.045), 0.12, 0.4, 0.7)
    floor_mat = material("Studio floor", (0.001, 0.0015, 0.0025), 0.06, 0.7)
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.12
    cameras = packet["static"] + packet["dynamic"]
    focus = packet.get("focus_views", [])
    context = bool(focus and packet.get("global_context"))
    highlight = packet.get("highlight_views", [])
    context_accent = material("Context cameras", (0.07, 0.19, 0.25), 0.3, 0.5, 0.25)
    context_floor = min(pose[2][3] for c in cameras for pose in c["poses"]) - 0.38
    if focus:
        cameras = [camera for camera in packet["dynamic"] if camera["id"] in focus]
        floor_shader = floor_mat.node_tree.nodes["Principled BSDF"]
        floor_shader.inputs["Metallic"].default_value = 0
        floor_shader.inputs["Roughness"].default_value = 1
        floor_shader.inputs["Specular IOR Level"].default_value = 0
    positions = [
        Vector(pose[0:3][i][3] for i in range(3)) for camera in cameras for pose in camera["poses"]
    ]
    glyph_scale = min(
        1.0,
        max(
            0.45,
            max(max(p[i] for p in positions) - min(p[i] for p in positions) for i in range(3))
            / 6.0,
        ),
    )
    if focus:
        centers = [Matrix(camera["poses"][0]).translation for camera in cameras]
        spacing = min((a - b).length for i, a in enumerate(centers) for b in centers[i + 1:])
        glyph_scale = max(spacing, 0.001) / 0.65
    glyph_matrix = Matrix.Scale(glyph_scale, 4)
    guide_scale = glyph_scale if focus else 1.0
    floor = min(p.z for p in positions) - (1.4 * spacing if focus else 0.38)
    center = Vector(
        (
            sum(p.x for p in positions) / len(positions),
            sum(p.y for p in positions) / len(positions),
            floor,
        )
    )
    radius = max((Vector((p.x, p.y, floor)) - center).length for p in positions)
    radius += spacing if focus else 0.4
    cube(
        "Ground", (200, 200, 0.025),
        (0, 0, (context_floor if context else floor) - (0.0125 if focus else 0.025)),
        floor_mat, bevel=0.0,
    )
    for fraction in (0.32, 0.62, 0.94, 1.0):
        r = radius * fraction
        curve(
            "Concentric guide",
            [
                (
                    center.x + r * math.cos(t * 2 * math.pi / 160),
                    center.y + r * math.sin(t * 2 * math.pi / 160),
                    floor + 0.002,
                )
                for t in range(160)
            ],
            guide,
            0.0025 * guide_scale,
            closed=True,
        )
    for i in range(64):
        theta = i * 2 * math.pi / 64
        curve(
            "Survey mark",
            [
                (center.x + r * math.cos(theta), center.y + r * math.sin(theta), floor + 0.003)
                for r in (radius, radius + (0.06 if i % 4 else 0.11) * guide_scale)
            ],
            guide,
            0.0025 * guide_scale,
        )
    objects, framing, motion = {}, list(positions), {}
    if context:
        cameras = packet["static"] + packet["dynamic"]
    for camera in cameras:
        dynamic = camera["dynamic"]
        distant = context and camera["id"] not in focus
        object_scale = 0.45 if distant else glyph_scale
        object_guide_scale = object_scale if distant else guide_scale
        object_floor = context_floor if distant else floor
        accent = amber if dynamic else cyan
        if distant or (highlight and dynamic and camera["id"] not in highlight):
            accent = context_accent
        root = bpy.data.objects.new(("Dynamic " if dynamic else "Static ") + camera["id"], None)
        bpy.context.collection.objects.link(root)
        root.rotation_mode = "QUATERNION"
        root["glyph_scale"] = object_scale
        camera_body(root, dynamic, graphite, accent, glass)
        mats = [Matrix(pose) for pose in camera["poses"]]
        root.matrix_world = mats[0] @ Matrix.Scale(object_scale, 4)
        foot = cylinder(
            "Station foot",
            (0.063 if dynamic else 0.052) * object_scale,
            0.014 * object_guide_scale,
            (mats[0].translation.x, mats[0].translation.y, object_floor + 0.012 * object_guide_scale),
            accent if dynamic else graphite,
        )
        mast = curve(
            "Support",
            [
                (mats[0].translation.x, mats[0].translation.y, object_floor + 0.02 * object_guide_scale),
                tuple(mats[0].translation),
            ],
            guide,
            0.006 * object_guide_scale,
        )
        if not distant:
            framing.extend(
                [Vector((matrix.translation.x, matrix.translation.y, floor)) for matrix in mats]
            )
        if dynamic and not distant and (not highlight or camera["id"] in highlight):
            k, width, height = camera["K"], camera["width"], camera["height"]
            length = min(0.94, 0.5 / max(width / k[0][0], height / k[1][1]))
            corners = [
                ((x - k[0][2]) / k[0][0] * length, -(y - k[1][2]) / k[1][1] * length, -length)
                for x, y in ((0, 0), (width, 0), (width, height), (0, height))
            ]
            curve("Field of view", corners, beam, 0.0025, root, True)
            for point in corners:
                curve("Optical ray", [(0, 0, -0.2), point], beam, 0.0015, root)
            for matrix in mats:
                framing.extend(
                    [
                        matrix @ Vector(point) * glyph_scale
                        + matrix.translation * (1 - glyph_scale)
                        for point in corners
                    ]
                )
            centers = [tuple(matrix.translation) for matrix in mats]
            if max((matrix.translation - mats[0].translation).length for matrix in mats) > 0.002:
                motion[camera["id"]] = [
                    (curve("Center trajectory", [centers[0]] * 20, trail, 0.004), centers)
                ]
            tips = [
                tuple(matrix @ Vector((0, 0, -(length + 0.12) * glyph_scale))) for matrix in mats
            ]
            motion.setdefault(camera["id"], []).append(
                (curve("Orientation trajectory", [tips[0]] * 20, trail, 0.003 * guide_scale), tips)
            )
            cylinder("Optical-axis marker", 0.014, 0.02, (0, 0, -length - 0.12), amber, root)
        objects[camera["id"]] = (root, foot, mast, mats, dynamic)
    az, el = map(math.radians, (packet["azimuth"], packet["elevation"]))
    depth = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    right = Vector((-math.sin(az), math.cos(az), 0.0))
    up = depth.cross(right)
    bounds = [
        (min(p.dot(axis) for p in framing), max(p.dot(axis) for p in framing))
        for axis in (right, up, depth)
    ]
    target = sum(
        (axis * ((lo + hi) / 2) for axis, (lo, hi) in zip((right, up, depth), bounds)), Vector()
    )
    view_data = bpy.data.cameras.new("Overview")
    view = bpy.data.objects.new("Overview", view_data)
    bpy.context.collection.objects.link(view)
    view.location = target + depth * 20
    aim(view, target)
    view_data.type = "ORTHO"
    aspect = packet["render_size"][0] / packet["render_size"][1]
    view_data.ortho_scale = (
        max(bounds[0][1] - bounds[0][0], (bounds[1][1] - bounds[1][0]) * aspect) * 1.2
    )
    if context:
        view_data.type = "PERSP"
        view_data.lens, view_data.sensor_width = 35, 36
        view_data.sensor_fit = "HORIZONTAL"
        view_data.clip_start = 0.001
        focal = 2 * view_data.lens / view_data.sensor_width
        distance = max(
            (p - target).dot(depth)
            + 1.25 * focal * max(abs((p - target).dot(right)), abs((p - target).dot(up)) * aspect)
            for p in framing
        )
        view.location = target + depth * distance
        aim(view, target)
    scene.camera = view
    for name, offset, color, energy, size in [
        ("Key", (0, -3, 7), (0.64, 0.82, 1.0), 1500, 5),
        ("Amber rim", (-4, 1, 4), (1.0, 0.57, 0.28), 1200, 4),
        ("Cyan rim", (4, 3, 5), (0.2, 0.62, 1.0), 1700, 4),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.color, data.shape, data.size = energy, color, "DISK", size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = center + Vector(offset)
        aim(obj, target)
    return scene, objects, floor, glyph_matrix, motion


def render_details(packet, output, frames, overview):
    """Render separate orientation comparisons in a fixed grid."""
    scene = bpy.data.scenes.new("Orientation details")
    bpy.context.window.scene = scene
    scene.world = overview.world
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 32
    scene.render.resolution_x, scene.render.resolution_y = 1280, 400
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    graphite = bpy.data.materials["Graphite housings"]
    amber = bpy.data.materials["Dynamic amber"]
    glass = bpy.data.materials["Optical glass"]
    beam = bpy.data.materials["Dynamic field of view"]
    trail = bpy.data.materials["Motion paths"]
    reference_mat = material("Initial orientation", (0.18, 0.23, 0.29), emission=0.8)
    view_data = bpy.data.cameras.new("Details lens")
    view = bpy.data.objects.new("Details view", view_data)
    scene.collection.objects.link(view)
    view.location = (0, 0, 15)
    aim(view, (0, 0, 0))
    view_data.type, view_data.ortho_scale = "ORTHO", 12.8
    scene.camera = view
    for location, energy, color in [
        ((-3, 2, 7), 2300, (0.65, 0.82, 1.0)),
        ((4, -2, 6), 2000, (1.0, 0.68, 0.38)),
    ]:
        light = bpy.data.lights.new("Details light", "AREA")
        light.energy, light.color, light.shape, light.size = energy, color, "DISK", 8
        obj = bpy.data.objects.new("Details light", light)
        scene.collection.objects.link(obj)
        obj.location = location
        aim(obj, (0, 0, 0))
    neutral = Vector((0.7, 0.45, -0.3)).to_track_quat("-Z", "Y").to_matrix()
    cameras = {camera["id"]: camera for camera in packet["dynamic"]}
    records = []
    for group_index, group in enumerate(packet["detail_groups"]):
        for index, camera_id in enumerate(group):
            camera = cameras[camera_id]
            poses = [Matrix(pose) for pose in camera["poses"]]
            first_inverse = poses[0].to_3x3().transposed()
            if packet.get("orientation_spread", 1.0) != 1:
                first_inverse = Matrix(cameras[group[0]]["poses"][0]).to_3x3().transposed()
            rotations = [neutral @ first_inverse @ pose.to_3x3() for pose in poses]
            k, width, height = camera["K"], camera["width"], camera["height"]
            length = min(1.05, 0.7 / max(width / k[0][0], height / k[1][1]))
            corners = [
                Vector(
                    ((x - k[0][2]) / k[0][0] * length, -(y - k[1][2]) / k[1][1] * length, -length)
                )
                for x, y in ((0, 0), (width, 0), (width, height), (0, height))
            ]
            envelope = [Vector((0, 0, 0)), *corners, Vector((0, 0, -length - 0.1))]
            envelope.extend(
                Vector((x, y, z))
                for x in (-0.12, 0.12)
                for y in (-0.09, 0.09)
                for z in (-0.22, 0.1)
            )
            bounds = [rotation @ point for rotation in rotations for point in envelope]
            limits = [(min(p[i] for p in bounds), max(p[i] for p in bounds)) for i in (0, 1)]
            scale = min(
                1.25 / max(limits[0][1] - limits[0][0], 0.3),
                1.28 / max(limits[1][1] - limits[1][0], 0.3),
                1.4,
            )
            column = (group_index % 2) * 4 + index
            center = Vector((-6.4 + (column + 0.5) * 1.6, 0.92 - (group_index // 2) * 2.0, 0))
            offset = center - Vector(
                (
                    (limits[0][0] + limits[0][1]) / 2 * scale,
                    (limits[1][0] + limits[1][1]) / 2 * scale,
                    0,
                )
            )
            matrices = []
            for rotation in rotations:
                matrix = rotation.to_4x4() @ Matrix.Scale(scale, 4)
                matrix.translation = offset
                matrices.append(matrix)
            root = bpy.data.objects.new("Detail " + camera_id, None)
            scene.collection.objects.link(root)
            camera_body(root, True, graphite, amber, glass)
            curve("Detail frustum", corners, beam, 0.004, root, True)
            for point in corners:
                curve("Detail ray", [(0, 0, -0.2), point], beam, 0.002, root)
            axis = Vector((0, 0, -length - 0.1))
            curve("Current axis", [(0, 0, -0.2), axis], amber, 0.008, root)
            curve("Initial axis", [offset, matrices[0] @ axis], reference_mat, 0.009)
            for a, b in zip(corners, corners[1:] + corners[:1]):
                curve("Initial frustum", [matrices[0] @ a, matrices[0] @ b], reference_mat, 0.003)
            cylinder("Detail optical marker", 0.022, 0.03, axis, amber, root)
            tips = [matrix @ axis for matrix in matrices]
            path = curve("Detail motion", [tips[0]] * 20, trail, 0.006)
            records.append((camera_id, root, matrices, path, tips))
    projections = {}
    for frame in frames:
        for camera_id, root, matrices, path, tips in records:
            root.matrix_world = matrices[frame]
            for i, point in enumerate(path.data.splines[0].points):
                point.co = (*tips[max(0, frame - 19 + i)], 1.0)
        bpy.context.view_layer.update()
        projections[str(frame)] = {}
        for camera_id, _, _, _, tips in records:
            uv = world_to_camera_view(scene, view, tips[frame])
            projections[str(frame)][camera_id] = [uv.x * 1280, (1 - uv.y) * 400]
        scene.render.filepath = str(output / f"detail_{frame:06d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"DETAIL {frame + 1}/{packet['frames']}", flush=True)
    (output / "detail_projections.json").write_text(json.dumps(projections) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    packet = json.loads(args.packet.read_text())
    scene, objects, floor, glyph_matrix, motion = build_scene(packet)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [args.frame] if args.frame is not None else range(packet["frames"])
    projected = {}
    for frame in frames:
        labels = {}
        for view, (root, foot, mast, poses, dynamic) in objects.items():
            pose = poses[frame] if dynamic else poses[0]
            root.matrix_world = pose @ Matrix.Scale(root["glyph_scale"], 4)
            foot.location.x, foot.location.y = pose.translation.x, pose.translation.y
            mast.data.splines[0].points[0].co = (
                pose.translation.x,
                pose.translation.y,
                mast.data.splines[0].points[0].co.z,
                1.0,
            )
            mast.data.splines[0].points[1].co = (*pose.translation, 1.0)
        for records in motion.values():
            for obj, points in records:
                for i, dst in enumerate(obj.data.splines[0].points):
                    dst.co = (*points[max(0, frame - 19 + i)], 1.0)
        bpy.context.view_layer.update()
        for view, (root, _, _, _, dynamic) in objects.items():
            if dynamic and (not packet.get("focus_views") or view in packet["focus_views"]):
                uv = world_to_camera_view(scene, scene.camera, root.matrix_world.translation)
                labels[view] = [
                    uv.x * packet["render_size"][0],
                    (1 - uv.y) * packet["render_size"][1],
                ]
        projected[str(frame)] = labels
        scene.render.filepath = str(args.output / f"{frame:06d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"FRAME {frame + 1}/{packet['frames']}", flush=True)
    (
        args.output / ("labels.json" if args.frame is None else f"labels_{args.frame:06d}.json")
    ).write_text(json.dumps(projected, indent=2) + "\n")
    if packet.get("detail_groups"):
        render_details(packet, args.output, frames, scene)


if __name__ == "__main__":
    main()
