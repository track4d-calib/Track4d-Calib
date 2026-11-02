#!/usr/bin/env python3
"""Render synchronized image panels and calibrated camera-array motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from track4d_calib.config import load_config

SIZE = (2560, 1440)
RENDER_SIZE = (1280, 1100)
INK, MUTED = "#EDF3F7", "#92A3B4"
CYAN, AMBER = "#62DBEB", "#F7B15C"
BACKGROUND = "#070B11"
FONTS = {}


@dataclass(frozen=True)
class Camera:
    rotation: np.ndarray
    translation: np.ndarray
    intrinsic: np.ndarray
    width: float
    height: float

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    @property
    def forward(self) -> np.ndarray:
        return self.rotation[2]


def read_cameras(directory: Path, views: list[str]) -> dict[str, Camera]:
    intrinsics = cv2.FileStorage(str(directory / "intri.yml"), cv2.FILE_STORAGE_READ)
    extrinsics = cv2.FileStorage(str(directory / "extri.yml"), cv2.FILE_STORAGE_READ)
    try:
        if not intrinsics.isOpened() or not extrinsics.isOpened():
            raise ValueError(f"camera files could not be opened: {directory}")
        cameras = {}
        for view in views:
            rotation = extrinsics.getNode(f"Rot_{view}").mat()
            translation = extrinsics.getNode(f"T_{view}").mat()
            intrinsic = intrinsics.getNode(f"K_{view}").mat()
            width = intrinsics.getNode(f"W_{view}").real()
            height = intrinsics.getNode(f"H_{view}").real()
            if any(value is None for value in (rotation, translation, intrinsic)):
                raise ValueError(f"incomplete camera {view}: {directory}")
            rotation, translation = rotation.reshape(3, 3), translation.reshape(3)
            if not all(np.isfinite(value).all() for value in (rotation, translation, intrinsic)):
                raise ValueError(f"nonfinite camera {view}: {directory}")
            if (
                not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
                or np.linalg.det(rotation) < 0
            ):
                raise ValueError(f"invalid camera rotation {view}: {directory}")
            if min(width, height, intrinsic[0, 0], intrinsic[1, 1]) <= 0:
                raise ValueError(f"invalid camera intrinsics {view}: {directory}")
            cameras[view] = Camera(rotation, translation, intrinsic, width, height)
        return cameras
    finally:
        intrinsics.release()
        extrinsics.release()


def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def scene_coordinates(cameras: list[Camera]) -> tuple[np.ndarray, np.ndarray, float]:
    """Choose one rigid display frame and a common scale for the entire clip."""
    centers = np.array([camera.center for camera in cameras])
    directions = np.array([camera.forward for camera in cameras])
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    origin = np.linalg.lstsq(
        projectors.sum(axis=0), np.einsum("nij,nj->i", projectors, centers), rcond=None
    )[0]
    up = normalized(np.mean([-camera.rotation[1] for camera in cameras], axis=0))
    lateral = centers[-1] - centers[0]
    lateral -= np.dot(lateral, up) * up
    if np.linalg.norm(lateral) < 1e-8:
        lateral = np.cross(up, np.eye(3)[np.argmin(np.abs(up))])
    x_axis = normalized(lateral)
    basis = np.column_stack((x_axis, normalized(np.cross(up, x_axis)), up))
    scale = max(float(np.median(np.linalg.norm(centers - origin, axis=1))), 1e-8)
    return origin, basis, scale


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size, bold=False, mono=False):
    key = size, bold, mono
    if key not in FONTS:
        name = "DejaVuSansMono" if mono else "DejaVuSans"
        try:
            FONTS[key] = ImageFont.truetype(name + ("-Bold" if bold else "") + ".ttf", size)
        except OSError:
            FONTS[key] = ImageFont.load_default(size=size)
    return FONTS[key]


def label(draw, xy, text, size=20, color=INK, bold=False, mono=False, anchor=None):
    draw.text(xy, text, fill=color, font=font(size, bold, mono), anchor=anchor)


def geometry(config, static, poses, azimuth, elevation, rotation_scale=1.0):
    origin, basis, scale = scene_coordinates(list(static.values()))

    def entry(view, cameras, dynamic):
        matrices = []
        for camera in cameras:
            transform = np.eye(4)
            transform[:3, :3] = basis.T @ camera.rotation.T @ np.diag([1, -1, -1])
            transform[:3, 3] = (camera.center - origin) @ basis / scale * 3
            matrices.append(transform.tolist())
        matrices = np.asarray(matrices)
        angles = []
        if dynamic:
            reference = matrices[0, :3, :3].copy()
            for matrix in matrices:
                delta = cv2.Rodrigues(reference.T @ matrix[:3, :3])[0]
                angles.append(float(np.rad2deg(np.linalg.norm(delta))))
                matrix[:3, :3] = reference @ cv2.Rodrigues(delta * rotation_scale)[0]
        camera = cameras[0]
        return dict(
            id=view,
            dynamic=dynamic,
            K=camera.intrinsic.tolist(),
            width=camera.width,
            height=camera.height,
            poses=matrices.tolist(),
            max_recorded_rotation_deg=max(angles, default=0.0),
            max_display_rotation_deg=max(angles, default=0.0) * rotation_scale,
        )

    return dict(
        frames=len(poses),
        render_size=list(RENDER_SIZE),
        azimuth=azimuth,
        elevation=elevation,
        rotation_scale=rotation_scale,
        static=[entry(view, [static[view]], False) for view in config["static_views"]],
        dynamic=[
            entry(view, [frame[view] for frame in poses], True) for view in config["dynamic_views"]
        ],
    )


def label_layout(projected, render_size=RENDER_SIZE):
    views = list(next(iter(projected.values())))
    centers = {
        view: np.mean([frame[view] for frame in projected.values()], axis=0) for view in views
    }
    groups = []
    for view in views:
        group = next(
            (g for g in groups if any(np.linalg.norm(centers[view] - centers[v]) < 44 for v in g)),
            None,
        )
        if group is None:
            groups.append([view])
        else:
            group.append(view)
    result, occupied = {}, []
    for group in sorted(groups, key=lambda g: float(np.mean([centers[v][1] for v in g]))):
        center = np.mean([centers[v] for v in group], axis=0)
        direction = center - np.array(render_size) / 2
        direction /= max(np.linalg.norm(direction), 1.0)
        midpoint = center + direction * 88
        width, height = len(group) * 49, 32
        x = float(np.clip(midpoint[0] - width / 2, 16, render_size[0] - width - 16))
        y = float(np.clip(midpoint[1] - height / 2, 24, render_size[1] - height - 76))
        for _ in range(40):
            if not any(
                x < b[2] + 8 and x + width > b[0] - 8 and y < b[3] + 8 and y + height > b[1] - 8
                for b in occupied
            ):
                break
            y += 38 if center[1] > render_size[1] / 2 else -38
            y = float(np.clip(y, 24, render_size[1] - height - 76))
        occupied.append((x, y, x + width, y + height))
        for i, view in enumerate(group):
            result[view] = (x + i * 49, y)
    return result


def spread_orientations(packet, groups, factor):
    """Expand initial orientations around each group's mean without moving centers."""
    if factor == 1:
        return
    cameras = {camera["id"]: camera for camera in packet["dynamic"]}
    for group in groups:
        first = np.array([cameras[view]["poses"][0] for view in group])[:, :3, :3]
        u, _, vt = np.linalg.svd(first.mean(axis=0))
        mean = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        for view, reference in zip(group, first):
            delta = cv2.Rodrigues(mean.T @ reference)[0]
            correction = mean @ cv2.Rodrigues(delta * (factor - 1))[0] @ mean.T
            poses = np.array(cameras[view]["poses"])
            poses[:, :3, :3] = correction @ poses[:, :3, :3]
            cameras[view]["poses"] = poses.tolist()


def compose_frame(
    config,
    images,
    static_views,
    array_image,
    projected,
    layout,
    frame,
    frame_count,
    title,
    rotation_scale=1.0,
    detail_image=None,
    detail_groups=(),
    rendered_dynamic=False,
    orientation_spread=1.0,
    focus_views=(),
    global_image=None,
    global_projected=None,
):
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    label(draw, (48, 36), "TRACK4D-CALIB", 20, CYAN, mono=True)
    label(draw, (48, 74), title, 42, bold=True)
    label(draw, (2512, 46), "CAMERA ARRAYS / CAPTURE IN MOTION", 18, MUTED, mono=True, anchor="ra")
    label(draw, (2512, 84), f"{frame + 1:03d} / {frame_count:03d}", 28, INK, mono=True, anchor="ra")
    draw.line((48, 140, 2512, 140), fill="#273341", width=1)
    panel_title = "RENDERED DYNAMIC VIEWS" if rendered_dynamic else "SYNCHRONIZED VIEWS"
    label(draw, (48, 164), panel_title, 19, INK, mono=True)
    label(
        draw,
        (1212, 164),
        f"{len(config['dynamic_views'])} DYNAMIC",
        17,
        AMBER,
        mono=True,
        anchor="ra",
    )
    array_title = (
        "CAMERA GROUP / " + " / ".join(focus_views) if focus_views else "CALIBRATED CAMERA ARRAY"
    )
    label(draw, (1280, 164), array_title, 19, INK, mono=True)
    rows = [list(row) for row in np.array_split(config["dynamic_views"], 3)]
    for row, views in enumerate([*rows, static_views]):
        y = 218 + 267 * row
        dynamic = row < 3
        color = AMBER if dynamic else CYAN
        if not dynamic:
            label(draw, (48, y - 28), "STATIC VIEWS", 16, CYAN, mono=True)
        width = (1164 - 12 * (len(views) - 1)) / len(views)
        for column, view in enumerate(views):
            x = round(48 + column * (width + 12))
            box = (x, y, x + round(width), y + 226)
            draw.rounded_rectangle(box, radius=10, fill="#101923", outline="#263443", width=1)
            if view in focus_views:
                draw.rounded_rectangle(box, radius=10, outline=AMBER, width=2)
            with Image.open(images / view / f"{frame:06d}.png") as source:
                photo = source.convert("RGB")
                photo.thumbnail((round(width) - 4, 192), Image.Resampling.LANCZOS)
            canvas.paste(
                photo, (x + (round(width) - photo.width) // 2, y + 2 + (192 - photo.height) // 2)
            )
            draw.line((x + 10, y + 200, x + 10, y + 216), fill=color, width=3)
            label(draw, (x + 22, y + 199), f"CAM {view}", 16, color, mono=True)
    array = Image.open(array_image).convert("RGB")
    canvas.paste(array, (1240, 206))
    draw = ImageDraw.Draw(canvas)
    for view, xy in layout.items():
        x, y = xy[0] + 1240, xy[1] + 206
        px, py = projected[view][0] + 1240, projected[view][1] + 206
        draw.line([(x + 22, y + 16), (px, py)], fill="#AA8759", width=1)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=AMBER)
    for view, xy in layout.items():
        x, y = xy[0] + 1240, xy[1] + 206
        draw.rounded_rectangle(
            (x, y, x + 44, y + 31), radius=6, fill="#111923", outline="#946D3C", width=1
        )
        label(draw, (x + 22, y + 5), view, 18, AMBER, mono=True, anchor="ma")
    if detail_image is not None:
        label(draw, (1254, 854), "ORIENTATION CHANGE / FRAME 001", 17, INK, mono=True)
        label(
            draw,
            (2510, 854),
            "GRAY / INITIAL     AMBER / CURRENT",
            14,
            MUTED,
            mono=True,
            anchor="ra",
        )
        for index, group in enumerate(detail_groups):
            x, y = 1240 + (index % 2) * 640, 880 + (index // 2) * 200
            draw.rounded_rectangle(
                (x + 6, y, x + 634, y + 194), radius=10, fill="#0E1721", outline="#293744", width=1
            )
            label(draw, (x + 18, y + 8), " / ".join(group), 14, MUTED, mono=True)
        with Image.open(detail_image) as details:
            canvas.paste(details, (1240, 880), details.getchannel("A"))
        draw = ImageDraw.Draw(canvas)
        for index, group in enumerate(detail_groups):
            x, y = 1240 + (index % 2) * 640, 880 + (index // 2) * 200
            for column, view in enumerate(group):
                label(
                    draw,
                    (x + column * 160 + 80, y + 175),
                    f"CAM {view}",
                    14,
                    AMBER,
                    mono=True,
                    anchor="ma",
                )
    if global_image is not None:
        x, y = 1942, 224
        draw.rounded_rectangle(
            (x - 12, y - 12, x + 572, y + 280), radius=14,
            fill=BACKGROUND, outline="#40505C", width=1,
        )
        label(draw, (x + 12, y + 4), "GLOBAL ARRAY", 16, INK, mono=True)
        with Image.open(global_image) as overview:
            canvas.paste(overview.convert("RGB"), (x, y + 34))
        draw = ImageDraw.Draw(canvas)
        points = np.array([global_projected[view] for view in focus_views])
        cx, cy = points.mean(axis=0) + [x, y + 34]
        radius = max(13, np.linalg.norm(points - points.mean(axis=0), axis=1).max() + 9)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=AMBER, width=2)
        draw.line([(cx, cy + radius), (cx, y + 248), (x + 24, y + 248)], fill=AMBER, width=1)
        label(draw, (x + 24, y + 258), " / ".join(focus_views) + "  ·  SELECTED GROUP", 14, AMBER, mono=True)
    draw.rectangle((1240, 1280, 2532, 1320), fill=BACKGROUND)
    if focus_views:
        label(draw, (1280, 1290), f"● {len(focus_views)} CAMERAS / CLOSE-UP", 19, AMBER, mono=True)
    else:
        label(draw, (1280, 1290), f"● {len(config['static_views'])} STATIC", 19, CYAN, mono=True)
        label(draw, (1570, 1290), f"● {len(config['dynamic_views'])} DYNAMIC", 19, AMBER, mono=True)
    motion_label = (
        "RECORDED MOTION ×1"
        if rotation_scale == 1
        else f"ROTATION ×{rotation_scale:g} / ILLUSTRATION"
    )
    if orientation_spread != 1:
        motion_label = (
            f"ROTATION ×{rotation_scale:g} / SPREAD ×{orientation_spread:g} / ILLUSTRATION"
        )
    label(draw, (2508, 1290), motion_label, 15, MUTED, mono=True, anchor="ra")
    draw.line((48, 1363, 2512, 1363), fill="#273341", width=3)
    x = 48 + (2512 - 48) * frame / max(frame_count - 1, 1)
    draw.line((48, 1363, x, 1363), fill=AMBER, width=3)
    draw.ellipse((x - 5, 1358, x + 5, 1368), fill=AMBER)
    label(
        draw,
        (48, 1384),
        "120 SYNCHRONIZED FRAMES" if frame_count == 120 else f"{frame_count} SYNCHRONIZED FRAMES",
        17,
        MUTED,
        mono=True,
    )
    label(
        draw, (2512, 1384), "STATIC / CYAN     DYNAMIC / AMBER", 17, MUTED, mono=True, anchor="ra"
    )
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--static-cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-views", nargs="+")
    parser.add_argument(
        "--focus-views", nargs="+", default=[],
        help="Show only these dynamic cameras in a close-up; retain all image panels",
    )
    parser.add_argument(
        "--global-context", action="store_true",
        help="Include distant cameras and a whole-array locator with a focused view",
    )
    parser.add_argument(
        "--detail-group",
        nargs="+",
        action="append",
        default=[],
        help="Camera IDs for an orientation detail panel (up to four groups of four)",
    )
    parser.add_argument(
        "--rendered-dynamic",
        action="store_true",
        help="Label dynamic image panels as rendered views",
    )
    parser.add_argument("--dynamic-views", nargs="+", help="Override the displayed dynamic roster")
    parser.add_argument(
        "--reference-views",
        nargs="+",
        default=[],
        help="Dynamic views whose poses come from reference calibration",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--scene")
    parser.add_argument("--fps", type=int, default=12, help="Playback frame rate")
    parser.add_argument("--azimuth", type=float, default=-65)
    parser.add_argument("--elevation", type=float, default=54)
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=1.0,
        help="Display multiplier for rotation relative to the first frame",
    )
    parser.add_argument(
        "--orientation-spread",
        type=float,
        default=1.0,
        help="Display multiplier for initial orientation differences within each detail group",
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-frame", type=int)
    parser.add_argument("--pose-source", choices=("estimated", "reference"), default="estimated")
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args(argv)
    if args.preview_frame is None:
        args.preview_frame = args.frames // 2
    if args.frames < 1 or not 1 <= args.fps <= 30 or not 0 <= args.preview_frame < args.frames:
        parser.error("invalid frame count, preview frame or playback rate")
    config = load_config(args.config)
    if not np.isfinite(args.rotation_scale) or args.rotation_scale <= 0:
        parser.error("rotation scale must be finite and positive")
    if not np.isfinite(args.orientation_spread) or args.orientation_spread <= 0:
        parser.error("orientation spread must be finite and positive")
    if args.orientation_spread != 1 and not args.detail_group:
        parser.error("orientation spread requires detail groups")
    if args.dynamic_views is not None:
        config["dynamic_views"] = args.dynamic_views
    dynamic_views = config["dynamic_views"]
    if (
        len(dynamic_views) < 3
        or len(set(dynamic_views)) != len(dynamic_views)
        or set(dynamic_views) & set(config["static_views"])
    ):
        parser.error("dynamic views must be distinct, nonstatic camera IDs (at least three)")
    if not set(args.reference_views) <= set(dynamic_views):
        parser.error("reference views must be members of the displayed dynamic roster")
    if args.focus_views and (
        len(args.focus_views) < 2
        or len(set(args.focus_views)) != len(args.focus_views)
        or not set(args.focus_views) <= set(dynamic_views)
    ):
        parser.error("focus views require at least two distinct dynamic camera IDs")
    detailed = [view for group in args.detail_group for view in group]
    if args.global_context and (not args.focus_views or args.detail_group):
        parser.error("global context requires focus views and no detail panels")
    if (
        len(args.detail_group) > 4
        or any(len(group) > 4 for group in args.detail_group)
        or len(set(detailed)) != len(detailed)
        or not set(detailed) <= set(dynamic_views)
    ):
        parser.error("detail groups require distinct dynamic IDs, at most four groups of four")
    reference_views = dynamic_views if args.pose_source == "reference" else args.reference_views
    camera_sources = {
        v: "reference" if v in reference_views else "estimated" for v in dynamic_views
    }
    pose_source = (
        next(iter(set(camera_sources.values())))
        if len(set(camera_sources.values())) == 1
        else "mixed"
    )
    if shutil.which(args.blender) is None:
        parser.error("Blender must be available on PATH or provided with --blender")
    if not args.preview_only and shutil.which("ffmpeg") is None:
        parser.error("FFmpeg must be available on PATH")
    static_views = args.static_views or [
        config["static_views"][len(config["static_views"]) * i // 4] for i in range(4)
    ]
    if not set(static_views) <= set(config["static_views"]) or len(set(static_views)) != len(
        static_views
    ):
        parser.error("static image views must be distinct members of the static roster")
    static = read_cameras(args.static_cameras, config["static_views"])
    poses = [
        read_cameras(args.poses / f"f{f:06d}/cameras_pred", config["dynamic_views"])
        for f in range(args.frames)
    ]
    sources = {}
    for view in [*config["dynamic_views"], *static_views]:
        for frame in range(args.frames):
            p = args.images / view / f"{frame:06d}.png"
            sources[f"images/{view}/{frame:06d}.png"] = sha256(p)
    for frame in range(args.frames):
        for name in ("intri.yml", "extri.yml"):
            sources[f"poses/f{frame:06d}/cameras_pred/{name}"] = sha256(
                args.poses / f"f{frame:06d}/cameras_pred" / name
            )
    for name in ("intri.yml", "extri.yml"):
        sources[f"static_cameras/{name}"] = sha256(args.static_cameras / name)
    dataset = {
        "dyn-neu3d": "Dyn-Neu3D",
        "dyn-dna-rendering": "Dyn-DNA-Rendering",
        "dyn-4dv": "Dyn-4DV",
    }.get(config["dataset"], config["dataset"])
    title = f"{dataset}  /  {args.scene or config['scene']}"
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Path(__file__).with_name("render_camera_array_blender.py")
    with tempfile.TemporaryDirectory(prefix=".array_", dir=args.output) as temporary:
        temp = Path(temporary)
        packet = temp / "cameras.json"
        geometry_data = geometry(
            config, static, poses, args.azimuth, args.elevation, args.rotation_scale
        )
        geometry_data["detail_groups"] = args.detail_group
        geometry_data["focus_views"] = args.focus_views
        geometry_data["global_context"] = args.global_context
        geometry_data["orientation_spread"] = args.orientation_spread
        spread_orientations(geometry_data, args.detail_group, args.orientation_spread)
        if args.detail_group:
            geometry_data["render_size"] = [1280, 640]
        packet.write_text(json.dumps(geometry_data))
        command = [
            args.blender,
            "-b",
            "-t",
            "6",
            "--python",
            str(backend),
            "--",
            "--packet",
            str(packet),
            "--output",
            str(temp / "frames"),
        ]
        if args.preview_only:
            command += ["--frame", str(args.preview_frame)]
        subprocess.run(command, check=True)
        name = f"labels_{args.preview_frame:06d}.json" if args.preview_only else "labels.json"
        projected = json.loads((temp / "frames" / name).read_text())
        layout = label_layout(projected, geometry_data["render_size"])
        global_projected = {}
        if args.global_context:
            overview = dict(
                geometry_data, focus_views=[], detail_groups=[], global_context=False,
                highlight_views=args.focus_views, render_size=[560, 210],
                azimuth=-65, elevation=54,
            )
            overview_packet = temp / "global.json"
            overview_packet.write_text(json.dumps(overview))
            overview_command = command.copy()
            overview_command[overview_command.index("--packet") + 1] = str(overview_packet)
            overview_command[overview_command.index("--output") + 1] = str(temp / "global_frames")
            subprocess.run(overview_command, check=True)
            global_projected = json.loads((temp / "global_frames" / name).read_text())
        detail_projections = (
            json.loads((temp / "frames/detail_projections.json").read_text())
            if args.detail_group
            else {}
        )

        def render(frame):
            return compose_frame(
                config,
                args.images,
                static_views,
                temp / "frames" / f"{frame:06d}.png",
                projected[str(frame)],
                layout,
                frame,
                args.frames,
                title,
                args.rotation_scale,
                temp / "frames" / f"detail_{frame:06d}.png" if args.detail_group else None,
                args.detail_group,
                args.rendered_dynamic,
                args.orientation_spread,
                args.focus_views,
                temp / "global_frames" / f"{frame:06d}.png" if args.global_context else None,
                global_projected.get(str(frame)),
            )

        render(args.preview_frame).save(args.output / "preview.png")
        if not args.preview_only:
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "2560x1440",
                "-r",
                str(args.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-threads",
                "4",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-movflags",
                "+faststart",
                str(args.output / "camera_array.mp4"),
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for frame in range(args.frames):
                    process.stdin.write(render(frame).tobytes())
                process.stdin.close()
                error = process.stderr.read().decode()
                if process.wait():
                    raise RuntimeError(error)
            except BaseException:
                process.kill()
                process.wait()
                (args.output / "camera_array.mp4").unlink(missing_ok=True)
                raise
    manifest = dict(
        dataset=config["dataset"],
        scene=args.scene or config["scene"],
        frame_ids=list(range(args.frames)),
        dynamic_views=config["dynamic_views"],
        static_views=config["static_views"],
        displayed_static_views=static_views,
        pose_source=pose_source,
        camera_sources=camera_sources,
        rendered_dynamic_images=args.rendered_dynamic,
        preview_frame=args.preview_frame,
        playback_fps=args.fps,
        display=dict(
            azimuth=args.azimuth,
            elevation=args.elevation,
            size=list(SIZE),
            rotation_scale=args.rotation_scale,
            orientation_spread=args.orientation_spread,
            focus_views=args.focus_views,
            global_context=args.global_context,
            global_locator=dict(azimuth=-65, elevation=54, size=[560, 210])
            if args.global_context else None,
            translation_scale=1.0,
            detail_groups=args.detail_group,
            detail_reference="first_frame_orientation" if args.detail_group else None,
            detail_coordinate_frame=(
                "group_first_camera" if args.orientation_spread != 1 else "individual_first_camera"
            ) if args.detail_group else None,
        ),
        motion={
            camera["id"]: {
                key: camera[key]
                for key in ("max_recorded_rotation_deg", "max_display_rotation_deg")
            }
            for camera in geometry_data["dynamic"]
        },
        source_sha256=sources,
        detail_motion_pixels={
            view: {
                "max_displacement": float(np.linalg.norm(points - points[0], axis=1).max()),
                "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
            }
            for view in detailed
            for points in [np.array([row[view] for row in detail_projections.values()])]
        }
        if not args.preview_only
        else {},
        config_sha256=sha256(args.config),
        generator_sha256={p.name: sha256(p) for p in (Path(__file__), backend)},
        outputs={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
