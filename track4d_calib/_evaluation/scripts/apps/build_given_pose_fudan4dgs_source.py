#!/usr/bin/env python3
"""Build a Fudan 4DGS source from explicit static+dynamic camera poses.

This is the shared source-construction entry point for experiments where camera
poses are already known or have already been estimated.  The script intentionally
keeps the two dataset conversion steps visible:

1. per-frame OpenCV cameras -> Panoptic-style train/test metadata
2. Panoptic metadata -> Fudan 4DGS source with time-aware init points

For rendered Neu3D-style roots, ``--pose_source neu3d_render_root_gt`` first
exports ``render_root/cameras + gt_long_poses.npz`` into the same per-frame
OpenCV camera layout used by calibration outputs, then runs the same path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader, CameraParams, save_cameras
from track4d_calib.fudan_trajectory_utils import estimate_similarity_transform, transform_points_similarity


def _canon_view(view: str | int) -> str:
    token = str(view).strip()
    if token.isdigit():
        return token.zfill(2)
    return token


def _view_aliases(view: str) -> list[str]:
    token = str(view).strip()
    if not token:
        return []
    aliases = [token]
    if token.isdigit():
        value = int(token)
        aliases.extend([str(value), f"{value:02d}", f"{value:03d}"])
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return out


def _resolve_image_size(render_scene_root: Path, views: Sequence[str], source_frame: int) -> tuple[int, int]:
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    for view in views:
        for alias in _view_aliases(view):
            frame_stem = f"{int(source_frame):06d}"
            candidates = sorted(
                path
                for path in (render_scene_root / "images" / alias).glob(f"{frame_stem}.*")
                if path.is_file() and path.suffix.lower() in image_suffixes
            )
            if len(candidates) > 1:
                raise FileNotFoundError(
                    f"Ambiguous images for view {view} frame {frame_stem}: {candidates}"
                )
            if candidates:
                with Image.open(candidates[0]) as image:
                    return int(image.width), int(image.height)
    raise FileNotFoundError(
        f"Could not infer image size from {render_scene_root / 'images'} for frame {source_frame:06d}"
    )


def _load_static_camera(loader: CameraLoader, view: str) -> CameraParams | None:
    for alias in _view_aliases(view):
        try:
            return loader.load_camera(alias)
        except Exception:
            continue
    return None


def export_neu3d_render_root_gt_per_frame(
    *,
    render_scene_root: Path,
    per_frame_root: Path,
    frames: Sequence[int],
    views: Sequence[str],
    source_frame_index_offset: int,
    overwrite: bool,
) -> dict:
    cameras_root = render_scene_root / "cameras"
    pose_path = render_scene_root / "gt_long_poses.npz"
    if not cameras_root.is_dir():
        raise FileNotFoundError(cameras_root)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    static_loader = CameraLoader(cameras_root)
    pose_npz = np.load(pose_path, allow_pickle=True)
    long_views = [_canon_view(item) for item in pose_npz["long_views"]]
    dyn_index = {view: idx for idx, view in enumerate(long_views)}
    dyn_R = np.asarray(pose_npz["R"], dtype=np.float64)
    dyn_T = np.asarray(pose_npz["T"], dtype=np.float64)
    dyn_K = np.asarray(pose_npz["K"], dtype=np.float64)

    views_canon = [_canon_view(view) for view in views]
    first_source_frame = int(frames[0]) + int(source_frame_index_offset)
    width, height = _resolve_image_size(render_scene_root, views_canon, first_source_frame)
    exported_frames = 0
    exported_cameras = 0
    reused_frames = 0
    validated_cameras = 0
    max_static_pose_diff = 0.0
    max_dynamic_pose_diff = 0.0

    static_export_views: set[str] = set()
    dynamic_export_views: set[str] = set()
    for frame in frames:
        frame_id = int(frame)
        source_frame = frame_id + int(source_frame_index_offset)
        frame_camera_dir = per_frame_root / f"f{frame_id:06d}" / "cameras_pred"
        reuse_frame = frame_camera_dir.is_dir() and not overwrite
        cameras: dict[str, CameraParams] = {}
        for view in views_canon:
            if view in dyn_index:
                if source_frame < 0 or source_frame >= dyn_R.shape[0]:
                    raise IndexError(f"source frame {source_frame} is outside gt_long_poses range [0,{dyn_R.shape[0] - 1}]")
                idx = dyn_index[view]
                cameras[view] = CameraParams(
                    view_id=view,
                    K=np.asarray(dyn_K[idx], dtype=np.float64).reshape(3, 3),
                    R=np.asarray(dyn_R[source_frame, idx], dtype=np.float64).reshape(3, 3),
                    T=np.asarray(dyn_T[source_frame, idx], dtype=np.float64).reshape(3),
                    D=np.zeros(5, dtype=np.float64),
                    H=int(height),
                    W=int(width),
                )
                dynamic_export_views.add(view)
                continue
            static_cam = _load_static_camera(static_loader, view)
            if static_cam is not None:
                cameras[view] = CameraParams(
                    view_id=view,
                    K=np.asarray(static_cam.K, dtype=np.float64),
                    R=np.asarray(static_cam.R, dtype=np.float64),
                    T=np.asarray(static_cam.T, dtype=np.float64).reshape(3),
                    D=np.asarray(static_cam.D, dtype=np.float64).reshape(-1),
                    H=int(static_cam.H) if int(static_cam.H) > 0 else int(height),
                    W=int(static_cam.W) if int(static_cam.W) > 0 else int(width),
                )
                static_export_views.add(view)
                continue
            raise KeyError(f"View {view} is neither in {cameras_root} nor {pose_path}:long_views")
        if reuse_frame:
            reused_frames += 1
        else:
            save_cameras(frame_camera_dir, cameras)
            exported_frames += 1
            exported_cameras += len(cameras)
        saved_loader = CameraLoader(frame_camera_dir)
        for view, camera in cameras.items():
            saved = saved_loader.load_camera(view)
            if view in dyn_index:
                idx = dyn_index[view]
                max_dynamic_pose_diff = max(
                    max_dynamic_pose_diff,
                    float(np.max(np.abs(saved.R - np.asarray(dyn_R[source_frame, idx], dtype=np.float64).reshape(3, 3)))),
                    float(np.max(np.abs(saved.T - np.asarray(dyn_T[source_frame, idx], dtype=np.float64).reshape(3)))),
                    float(np.max(np.abs(saved.K - np.asarray(dyn_K[idx], dtype=np.float64).reshape(3, 3)))),
                )
            else:
                static_ref = _load_static_camera(static_loader, view)
                if static_ref is None:
                    raise KeyError(f"Static validation could not reload view {view} from {cameras_root}")
                max_static_pose_diff = max(
                    max_static_pose_diff,
                    float(np.max(np.abs(saved.R - np.asarray(static_ref.R, dtype=np.float64).reshape(3, 3)))),
                    float(np.max(np.abs(saved.T - np.asarray(static_ref.T, dtype=np.float64).reshape(3)))),
                    float(np.max(np.abs(saved.K - np.asarray(static_ref.K, dtype=np.float64).reshape(3, 3)))),
                )
            validated_cameras += 1

    tolerance = 1e-8
    if max_static_pose_diff > tolerance or max_dynamic_pose_diff > tolerance:
        raise RuntimeError(
            "Neu3D GT pose export validation failed: "
            f"max_static_pose_diff={max_static_pose_diff:g}, "
            f"max_dynamic_pose_diff={max_dynamic_pose_diff:g}, tolerance={tolerance:g}"
        )

    summary = {
        "pose_source": "neu3d_render_root_gt",
        "render_scene_root": str(render_scene_root),
        "per_frame_root": str(per_frame_root),
        "gt_long_poses": str(pose_path),
        "frames": [int(frames[0]), int(frames[-1])],
        "source_frame_index_offset": int(source_frame_index_offset),
        "views": views_canon,
        "static_views_from_cameras": sorted(static_export_views, key=lambda x: int(x) if x.isdigit() else x),
        "dynamic_views_from_gt_long_poses": sorted(dynamic_export_views, key=lambda x: int(x) if x.isdigit() else x),
        "image_wh": [int(width), int(height)],
        "exported_frames": int(exported_frames),
        "reused_frames": int(reused_frames),
        "exported_cameras": int(exported_cameras),
        "validated_cameras": int(validated_cameras),
        "pose_validation": {
            "enabled": True,
            "tolerance": float(tolerance),
            "max_static_pose_diff": float(max_static_pose_diff),
            "max_dynamic_pose_diff": float(max_dynamic_pose_diff),
        },
    }
    per_frame_root.mkdir(parents=True, exist_ok=True)
    (per_frame_root / "neu3d_render_root_gt_export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _append_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _parse_panoptic_view_frame(filename: str) -> tuple[str, int]:
    match = re.search(r"v(\d+)_f(\d+)", Path(str(filename)).name)
    if match is None:
        raise ValueError(f"Cannot parse panoptic view/frame from {filename}")
    return f"{int(match.group(1)):02d}", int(match.group(2))


def _find_render_mask(render_scene_root: Path, view: str, source_frame: int) -> Path | None:
    for mask_root_name in ["fmasks", "masks", "seg"]:
        for alias in _view_aliases(view):
            candidate = render_scene_root / mask_root_name / alias / f"{int(source_frame):06d}.png"
            if candidate.is_file():
                return candidate
    return None


def _replace_with_link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        relative_source = os.path.relpath(source, start=destination.parent)
        os.symlink(relative_source, destination)
        return "symlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def link_panoptic_foreground_masks(
    *,
    render_scene_root: Path,
    panoptic_root: Path,
    frame_start: int,
    frame_end: int,
    source_frame_index_offset: int,
) -> dict:
    linked = 0
    missing: list[str] = []
    actions: dict[str, int] = {}
    seg_root = panoptic_root / "seg"
    for meta_name in ["train_meta.json", "test_meta.json"]:
        meta_path = panoptic_root / meta_name
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for frame_fns in meta.get("fn", []):
            for filename in frame_fns:
                view, frame_id = _parse_panoptic_view_frame(str(filename))
                if frame_id < int(frame_start) or frame_id > int(frame_end):
                    continue
                source_frame = frame_id + int(source_frame_index_offset)
                source = _find_render_mask(render_scene_root, view, source_frame)
                destination = seg_root / Path(str(filename)).with_suffix(".png").name
                if source is None:
                    missing.append(str(destination))
                    continue
                action = _replace_with_link_or_copy(source, destination)
                actions[action] = actions.get(action, 0) + 1
                linked += 1
    summary = {
        "enabled": True,
        "render_scene_root": str(render_scene_root),
        "panoptic_root": str(panoptic_root),
        "seg_root": str(seg_root),
        "frame_range": [int(frame_start), int(frame_end)],
        "source_frame_index_offset": int(source_frame_index_offset),
        "linked_masks": int(linked),
        "missing_masks": int(len(missing)),
        "missing_examples": missing[:20],
        "actions": actions,
    }
    (panoptic_root / "foreground_mask_link_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _rotation_delta_deg(rotation: np.ndarray, reference: np.ndarray) -> float:
    delta = np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ np.asarray(reference, dtype=np.float64).reshape(3, 3).T
    cosine = (float(np.trace(delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def audit_per_frame_pose_motion(
    *,
    per_frame_root: Path,
    frames: Sequence[int],
    require_motion_views: Sequence[str],
    min_center_delta: float,
    min_rotation_deg: float,
    fail_without_motion: bool = True,
) -> dict:
    """Verify required views are not just a repeated single-frame pose."""

    required_views = [str(view) for view in require_motion_views if str(view).strip()]
    summary = {
        "enabled": bool(required_views),
        "per_frame_root": str(per_frame_root),
        "frames": [int(frames[0]), int(frames[-1])] if frames else [],
        "required_views": required_views,
        "min_center_delta": float(min_center_delta),
        "min_rotation_deg": float(min_rotation_deg),
        "failure_policy": "reject" if fail_without_motion else "record_only",
        "per_view": {},
        "required_views_without_motion": [],
        "all_required_views_have_motion": True,
    }
    if not required_views:
        return summary
    if not frames:
        raise ValueError("pose motion audit requires at least one frame")

    missing: list[str] = []
    without_motion: list[str] = []
    for view in required_views:
        cameras = []
        for frame in frames:
            cam_dir = Path(per_frame_root) / f"f{int(frame):06d}" / "cameras_pred"
            try:
                cameras.append(CameraLoader(cam_dir).load_camera(view))
            except Exception as exc:
                missing.append(f"{view}@f{int(frame):06d}: {exc}")
                break
        if len(cameras) != len(frames):
            continue
        centers = np.stack([np.asarray(cam.camera_center, dtype=np.float64).reshape(3) for cam in cameras], axis=0)
        center0 = centers[0]
        max_center_delta = float(np.max(np.linalg.norm(centers - center0.reshape(1, 3), axis=1)))
        ref_R = np.asarray(cameras[0].R, dtype=np.float64).reshape(3, 3)
        max_rotation_delta = float(max(_rotation_delta_deg(np.asarray(cam.R, dtype=np.float64), ref_R) for cam in cameras))
        moved = max_center_delta > float(min_center_delta) or max_rotation_delta > float(min_rotation_deg)
        summary["per_view"][view] = {
            "frames": int(len(cameras)),
            "max_center_delta": max_center_delta,
            "max_rotation_delta_deg": max_rotation_delta,
            "has_motion": bool(moved),
        }
        if not moved:
            without_motion.append(view)

    if missing:
        raise ValueError("pose motion audit could not load required cameras: " + "; ".join(missing[:8]))
    summary["required_views_without_motion"] = without_motion
    summary["all_required_views_have_motion"] = not without_motion
    if without_motion and fail_without_motion:
        raise ValueError(
            "required dynamic pose views have no measurable per-frame pose motion: "
            f"{without_motion}. per_frame_root={per_frame_root}. "
            "This usually means a single camera rig was repeated across frames before 4DGS source construction."
        )
    return summary


def build_panoptic_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        "scripts/build_4dgs_panoptic_from_calib.py",
        "--render_scene_root",
        str(args.render_scene_root),
        "--calib_run_root",
        str(args.per_frame_root),
        "--output_dir",
        str(args.panoptic_root),
        "--frame_start",
        str(int(args.frame_start)),
        "--frame_end",
        str(int(args.frame_end)),
        "--source_frame_index_offset",
        str(int(args.source_frame_index_offset)),
        "--views",
        *[str(view) for view in args.views],
        "--test_views",
        *[str(view) for view in args.test_views],
        "--test_stride",
        str(int(args.test_stride)),
        "--test_first_frame",
        str(int(args.test_first_frame)),
        "--resize_wh",
        str(int(args.resize_wh[0])),
        str(int(args.resize_wh[1])),
        # ``build_4dgs_panoptic_from_calib.py`` selects its point-cloud source
        # solely through ``--point_cloud_ply``.  It does not expose the wrapper's
        # higher-level point_cloud_mode flag; forwarding it makes the delegated
        # command fail before any dataset is built.
        "--num_init_points",
        str(int(args.num_init_points)),
        "--seed",
        str(int(args.seed)),
    ]
    if args.test_frame_start is not None:
        command.extend(
            [
                "--test_frame_start",
                str(int(args.test_frame_start)),
                "--test_frame_end",
                str(int(args.test_frame_end)),
            ]
        )
    if args.test_per_frame_root is not None:
        command.extend(["--test_calib_run_root", str(args.test_per_frame_root)])
    if args.test_static_gauge_views:
        command.extend(["--test_calib_static_gauge_views", *[str(view) for view in args.test_static_gauge_views]])
    if bool(args.drop_test_frames_from_train_except_first):
        command.append("--drop_test_frames_from_train_except_first")
    else:
        command.append("--no-drop_test_frames_from_train_except_first")
    _append_flag(command, bool(args.drop_all_test_frames_from_train), "--drop_all_test_frames_from_train")
    _append_flag(command, bool(args.allow_missing_calib_cameras), "--allow_missing_calib_cameras")
    if str(args.point_cloud_ply).strip():
        command.extend(["--point_cloud_ply", str(args.point_cloud_ply)])
    _append_flag(command, bool(args.overwrite_images), "--overwrite_images")
    _append_flag(command, bool(args.source_images_from_per_frame), "--source_images_from_per_frame")
    _append_flag(command, bool(args.center_crop_to_camera_wh), "--center_crop_to_camera_wh")
    return command


def build_fudan_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        "scripts/build_fudan_4dgs_time_init_pointcloud.py",
        "--panoptic_root",
        str(args.panoptic_root),
        "--per_frame_root",
        str(args.per_frame_root),
        "--output_dir",
        str(args.fudan_source),
        "--mode",
        str(args.fudan_mode),
        "--start_frame",
        str(int(args.frame_start)),
        "--end_frame",
        str(int(args.frame_end)),
        "--total_points",
        str(int(args.fudan_total_points)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--roma_setting",
        str(args.roma_setting),
        "--roma_ref_view",
        str(args.roma_ref_view),
        "--roma_pairs_per_frame",
        str(int(args.roma_pairs_per_frame)),
        "--roma_pairs_per_ref",
        str(int(args.roma_pairs_per_ref)),
        "--roma_matches_per_pair",
        str(int(args.roma_matches_per_pair)),
        "--roma_candidate_multiplier",
        f"{float(args.roma_candidate_multiplier):g}",
        "--roma_foreground_mode",
        str(args.roma_foreground_mode),
        "--roma_foreground_min_ratio",
        f"{float(args.roma_foreground_min_ratio):g}",
        "--roma_foreground_dilate_px",
        str(int(args.roma_foreground_dilate_px)),
        "--roma_reproj_px",
        f"{float(args.roma_reproj_px):g}",
        "--roma_max_depth",
        f"{float(args.roma_max_depth):g}",
        "--roma_target_strategy",
        str(args.roma_target_strategy),
        "--timestamp_mode",
        str(args.timestamp_mode),
        "--fps",
        f"{float(args.fps):g}",
        "--gauge_min_common",
        str(int(args.gauge_min_common)),
    ]
    if args.roma_ref_views:
        command.extend(["--roma_ref_views", *[str(view) for view in args.roma_ref_views]])
    if args.roma_target_views:
        command.extend(["--roma_target_views", *[str(view) for view in args.roma_target_views]])
    if bool(args.init_point_filter_outliers):
        command.extend(
            [
                "--init_point_filter_outliers",
                "--init_point_filter_mad",
                f"{float(args.init_point_filter_mad):g}",
                "--init_point_filter_radial_mad",
                f"{float(args.init_point_filter_radial_mad):g}",
                "--init_point_filter_min_points",
                str(int(args.init_point_filter_min_points)),
            ]
        )
    else:
        command.append("--no-init_point_filter_outliers")
    if str(args.romav2_weights).strip():
        command.extend(["--romav2_weights", str(args.romav2_weights)])
    if str(args.dinov3_src).strip():
        command.extend(["--dinov3_src", str(args.dinov3_src)])
    _append_flag(command, bool(args.roma_compile), "--roma_compile")
    if bool(args.align_frame_gauge):
        command.append("--align_frame_gauge")
        if args.gauge_anchor_frame is not None:
            command.extend(["--gauge_anchor_frame", str(int(args.gauge_anchor_frame))])
        if args.gauge_align_views:
            command.extend(["--gauge_align_views", *[str(view) for view in args.gauge_align_views]])
    if args.fudan_freeze_static_views:
        command.extend(["--freeze_static_views", *[str(view) for view in args.fudan_freeze_static_views]])
    _append_flag(command, bool(args.force_fudan), "--force")
    return command


def build_validate_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "scripts/validate_fudan_4dgs_dataset.py",
        "--source_path",
        str(args.fudan_source),
        "--extension",
        str(args.image_extension),
    ]


def _inspect_time_ply_header(path: Path) -> dict:
    vertex_count: int | None = None
    has_time = False
    header_lines = 0
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            elif stripped == "property float time" or stripped == "property double time":
                has_time = True
            elif stripped == "end_header":
                break
        else:
            raise ValueError(f"{path} is not a valid ASCII PLY: missing end_header")
    if vertex_count is None:
        raise ValueError(f"{path} is not a valid Fudan init PLY: missing vertex count")
    if not has_time:
        raise ValueError(f"{path} is not a Fudan time-aware points3d.ply: missing time property")
    return {
        "vertex_count": int(vertex_count),
        "has_time_property": True,
        "header_lines": int(header_lines),
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_fudan_view(file_path: str) -> str | None:
    matches = re.findall(r"_v(\d+)(?:_|\.|$)", str(file_path))
    if not matches:
        return None
    return str(int(matches[-1]))


def _canon_numeric_view(view: str | int) -> str:
    token = str(view).strip()
    return str(int(token)) if token.isdigit() else token


def _load_fudan_frames(root: Path, split: str) -> list[dict]:
    path = Path(root) / f"transforms_{split}.json"
    return list(json.loads(path.read_text(encoding="utf-8")).get("frames", []))


def _infer_fudan_source_root(points3d: Path) -> Path | None:
    root = Path(points3d).expanduser().resolve().parent
    if (root / "transforms_train.json").is_file() and (root / "transforms_test.json").is_file():
        return root
    return None


def _transform_matrix(frame: dict) -> np.ndarray:
    return np.asarray(frame["transform_matrix"], dtype=np.float64).reshape(4, 4)


def _first_frame_by_view(root: Path, *, preferred_views: Sequence[str]) -> dict[str, np.ndarray]:
    preferred = {_canon_numeric_view(view) for view in preferred_views if str(view).strip()}
    out: dict[str, np.ndarray] = {}
    for frame in _load_fudan_frames(root, "train"):
        view = _parse_fudan_view(str(frame.get("file_path", "")))
        if view is None:
            continue
        if preferred and view not in preferred:
            continue
        out.setdefault(view, _transform_matrix(frame))
    return out


def _detect_static_views(root: Path, *, max_center_std: float = 1e-7) -> set[str]:
    by_view: dict[str, list[np.ndarray]] = {}
    for frame in _load_fudan_frames(root, "train"):
        view = _parse_fudan_view(str(frame.get("file_path", "")))
        if view is None:
            continue
        by_view.setdefault(view, []).append(_transform_matrix(frame)[:3, 3].reshape(3))
    static_views: set[str] = set()
    for view, centers in by_view.items():
        if len(centers) < 2:
            continue
        arr = np.stack(centers, axis=0)
        if float(np.max(np.std(arr, axis=0))) <= float(max_center_std):
            static_views.add(view)
    return static_views


def estimate_stable_points3d_gauge_alignment(
    *,
    stable_source_root: Path,
    target_source_root: Path,
    preferred_views: Sequence[str],
    min_common_views: int,
) -> dict:
    """Estimate the Sim(3) that maps a stable Fudan init cloud into target gauge."""

    stable_source_root = Path(stable_source_root).expanduser().resolve()
    target_source_root = Path(target_source_root).expanduser().resolve()
    source_frames = _first_frame_by_view(stable_source_root, preferred_views=preferred_views)
    target_frames = _first_frame_by_view(target_source_root, preferred_views=preferred_views)
    common = sorted(set(source_frames) & set(target_frames), key=lambda view: int(view) if view.isdigit() else view)
    if not preferred_views:
        common_static = _detect_static_views(stable_source_root) & _detect_static_views(target_source_root)
        common = [view for view in common if view in common_static]
    if len(common) < int(min_common_views):
        raise ValueError(
            "Could not estimate stable points3d gauge alignment: "
            f"common static/support views={len(common)} < min_common_views={int(min_common_views)}. "
            f"stable_source_root={stable_source_root}, target_source_root={target_source_root}, "
            f"preferred_views={[str(view) for view in preferred_views]}"
        )

    source_centers = np.stack([source_frames[view][:3, 3] for view in common], axis=0)
    target_centers = np.stack([target_frames[view][:3, 3] for view in common], axis=0)
    transform = estimate_similarity_transform(source_centers, target_centers)
    mapped_centers = transform_points_similarity(source_centers, transform)
    center_residuals = np.linalg.norm(mapped_centers - target_centers, axis=1)

    rot = np.asarray(transform["rotation"], dtype=np.float64).reshape(3, 3)
    rotation_residuals = [
        _rotation_delta_deg(rot @ source_frames[view][:3, :3], target_frames[view][:3, :3])
        for view in common
    ]
    return {
        "enabled": True,
        "stable_source_root": str(stable_source_root),
        "target_source_root": str(target_source_root),
        "support_views": common,
        "support_count": int(len(common)),
        "scale": float(transform["scale"]),
        "rotation": rot.tolist(),
        "translation": np.asarray(transform["translation"], dtype=np.float64).reshape(3).tolist(),
        "center_residual_mean": float(np.mean(center_residuals)),
        "center_residual_max": float(np.max(center_residuals)),
        "rotation_residual_deg_mean": float(np.mean(rotation_residuals)),
        "rotation_residual_deg_max": float(np.max(rotation_residuals)),
    }


def _read_ascii_ply_rows(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with Path(path).open("r", encoding="ascii", errors="strict") as handle:
        first = handle.readline()
        if first.strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = handle.readline()
        if fmt.strip() != "format ascii 1.0":
            raise ValueError(f"{path} must be ascii PLY; got {fmt.strip()!r}")
        header = [first, fmt]
        properties: list[str] = []
        vertex_count: int | None = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} ended before end_header")
            header.append(line)
            stripped = line.strip()
            if stripped == "end_header":
                break
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
            elif len(parts) >= 3 and parts[0] == "property":
                properties.append(parts[-1])
        if vertex_count is None:
            raise ValueError(f"{path} does not declare element vertex")
        if vertex_count == 0:
            rows = np.zeros((0, len(properties)), dtype=np.float64)
        else:
            rows = np.loadtxt(handle, dtype=np.float64, max_rows=vertex_count)
            rows = np.asarray(rows, dtype=np.float64)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
        if rows.shape[1] != len(properties):
            raise ValueError(f"{path} has {rows.shape[1]} columns but declares {len(properties)} properties")
    return header, properties, rows


def _write_ascii_ply_rows(path: Path, header: list[str], rows: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.writelines(header)
        if rows.size:
            np.savetxt(handle, rows, fmt="%.10g")


def transform_stable_points3d_to_target_gauge(
    *,
    input_ply: Path,
    output_ply: Path,
    alignment: dict,
    report_path: Path | None,
) -> dict:
    header, properties, rows = _read_ascii_ply_rows(input_ply)
    prop_index = {name: idx for idx, name in enumerate(properties)}
    missing = [name for name in ("x", "y", "z") if name not in prop_index]
    if missing:
        raise ValueError(f"{input_ply} missing required coordinate properties: {missing}")

    xyz_cols = [prop_index["x"], prop_index["y"], prop_index["z"]]
    before_xyz = rows[:, xyz_cols].copy() if rows.size else np.zeros((0, 3), dtype=np.float64)
    transform = {
        "scale": float(alignment["scale"]),
        "rotation": np.asarray(alignment["rotation"], dtype=np.float64).reshape(3, 3),
        "translation": np.asarray(alignment["translation"], dtype=np.float64).reshape(3),
    }
    if rows.size:
        rows[:, xyz_cols] = transform_points_similarity(rows[:, xyz_cols], transform)
    after_xyz = rows[:, xyz_cols].copy() if rows.size else np.zeros((0, 3), dtype=np.float64)
    _write_ascii_ply_rows(output_ply, header, rows)

    report = {
        "input_ply": str(Path(input_ply).expanduser().resolve()),
        "output_ply": str(Path(output_ply).expanduser().resolve()),
        "vertices": int(rows.shape[0]),
        "scale": float(transform["scale"]),
        "rotation": np.asarray(transform["rotation"], dtype=np.float64).tolist(),
        "translation": np.asarray(transform["translation"], dtype=np.float64).reshape(3).tolist(),
        "bbox_before": {
            "min": before_xyz.min(axis=0).tolist() if len(before_xyz) else [0.0, 0.0, 0.0],
            "max": before_xyz.max(axis=0).tolist() if len(before_xyz) else [0.0, 0.0, 0.0],
        },
        "bbox_after": {
            "min": after_xyz.min(axis=0).tolist() if len(after_xyz) else [0.0, 0.0, 0.0],
            "max": after_xyz.max(axis=0).tolist() if len(after_xyz) else [0.0, 0.0, 0.0],
        },
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def install_stable_fudan_points3d(args: argparse.Namespace) -> dict:
    source_text = str(args.stable_fudan_points3d_ply).strip()
    if not source_text:
        return {"enabled": False}
    source = Path(source_text).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"--stable_fudan_points3d_ply does not exist: {source}")
    source_info = _inspect_time_ply_header(source)
    destination = args.fudan_source / "points3d.ply"
    destination.parent.mkdir(parents=True, exist_ok=True)

    gauge_alignment: dict = {"enabled": False}
    transform_report: dict | None = None
    stable_source_root = Path(str(args.stable_fudan_points3d_source)).expanduser().resolve() if str(args.stable_fudan_points3d_source).strip() else _infer_fudan_source_root(source)
    if bool(args.align_stable_points3d_to_target_gauge) and stable_source_root is not None and stable_source_root.resolve() != args.fudan_source.resolve():
        gauge_alignment = estimate_stable_points3d_gauge_alignment(
            stable_source_root=stable_source_root,
            target_source_root=args.fudan_source,
            preferred_views=[str(view) for view in args.gauge_align_views],
            min_common_views=int(args.gauge_min_common),
        )
        alignment_path = args.fudan_source / "stable_points3d_gauge_alignment_summary.json"
        alignment_path.write_text(json.dumps(gauge_alignment, indent=2) + "\n", encoding="utf-8")
        transform_report = transform_stable_points3d_to_target_gauge(
            input_ply=source,
            output_ply=destination,
            alignment=gauge_alignment,
            report_path=args.fudan_source / "stable_points3d_transform_report.json",
        )
        action = "transform_to_target_gauge"
    elif destination.exists() and source.samefile(destination):
        action = "reuse_existing"
    else:
        shutil.copy2(source, destination)
        action = "copy"
        if bool(args.align_stable_points3d_to_target_gauge):
            gauge_alignment = {
                "enabled": False,
                "reason": "no distinct stable source root with transforms_train/test was available",
                "stable_source_root": None if stable_source_root is None else str(stable_source_root),
                "target_source_root": str(args.fudan_source.resolve()),
            }
    destination_info = _inspect_time_ply_header(destination)
    return {
        "enabled": True,
        "source": str(source),
        "source_fudan_root": "" if stable_source_root is None else str(stable_source_root),
        "align_to_target_gauge": bool(args.align_stable_points3d_to_target_gauge),
        "destination": str(destination.resolve()),
        "action": action,
        "source_md5": _md5(source),
        "destination_md5": _md5(destination),
        "source_info": source_info,
        "destination_info": destination_info,
        "gauge_alignment": gauge_alignment,
        "transform_report": transform_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a shared Panoptic->Fudan 4DGS source from known static+dynamic poses.",
    )
    parser.add_argument("--render_scene_root", type=Path, required=True)
    parser.add_argument("--per_frame_root", type=Path, required=True)
    parser.add_argument(
        "--test_pose_source",
        choices=[
            "same_as_train",
            "neu3d_render_root_gt",
            "existing_per_frame",
        ],
        default="same_as_train",
        help=(
            "Camera source for held-out NVS views. existing_per_frame uses an "
            "already exported camera root without reading hidden GT from the "
            "render scene."
        ),
    )
    parser.add_argument(
        "--test_per_frame_root",
        type=Path,
        default=None,
        help="Separate per-frame camera root used when --test_pose_source is not same_as_train.",
    )
    parser.add_argument(
        "--test_static_gauge_views",
        nargs="*",
        default=[],
        help="Static views used to align GT test cameras into the method train-camera gauge.",
    )
    parser.add_argument("--panoptic_root", type=Path, required=True)
    parser.add_argument("--fudan_source", type=Path, required=True)
    parser.add_argument(
        "--pose_source",
        choices=["per_frame_calib", "neu3d_render_root_gt"],
        default="per_frame_calib",
        help="Where the explicit camera poses come from before Panoptic conversion.",
    )
    parser.add_argument("--pose_source_label", default="given_pose")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, required=True)
    parser.add_argument("--source_frame_index_offset", type=int, default=0)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--test_views", nargs="+", required=True)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--test_first_frame", type=int, default=None)
    parser.add_argument("--test_frame_start", type=int, default=None)
    parser.add_argument("--test_frame_end", type=int, default=None)
    parser.add_argument("--drop_test_frames_from_train_except_first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop_all_test_frames_from_train", action="store_true")
    parser.add_argument(
        "--allow_missing_calib_cameras",
        action="store_true",
        help="Skip frame/view samples with missing calibrated cameras when building Panoptic/Fudan metadata.",
    )
    parser.add_argument("--resize_wh", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--point_cloud_mode", choices=["auto", "ply", "synthetic"], default="synthetic")
    parser.add_argument("--point_cloud_ply", default="")
    parser.add_argument("--num_init_points", type=int, default=1000)
    parser.add_argument(
        "--stable_fudan_points3d_ply",
        default="",
        help=(
            "Optional final Fudan time-aware points3d.ply to install after Fudan source "
            "construction. This decouples learned pose updates from RoMA init-point "
            "re-triangulation in diagnostic/fair NVS runs."
        ),
    )
    parser.add_argument(
        "--stable_fudan_points3d_source",
        default="",
        help=(
            "Optional Fudan source root that owns --stable_fudan_points3d_ply. "
            "When omitted, the parent directory of the PLY is used if it contains "
            "transforms_train/test.json. The builder uses this to map stable init "
            "points into the target camera gauge before training."
        ),
    )
    parser.add_argument(
        "--align_stable_points3d_to_target_gauge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transform stable Fudan points3d.ply from its source gauge into the target Fudan source gauge.",
    )
    parser.add_argument("--overwrite_images", action="store_true")
    parser.add_argument("--source_images_from_per_frame", action="store_true")
    parser.add_argument("--center_crop_to_camera_wh", action="store_true")
    parser.add_argument(
        "--fudan_mode",
        choices=["sfm_points", "roma_ref_pairs", "panoptic_init_npz"],
        default="roma_ref_pairs",
    )
    parser.add_argument("--fudan_total_points", type=int, default=300000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--romav2_weights", default="")
    parser.add_argument("--dinov3_src", default="")
    parser.add_argument("--roma_setting", default="fast")
    parser.add_argument("--roma_compile", action="store_true")
    parser.add_argument("--roma_ref_view", default="00")
    parser.add_argument("--roma_ref_views", nargs="*", default=["2", "9", "14", "23"])
    parser.add_argument(
        "--roma_target_views",
        nargs="*",
        default=[],
        help="Optional RoMA target allowlist; use static anchor views for pose-neutral shared initialization.",
    )
    parser.add_argument("--roma_pairs_per_frame", type=int, default=2)
    parser.add_argument("--roma_pairs_per_ref", type=int, default=4)
    parser.add_argument("--roma_matches_per_pair", type=int, default=1024)
    parser.add_argument("--roma_candidate_multiplier", type=float, default=4.0)
    parser.add_argument("--roma_foreground_mode", choices=["none", "ref", "either", "both"], default="ref")
    parser.add_argument("--roma_foreground_min_ratio", type=float, default=0.6)
    parser.add_argument("--roma_foreground_dilate_px", type=int, default=5)
    parser.add_argument("--roma_reproj_px", type=float, default=2.0)
    parser.add_argument("--roma_max_depth", type=float, default=0.0)
    parser.add_argument("--roma_target_strategy", choices=["nearest", "spread"], default="spread")
    parser.add_argument("--timestamp_mode", choices=["normalized", "fps"], default="normalized")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--align_frame_gauge", action="store_true")
    parser.add_argument("--gauge_anchor_frame", type=int, default=None)
    parser.add_argument("--gauge_align_views", nargs="*", default=[])
    parser.add_argument("--gauge_min_common", type=int, default=6)
    parser.add_argument("--fudan_freeze_static_views", nargs="*", default=[])
    parser.add_argument(
        "--require_pose_motion_views",
        nargs="*",
        default=[],
        help=(
            "Views that must show measurable per-frame pose motion before building the 4DGS source. "
            "Use this for moving-camera/dynamic-pose protocols to catch accidentally repeated static rigs."
        ),
    )
    parser.add_argument("--pose_motion_min_center_delta", type=float, default=1e-6)
    parser.add_argument("--pose_motion_min_rotation_deg", type=float, default=1e-4)
    parser.add_argument(
        "--pose_motion_audit_mode",
        choices=("reject", "record_only"),
        default="reject",
        help="Whether a required view without measurable motion aborts or is only recorded.",
    )
    parser.add_argument("--init_point_filter_outliers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init_point_filter_mad", type=float, default=4.0)
    parser.add_argument("--init_point_filter_radial_mad", type=float, default=4.0)
    parser.add_argument("--init_point_filter_min_points", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--image_extension", default=".png")
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_validate", action="store_true")
    parser.add_argument("--overwrite_pose_export", action="store_true")
    parser.add_argument("--force_fudan", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.test_first_frame is None:
        args.test_first_frame = int(args.frame_start)
    if int(args.frame_start) > int(args.frame_end):
        parser.error("--frame_start must be <= --frame_end")
    has_explicit_test_start = args.test_frame_start is not None
    has_explicit_test_end = args.test_frame_end is not None
    if has_explicit_test_start != has_explicit_test_end:
        parser.error("--test_frame_start and --test_frame_end must be provided together")
    if has_explicit_test_start and not (
        int(args.frame_start)
        <= int(args.test_frame_start)
        <= int(args.test_frame_end)
        <= int(args.frame_end)
    ):
        parser.error("test frame range must be inside source frame range")
    if str(args.test_pose_source) != "same_as_train" and args.test_per_frame_root is None:
        parser.error("--test_per_frame_root is required with a separate test pose source")
    if str(args.test_pose_source) != "same_as_train" and not args.test_static_gauge_views:
        parser.error("--test_static_gauge_views is required with a separate test pose source")
    if str(args.test_pose_source) == "same_as_train" and args.test_per_frame_root is not None:
        parser.error("--test_per_frame_root requires a non-default --test_pose_source")
    return args


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    frames = list(range(int(args.frame_start), int(args.frame_end) + 1))
    panoptic_command = build_panoptic_command(args)
    fudan_command = build_fudan_command(args)
    validate_command = build_validate_command(args)
    summary_path = args.summary_json or (args.fudan_source / "given_pose_source_summary.json")
    summary: dict = {
        "dry_run": bool(args.dry_run),
        "pose_source": str(args.pose_source),
        "pose_source_label": str(args.pose_source_label),
        "calib_source_label": str(args.pose_source_label),
        "render_scene_root": str(args.render_scene_root),
        "per_frame_root": str(args.per_frame_root),
        "test_pose_source": str(args.test_pose_source),
        "test_per_frame_root": "" if args.test_per_frame_root is None else str(args.test_per_frame_root),
        "test_static_gauge_views": [str(view) for view in args.test_static_gauge_views],
        "panoptic_root": str(args.panoptic_root),
        "fudan_source": str(args.fudan_source),
        "frames": [int(args.frame_start), int(args.frame_end)],
        "total_frames": int(len(frames)),
        "source_frame_index_offset": int(args.source_frame_index_offset),
        "views": [str(view) for view in args.views],
        "test_views": [str(view) for view in args.test_views],
        "test_stride": int(args.test_stride),
        "test_first_frame": int(args.test_first_frame),
        "test_frame_range": (
            [int(args.test_frame_start), int(args.test_frame_end)]
            if args.test_frame_start is not None
            else None
        ),
        "drop_test_frames_from_train_except_first": bool(args.drop_test_frames_from_train_except_first),
        "drop_all_test_frames_from_train": bool(args.drop_all_test_frames_from_train),
        "train_views": [str(view) for view in args.views if str(view) not in {str(v) for v in args.test_views}],
        "fairness_checklist": {
            "train_pose_source": str(args.pose_source),
            "test_pose_source": str(args.test_pose_source),
            "test_static_gauge_views": [str(view) for view in args.test_static_gauge_views],
            "resolution": [int(args.resize_wh[0]), int(args.resize_wh[1])],
            "frame_ids": [int(args.frame_start), int(args.frame_end)],
            "source_frame_ids": [int(frame) for frame in frames],
            "test_frame_ids": (
                list(range(int(args.test_frame_start), int(args.test_frame_end) + 1, int(args.test_stride)))
                if args.test_frame_start is not None
                else [
                    int(frame)
                    for frame in frames
                    if int(frame) >= int(args.test_first_frame)
                    and (int(frame) - int(args.test_first_frame)) % int(args.test_stride) == 0
                ]
            ),
            "seed": int(args.seed),
            "preprocess": {
                "source_frame_index_offset": int(args.source_frame_index_offset),
                "source_images_from_per_frame": bool(args.source_images_from_per_frame),
                "center_crop_to_camera_wh": bool(args.center_crop_to_camera_wh),
                "image_extension": str(args.image_extension),
                "allow_missing_calib_cameras": bool(args.allow_missing_calib_cameras),
            },
            "init_points": int(args.fudan_total_points),
            "init_point_source": {
                "fudan_mode": str(args.fudan_mode),
                "roma_target_views": [str(view) for view in args.roma_target_views],
                "dinov3_src": str(args.dinov3_src),
                "stable_fudan_points3d_ply": str(Path(str(args.stable_fudan_points3d_ply)).expanduser())
                if str(args.stable_fudan_points3d_ply).strip()
                else "",
                "stable_fudan_points3d_source": str(Path(str(args.stable_fudan_points3d_source)).expanduser())
                if str(args.stable_fudan_points3d_source).strip()
                else "",
                "align_stable_points3d_to_target_gauge": bool(args.align_stable_points3d_to_target_gauge),
            },
            "timestamp_mode": str(args.timestamp_mode),
        },
        "panoptic_command": panoptic_command,
        "fudan_command": fudan_command,
        "validate_command": validate_command,
        "stable_fudan_points3d": {
            "enabled": bool(str(args.stable_fudan_points3d_ply).strip()),
            "source": str(Path(str(args.stable_fudan_points3d_ply)).expanduser()) if str(args.stable_fudan_points3d_ply).strip() else "",
            "source_fudan_root": str(Path(str(args.stable_fudan_points3d_source)).expanduser())
            if str(args.stable_fudan_points3d_source).strip()
            else "",
            "align_to_target_gauge": bool(args.align_stable_points3d_to_target_gauge),
        },
        "pose_motion_audit": {
            "enabled": bool(args.require_pose_motion_views),
            "required_views": [str(view) for view in args.require_pose_motion_views],
            "min_center_delta": float(args.pose_motion_min_center_delta),
            "min_rotation_deg": float(args.pose_motion_min_rotation_deg),
            "mode": str(args.pose_motion_audit_mode),
        },
    }

    if bool(args.dry_run):
        write_summary(summary_path, summary)
        print(json.dumps(summary, indent=2))
        return 0

    if str(args.pose_source) == "neu3d_render_root_gt":
        summary["pose_export"] = export_neu3d_render_root_gt_per_frame(
            render_scene_root=args.render_scene_root,
            per_frame_root=args.per_frame_root,
            frames=frames,
            views=[str(view) for view in args.views],
            source_frame_index_offset=int(args.source_frame_index_offset),
            overwrite=bool(args.overwrite_pose_export),
        )

    if str(args.test_pose_source) == "neu3d_render_root_gt":
        gt_export_views = list(
            dict.fromkeys([str(view) for view in args.test_static_gauge_views] + [str(view) for view in args.test_views])
        )
        summary["test_pose_export"] = export_neu3d_render_root_gt_per_frame(
            render_scene_root=args.render_scene_root,
            per_frame_root=args.test_per_frame_root,
            frames=frames,
            views=gt_export_views,
            source_frame_index_offset=int(args.source_frame_index_offset),
            overwrite=bool(args.overwrite_pose_export),
        )

    summary["pose_motion_audit"] = audit_per_frame_pose_motion(
        per_frame_root=args.per_frame_root,
        frames=frames,
        require_motion_views=[str(view) for view in args.require_pose_motion_views],
        min_center_delta=float(args.pose_motion_min_center_delta),
        min_rotation_deg=float(args.pose_motion_min_rotation_deg),
        fail_without_motion=str(args.pose_motion_audit_mode) == "reject",
    )

    subprocess.run(panoptic_command, check=True, cwd=str(PROJECT_ROOT))
    if str(args.roma_foreground_mode) != "none":
        summary["foreground_masks"] = link_panoptic_foreground_masks(
            render_scene_root=args.render_scene_root,
            panoptic_root=args.panoptic_root,
            frame_start=int(args.frame_start),
            frame_end=int(args.frame_end),
            source_frame_index_offset=int(args.source_frame_index_offset),
        )
    subprocess.run(fudan_command, check=True, cwd=str(PROJECT_ROOT))
    summary["stable_fudan_points3d"] = install_stable_fudan_points3d(args)
    if not bool(args.skip_validate):
        subprocess.run(validate_command, check=True, cwd=str(PROJECT_ROOT))
    summary["dry_run"] = False
    summary["status"] = "ok"
    write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
