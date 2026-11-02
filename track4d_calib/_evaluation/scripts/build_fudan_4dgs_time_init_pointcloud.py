#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "hloc" / "hloc" / "utils"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "colmap_glomap_src" / "scripts" / "python"))

from read_write_model import read_images_binary, read_points3D_binary, qvec2rotmat  # type: ignore


def _seed_random_generators(
    seed: int,
) -> tuple[np.random.Generator, dict[str, object]]:
    """Bind RoMA sampling and point subsampling to the source-build seed."""
    value = int(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(value)
    rng = np.random.default_rng(value)

    import torch

    torch.manual_seed(value)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return rng, {
        "seed": value,
        "python_random_seeded": True,
        "numpy_generator_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": cuda_available,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
    }


class SimilarityTransform:
    def __init__(self, scale: float, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.scale = float(scale)
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)


def write_ply_with_time(path: Path, xyz: np.ndarray, rgb: np.ndarray, times: np.ndarray) -> None:
    xyz_f = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb_u8 = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    times_f = np.asarray(times, dtype=np.float32).reshape(-1)
    if xyz_f.shape[0] != rgb_u8.shape[0] or xyz_f.shape[0] != times_f.shape[0]:
        raise ValueError("xyz, rgb, times must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {int(xyz_f.shape[0])}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float nx\n")
        handle.write("property float ny\n")
        handle.write("property float nz\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property float time\n")
        handle.write("end_header\n")
        for point, color, time in zip(xyz_f, rgb_u8, times_f):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"0.0 0.0 0.0 {int(color[0])} {int(color[1])} {int(color[2])} {float(time):.8f}\n"
            )


def estimate_similarity_from_centers(
    src_centers: Dict[str, np.ndarray],
    dst_centers: Dict[str, np.ndarray],
) -> SimilarityTransform:
    common = [key for key in src_centers.keys() if key in dst_centers]
    if len(common) < 3:
        raise ValueError("Need at least 3 common centers to estimate similarity")
    src = np.stack([np.asarray(src_centers[key], dtype=np.float64).reshape(3) for key in common], axis=0)
    dst = np.stack([np.asarray(dst_centers[key], dtype=np.float64).reshape(3) for key in common], axis=0)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (src_centered.T @ dst_centered) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(cov)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= 1e-12:
        raise ValueError("Degenerate source centers for similarity estimation")
    scale = float(np.sum(singular_values) / src_var)
    translation = dst_mean - scale * (rotation @ src_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def apply_similarity(points: np.ndarray, sim: SimilarityTransform) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    transformed = (sim.scale * (sim.rotation @ pts.T)).T + sim.translation.reshape(1, 3)
    return transformed.astype(np.float32)


def _filter_init_point_spatial_outliers(
    xyz: np.ndarray,
    rgb: np.ndarray,
    times: np.ndarray,
    *,
    mad_threshold: float,
    radial_mad_threshold: float,
    min_points_per_group: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    xyz_f = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb_u8 = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    times_f = np.asarray(times, dtype=np.float32).reshape(-1)
    if xyz_f.shape[0] != rgb_u8.shape[0] or xyz_f.shape[0] != times_f.shape[0]:
        raise ValueError("xyz, rgb, times must have the same length")
    if xyz_f.shape[0] == 0:
        return xyz_f, rgb_u8, times_f, {
            "enabled": True,
            "input_points": 0,
            "kept_points": 0,
            "removed_points": 0,
            "groups": {},
        }

    mad_threshold_f = float(mad_threshold)
    radial_threshold_f = float(radial_mad_threshold)
    min_group = max(1, int(min_points_per_group))
    keep = np.zeros((xyz_f.shape[0],), dtype=bool)
    groups: dict[str, dict] = {}
    eps = 1e-6

    for time_value in np.unique(times_f):
        idx = np.flatnonzero(times_f == time_value)
        key = f"{float(time_value):.8f}"
        if idx.size < min_group:
            keep[idx] = True
            groups[key] = {
                "input_points": int(idx.size),
                "kept_points": int(idx.size),
                "removed_points": 0,
                "skipped": True,
                "reason": "too_few_points",
            }
            continue

        pts = xyz_f[idx].astype(np.float64)
        median = np.median(pts, axis=0)
        abs_dev = np.abs(pts - median.reshape(1, 3))
        mad = np.median(abs_dev, axis=0)
        std = np.std(pts, axis=0)
        robust_scale = 1.4826 * mad
        scale = np.where(robust_scale > eps, robust_scale, np.maximum(std, 1.0))
        normalized = abs_dev / scale.reshape(1, 3)
        axis_mask = np.all(normalized <= mad_threshold_f, axis=1) if mad_threshold_f > 0.0 else np.ones(idx.size, dtype=bool)
        radial = np.linalg.norm(normalized, axis=1)
        radial_mask = radial <= radial_threshold_f if radial_threshold_f > 0.0 else np.ones(idx.size, dtype=bool)
        group_keep = axis_mask & radial_mask
        if int(np.count_nonzero(group_keep)) < min_group:
            group_keep = np.ones(idx.size, dtype=bool)
            skipped = True
            reason = "would_drop_below_min_points"
        else:
            skipped = False
            reason = ""
        keep[idx[group_keep]] = True
        groups[key] = {
            "input_points": int(idx.size),
            "kept_points": int(np.count_nonzero(group_keep)),
            "removed_points": int(idx.size - np.count_nonzero(group_keep)),
            "skipped": bool(skipped),
            "reason": reason,
            "median": median.tolist(),
            "robust_scale": scale.tolist(),
            "max_radial_mad": float(np.max(radial)) if radial.size else 0.0,
        }

    filtered_xyz = xyz_f[keep]
    filtered_rgb = rgb_u8[keep]
    filtered_times = times_f[keep]
    summary = {
        "enabled": True,
        "input_points": int(xyz_f.shape[0]),
        "kept_points": int(filtered_xyz.shape[0]),
        "removed_points": int(xyz_f.shape[0] - filtered_xyz.shape[0]),
        "mad_threshold": mad_threshold_f,
        "radial_mad_threshold": radial_threshold_f,
        "min_points_per_group": int(min_group),
        "bbox_before": {
            "min": np.min(xyz_f, axis=0).astype(float).tolist(),
            "max": np.max(xyz_f, axis=0).astype(float).tolist(),
        },
        "bbox_after": {
            "min": np.min(filtered_xyz, axis=0).astype(float).tolist() if filtered_xyz.size else [],
            "max": np.max(filtered_xyz, axis=0).astype(float).tolist() if filtered_xyz.size else [],
        },
        "groups": groups,
    }
    return filtered_xyz, filtered_rgb, filtered_times, summary


def _w2c_center(w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
    return (-pose[:3, :3].T @ pose[:3, 3]).reshape(3)


def _transform_w2c_world(w2c: np.ndarray, sim: SimilarityTransform) -> np.ndarray:
    pose = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
    c2w = np.linalg.inv(pose)
    c2w_out = np.eye(4, dtype=np.float64)
    c2w_out[:3, :3] = sim.rotation @ c2w[:3, :3]
    c2w_out[:3, 3] = sim.scale * (sim.rotation @ c2w[:3, 3]) + sim.translation
    return np.linalg.inv(c2w_out)


def _meta_centers_by_view(meta: dict, frame_index: int) -> Dict[str, np.ndarray]:
    centers: Dict[str, np.ndarray] = {}
    if frame_index < 0 or frame_index >= len(meta.get("w2c", [])):
        return centers
    cam_ids = meta.get("cam_id", [[]])[frame_index]
    w2c_list = meta.get("w2c", [[]])[frame_index]
    for view_id, w2c in zip(cam_ids, w2c_list):
        key = f"{int(view_id):03d}" if str(view_id).isdigit() else str(view_id)
        centers[key] = _w2c_center(np.asarray(w2c, dtype=np.float64))
    return centers


def _estimate_frame_gauge_sims(
    *,
    train_meta: dict,
    test_meta: dict | None,
    frame_ids: Sequence[int],
    anchor_frame: int,
    align_views: Sequence[str],
    min_common: int,
) -> tuple[Dict[int, SimilarityTransform], dict]:
    frame_ids_list = [int(frame_id) for frame_id in frame_ids]
    if int(anchor_frame) not in frame_ids_list:
        raise ValueError(f"anchor_frame {anchor_frame} is not in frame_ids")
    anchor_index = frame_ids_list.index(int(anchor_frame))
    align_set = {f"{int(view):03d}" if str(view).isdigit() else str(view) for view in align_views}
    anchor_centers = _meta_centers_by_view(train_meta, anchor_index)
    if test_meta is not None:
        anchor_centers.update(_meta_centers_by_view(test_meta, anchor_index))
    if align_set:
        anchor_centers = {view: center for view, center in anchor_centers.items() if view in align_set}
    if len(anchor_centers) < int(min_common):
        raise ValueError(f"Need at least {min_common} anchor cameras for gauge alignment, got {len(anchor_centers)}")

    sims: Dict[int, SimilarityTransform] = {}
    frame_summaries: Dict[str, dict] = {}
    identity = SimilarityTransform(scale=1.0, rotation=np.eye(3, dtype=np.float64), translation=np.zeros(3, dtype=np.float64))
    for frame_index, frame_id in enumerate(frame_ids_list):
        cur_centers = _meta_centers_by_view(train_meta, frame_index)
        if test_meta is not None:
            cur_centers.update(_meta_centers_by_view(test_meta, frame_index))
        if align_set:
            cur_centers = {view: center for view, center in cur_centers.items() if view in align_set}
        common = sorted([view for view in cur_centers if view in anchor_centers])
        if frame_index == anchor_index:
            sim = identity
        elif len(common) >= int(min_common):
            sim = estimate_similarity_from_centers(
                {view: cur_centers[view] for view in common},
                {view: anchor_centers[view] for view in common},
            )
        else:
            sim = identity
        sims[int(frame_id)] = sim
        residual = 0.0
        if common:
            transformed = apply_similarity(np.stack([cur_centers[view] for view in common], axis=0), sim)
            target = np.stack([anchor_centers[view] for view in common], axis=0)
            residual = float(np.mean(np.linalg.norm(transformed - target, axis=1)))
        frame_summaries[f"f{int(frame_id):06d}"] = {
            "num_common": int(len(common)),
            "scale": float(sim.scale),
            "mean_center_residual": float(residual),
        }
    return sims, {
        "enabled": True,
        "anchor_frame": int(anchor_frame),
        "min_common": int(min_common),
        "align_views": sorted(align_set) if align_set else [],
        "frames_aligned": int(len(sims)),
        "frame_summaries": frame_summaries,
    }


def _apply_frame_gauge_to_meta(meta: dict, *, sims: Dict[int, SimilarityTransform], frame_ids: Sequence[int]) -> dict:
    out = json.loads(json.dumps(meta))
    frame_ids_list = [int(frame_id) for frame_id in frame_ids]
    for frame_index, frame_id in enumerate(frame_ids_list):
        sim = sims.get(int(frame_id))
        if sim is None:
            continue
        for camera_index, w2c in enumerate(out.get("w2c", [[]])[frame_index]):
            out["w2c"][frame_index][camera_index] = _transform_w2c_world(np.asarray(w2c, dtype=np.float64), sim).tolist()
    return out


def _canonical_cam_id(view: str | int) -> str:
    return f"{int(view):03d}" if str(view).isdigit() else str(view)


def _freeze_meta_view_poses_to_anchor(
    meta: dict,
    *,
    frame_ids: Sequence[int],
    anchor_frame: int,
    freeze_views: Sequence[str],
) -> dict:
    freeze_set = {_canonical_cam_id(view) for view in freeze_views}
    if not freeze_set:
        return meta
    frame_ids_list = [int(frame_id) for frame_id in frame_ids]
    if int(anchor_frame) not in frame_ids_list:
        raise ValueError(f"anchor_frame {anchor_frame} is not in frame_ids")
    anchor_index = frame_ids_list.index(int(anchor_frame))
    out = json.loads(json.dumps(meta))
    anchor_by_view: Dict[str, list] = {}
    for cam_id, w2c in zip(out.get("cam_id", [[]])[anchor_index], out.get("w2c", [[]])[anchor_index]):
        key = _canonical_cam_id(cam_id)
        if key in freeze_set:
            anchor_by_view[key] = w2c
    for frame_index in range(len(out.get("w2c", []))):
        for camera_index, cam_id in enumerate(out.get("cam_id", [[]])[frame_index]):
            key = _canonical_cam_id(cam_id)
            if key in anchor_by_view:
                out["w2c"][frame_index][camera_index] = json.loads(json.dumps(anchor_by_view[key]))
    return out


def _align_frame_gauge_then_freeze_static_views(
    *,
    train_meta: dict,
    test_meta: dict | None,
    frame_ids: Sequence[int],
    anchor_frame: int,
    align_views: Sequence[str],
    min_common: int,
    freeze_views: Sequence[str],
) -> tuple[dict, dict | None, Dict[int, SimilarityTransform], dict, dict]:
    gauge_sims, gauge_summary = _estimate_frame_gauge_sims(
        train_meta=train_meta,
        test_meta=test_meta,
        frame_ids=frame_ids,
        anchor_frame=anchor_frame,
        align_views=align_views,
        min_common=min_common,
    )
    aligned_train = _apply_frame_gauge_to_meta(train_meta, sims=gauge_sims, frame_ids=frame_ids)
    aligned_test = (
        _apply_frame_gauge_to_meta(test_meta, sims=gauge_sims, frame_ids=frame_ids)
        if test_meta is not None
        else None
    )
    freeze_summary = {
        "enabled": bool(freeze_views),
        "anchor_frame": int(anchor_frame),
        "views": [str(view) for view in freeze_views],
        "order": "after_frame_gauge_alignment",
    }
    if freeze_views:
        aligned_train = _freeze_meta_view_poses_to_anchor(
            aligned_train,
            frame_ids=frame_ids,
            anchor_frame=anchor_frame,
            freeze_views=freeze_views,
        )
        if aligned_test is not None:
            aligned_test = _freeze_meta_view_poses_to_anchor(
                aligned_test,
                frame_ids=frame_ids,
                anchor_frame=anchor_frame,
                freeze_views=freeze_views,
            )
    return aligned_train, aligned_test, gauge_sims, gauge_summary, freeze_summary


def triangulate_pair_with_reprojection_filter(
    *,
    k_a: np.ndarray,
    w2c_a: np.ndarray,
    uv_a: np.ndarray,
    k_b: np.ndarray,
    w2c_b: np.ndarray,
    uv_b: np.ndarray,
    max_reproj_px: float,
    max_depth: float = 0.0,
) -> tuple[np.ndarray | None, float]:
    k_a = np.asarray(k_a, dtype=np.float64).reshape(3, 3)
    k_b = np.asarray(k_b, dtype=np.float64).reshape(3, 3)
    w2c_a = np.asarray(w2c_a, dtype=np.float64).reshape(4, 4)
    w2c_b = np.asarray(w2c_b, dtype=np.float64).reshape(4, 4)
    uv_a = np.asarray(uv_a, dtype=np.float64).reshape(2)
    uv_b = np.asarray(uv_b, dtype=np.float64).reshape(2)
    proj_a = k_a @ w2c_a[:3, :]
    proj_b = k_b @ w2c_b[:3, :]
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0] = uv_a[0] * proj_a[2] - proj_a[0]
    matrix[1] = uv_a[1] * proj_a[2] - proj_a[1]
    matrix[2] = uv_b[0] * proj_b[2] - proj_b[0]
    matrix[3] = uv_b[1] * proj_b[2] - proj_b[1]
    try:
        _, _, vt = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return None, float("inf")
    point_h = vt[-1]
    if abs(float(point_h[3])) < 1e-12:
        return None, float("inf")
    point = point_h[:3] / point_h[3]
    total = 0.0
    for proj, uv in ((proj_a, uv_a), (proj_b, uv_b)):
        projected = proj @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
        if projected[2] <= 1e-12:
            return None, float("inf")
        if float(max_depth) > 0.0 and projected[2] > float(max_depth):
            return None, float("inf")
        total += float(np.linalg.norm(projected[:2] / projected[2] - uv))
    mean_error = total / 2.0
    if mean_error > float(max_reproj_px):
        return None, mean_error
    return point.astype(np.float32), mean_error


def w2c_to_blender_c2w(w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
    c2w_colmap = np.linalg.inv(w2c)
    c2w_blender = c2w_colmap.copy()
    c2w_blender[:3, 1:3] *= -1.0
    return c2w_blender


def _read_train_test_meta(panoptic_root: Path) -> tuple[dict, dict]:
    train_meta = json.loads((panoptic_root / "train_meta.json").read_text(encoding="utf-8"))
    test_meta = json.loads((panoptic_root / "test_meta.json").read_text(encoding="utf-8"))
    return train_meta, test_meta


def _timestamp_from_index(index: int, *, num_t: int, mode: str, fps: float) -> float:
    if str(mode) == "fps":
        if float(fps) <= 0.0:
            raise ValueError(f"fps must be > 0 for fps timestamp mode, got {fps}")
        return float(index) / float(fps)
    if str(mode) == "normalized":
        return float(index / max(1, num_t))
    raise ValueError(f"Unsupported timestamp mode: {mode}")


def _build_frames(
    meta: dict,
    split_name: str,
    src_root: Path,
    out_dir: Path,
    *,
    timestamp_mode: str,
    fps: float,
) -> list[dict]:
    frames = []
    num_t = len(meta["fn"])
    for t_idx in range(num_t):
        timestamp = _timestamp_from_index(t_idx, num_t=num_t, mode=str(timestamp_mode), fps=float(fps))
        for j, fn in enumerate(meta["fn"][t_idx]):
            src_img = src_root / "ims" / fn
            cam_ids = meta.get("cam_id", [])
            if t_idx < len(cam_ids) and j < len(cam_ids[t_idx]):
                view = str(cam_ids[t_idx][j])
                view = str(int(view)) if view.isdigit() else view
            else:
                view = Path(fn).parent.name
                view = str(int(view)) if view.isdigit() else view
            rel = f"images/{split_name}_{t_idx:04d}_{j:03d}_v{view}_f{t_idx:06d}.png"
            dst_img = out_dir / rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                try:
                    dst_img.hardlink_to(src_img)
                except Exception:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    try:
                        dst_img.symlink_to(src_img)
                    except Exception:
                        if dst_img.exists() or dst_img.is_symlink():
                            dst_img.unlink()
                        shutil.copy2(src_img, dst_img)
            k = np.asarray(meta["k"][t_idx][j], dtype=np.float64)
            if k.shape == (3, 3):
                fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
            else:
                fx, fy, cx, cy = map(float, k)
            c2w = w2c_to_blender_c2w(np.asarray(meta["w2c"][t_idx][j], dtype=np.float64))
            frames.append(
                {
                    "file_path": rel[:-4],
                    "time": timestamp,
                    "transform_matrix": c2w.tolist(),
                    "fl_x": fx,
                    "fl_y": fy,
                    "cx": cx,
                    "cy": cy,
                }
            )
    return frames


def _write_transforms_from_panoptic(
    panoptic_root: Path,
    train_meta: dict,
    test_meta: dict,
    out_dir: Path,
    *,
    timestamp_mode: str,
    fps: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    train_frames = _build_frames(
        train_meta,
        "train",
        panoptic_root,
        out_dir,
        timestamp_mode=str(timestamp_mode),
        fps=float(fps),
    )
    test_frames = _build_frames(
        test_meta,
        "test",
        panoptic_root,
        out_dir,
        timestamp_mode=str(timestamp_mode),
        fps=float(fps),
    )
    k0 = np.asarray(train_meta["k"][0][0], dtype=np.float64)
    common = {
        "w": int(train_meta["w"]),
        "h": int(train_meta["h"]),
        "fl_x": float(k0[0, 0]),
        "fl_y": float(k0[1, 1]),
        "cx": float(k0[0, 2]),
        "cy": float(k0[1, 2]),
    }
    (out_dir / "transforms_train.json").write_text(
        json.dumps({**common, "frames": train_frames}), encoding="utf-8"
    )
    (out_dir / "transforms_test.json").write_text(
        json.dumps({**common, "frames": test_frames}), encoding="utf-8"
    )
    return {
        "w": int(train_meta["w"]),
        "h": int(train_meta["h"]),
        "train_samples": len(train_frames),
        "test_samples": len(test_frames),
        "timestamp_mode": str(timestamp_mode),
        "fps": float(fps),
    }


def _canon_view_id_from_name(name: str) -> str | None:
    match = re.search(r"(\d+)", str(name))
    if match is None:
        return None
    return f"{int(match.group(1)):03d}"


def _resolve_sparse_model_dir(frame_root: Path) -> Path:
    candidates = [
        frame_root / "sfm_opencv_refined",
        frame_root / "undistort" / "sparse",
        frame_root / "hloc_sfm" / "hloc_sfm" / "sfm",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Could not find sparse model dir under {frame_root}")


def _load_sfm_camera_centers(frame_root: Path) -> Dict[str, np.ndarray]:
    sparse_dir = _resolve_sparse_model_dir(frame_root)
    images = read_images_binary(str(sparse_dir / "images.bin"))
    out: Dict[str, np.ndarray] = {}
    for image in images.values():
        view_id = _canon_view_id_from_name(str(image.name))
        if view_id is None:
            continue
        rotation = qvec2rotmat(image.qvec)
        center = (-rotation.T @ np.asarray(image.tvec, dtype=np.float64).reshape(3)).reshape(3)
        out[view_id] = center
    return out


def _load_exported_camera_centers(frame_root: Path) -> Dict[str, np.ndarray]:
    fs = cv2.FileStorage(str(frame_root / "cameras_pred" / "extri.yml"), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open extri.yml under {frame_root / 'cameras_pred'}")
    out: Dict[str, np.ndarray] = {}
    try:
        text = (frame_root / "cameras_pred" / "extri.yml").read_text(encoding="utf-8", errors="ignore")
        for raw_view in re.findall(r"^Rot_(.+?):", text, flags=re.MULTILINE):
            view_id = f"{int(str(raw_view).strip()):03d}"
            rot = fs.getNode(f"Rot_{raw_view}").mat()
            trans = fs.getNode(f"T_{raw_view}").mat()
            if rot is None or trans is None:
                continue
            center = (-np.asarray(rot, dtype=np.float64).T @ np.asarray(trans, dtype=np.float64).reshape(3)).reshape(3)
            out[view_id] = center
    finally:
        fs.release()
    return out


def _collect_sfm_points_for_frame(
    frame_root: Path,
    timestamp: float,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    src_centers = _load_sfm_camera_centers(frame_root)
    dst_centers = _load_exported_camera_centers(frame_root)
    sim = estimate_similarity_from_centers(src_centers, dst_centers)
    sparse_dir = _resolve_sparse_model_dir(frame_root)
    points = read_points3D_binary(str(sparse_dir / "points3D.bin"))
    xyz = np.stack([np.asarray(point.xyz, dtype=np.float64) for point in points.values()], axis=0)
    rgb = np.stack([np.asarray(point.rgb, dtype=np.uint8) for point in points.values()], axis=0)
    if xyz.shape[0] > int(max_points):
        idx = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
    xyz_aligned = apply_similarity(xyz, sim)
    times = np.full((xyz_aligned.shape[0],), float(timestamp), dtype=np.float32)
    summary = {
        "raw_points": int(len(points)),
        "kept_points": int(xyz_aligned.shape[0]),
        "sim_scale": float(sim.scale),
    }
    return xyz_aligned, rgb, times, summary


def _build_time_init_from_sfm_points(
    *,
    per_frame_root: Path,
    frame_ids: Sequence[int],
    total_points: int,
    rng: np.random.Generator,
    timestamp_mode: str,
    fps: float,
    gauge_sims: Dict[int, SimilarityTransform] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    xyz_all: List[np.ndarray] = []
    rgb_all: List[np.ndarray] = []
    time_all: List[np.ndarray] = []
    frame_summaries: Dict[str, dict] = {}
    per_frame_budget = max(1, int(np.ceil(float(total_points) / max(1, len(frame_ids)))))
    for index, frame_id in enumerate(frame_ids):
        frame_root = per_frame_root / f"f{int(frame_id):06d}"
        timestamp = _timestamp_from_index(index, num_t=len(frame_ids), mode=str(timestamp_mode), fps=float(fps))
        xyz, rgb, times, summary = _collect_sfm_points_for_frame(
            frame_root=frame_root,
            timestamp=timestamp,
            max_points=per_frame_budget,
            rng=rng,
        )
        gauge_sim = gauge_sims.get(int(frame_id)) if gauge_sims is not None else None
        if gauge_sim is not None:
            xyz = apply_similarity(xyz, gauge_sim)
            summary.update(
                {
                    "gauge_scale": float(gauge_sim.scale),
                    "gauge_translation_norm": float(np.linalg.norm(gauge_sim.translation)),
                }
            )
        xyz_all.append(xyz)
        rgb_all.append(rgb)
        time_all.append(times)
        frame_summaries[f"f{int(frame_id):06d}"] = summary
    xyz_cat = np.concatenate(xyz_all, axis=0)
    rgb_cat = np.concatenate(rgb_all, axis=0)
    time_cat = np.concatenate(time_all, axis=0)
    if xyz_cat.shape[0] > int(total_points):
        keep = rng.choice(xyz_cat.shape[0], size=int(total_points), replace=False)
        xyz_cat = xyz_cat[keep]
        rgb_cat = rgb_cat[keep]
        time_cat = time_cat[keep]
    summary = {
        "frames": [int(frame_id) for frame_id in frame_ids],
        "per_frame_budget": int(per_frame_budget),
        "frame_summaries": frame_summaries,
        "mode": "sfm_points",
        "timestamp_mode": str(timestamp_mode),
        "fps": float(fps),
    }
    return xyz_cat, rgb_cat, time_cat, summary


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb_f = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    if rgb_f.size == 0:
        return rgb_f.astype(np.uint8)
    if float(np.nanmax(rgb_f)) <= 1.0 + 1e-6:
        rgb_f = rgb_f * 255.0
    return np.clip(np.rint(rgb_f), 0.0, 255.0).astype(np.uint8)


def _build_time_init_from_panoptic_init_npz(
    *,
    panoptic_root: Path,
    frame_ids: Sequence[int],
    total_points: int,
    rng: np.random.Generator,
    timestamp_mode: str,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    init_path = panoptic_root / "init_pt_cld.npz"
    if not init_path.is_file():
        raise FileNotFoundError(f"Missing PanopticSports init point cloud: {init_path}")
    payload = np.load(init_path)
    if "data" not in payload:
        raise KeyError(f"{init_path} must contain a `data` array")
    data = np.asarray(payload["data"], dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"{init_path} data must have shape N x >=6, got {data.shape}")
    xyz_base = data[:, :3].astype(np.float32)
    rgb_base = _rgb_to_uint8(data[:, 3:6])
    if xyz_base.shape[0] == 0:
        raise ValueError(f"{init_path} contains no init points")

    xyz_all: List[np.ndarray] = []
    rgb_all: List[np.ndarray] = []
    time_all: List[np.ndarray] = []
    frame_summaries: Dict[str, dict] = {}
    per_frame_budget = max(1, int(np.ceil(float(total_points) / max(1, len(frame_ids)))))
    for index, frame_id in enumerate(frame_ids):
        if xyz_base.shape[0] > per_frame_budget:
            keep = rng.choice(xyz_base.shape[0], size=per_frame_budget, replace=False)
        else:
            keep = np.arange(xyz_base.shape[0])
        timestamp = _timestamp_from_index(index, num_t=len(frame_ids), mode=str(timestamp_mode), fps=float(fps))
        xyz = xyz_base[keep]
        rgb = rgb_base[keep]
        times = np.full((xyz.shape[0],), float(timestamp), dtype=np.float32)
        xyz_all.append(xyz)
        rgb_all.append(rgb)
        time_all.append(times)
        frame_summaries[f"f{int(frame_id):06d}"] = {
            "raw_points": int(xyz_base.shape[0]),
            "kept_points": int(xyz.shape[0]),
            "timestamp": float(timestamp),
        }

    xyz_cat = np.concatenate(xyz_all, axis=0)
    rgb_cat = np.concatenate(rgb_all, axis=0)
    time_cat = np.concatenate(time_all, axis=0)
    if xyz_cat.shape[0] > int(total_points):
        keep = rng.choice(xyz_cat.shape[0], size=int(total_points), replace=False)
        xyz_cat = xyz_cat[keep]
        rgb_cat = rgb_cat[keep]
        time_cat = time_cat[keep]
        order = np.argsort(time_cat, kind="stable")
        xyz_cat = xyz_cat[order]
        rgb_cat = rgb_cat[order]
        time_cat = time_cat[order]

    segmentation_counts: dict[str, int] = {}
    if data.shape[1] >= 7:
        labels, counts = np.unique(data[:, 6].astype(np.int64), return_counts=True)
        segmentation_counts = {str(int(label)): int(count) for label, count in zip(labels, counts)}

    summary = {
        "mode": "panoptic_init_npz",
        "source_init_npz": str(init_path),
        "raw_init_points": int(xyz_base.shape[0]),
        "frames": [int(frame_id) for frame_id in frame_ids],
        "per_frame_budget": int(per_frame_budget),
        "frame_summaries": frame_summaries,
        "timestamp_mode": str(timestamp_mode),
        "fps": float(fps),
        "segmentation_counts": segmentation_counts,
    }
    return xyz_cat.astype(np.float32), rgb_cat.astype(np.uint8), time_cat.astype(np.float32), summary


def _view_id_from_panoptic_name(name: str) -> str:
    match = re.search(r"v(\d+)", str(name))
    if match is not None:
        return f"{int(match.group(1)):03d}"
    for part in Path(str(name)).parts[:-1]:
        if part.isdigit():
            return f"{int(part):03d}"
    raise ValueError(f"Cannot parse view id from {name}")


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _camera_centers_from_w2c(w2c_list: Sequence[np.ndarray]) -> np.ndarray:
    centers = []
    for w2c in w2c_list:
        pose = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
        centers.append((-pose[:3, :3].T @ pose[:3, 3]).reshape(3))
    return np.stack(centers, axis=0)


def _select_roma_reference_and_targets(
    *,
    view_ids: Sequence[str],
    w2c_list: Sequence[np.ndarray],
    ref_view: str,
    num_pairs: int,
    target_strategy: str = "nearest",
    target_views: Sequence[str] | str | None = None,
) -> tuple[int, list[int]]:
    centers = _camera_centers_from_w2c(w2c_list)
    if str(ref_view).lower() == "auto":
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        ref_idx = int(np.argmin(distances.mean(axis=1)))
    else:
        canonical = f"{int(ref_view):03d}"
        if canonical not in view_ids:
            raise ValueError(f"ref view {canonical} not in frame views")
        ref_idx = int(list(view_ids).index(canonical))
    allowed_targets = set(_normalize_ref_views(target_views))
    missing_targets = sorted(allowed_targets - set(view_ids))
    if missing_targets:
        raise ValueError(f"target views are not in frame views: {missing_targets}")
    distances = np.linalg.norm(centers - centers[ref_idx], axis=1)
    order = [
        int(i)
        for i in np.argsort(distances)
        if int(i) != ref_idx and (not allowed_targets or view_ids[int(i)] in allowed_targets)
    ]
    if not order:
        raise ValueError("No eligible RoMA target views remain after filtering")
    count = max(1, int(num_pairs))
    if str(target_strategy) == "nearest" or len(order) <= count:
        return ref_idx, order[:count]
    if str(target_strategy) != "spread":
        raise ValueError(f"Unsupported RoMA target strategy: {target_strategy}")
    positions = np.linspace(0, len(order) - 1, num=count)
    selected: list[int] = []
    seen: set[int] = set()
    for position in positions:
        candidate = int(order[int(round(float(position)))])
        if candidate in seen:
            continue
        seen.add(candidate)
        selected.append(candidate)
    for candidate in order:
        if len(selected) >= count:
            break
        if candidate in seen:
            continue
        seen.add(candidate)
        selected.append(candidate)
    return ref_idx, selected


def _canonical_view_id(view: str | int) -> str:
    view_text = str(view)
    return f"{int(view_text):03d}" if view_text.isdigit() else view_text


def _normalize_ref_views(ref_views: Sequence[str] | str | None) -> list[str]:
    if ref_views is None:
        return []
    if isinstance(ref_views, str):
        tokens = [token.strip() for token in re.split(r"[,\s]+", ref_views) if token.strip()]
    else:
        tokens = []
        for item in ref_views:
            tokens.extend(token.strip() for token in re.split(r"[,\s]+", str(item)) if token.strip())
    return [_canonical_view_id(token) for token in tokens]


def _build_roma_reference_target_plan(
    *,
    view_ids: Sequence[str],
    w2c_list: Sequence[np.ndarray],
    ref_view: str,
    ref_views: Sequence[str] | str | None,
    target_views: Sequence[str] | str | None = None,
    pairs_per_ref: int,
    target_strategy: str,
) -> list[dict]:
    canonical_view_ids = [_canonical_view_id(view) for view in view_ids]
    requested_refs = _normalize_ref_views(ref_views)
    if not requested_refs:
        ref_idx, target_indices = _select_roma_reference_and_targets(
            view_ids=canonical_view_ids,
            w2c_list=w2c_list,
            ref_view=ref_view,
            num_pairs=int(pairs_per_ref),
            target_strategy=str(target_strategy),
            target_views=target_views,
        )
        requested_ref_indices = [ref_idx]
        targets_by_ref = {ref_idx: target_indices}
    else:
        requested_ref_indices = []
        for requested_ref in requested_refs:
            if requested_ref not in canonical_view_ids:
                raise ValueError(f"ref view {requested_ref} not in frame views")
            requested_ref_indices.append(int(canonical_view_ids.index(requested_ref)))
        targets_by_ref = {}
        for ref_idx in requested_ref_indices:
            _unused_ref, target_indices = _select_roma_reference_and_targets(
                view_ids=canonical_view_ids,
                w2c_list=w2c_list,
                ref_view=canonical_view_ids[ref_idx],
                num_pairs=int(pairs_per_ref),
                target_strategy=str(target_strategy),
                target_views=target_views,
            )
            targets_by_ref[ref_idx] = target_indices

    plan: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    for ref_idx in requested_ref_indices:
        for target_idx in targets_by_ref[ref_idx]:
            pair_key = tuple(sorted((int(ref_idx), int(target_idx))))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            plan.append(
                {
                    "ref_idx": int(ref_idx),
                    "target_idx": int(target_idx),
                    "ref_view": canonical_view_ids[ref_idx],
                    "target_view": canonical_view_ids[target_idx],
                }
            )
    return plan


def _sample_rgb_at_uv(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    xy = np.rint(np.asarray(uv, dtype=np.float64)).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
    return image[xy[:, 1], xy[:, 0]].astype(np.uint8)


def _load_foreground_mask(path: Path | None, *, dilate_px: int = 0) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as image:
        mask_image = image.convert("L")
        radius = max(0, int(dilate_px))
        if radius > 0:
            from PIL import ImageFilter

            mask_image = mask_image.filter(ImageFilter.MaxFilter(size=2 * radius + 1))
        mask = np.asarray(mask_image, dtype=np.uint8) > 0
    return mask.astype(bool)


def _foreground_hits_at_uv(mask: np.ndarray | None, uv: np.ndarray) -> np.ndarray:
    coords = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if mask is None:
        return np.zeros((coords.shape[0],), dtype=bool)
    mask_bool = np.asarray(mask, dtype=bool)
    h, w = mask_bool.shape[:2]
    xy = np.rint(coords).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
    return mask_bool[xy[:, 1], xy[:, 0]].reshape(-1)


def _select_roma_match_indices_with_foreground_priority(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    overlaps: np.ndarray,
    *,
    mask_a: np.ndarray | None,
    mask_b: np.ndarray | None,
    mode: str,
    num_matches: int,
    min_foreground_ratio: float,
) -> tuple[np.ndarray, dict]:
    uv_a_np = np.asarray(uv_a, dtype=np.float64).reshape(-1, 2)
    uv_b_np = np.asarray(uv_b, dtype=np.float64).reshape(-1, 2)
    overlaps_np = np.asarray(overlaps, dtype=np.float64).reshape(-1)
    count = min(max(0, int(num_matches)), uv_a_np.shape[0])
    if uv_a_np.shape[0] != uv_b_np.shape[0] or uv_a_np.shape[0] != overlaps_np.shape[0]:
        raise ValueError("uv_a, uv_b, and overlaps must have the same length")
    if count == 0:
        return np.zeros((0,), dtype=np.int64), {
            "foreground_mode": str(mode),
            "candidate_matches": int(uv_a_np.shape[0]),
            "candidate_foreground_matches": 0,
            "selected_matches": 0,
            "selected_foreground_matches": 0,
            "selected_background_matches": 0,
        }

    mode_key = str(mode).lower()
    hit_a = _foreground_hits_at_uv(mask_a, uv_a_np)
    hit_b = _foreground_hits_at_uv(mask_b, uv_b_np)
    if mode_key == "none":
        foreground = np.zeros((uv_a_np.shape[0],), dtype=bool)
    elif mode_key == "ref":
        foreground = hit_a
    elif mode_key == "either":
        foreground = hit_a | hit_b
    elif mode_key == "both":
        foreground = hit_a & hit_b
    else:
        raise ValueError(f"Unsupported RoMA foreground mode: {mode}")

    order = np.argsort(-overlaps_np, kind="mergesort")
    if mode_key == "none":
        selected = order[:count]
    else:
        foreground_order = [int(index) for index in order if bool(foreground[index])]
        background_order = [int(index) for index in order if not bool(foreground[index])]
        target_foreground = int(np.ceil(float(count) * max(0.0, min(1.0, float(min_foreground_ratio)))))
        selected_list = foreground_order[: min(target_foreground, len(foreground_order))]
        for index in foreground_order[len(selected_list) :] + background_order:
            if len(selected_list) >= count:
                break
            selected_list.append(int(index))
        selected = np.asarray(selected_list, dtype=np.int64)

    selected = np.asarray(selected, dtype=np.int64)
    selected_foreground = foreground[selected] if selected.size else np.zeros((0,), dtype=bool)
    stats = {
        "foreground_mode": mode_key,
        "candidate_matches": int(uv_a_np.shape[0]),
        "candidate_foreground_matches": int(np.count_nonzero(foreground)),
        "selected_matches": int(selected.size),
        "selected_foreground_matches": int(np.count_nonzero(selected_foreground)),
        "selected_background_matches": int(selected.size - np.count_nonzero(selected_foreground)),
        "min_foreground_ratio": float(min_foreground_ratio),
    }
    return selected, stats


def _triangulate_roma_pair_points(
    *,
    roma,
    image_a_path: Path,
    image_b_path: Path,
    mask_a_path: Path | None = None,
    mask_b_path: Path | None = None,
    k_a: np.ndarray,
    w2c_a: np.ndarray,
    k_b: np.ndarray,
    w2c_b: np.ndarray,
    num_matches: int,
    candidate_multiplier: float,
    foreground_mode: str,
    foreground_min_ratio: float,
    foreground_dilate_px: int,
    max_reproj_px: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch

    image_a = _load_rgb_image(image_a_path)
    image_b = _load_rgb_image(image_b_path)
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]
    requested_matches = max(0, int(num_matches))
    sampled_candidates = int(np.ceil(float(requested_matches) * max(1.0, float(candidate_multiplier))))
    with torch.inference_mode():
        preds = roma.match(str(image_a_path), str(image_b_path))
        matches, overlaps, _precision_ab, _precision_ba = roma.sample(preds, sampled_candidates)
        uv_a_t, uv_b_t = roma.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
    uv_a_all = uv_a_t.detach().cpu().numpy().reshape(-1, 2)
    uv_b_all = uv_b_t.detach().cpu().numpy().reshape(-1, 2)
    overlaps_np = overlaps.detach().cpu().numpy().reshape(-1)
    valid = (
        np.isfinite(uv_a_all).all(axis=1)
        & np.isfinite(uv_b_all).all(axis=1)
        & (uv_a_all[:, 0] >= 0.0)
        & (uv_a_all[:, 0] < float(w_a))
        & (uv_a_all[:, 1] >= 0.0)
        & (uv_a_all[:, 1] < float(h_a))
        & (uv_b_all[:, 0] >= 0.0)
        & (uv_b_all[:, 0] < float(w_b))
        & (uv_b_all[:, 1] >= 0.0)
        & (uv_b_all[:, 1] < float(h_b))
    )
    uv_a_all = uv_a_all[valid]
    uv_b_all = uv_b_all[valid]
    overlaps_np = overlaps_np[valid]
    mask_a = _load_foreground_mask(mask_a_path, dilate_px=int(foreground_dilate_px))
    mask_b = _load_foreground_mask(mask_b_path, dilate_px=int(foreground_dilate_px))
    selected, foreground_stats = _select_roma_match_indices_with_foreground_priority(
        uv_a_all,
        uv_b_all,
        overlaps_np,
        mask_a=mask_a,
        mask_b=mask_b,
        mode=str(foreground_mode),
        num_matches=requested_matches,
        min_foreground_ratio=float(foreground_min_ratio),
    )
    uv_a_all = uv_a_all[selected]
    uv_b_all = uv_b_all[selected]
    overlaps_np = overlaps_np[selected]
    points = []
    uv_kept = []
    errors = []
    for uv_a, uv_b in zip(uv_a_all, uv_b_all):
        point, err = triangulate_pair_with_reprojection_filter(
            k_a=k_a,
            w2c_a=w2c_a,
            uv_a=uv_a,
            k_b=k_b,
            w2c_b=w2c_b,
            uv_b=uv_b,
            max_reproj_px=float(max_reproj_px),
            max_depth=float(max_depth),
        )
        if point is None:
            continue
        points.append(point)
        uv_kept.append(uv_a)
        errors.append(err)
    if not points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            {
                "requested_matches": int(requested_matches),
                "sampled_candidate_matches": int(sampled_candidates),
                "sampled_matches": int(len(uv_a_all)),
                "kept_points": 0,
                "mean_overlap": float(np.mean(overlaps_np)) if overlaps_np.size else 0.0,
                "mean_reproj_px": float("inf"),
                "foreground": foreground_stats,
            },
        )
    xyz = np.stack(points, axis=0).astype(np.float32)
    rgb = _sample_rgb_at_uv(image_a, np.stack(uv_kept, axis=0))
    return (
        xyz,
        rgb,
        {
            "requested_matches": int(requested_matches),
            "sampled_candidate_matches": int(sampled_candidates),
            "sampled_matches": int(len(uv_a_all)),
            "kept_points": int(xyz.shape[0]),
            "mean_overlap": float(np.mean(overlaps_np)) if overlaps_np.size else 0.0,
            "mean_reproj_px": float(np.mean(errors)) if errors else float("inf"),
            "max_depth": float(max_depth),
            "foreground": foreground_stats,
        },
    )


def _construct_roma_with_explicit_weights(
    RoMaV2,
    *,
    weights: Path,
    dinov3_src: Path,
    setting: str,
    compile_model: bool,
):
    import torch

    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"RoMaV2 weights not found: {weights}")
    dinov3_src = Path(dinov3_src).resolve()
    if not (dinov3_src / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"DINOv3 source root must contain hubconf.py: {dinov3_src}"
        )
    original_loader = torch.hub.load_state_dict_from_url
    original_hub_load = torch.hub.load

    def load_local_state_dict(_url, *args, **kwargs):
        map_location = kwargs.get("map_location", "cpu")
        try:
            return torch.load(
                weights,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:
            return torch.load(weights, map_location=map_location)

    def load_local_dinov3(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).startswith("facebookresearch/dinov3:"):
            return original_hub_load(
                str(dinov3_src),
                model,
                *args,
                source="local",
                **kwargs,
            )
        return original_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load_state_dict_from_url = load_local_state_dict
    torch.hub.load = load_local_dinov3
    try:
        return RoMaV2(
            RoMaV2.Cfg(setting=str(setting), compile=bool(compile_model))
        )
    finally:
        torch.hub.load_state_dict_from_url = original_loader
        torch.hub.load = original_hub_load


def _init_roma_model(
    *,
    weights: Path,
    dinov3_src: Path,
    setting: str,
    compile_model: bool,
    device: str,
):
    import torch

    roma_src_candidates = []
    env_roma_src = os.environ.get("ROMAV2_SRC", "").strip()
    if env_roma_src:
        roma_src_candidates.append(Path(env_roma_src))
    roma_src_candidates.append(PROJECT_ROOT / "third_party" / "RoMaV2" / "src")
    # Worktrees may not contain third_party weights/sources.  When the caller
    # passes a checkpoint from the main checkout, infer that checkout as a
    # fallback source location.
    if weights.name == "romav2.pt" and weights.parent.name == "checkpoints":
        roma_src_candidates.append(weights.parent.parent / "third_party" / "RoMaV2" / "src")
    for roma_src in reversed(roma_src_candidates):
        if (roma_src / "romav2").is_dir() and str(roma_src) not in sys.path:
            sys.path.insert(0, str(roma_src))
    from romav2 import RoMaV2

    model = _construct_roma_with_explicit_weights(
        RoMaV2,
        weights=weights,
        dinov3_src=dinov3_src,
        setting=setting,
        compile_model=compile_model,
    )
    torch_device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    return model.to(torch_device).eval()


def _panoptic_meta_index(
    frame_id: int,
    *,
    frame_ids: Sequence[int],
    meta_start_frame: int | None,
) -> int:
    if not frame_ids:
        raise ValueError("frame_ids must be non-empty")
    start = int(frame_ids[0]) if meta_start_frame is None else int(meta_start_frame)
    return int(frame_id) - start


def _build_time_init_from_roma_ref_pairs(
    *,
    panoptic_root: Path,
    frame_ids: Sequence[int],
    total_points: int,
    rng: np.random.Generator,
    roma_weights: Path,
    dinov3_src: Path,
    roma_setting: str,
    roma_compile: bool,
    device: str,
    roma_ref_view: str,
    roma_ref_views: Sequence[str] | str | None,
    roma_target_views: Sequence[str] | str | None,
    roma_pairs_per_frame: int,
    roma_pairs_per_ref: int,
    roma_matches_per_pair: int,
    roma_candidate_multiplier: float,
    roma_foreground_mode: str,
    roma_foreground_min_ratio: float,
    roma_foreground_dilate_px: int,
    roma_reproj_px: float,
    roma_max_depth: float,
    timestamp_mode: str,
    fps: float,
    roma_target_strategy: str,
    panoptic_meta_start_frame: int | None = None,
    gauge_sims: Dict[int, SimilarityTransform] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    train_meta, _test_meta = _read_train_test_meta(panoptic_root)
    roma = _init_roma_model(
        weights=roma_weights,
        dinov3_src=dinov3_src,
        setting=roma_setting,
        compile_model=roma_compile,
        device=device,
    )
    normalized_ref_views = _normalize_ref_views(roma_ref_views)
    effective_pairs_per_ref = (
        int(roma_pairs_per_ref)
        if normalized_ref_views and int(roma_pairs_per_ref) > 0
        else int(roma_pairs_per_frame)
    )
    xyz_all: List[np.ndarray] = []
    rgb_all: List[np.ndarray] = []
    time_all: List[np.ndarray] = []
    frame_summaries: Dict[str, dict] = {}
    for local_index, frame_id in enumerate(frame_ids):
        meta_index = _panoptic_meta_index(
            int(frame_id),
            frame_ids=frame_ids,
            meta_start_frame=panoptic_meta_start_frame,
        )
        if meta_index < 0 or meta_index >= len(train_meta["fn"]):
            raise IndexError(f"frame {frame_id} maps to invalid train_meta index {meta_index}")
        view_ids = [_view_id_from_panoptic_name(name) for name in train_meta["fn"][meta_index]]
        w2c_list = [np.asarray(x, dtype=np.float64).reshape(4, 4) for x in train_meta["w2c"][meta_index]]
        k_list = [np.asarray(x, dtype=np.float64).reshape(3, 3) for x in train_meta["k"][meta_index]]
        pair_plan = _build_roma_reference_target_plan(
            view_ids=view_ids,
            w2c_list=w2c_list,
            ref_view=roma_ref_view,
            ref_views=normalized_ref_views,
            target_views=roma_target_views,
            pairs_per_ref=int(effective_pairs_per_ref),
            target_strategy=str(roma_target_strategy),
        )
        timestamp = _timestamp_from_index(local_index, num_t=len(frame_ids), mode=str(timestamp_mode), fps=float(fps))
        pair_summaries = []
        ref_views_used: list[str] = []
        for pair_item in pair_plan:
            ref_idx = int(pair_item["ref_idx"])
            target_idx = int(pair_item["target_idx"])
            ref_name = train_meta["fn"][meta_index][ref_idx]
            ref_path = panoptic_root / "ims" / ref_name
            ref_mask_path = panoptic_root / "seg" / Path(str(ref_name)).with_suffix(".png")
            target_name = train_meta["fn"][meta_index][target_idx]
            target_path = panoptic_root / "ims" / target_name
            target_mask_path = panoptic_root / "seg" / Path(str(target_name)).with_suffix(".png")
            xyz, rgb, pair_summary = _triangulate_roma_pair_points(
                roma=roma,
                image_a_path=ref_path,
                image_b_path=target_path,
                mask_a_path=ref_mask_path,
                mask_b_path=target_mask_path,
                k_a=k_list[ref_idx],
                w2c_a=w2c_list[ref_idx],
                k_b=k_list[target_idx],
                w2c_b=w2c_list[target_idx],
                num_matches=int(roma_matches_per_pair),
                candidate_multiplier=float(roma_candidate_multiplier),
                foreground_mode=str(roma_foreground_mode),
                foreground_min_ratio=float(roma_foreground_min_ratio),
                foreground_dilate_px=int(roma_foreground_dilate_px),
                max_reproj_px=float(roma_reproj_px),
                max_depth=float(roma_max_depth),
            )
            if xyz.shape[0] > 0:
                gauge_sim = gauge_sims.get(int(frame_id)) if gauge_sims is not None else None
                if gauge_sim is not None:
                    xyz = apply_similarity(xyz, gauge_sim)
                xyz_all.append(xyz)
                rgb_all.append(rgb)
                time_all.append(np.full((xyz.shape[0],), timestamp, dtype=np.float32))
            ref_view_used = view_ids[ref_idx]
            if ref_view_used not in ref_views_used:
                ref_views_used.append(ref_view_used)
            pair_summary.update({"ref_view": ref_view_used, "target_view": view_ids[target_idx]})
            pair_summaries.append(pair_summary)
        frame_summaries[f"f{int(frame_id):06d}"] = {
            "ref_view": ref_views_used[0] if len(ref_views_used) == 1 else "",
            "ref_views": ref_views_used,
            "pair_summaries": pair_summaries,
            "kept_points": int(sum(item["kept_points"] for item in pair_summaries)),
        }
    if not xyz_all:
        raise RuntimeError("RoMA triangulation produced no points")
    xyz_cat = np.concatenate(xyz_all, axis=0)
    rgb_cat = np.concatenate(rgb_all, axis=0)
    time_cat = np.concatenate(time_all, axis=0)
    if xyz_cat.shape[0] > int(total_points):
        keep = rng.choice(xyz_cat.shape[0], size=int(total_points), replace=False)
        xyz_cat = xyz_cat[keep]
        rgb_cat = rgb_cat[keep]
        time_cat = time_cat[keep]
    summary = {
        "frames": [int(frame_id) for frame_id in frame_ids],
        "mode": "roma_ref_pairs",
        "roma_setting": str(roma_setting),
        "roma_ref_view": str(roma_ref_view),
        "roma_ref_views": normalized_ref_views,
        "roma_target_views": _normalize_ref_views(roma_target_views),
        "roma_pairs_per_frame": int(roma_pairs_per_frame),
        "roma_pairs_per_ref": int(effective_pairs_per_ref),
        "roma_matches_per_pair": int(roma_matches_per_pair),
        "roma_candidate_multiplier": float(roma_candidate_multiplier),
        "roma_foreground_mode": str(roma_foreground_mode),
        "roma_foreground_min_ratio": float(roma_foreground_min_ratio),
        "roma_foreground_dilate_px": int(roma_foreground_dilate_px),
        "roma_reproj_px": float(roma_reproj_px),
        "roma_max_depth": float(roma_max_depth),
        "roma_target_strategy": str(roma_target_strategy),
        "timestamp_mode": str(timestamp_mode),
        "fps": float(fps),
        "raw_kept_points_before_global_sample": int(sum(v["kept_points"] for v in frame_summaries.values())),
        "frame_summaries": frame_summaries,
    }
    return xyz_cat, rgb_cat, time_cat, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Fudan 4DGS dataset with time-aware init point cloud")
    parser.add_argument("--panoptic_root", type=Path, required=True)
    parser.add_argument("--per_frame_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="sfm_points",
        choices=["sfm_points", "roma_ref_pairs", "panoptic_init_npz"],
        help="sfm_points uses existing per-frame SfM point clouds; roma_ref_pairs triangulates RoMA matches; panoptic_init_npz reuses panoptic_root/init_pt_cld.npz.",
    )
    parser.add_argument("--start_frame", type=int, default=1)
    parser.add_argument("--end_frame", type=int, default=120)
    parser.add_argument("--total_points", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--romav2_weights", type=Path, default=PROJECT_ROOT / "checkpoints" / "romav2.pt")
    parser.add_argument(
        "--dinov3_src",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "dinov3",
    )
    parser.add_argument("--roma_setting", type=str, default="fast")
    parser.add_argument("--roma_compile", action="store_true")
    parser.add_argument("--roma_ref_view", type=str, default="auto")
    parser.add_argument("--roma_ref_views", nargs="*", default=[], help="Optional explicit multi-reference views for RoMA init, e.g. --roma_ref_views 2 9 14 23. Empty keeps single-ref behavior.")
    parser.add_argument(
        "--roma_target_views",
        nargs="*",
        default=[],
        help="Optional target-view allowlist for pose-independent RoMA initialization, e.g. static anchor views only.",
    )
    parser.add_argument("--roma_pairs_per_frame", type=int, default=2)
    parser.add_argument("--roma_pairs_per_ref", type=int, default=0, help="Number of target pairs per explicit RoMA reference view. <=0 falls back to roma_pairs_per_frame.")
    parser.add_argument("--roma_matches_per_pair", type=int, default=256)
    parser.add_argument("--roma_candidate_multiplier", type=float, default=1.0, help="Sample this many more RoMA candidates before keeping roma_matches_per_pair matches after foreground priority.")
    parser.add_argument(
        "--roma_foreground_mode",
        type=str,
        default="none",
        choices=["none", "ref", "either", "both"],
        help="Use PanopticSports seg masks to prioritize RoMA matches on the foreground person.",
    )
    parser.add_argument("--roma_foreground_min_ratio", type=float, default=0.0, help="Requested minimum foreground ratio among selected RoMA matches when foreground mode is enabled.")
    parser.add_argument("--roma_foreground_dilate_px", type=int, default=0, help="Dilate foreground masks by this radius in pixels before prioritizing matches.")
    parser.add_argument("--roma_reproj_px", type=float, default=2.0)
    parser.add_argument("--roma_max_depth", type=float, default=0.0, help="Reject RoMA-triangulated init points whose positive depth exceeds this value in either pair camera. <=0 disables the filter.")
    parser.add_argument(
        "--roma_target_strategy",
        type=str,
        default="nearest",
        choices=["nearest", "spread"],
        help="How to choose RoMA pair targets around the reference view.",
    )
    parser.add_argument(
        "--timestamp_mode",
        type=str,
        default="normalized",
        choices=["normalized", "fps"],
        help="normalized keeps legacy [0,1) timestamps; fps uses frame_idx / fps like native N3V conversion.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--panoptic_meta_start_frame",
        type=int,
        default=None,
        help=(
            "Absolute frame represented by metadata index 0. Defaults to "
            "start_frame for backward compatibility; set this when building "
            "an absolute sub-window from a larger Panoptic package."
        ),
    )
    parser.add_argument("--align_frame_gauge", action="store_true", help="Align each frame gauge to an anchor frame using common camera centers before writing Fudan transforms and points.")
    parser.add_argument("--gauge_anchor_frame", type=int, default=None, help="Frame id used as gauge anchor. Defaults to start_frame.")
    parser.add_argument("--gauge_align_views", nargs="*", default=[], help="Optional camera ids used for gauge alignment. Empty means all common cameras.")
    parser.add_argument("--gauge_min_common", type=int, default=6, help="Minimum common cameras required to estimate per-frame Sim(3) gauge alignment.")
    parser.add_argument("--freeze_static_views", nargs="*", default=[], help="Camera ids whose poses should be frozen to the anchor frame after frame-gauge alignment when --align_frame_gauge is enabled.")
    parser.add_argument("--init_point_filter_outliers", action=argparse.BooleanOptionalAction, default=True, help="Filter per-time init point outliers using robust MAD statistics before writing points3d.ply.")
    parser.add_argument("--init_point_filter_mad", type=float, default=4.0, help="Per-axis MAD threshold for init point outlier filtering. <=0 disables this part.")
    parser.add_argument("--init_point_filter_radial_mad", type=float, default=4.0, help="Radial normalized MAD threshold for init point outlier filtering. <=0 disables this part.")
    parser.add_argument("--init_point_filter_min_points", type=int, default=16, help="Do not filter a timestamp group if fewer points than this would remain.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng, randomness = _seed_random_generators(int(args.seed))
    frame_ids = list(range(int(args.start_frame), int(args.end_frame) + 1))
    if not frame_ids:
        raise ValueError("Empty frame range")
    if args.output_dir.exists() and bool(args.force):
        shutil.rmtree(args.output_dir)
    train_meta, test_meta = _read_train_test_meta(args.panoptic_root)
    gauge_sims: Dict[int, SimilarityTransform] | None = None
    gauge_summary: dict = {"enabled": False}
    anchor_frame = int(args.start_frame) if args.gauge_anchor_frame is None else int(args.gauge_anchor_frame)
    freeze_views = [str(view) for view in args.freeze_static_views]
    freeze_summary = {
        "enabled": bool(freeze_views),
        "anchor_frame": int(anchor_frame),
        "views": freeze_views,
        "order": "not_applied",
    }
    if bool(args.align_frame_gauge):
        train_meta, test_meta, gauge_sims, gauge_summary, freeze_summary = _align_frame_gauge_then_freeze_static_views(
            train_meta=train_meta,
            test_meta=test_meta,
            frame_ids=frame_ids,
            anchor_frame=anchor_frame,
            align_views=[str(view).zfill(3) if str(view).isdigit() else str(view) for view in args.gauge_align_views],
            min_common=int(args.gauge_min_common),
            freeze_views=freeze_views,
        )
    elif freeze_views:
        train_meta = _freeze_meta_view_poses_to_anchor(
            train_meta,
            frame_ids=frame_ids,
            anchor_frame=anchor_frame,
            freeze_views=freeze_views,
        )
        test_meta = _freeze_meta_view_poses_to_anchor(
            test_meta,
            frame_ids=frame_ids,
            anchor_frame=anchor_frame,
            freeze_views=freeze_views,
        )
        freeze_summary["order"] = "without_frame_gauge_alignment"
    dataset_summary = _write_transforms_from_panoptic(
        args.panoptic_root,
        train_meta,
        test_meta,
        args.output_dir,
        timestamp_mode=str(args.timestamp_mode),
        fps=float(args.fps),
    )
    if str(args.mode) == "sfm_points":
        if args.per_frame_root is None:
            raise ValueError("--per_frame_root is required when --mode sfm_points")
        xyz, rgb, times, init_summary = _build_time_init_from_sfm_points(
            per_frame_root=args.per_frame_root,
            frame_ids=frame_ids,
            total_points=int(args.total_points),
            rng=rng,
            timestamp_mode=str(args.timestamp_mode),
            fps=float(args.fps),
            gauge_sims=gauge_sims,
        )
    elif str(args.mode) == "roma_ref_pairs":
        xyz, rgb, times, init_summary = _build_time_init_from_roma_ref_pairs(
            panoptic_root=args.panoptic_root,
            frame_ids=frame_ids,
            total_points=int(args.total_points),
            rng=rng,
            roma_weights=args.romav2_weights,
            dinov3_src=args.dinov3_src,
            roma_setting=str(args.roma_setting),
            roma_compile=bool(args.roma_compile),
            device=str(args.device),
            roma_ref_view=str(args.roma_ref_view),
            roma_ref_views=args.roma_ref_views,
            roma_target_views=args.roma_target_views,
            roma_pairs_per_frame=int(args.roma_pairs_per_frame),
            roma_pairs_per_ref=int(args.roma_pairs_per_ref),
            roma_matches_per_pair=int(args.roma_matches_per_pair),
            roma_candidate_multiplier=float(args.roma_candidate_multiplier),
            roma_foreground_mode=str(args.roma_foreground_mode),
            roma_foreground_min_ratio=float(args.roma_foreground_min_ratio),
            roma_foreground_dilate_px=int(args.roma_foreground_dilate_px),
            roma_reproj_px=float(args.roma_reproj_px),
            roma_max_depth=float(args.roma_max_depth),
            timestamp_mode=str(args.timestamp_mode),
            fps=float(args.fps),
            roma_target_strategy=str(args.roma_target_strategy),
            panoptic_meta_start_frame=args.panoptic_meta_start_frame,
            gauge_sims=gauge_sims,
        )
    else:
        xyz, rgb, times, init_summary = _build_time_init_from_panoptic_init_npz(
            panoptic_root=args.panoptic_root,
            frame_ids=frame_ids,
            total_points=int(args.total_points),
            rng=rng,
            timestamp_mode=str(args.timestamp_mode),
            fps=float(args.fps),
        )
    init_point_filter_summary = {"enabled": False}
    if bool(args.init_point_filter_outliers):
        xyz, rgb, times, init_point_filter_summary = _filter_init_point_spatial_outliers(
            xyz,
            rgb,
            times,
            mad_threshold=float(args.init_point_filter_mad),
            radial_mad_threshold=float(args.init_point_filter_radial_mad),
            min_points_per_group=int(args.init_point_filter_min_points),
        )
    write_ply_with_time(args.output_dir / "points3d.ply", xyz, rgb, times)
    summary = {
        "source_panoptic": str(args.panoptic_root),
        "per_frame_root": str(args.per_frame_root) if args.per_frame_root is not None else "",
        "output_dir": str(args.output_dir),
        "mode": str(args.mode),
        "total_points": int(xyz.shape[0]),
        "frame_range": [int(args.start_frame), int(args.end_frame)],
        "timestamp_mode": str(args.timestamp_mode),
        "fps": float(args.fps),
        "dataset": dataset_summary,
        "init_summary": init_summary,
        "init_point_filter": init_point_filter_summary,
        "gauge_alignment": gauge_summary,
        "static_pose_freeze": freeze_summary,
        "randomness": randomness,
        "note": "sfm_points is a fast proxy; roma_ref_pairs triangulates RoMA matches; panoptic_init_npz reuses official PanopticSports init_pt_cld.npz.",
    }
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
