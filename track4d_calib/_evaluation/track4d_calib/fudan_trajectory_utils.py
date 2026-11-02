from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


_CAM_VIEW_RE = re.compile(r"cam(?P<view>\d+)_(?P<frame>\d+)$")
_TRAIN_TEST_VIEW_RE = re.compile(r"(?P<split>train|test)_f(?P<frame>\d+)_v(?P<view>.+)$")


def parse_frame_view_id(file_path: str) -> str | None:
    stem = Path(str(file_path)).stem
    match = _CAM_VIEW_RE.match(stem)
    if match is not None:
        return match.group("view")
    match = _TRAIN_TEST_VIEW_RE.search(stem)
    if match is not None:
        view = str(match.group("view"))
        return str(int(view)) if view.isdigit() else view
    return None


def blender_c2w_to_opencv_w2c(c2w_blender: np.ndarray) -> np.ndarray:
    c2w_blender = np.asarray(c2w_blender, dtype=np.float64).reshape(4, 4)
    c2w_colmap = c2w_blender.copy()
    c2w_colmap[:3, 1:3] *= -1.0
    return np.linalg.inv(c2w_colmap)


def opencv_w2c_to_blender_c2w(w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
    c2w_colmap = np.linalg.inv(w2c)
    c2w_blender = c2w_colmap.copy()
    c2w_blender[:3, 1:3] *= -1.0
    return c2w_blender


def opencv_camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return (-rotation.T @ translation).reshape(3)


def load_named_opencv_cameras_from_transforms(transform_paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    cameras: Dict[str, Dict[str, Any]] = {}
    for path in transform_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for frame in payload.get("frames", []):
            view_id = parse_frame_view_id(frame.get("file_path", ""))
            if view_id is None or view_id in cameras:
                continue
            w2c = blender_c2w_to_opencv_w2c(np.asarray(frame["transform_matrix"], dtype=np.float64))
            cameras[str(view_id)] = {
                "view_id": str(view_id),
                "w2c": w2c,
                "R": w2c[:3, :3].copy(),
                "T": w2c[:3, 3].copy(),
                "center": opencv_camera_center_from_w2c(w2c),
                "fl_x": float(frame.get("fl_x", payload.get("fl_x", -1.0))),
                "fl_y": float(frame.get("fl_y", payload.get("fl_y", -1.0))),
                "cx": float(frame.get("cx", payload.get("cx", -1.0))),
                "cy": float(frame.get("cy", payload.get("cy", -1.0))),
                "width": int(round(float(payload.get("w", 0.0)))),
                "height": int(round(float(payload.get("h", 0.0)))),
                "source_transform": str(path),
                "image_path": str(frame.get("file_path", "")),
                "time": float(frame.get("time", 0.0)),
            }
    return cameras


def estimate_similarity_transform(source_points: np.ndarray, target_points: np.ndarray) -> Dict[str, np.ndarray | float]:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError(f"source/target shape mismatch: {source.shape} vs {target.shape}")
    if source.shape[0] < 3:
        raise ValueError("At least three point correspondences are required for a similarity transform")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / float(source.shape[0])
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    handedness = np.eye(3, dtype=np.float64)
    if float(np.linalg.det(u_mat @ vt_mat)) < 0.0:
        handedness[-1, -1] = -1.0
    rotation = u_mat @ handedness @ vt_mat
    variance = float((source_centered**2).sum() / float(source.shape[0]))
    if variance <= 0.0:
        raise ValueError("Degenerate source points for similarity transform")
    scale = float(np.trace(np.diag(singular_values) @ handedness) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return {"scale": scale, "rotation": rotation, "translation": translation}


def transform_points_similarity(points: np.ndarray, transform: Dict[str, np.ndarray | float]) -> np.ndarray:
    points_arr = np.asarray(points, dtype=np.float64)
    original_shape = points_arr.shape
    points_flat = points_arr.reshape(-1, 3)
    scale = float(transform["scale"])
    rotation = np.asarray(transform["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(transform["translation"], dtype=np.float64).reshape(3)
    mapped = scale * (rotation @ points_flat.T).T + translation
    return mapped.reshape(original_shape)


def look_at_opencv(center: np.ndarray, target: np.ndarray, base_y_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    base_y_axis = np.asarray(base_y_axis, dtype=np.float64).reshape(3)

    z_axis = target - center
    z_axis /= np.linalg.norm(z_axis) + 1e-12

    y_axis = base_y_axis - z_axis * float(np.dot(base_y_axis, z_axis))
    if np.linalg.norm(y_axis) < 1e-6:
        fallback = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        y_axis = fallback - z_axis * float(np.dot(fallback, z_axis))
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)
    translation = -rotation @ center
    return rotation, translation
