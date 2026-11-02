from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from track4d_calib.cameras import CameraParams


def anchor_relative_world_rotation(
    *,
    anchor_correction: np.ndarray,
    frame_correction: np.ndarray,
) -> np.ndarray:
    """Remove a seed map's constant world-rotation gauge from a frame correction."""

    anchor = np.asarray(anchor_correction, dtype=np.float64).reshape(3, 3)
    frame = np.asarray(frame_correction, dtype=np.float64).reshape(3, 3)
    return anchor.T @ frame


def anchor_relative_world_transform(
    *,
    anchor_correction: np.ndarray,
    anchor_translation: np.ndarray,
    frame_correction: np.ndarray,
    frame_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a seed map's constant SE(3) gauge from a frame correction."""

    anchor_rotation = np.asarray(anchor_correction, dtype=np.float64).reshape(3, 3)
    anchor_offset = np.asarray(anchor_translation, dtype=np.float64).reshape(3)
    frame_rotation = np.asarray(frame_correction, dtype=np.float64).reshape(3, 3)
    frame_offset = np.asarray(frame_translation, dtype=np.float64).reshape(3)
    return (
        anchor_rotation.T @ frame_rotation,
        anchor_rotation.T @ (frame_offset - anchor_offset),
    )


def interpolate_anchor_vectors(
    *,
    frame: int,
    anchor_frames: Sequence[int],
    vectors_by_anchor: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Linearly blend the two anchor-relative vectors bracketing a frame."""

    anchors = sorted({int(value) for value in anchor_frames})
    if not anchors:
        raise ValueError("anchor_frames must not be empty")
    missing = [anchor for anchor in anchors if anchor not in vectors_by_anchor]
    if missing:
        raise KeyError(f"anchor rotation vectors are missing: {missing}")
    frame = int(frame)
    if frame <= anchors[0]:
        return np.asarray(vectors_by_anchor[anchors[0]], dtype=np.float64).reshape(3).copy()
    if frame >= anchors[-1]:
        return np.asarray(vectors_by_anchor[anchors[-1]], dtype=np.float64).reshape(3).copy()

    right_index = next(index for index, anchor in enumerate(anchors) if anchor >= frame)
    left = anchors[right_index - 1]
    right = anchors[right_index]
    alpha = float(frame - left) / float(right - left)
    left_vector = np.asarray(vectors_by_anchor[left], dtype=np.float64).reshape(3)
    right_vector = np.asarray(vectors_by_anchor[right], dtype=np.float64).reshape(3)
    return (1.0 - alpha) * left_vector + alpha * right_vector


def interpolate_anchor_rotation_vectors(
    *,
    frame: int,
    anchor_frames: Sequence[int],
    vectors_by_anchor: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Linearly blend anchor-relative rotation vectors bracketing a frame."""

    return interpolate_anchor_vectors(
        frame=frame,
        anchor_frames=anchor_frames,
        vectors_by_anchor=vectors_by_anchor,
    )


def apply_common_world_rotation(
    camera: CameraParams,
    correction: np.ndarray,
) -> CameraParams:
    """Right-multiply one world-to-camera pose while preserving its center."""

    rotation = np.asarray(camera.R, dtype=np.float64).reshape(3, 3) @ np.asarray(
        correction, dtype=np.float64
    ).reshape(3, 3)
    center = np.asarray(camera.camera_center, dtype=np.float64).reshape(3)
    return replace(camera, R=rotation, T=(-rotation @ center).reshape(3))


def estimate_common_world_rotation_from_pose_pairs(
    *,
    base_cameras: Mapping[str, CameraParams],
    candidate_cameras: Mapping[str, CameraParams],
    contributor_views: Sequence[str],
    min_views: int = 3,
) -> dict[str, Any]:
    """Estimate the shared world-right rotation in ``R_candidate = R_base Q``."""

    required = max(1, int(min_views))
    contributors = []
    seen = set()
    for raw_view in contributor_views:
        view = str(raw_view)
        if view in seen or view not in base_cameras or view not in candidate_cameras:
            continue
        seen.add(view)
        contributors.append(view)
    if len(contributors) < required:
        return {
            "success": False,
            "reason": "insufficient_pose_pairs",
            "correction": np.eye(3, dtype=np.float64),
            "correction_vector": np.zeros(3, dtype=np.float64),
            "correction_deg": 0.0,
            "contributor_views": contributors,
            "contributor_count": int(len(contributors)),
        }

    corrections = [
        np.asarray(base_cameras[view].R, dtype=np.float64).reshape(3, 3).T
        @ np.asarray(candidate_cameras[view].R, dtype=np.float64).reshape(3, 3)
        for view in contributors
    ]
    vectors = np.stack(
        [cv2.Rodrigues(correction)[0].reshape(3) for correction in corrections]
    )
    correction_vector = np.median(vectors, axis=0)
    correction = cv2.Rodrigues(correction_vector.reshape(3, 1))[0]
    residuals_deg = np.asarray(
        [
            np.linalg.norm(
                cv2.Rodrigues(correction @ value.T)[0].reshape(3)
            )
            * 180.0
            / np.pi
            for value in corrections
        ],
        dtype=np.float64,
    )
    return {
        "success": True,
        "reason": "robust_world_right_rotation_estimated",
        "correction": correction,
        "correction_vector": correction_vector,
        "correction_deg": float(np.linalg.norm(correction_vector) * 180.0 / np.pi),
        "contributor_views": contributors,
        "contributor_count": int(len(contributors)),
        "residual_deg_median": float(np.median(residuals_deg)),
        "residual_deg_p90": float(np.percentile(residuals_deg, 90.0)),
        "residual_deg_max": float(np.max(residuals_deg)),
    }


def estimate_common_world_transform_from_pose_pairs(
    *,
    base_cameras: Mapping[str, CameraParams],
    candidate_cameras: Mapping[str, CameraParams],
    contributor_views: Sequence[str],
    min_views: int = 3,
) -> dict[str, Any]:
    """Estimate ``R' = R Q, T' = T + R t`` from paired rig poses."""

    required = max(1, int(min_views))
    contributors = []
    seen = set()
    for raw_view in contributor_views:
        view = str(raw_view)
        if view in seen or view not in base_cameras or view not in candidate_cameras:
            continue
        seen.add(view)
        contributors.append(view)
    if len(contributors) < required:
        return {
            "success": False,
            "reason": "insufficient_pose_pairs",
            "correction": np.eye(3, dtype=np.float64),
            "correction_vector": np.zeros(3, dtype=np.float64),
            "correction_deg": 0.0,
            "translation": np.zeros(3, dtype=np.float64),
            "translation_norm": 0.0,
            "contributor_views": contributors,
            "contributor_count": int(len(contributors)),
        }

    corrections = []
    translations = []
    for view in contributors:
        base = base_cameras[view]
        candidate = candidate_cameras[view]
        base_rotation = np.asarray(base.R, dtype=np.float64).reshape(3, 3)
        candidate_rotation = np.asarray(candidate.R, dtype=np.float64).reshape(3, 3)
        corrections.append(base_rotation.T @ candidate_rotation)
        translations.append(
            base_rotation.T
            @ (
                np.asarray(candidate.T, dtype=np.float64).reshape(3)
                - np.asarray(base.T, dtype=np.float64).reshape(3)
            )
        )
    vectors = np.stack(
        [cv2.Rodrigues(correction)[0].reshape(3) for correction in corrections]
    )
    translation_values = np.stack(translations)
    correction_vector = np.median(vectors, axis=0)
    translation = np.median(translation_values, axis=0)
    correction = cv2.Rodrigues(correction_vector.reshape(3, 1))[0]
    rotation_residuals_deg = np.asarray(
        [
            np.linalg.norm(cv2.Rodrigues(correction @ value.T)[0].reshape(3))
            * 180.0
            / np.pi
            for value in corrections
        ],
        dtype=np.float64,
    )
    translation_residuals = np.linalg.norm(
        translation_values - translation.reshape(1, 3), axis=1
    )
    return {
        "success": True,
        "reason": "robust_world_right_transform_estimated",
        "correction": correction,
        "correction_vector": correction_vector,
        "correction_deg": float(np.linalg.norm(correction_vector) * 180.0 / np.pi),
        "translation": translation,
        "translation_norm": float(np.linalg.norm(translation)),
        "contributor_views": contributors,
        "contributor_count": int(len(contributors)),
        "rotation_residual_deg_median": float(np.median(rotation_residuals_deg)),
        "rotation_residual_deg_p90": float(
            np.percentile(rotation_residuals_deg, 90.0)
        ),
        "rotation_residual_deg_max": float(np.max(rotation_residuals_deg)),
        "translation_residual_median": float(np.median(translation_residuals)),
        "translation_residual_p90": float(
            np.percentile(translation_residuals, 90.0)
        ),
        "translation_residual_max": float(np.max(translation_residuals)),
    }


def apply_common_world_rotation_about_pivot(
    camera: CameraParams,
    correction: np.ndarray,
    *,
    pivot: np.ndarray,
) -> CameraParams:
    """Apply one world rotation to the full camera pose about a scene pivot."""

    world_rotation = np.asarray(correction, dtype=np.float64).reshape(3, 3)
    rotation = np.asarray(camera.R, dtype=np.float64).reshape(3, 3) @ world_rotation
    center = np.asarray(camera.camera_center, dtype=np.float64).reshape(3)
    scene_pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
    corrected_center = world_rotation.T @ (center - scene_pivot) + scene_pivot
    return replace(
        camera,
        R=rotation,
        T=(-rotation @ corrected_center).reshape(3),
    )


def apply_common_world_transform(
    camera: CameraParams,
    correction: np.ndarray,
    *,
    translation: np.ndarray,
) -> CameraParams:
    """Apply one shared world SE(3) correction to a camera rig pose."""

    world_rotation = np.asarray(correction, dtype=np.float64).reshape(3, 3)
    world_translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera_rotation = np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
    rotation = camera_rotation @ world_rotation
    offset = np.asarray(camera.T, dtype=np.float64).reshape(3) + camera_rotation @ world_translation
    return replace(camera, R=rotation, T=offset)


def _collect_observations(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_views: Sequence[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    views = [str(view) for view in view_ids]
    tracks = np.asarray(tracks_frame, dtype=np.float64)
    visibility = np.asarray(visibility_frame, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks_frame must have shape (views, points, 2)")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError("visibility_frame shape must match tracks_frame[:2]")
    if tracks.shape[0] != len(views):
        raise ValueError("view_ids length must match the track view dimension")

    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    xyz = np.asarray(point_bank.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    if len(point_ids) != len(xyz):
        raise ValueError("point bank point_ids and xyz have different lengths")
    if np.any(point_ids < 0) or np.any(point_ids >= tracks.shape[1]):
        raise ValueError("point bank contains an out-of-range point id")

    observations = []
    for focus_view in focus_views:
        view = str(focus_view)
        if view not in views:
            raise KeyError(f"focus view {view!r} is absent from track bundle")
        view_index = views.index(view)
        valid = (
            visibility[view_index, point_ids]
            & np.isfinite(xyz).all(axis=1)
            & np.isfinite(tracks[view_index, point_ids]).all(axis=1)
        )
        if np.any(valid):
            observations.append(
                (
                    view,
                    xyz[valid],
                    tracks[view_index, point_ids[valid]],
                )
            )
    return observations


def _reprojection_residuals(
    rotation_vector: np.ndarray,
    *,
    observations: Sequence[tuple[str, np.ndarray, np.ndarray]],
    cameras_by_view: Mapping[str, CameraParams],
) -> np.ndarray:
    correction, _ = cv2.Rodrigues(
        np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1)
    )
    residuals = []
    for view, xyz, image_points in observations:
        camera = apply_common_world_rotation(cameras_by_view[view], correction)
        camera_rotation_vector, _ = cv2.Rodrigues(camera.R)
        projected, _ = cv2.projectPoints(
            xyz,
            camera_rotation_vector,
            camera.T,
            camera.K,
            camera.D,
        )
        residuals.append((projected.reshape(-1, 2) - image_points).reshape(-1))
    return np.concatenate(residuals) if residuals else np.empty((0,), dtype=np.float64)


def _reprojection_transform_residuals(
    transform_vector: np.ndarray,
    *,
    observations: Sequence[tuple[str, np.ndarray, np.ndarray]],
    cameras_by_view: Mapping[str, CameraParams],
) -> np.ndarray:
    vector = np.asarray(transform_vector, dtype=np.float64).reshape(6)
    correction, _ = cv2.Rodrigues(vector[:3].reshape(3, 1))
    residuals = []
    for view, xyz, image_points in observations:
        camera = apply_common_world_transform(
            cameras_by_view[view],
            correction,
            translation=vector[3:],
        )
        camera_rotation_vector, _ = cv2.Rodrigues(camera.R)
        projected, _ = cv2.projectPoints(
            xyz,
            camera_rotation_vector,
            camera.T,
            camera.K,
            camera.D,
        )
        residuals.append((projected.reshape(-1, 2) - image_points).reshape(-1))
    return np.concatenate(residuals) if residuals else np.empty((0,), dtype=np.float64)


def _summarize_pixel_residuals(residuals: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(np.asarray(residuals, dtype=np.float64).reshape(-1, 2), axis=1)
    if not len(errors):
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90.0)),
    }


def _collect_grouped_observations(
    *,
    observations_by_view: Mapping[str, Mapping[str, np.ndarray]],
    cameras_by_view: Mapping[str, CameraParams],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    observations = []
    for raw_view, payload in observations_by_view.items():
        view = str(raw_view)
        if view not in cameras_by_view:
            raise KeyError(f"focus camera {view!r} is missing")
        xyz = np.asarray(payload.get("xyz", []), dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(
            payload.get("image_points", []), dtype=np.float64
        ).reshape(-1, 2)
        if len(xyz) != len(image_points):
            raise ValueError(f"view {view} xyz and image_points have different lengths")
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(image_points).all(axis=1)
        if np.any(valid):
            observations.append((view, xyz[valid], image_points[valid]))
    return observations


def score_common_world_rotation_observations(
    *,
    observations_by_view: Mapping[str, Mapping[str, np.ndarray]],
    cameras_by_view: Mapping[str, CameraParams],
    correction: np.ndarray,
) -> dict[str, Any]:
    """Score a center-preserving shared rotation on grouped observations."""

    observations = _collect_grouped_observations(
        observations_by_view=observations_by_view,
        cameras_by_view=cameras_by_view,
    )
    correction_vector, _ = cv2.Rodrigues(
        np.asarray(correction, dtype=np.float64).reshape(3, 3)
    )
    by_view = {}
    pooled_values = []
    for view, xyz, image_points in observations:
        residuals = _reprojection_residuals(
            correction_vector.reshape(3),
            observations=[(view, xyz, image_points)],
            cameras_by_view=cameras_by_view,
        )
        values = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
        by_view[view] = {
            "count": int(len(values)),
            **_summarize_pixel_residuals(residuals),
            "values_px": values.tolist(),
        }
        pooled_values.append(values)
    pooled = (
        np.concatenate(pooled_values)
        if pooled_values
        else np.empty((0,), dtype=np.float64)
    )
    pooled_stats = (
        {
            "mean": float(np.mean(pooled)),
            "median": float(np.median(pooled)),
            "p90": float(np.percentile(pooled, 90.0)),
        }
        if len(pooled)
        else {"mean": None, "median": None, "p90": None}
    )
    return {
        "observation_count": int(len(pooled)),
        "focus_view_count": int(len(by_view)),
        "pooled": {
            "count": int(len(pooled)),
            **pooled_stats,
            "values_px": pooled.tolist(),
        },
        "by_view": by_view,
    }


def estimate_common_world_rotation_from_observations(
    *,
    observations_by_view: Mapping[str, Mapping[str, np.ndarray]],
    cameras_by_view: Mapping[str, CameraParams],
    robust_scale_px: float = 4.0,
    max_evaluations: int = 30,
) -> dict[str, Any]:
    """Fit a center-preserving shared rotation with redescending robustness."""

    if float(robust_scale_px) <= 0.0:
        raise ValueError("robust_scale_px must be positive")
    observations = _collect_grouped_observations(
        observations_by_view=observations_by_view,
        cameras_by_view=cameras_by_view,
    )
    observation_count = int(sum(len(xyz) for _view, xyz, _pixels in observations))
    if observation_count < 4:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "observation_count": observation_count,
            "correction": np.eye(3, dtype=np.float64),
            "correction_vector": np.zeros(3, dtype=np.float64),
        }

    zero = np.zeros(3, dtype=np.float64)
    before = _reprojection_residuals(
        zero,
        observations=observations,
        cameras_by_view=cameras_by_view,
    )
    solution = least_squares(
        lambda vector: _reprojection_residuals(
            vector,
            observations=observations,
            cameras_by_view=cameras_by_view,
        ),
        zero,
        loss="cauchy",
        f_scale=float(robust_scale_px),
        max_nfev=int(max_evaluations),
    )
    correction_vector = np.asarray(solution.x, dtype=np.float64).reshape(3)
    correction, _ = cv2.Rodrigues(correction_vector.reshape(3, 1))
    after = _reprojection_residuals(
        correction_vector,
        observations=observations,
        cameras_by_view=cameras_by_view,
    )
    before_stats = _summarize_pixel_residuals(before)
    after_stats = _summarize_pixel_residuals(after)
    return {
        "success": bool(solution.success and np.isfinite(correction_vector).all()),
        "reason": "ok" if solution.success else str(solution.message),
        "observation_count": observation_count,
        "focus_view_count": int(len(observations)),
        "robust_loss": "cauchy",
        "correction": correction,
        "correction_vector": correction_vector,
        "correction_deg": float(np.linalg.norm(correction_vector) * 180.0 / np.pi),
        "reprojection_mean_px_before": before_stats["mean"],
        "reprojection_median_px_before": before_stats["median"],
        "reprojection_p90_px_before": before_stats["p90"],
        "reprojection_mean_px_after": after_stats["mean"],
        "reprojection_median_px_after": after_stats["median"],
        "reprojection_p90_px_after": after_stats["p90"],
        "optimizer_evaluations": int(solution.nfev),
    }


def score_common_world_transform_observations(
    *,
    observations_by_view: Mapping[str, Mapping[str, np.ndarray]],
    cameras_by_view: Mapping[str, CameraParams],
    correction: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    """Score one shared world transform on grouped, possibly disjoint points."""

    observations = _collect_grouped_observations(
        observations_by_view=observations_by_view,
        cameras_by_view=cameras_by_view,
    )
    correction_vector, _ = cv2.Rodrigues(
        np.asarray(correction, dtype=np.float64).reshape(3, 3)
    )
    transform_vector = np.concatenate(
        [
            correction_vector.reshape(3),
            np.asarray(translation, dtype=np.float64).reshape(3),
        ]
    )
    by_view = {}
    pooled_values = []
    for view, xyz, image_points in observations:
        residuals = _reprojection_transform_residuals(
            transform_vector,
            observations=[(view, xyz, image_points)],
            cameras_by_view=cameras_by_view,
        )
        values = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
        stats = _summarize_pixel_residuals(residuals)
        by_view[view] = {
            "count": int(len(values)),
            **stats,
            "values_px": values.tolist(),
        }
        pooled_values.append(values)
    pooled = (
        np.concatenate(pooled_values)
        if pooled_values
        else np.empty((0,), dtype=np.float64)
    )
    pooled_stats = (
        {
            "mean": float(np.mean(pooled)),
            "median": float(np.median(pooled)),
            "p90": float(np.percentile(pooled, 90.0)),
        }
        if len(pooled)
        else {"mean": None, "median": None, "p90": None}
    )
    return {
        "observation_count": int(len(pooled)),
        "focus_view_count": int(len(by_view)),
        "pooled": {
            "count": int(len(pooled)),
            **pooled_stats,
            "values_px": pooled.tolist(),
        },
        "by_view": by_view,
    }


def estimate_common_world_transform_from_observations(
    *,
    observations_by_view: Mapping[str, Mapping[str, np.ndarray]],
    cameras_by_view: Mapping[str, CameraParams],
    robust_scale_px: float = 4.0,
    max_evaluations: int = 30,
) -> dict[str, Any]:
    """Fit one shared world SE(3) with redescending outlier robustness."""

    if float(robust_scale_px) <= 0.0:
        raise ValueError("robust_scale_px must be positive")
    observations = _collect_grouped_observations(
        observations_by_view=observations_by_view,
        cameras_by_view=cameras_by_view,
    )
    observation_count = int(sum(len(xyz) for _view, xyz, _pixels in observations))
    if observation_count < 4:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "observation_count": observation_count,
            "correction": np.eye(3, dtype=np.float64),
            "correction_vector": np.zeros(3, dtype=np.float64),
            "translation": np.zeros(3, dtype=np.float64),
        }

    zero = np.zeros(6, dtype=np.float64)
    before = _reprojection_transform_residuals(
        zero,
        observations=observations,
        cameras_by_view=cameras_by_view,
    )
    solution = least_squares(
        lambda vector: _reprojection_transform_residuals(
            vector,
            observations=observations,
            cameras_by_view=cameras_by_view,
        ),
        zero,
        loss="cauchy",
        f_scale=float(robust_scale_px),
        max_nfev=int(max_evaluations),
    )
    vector = np.asarray(solution.x, dtype=np.float64).reshape(6)
    correction_vector = vector[:3]
    translation = vector[3:]
    correction, _ = cv2.Rodrigues(correction_vector.reshape(3, 1))
    after = _reprojection_transform_residuals(
        vector,
        observations=observations,
        cameras_by_view=cameras_by_view,
    )
    before_stats = _summarize_pixel_residuals(before)
    after_stats = _summarize_pixel_residuals(after)
    return {
        "success": bool(solution.success and np.isfinite(vector).all()),
        "reason": "ok" if solution.success else str(solution.message),
        "observation_count": observation_count,
        "focus_view_count": int(len(observations)),
        "robust_loss": "cauchy",
        "correction": correction,
        "correction_vector": correction_vector,
        "correction_deg": float(np.linalg.norm(correction_vector) * 180.0 / np.pi),
        "translation": translation,
        "translation_norm": float(np.linalg.norm(translation)),
        "reprojection_mean_px_before": before_stats["mean"],
        "reprojection_median_px_before": before_stats["median"],
        "reprojection_p90_px_before": before_stats["p90"],
        "reprojection_mean_px_after": after_stats["mean"],
        "reprojection_median_px_after": after_stats["median"],
        "reprojection_p90_px_after": after_stats["p90"],
        "optimizer_evaluations": int(solution.nfev),
    }


def estimate_common_world_rotation(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_views: Sequence[str],
    cameras_by_view: Mapping[str, CameraParams],
    robust_scale_px: float = 4.0,
    max_evaluations: int = 30,
) -> dict[str, Any]:
    """Fit one center-preserving world rotation to all focus-view track observations."""

    missing = [str(view) for view in focus_views if str(view) not in cameras_by_view]
    if missing:
        raise KeyError(f"focus cameras are missing: {missing}")
    if float(robust_scale_px) <= 0.0:
        raise ValueError("robust_scale_px must be positive")

    observations = _collect_observations(
        point_bank=point_bank,
        tracks_frame=tracks_frame,
        visibility_frame=visibility_frame,
        view_ids=view_ids,
        focus_views=focus_views,
    )
    observation_count = int(sum(len(xyz) for _view, xyz, _image_points in observations))
    if observation_count < 4:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "observation_count": observation_count,
            "correction": np.eye(3, dtype=np.float64),
            "correction_vector": np.zeros(3, dtype=np.float64),
        }

    zero = np.zeros(3, dtype=np.float64)
    before = _reprojection_residuals(
        zero,
        observations=observations,
        cameras_by_view=cameras_by_view,
    )
    solution = least_squares(
        lambda vector: _reprojection_residuals(
            vector,
            observations=observations,
            cameras_by_view=cameras_by_view,
        ),
        zero,
        loss="huber",
        f_scale=float(robust_scale_px),
        max_nfev=int(max_evaluations),
    )
    correction_vector = np.asarray(solution.x, dtype=np.float64).reshape(3)
    correction, _ = cv2.Rodrigues(correction_vector.reshape(3, 1))
    after = _reprojection_residuals(
        correction_vector,
        observations=observations,
        cameras_by_view=cameras_by_view,
    )
    before_stats = _summarize_pixel_residuals(before)
    after_stats = _summarize_pixel_residuals(after)
    return {
        "success": bool(solution.success and np.isfinite(correction_vector).all()),
        "reason": "ok" if solution.success else str(solution.message),
        "observation_count": observation_count,
        "focus_view_count": int(len(observations)),
        "correction": correction,
        "correction_vector": correction_vector,
        "correction_deg": float(np.linalg.norm(correction_vector) * 180.0 / np.pi),
        "reprojection_mean_px_before": before_stats["mean"],
        "reprojection_median_px_before": before_stats["median"],
        "reprojection_p90_px_before": before_stats["p90"],
        "reprojection_mean_px_after": after_stats["mean"],
        "reprojection_median_px_after": after_stats["median"],
        "reprojection_p90_px_after": after_stats["p90"],
        "optimizer_evaluations": int(solution.nfev),
    }


def estimate_common_world_transform(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_views: Sequence[str],
    cameras_by_view: Mapping[str, CameraParams],
    robust_scale_px: float = 4.0,
    max_evaluations: int = 30,
) -> dict[str, Any]:
    """Fit one shared world SE(3) correction to all focus-view track observations."""

    missing = [str(view) for view in focus_views if str(view) not in cameras_by_view]
    if missing:
        raise KeyError(f"focus cameras are missing: {missing}")
    if float(robust_scale_px) <= 0.0:
        raise ValueError("robust_scale_px must be positive")

    observations = _collect_observations(
        point_bank=point_bank,
        tracks_frame=tracks_frame,
        visibility_frame=visibility_frame,
        view_ids=view_ids,
        focus_views=focus_views,
    )
    observation_count = int(sum(len(xyz) for _view, xyz, _image_points in observations))
    if observation_count < 4:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "observation_count": observation_count,
            "correction": np.eye(3, dtype=np.float64),
            "correction_vector": np.zeros(3, dtype=np.float64),
            "translation": np.zeros(3, dtype=np.float64),
        }

    zero = np.zeros(6, dtype=np.float64)
    before = _reprojection_transform_residuals(
        zero,
        observations=observations,
        cameras_by_view=cameras_by_view,
    )
    solution = least_squares(
        lambda vector: _reprojection_transform_residuals(
            vector,
            observations=observations,
            cameras_by_view=cameras_by_view,
        ),
        zero,
        loss="huber",
        f_scale=float(robust_scale_px),
        max_nfev=int(max_evaluations),
    )
    vector = np.asarray(solution.x, dtype=np.float64).reshape(6)
    correction_vector = vector[:3]
    translation = vector[3:]
    correction, _ = cv2.Rodrigues(correction_vector.reshape(3, 1))
    after = _reprojection_transform_residuals(
        vector,
        observations=observations,
        cameras_by_view=cameras_by_view,
    )
    before_stats = _summarize_pixel_residuals(before)
    after_stats = _summarize_pixel_residuals(after)
    return {
        "success": bool(solution.success and np.isfinite(vector).all()),
        "reason": "ok" if solution.success else str(solution.message),
        "observation_count": observation_count,
        "focus_view_count": int(len(observations)),
        "correction": correction,
        "correction_vector": correction_vector,
        "correction_deg": float(np.linalg.norm(correction_vector) * 180.0 / np.pi),
        "translation": translation,
        "translation_norm": float(np.linalg.norm(translation)),
        "reprojection_mean_px_before": before_stats["mean"],
        "reprojection_median_px_before": before_stats["median"],
        "reprojection_p90_px_before": before_stats["p90"],
        "reprojection_mean_px_after": after_stats["mean"],
        "reprojection_median_px_after": after_stats["median"],
        "reprojection_p90_px_after": after_stats["p90"],
        "optimizer_evaluations": int(solution.nfev),
    }
