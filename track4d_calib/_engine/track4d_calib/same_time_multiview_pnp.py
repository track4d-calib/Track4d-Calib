from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Mapping

import cv2
import numpy as np

from track4d_calib.cameras import CameraParams
from track4d_calib.track_crossview_verify import stable_point_id_validation_groups
from track4d_calib.triangulation import triangulate_point_dlt


@dataclass(frozen=True)
class SameTimePnPConfig:
    min_support_views: int = 2
    min_support_baseline: float = 1e-4
    min_triangulation_angle_deg: float = 1.0
    support_reprojection_error_px: float = 2.0
    pnp_reprojection_error_px: float = 4.0
    min_pnp_inliers: int = 12
    pnp_iterations: int = 2000
    pnp_confidence: float = 0.999
    rng_seed: int = 0


def _normalized_point(camera: CameraParams, point: np.ndarray) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64).reshape(1, 1, 2)
    normalized = cv2.undistortPoints(
        value,
        np.asarray(camera.K, dtype=np.float64),
        np.asarray(camera.D, dtype=np.float64).reshape(-1),
    )
    return np.asarray(normalized, dtype=np.float64).reshape(2)


def _project(camera: CameraParams, point: np.ndarray) -> tuple[np.ndarray, float]:
    rotation_vector, _ = cv2.Rodrigues(
        np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
    )
    projected, _ = cv2.projectPoints(
        np.asarray(point, dtype=np.float64).reshape(1, 3),
        rotation_vector,
        np.asarray(camera.T, dtype=np.float64).reshape(3),
        np.asarray(camera.K, dtype=np.float64).reshape(3, 3),
        np.asarray(camera.D, dtype=np.float64).reshape(-1),
    )
    depth = float(
        (
            np.asarray(camera.R, dtype=np.float64)
            @ np.asarray(point, dtype=np.float64).reshape(3)
            + np.asarray(camera.T, dtype=np.float64).reshape(3)
        )[2]
    )
    return projected.reshape(2), depth


def _world_ray(camera: CameraParams, point: np.ndarray) -> np.ndarray:
    normalized = _normalized_point(camera, point)
    camera_ray = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    world_ray = np.asarray(camera.R, dtype=np.float64).T @ camera_ray
    return world_ray / np.linalg.norm(world_ray)


def _pair_angle_deg(
    first_camera: CameraParams,
    first_point: np.ndarray,
    second_camera: CameraParams,
    second_point: np.ndarray,
) -> float:
    first_ray = _world_ray(first_camera, first_point)
    second_ray = _world_ray(second_camera, second_point)
    cosine = float(np.clip(np.dot(first_ray, second_ray), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _triangulate_normalized(
    observations: list[tuple[str, CameraParams, np.ndarray]],
) -> np.ndarray | None:
    projections = []
    points = []
    for _view, camera, point in observations:
        projections.append(
            np.hstack(
                [
                    np.asarray(camera.R, dtype=np.float64).reshape(3, 3),
                    np.asarray(camera.T, dtype=np.float64).reshape(3, 1),
                ]
            )
        )
        points.append(_normalized_point(camera, point))
    return triangulate_point_dlt(projections, points)


def _triangulate_robust(
    *,
    observations: list[tuple[str, CameraParams, np.ndarray]],
    config: SameTimePnPConfig,
) -> dict[str, Any]:
    """Score all pair hypotheses with one projection call per support camera."""
    minimum = max(2, int(config.min_support_views))
    if len(observations) < minimum:
        return {"success": False, "reason": "insufficient_support_views"}

    projections = []
    normalized_points = []
    rays = []
    centers = []
    for _view, camera, observed in observations:
        rotation = np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(camera.T, dtype=np.float64).reshape(3, 1)
        projections.append(np.hstack([rotation, translation]))
        normalized = _normalized_point(camera, observed)
        normalized_points.append(normalized)
        ray = rotation.T @ np.array([normalized[0], normalized[1], 1.0])
        rays.append(ray / np.linalg.norm(ray))
        centers.append(camera.camera_center)

    pair_angles = {}
    candidate_points = []
    candidate_angles = []
    for first_index, second_index in combinations(range(len(observations)), 2):
        baseline = float(np.linalg.norm(centers[first_index] - centers[second_index]))
        if baseline < float(config.min_support_baseline):
            continue
        cosine = float(np.clip(np.dot(rays[first_index], rays[second_index]), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cosine)))
        pair_angles[first_index, second_index] = angle_deg
        if angle_deg < float(config.min_triangulation_angle_deg):
            continue
        point = triangulate_point_dlt(
            [projections[first_index], projections[second_index]],
            [normalized_points[first_index], normalized_points[second_index]],
        )
        if point is None or not np.isfinite(point).all():
            continue
        candidate_points.append(point)
        candidate_angles.append(angle_deg)

    if not candidate_points:
        return {"success": False, "reason": "no_valid_triangulation_hypothesis"}
    points = np.asarray(candidate_points, dtype=np.float64).reshape(-1, 3)
    errors = np.empty((len(points), len(observations)), dtype=np.float64)
    positive = np.empty(errors.shape, dtype=bool)
    for index, (_view, camera, observed) in enumerate(observations):
        rotation = np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(camera.T, dtype=np.float64).reshape(3)
        rotation_vector, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            points, rotation_vector, translation,
            np.asarray(camera.K, dtype=np.float64).reshape(3, 3),
            np.asarray(camera.D, dtype=np.float64).reshape(-1),
        )
        errors[:, index] = [float(np.linalg.norm(pixel - observed))
                            for pixel in projected.reshape(-1, 2)]
        positive[:, index] = [(rotation @ point + translation)[2] > 1e-8
                              for point in points]

    hypotheses: list[tuple[tuple[int, float, float], np.ndarray, np.ndarray]] = []
    for point, angle_deg, error_array, positive_array in zip(
        points, candidate_angles, errors, positive
    ):
        inliers = (
            positive_array
            & np.isfinite(error_array)
            & (error_array <= float(config.support_reprojection_error_px))
        )
        if int(np.count_nonzero(inliers)) < minimum:
            continue
        inlier_errors = error_array[inliers]
        score = (
            int(np.count_nonzero(inliers)),
            -float(np.median(inlier_errors)),
            angle_deg,
        )
        hypotheses.append((score, point, inliers))

    if not hypotheses:
        return {"success": False, "reason": "no_valid_triangulation_hypothesis"}
    _score, _point, inliers = max(hypotheses, key=lambda row: row[0])
    refined = _triangulate_normalized(
        [observation for observation, keep in zip(observations, inliers) if keep]
    )
    if refined is None or not np.isfinite(refined).all():
        return {"success": False, "reason": "triangulation_refinement_failed"}

    final_errors = []
    final_positive = []
    for _view, camera, observed in observations:
        projected, depth = _project(camera, refined)
        final_errors.append(float(np.linalg.norm(projected - observed)))
        final_positive.append(depth > 1e-8)
    final_errors_array = np.asarray(final_errors, dtype=np.float64)
    final_inliers = (
        np.asarray(final_positive, dtype=bool)
        & np.isfinite(final_errors_array)
        & (final_errors_array <= float(config.support_reprojection_error_px))
    )
    if int(np.count_nonzero(final_inliers)) < minimum:
        return {"success": False, "reason": "too_few_refined_support_inliers"}

    inlier_indices = np.flatnonzero(final_inliers)
    maximum_angle = 0.0
    for first_index, second_index in combinations(inlier_indices.tolist(), 2):
        maximum_angle = max(maximum_angle, pair_angles.get((first_index, second_index), 0.0))
    if maximum_angle < float(config.min_triangulation_angle_deg):
        return {"success": False, "reason": "insufficient_triangulation_angle"}

    return {
        "success": True,
        "reason": "ok",
        "point": refined,
        "inlier_mask": final_inliers,
        "inlier_count": int(np.count_nonzero(final_inliers)),
        "observation_count": int(len(observations)),
        "reprojection_median_px": float(np.median(final_errors_array[final_inliers])),
        "reprojection_p90_px": float(
            np.percentile(final_errors_array[final_inliers], 90.0)
        ),
        "max_triangulation_angle_deg": float(maximum_angle),
    }


def _triangulate_same_time_score_points(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    config: SameTimePnPConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    result = triangulate_same_time_multiview_points(
        support_cameras=support_cameras,
        support_observations=support_observations,
        target_observations=target_observations,
        config=config,
    )
    diagnostics = {
        key: result[key]
        for key in (
            "triangulated_count",
            "candidate_target_observation_count",
            "support_observation_count",
            "support_observation_inlier_count",
            "support_observation_rejected_count",
        )
    }
    return result["object_points"], result["image_points"], diagnostics


def triangulate_same_time_multiview_points(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    config: SameTimePnPConfig | None = None,
) -> dict[str, Any]:
    """Triangulate static-support tracks while preserving target point IDs."""

    cfg = config or SameTimePnPConfig()
    points_3d: list[np.ndarray] = []
    points_2d: list[np.ndarray] = []
    point_ids: list[int] = []
    triangulation_rows: list[dict[str, Any]] = []
    support_observation_count = 0
    support_inlier_count = 0
    for point_id in sorted(set(support_observations) & set(target_observations)):
        target_point = np.asarray(
            target_observations[point_id], dtype=np.float64
        ).reshape(2)
        if not np.isfinite(target_point).all():
            continue
        observations = []
        for view, point in support_observations[point_id].items():
            normalized_view = str(view)
            if normalized_view not in support_cameras:
                continue
            value = np.asarray(point, dtype=np.float64).reshape(2)
            if not np.isfinite(value).all():
                continue
            observations.append(
                (normalized_view, support_cameras[normalized_view], value)
            )
        support_observation_count += len(observations)
        triangulated = _triangulate_robust(observations=observations, config=cfg)
        if not triangulated["success"]:
            continue
        support_inlier_count += int(triangulated["inlier_count"])
        points_3d.append(np.asarray(triangulated["point"], dtype=np.float64).reshape(3))
        points_2d.append(target_point)
        point_ids.append(int(point_id))
        triangulation_rows.append(
            {
                "point_id": int(point_id),
                "support_inlier_count": int(triangulated["inlier_count"]),
                "support_observation_count": int(
                    triangulated["observation_count"]
                ),
                "support_reprojection_median_px": float(
                    triangulated["reprojection_median_px"]
                ),
                "support_reprojection_p90_px": float(
                    triangulated["reprojection_p90_px"]
                ),
                "max_triangulation_angle_deg": float(
                    triangulated["max_triangulation_angle_deg"]
                ),
            }
        )
    object_points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    return {
        "success": bool(len(object_points)),
        "reason": "ok" if len(object_points) else "no_triangulated_points",
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "object_points": object_points,
        "image_points": image_points,
        "triangulation_rows": triangulation_rows,
        "triangulated_count": int(len(object_points)),
        "candidate_target_observation_count": int(len(target_observations)),
        "support_observation_count": int(support_observation_count),
        "support_observation_inlier_count": int(support_inlier_count),
        "support_observation_rejected_count": int(
            support_observation_count - support_inlier_count
        ),
    }


def pretriangulate_static_support_points(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    config: SameTimePnPConfig | None = None,
) -> dict[str, Any]:
    """Triangulate each static-supported track exactly once.

    The resulting cache is target-camera agnostic.  It can therefore be reused
    by every PTZ query in the same frame without changing the tracks, geometric
    gates, or PnP fit performed for an individual target.
    """

    cfg = config or SameTimePnPConfig()
    point_results: dict[int, dict[str, Any]] = {}
    successful_count = 0
    for point_id in sorted(support_observations):
        observations = []
        for view, point in support_observations[point_id].items():
            normalized_view = str(view)
            if normalized_view not in support_cameras:
                continue
            value = np.asarray(point, dtype=np.float64).reshape(2)
            if not np.isfinite(value).all():
                continue
            observations.append(
                (normalized_view, support_cameras[normalized_view], value)
            )
        triangulated = _triangulate_robust(observations=observations, config=cfg)
        if bool(triangulated["success"]):
            successful_count += 1
        point_results[int(point_id)] = {
            "support_observation_count": int(len(observations)),
            "triangulation": triangulated,
        }
    return {
        "point_results": point_results,
        "candidate_point_count": int(len(point_results)),
        "triangulated_point_count": int(successful_count),
    }


def solve_pretriangulated_same_time_multiview_pnp(
    *,
    pretriangulated: Mapping[str, Any],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    target_template: CameraParams,
    config: SameTimePnPConfig | None = None,
) -> dict[str, Any]:
    """Solve one target PnP from a shared static-track triangulation cache."""

    cfg = config or SameTimePnPConfig()
    cached_results = pretriangulated.get("point_results")
    if not isinstance(cached_results, Mapping):
        raise ValueError("pretriangulated cache is missing point_results")

    points_3d: list[np.ndarray] = []
    points_2d: list[np.ndarray] = []
    point_ids: list[int] = []
    triangulation_rows: list[dict[str, Any]] = []
    support_observation_count = 0
    support_inlier_count = 0
    for point_id in sorted(set(support_observations) & set(target_observations)):
        target_point = np.asarray(
            target_observations[point_id], dtype=np.float64
        ).reshape(2)
        if not np.isfinite(target_point).all():
            continue
        if point_id not in cached_results:
            raise KeyError(f"point {point_id} is absent from triangulation cache")
        cached = cached_results[point_id]
        support_observation_count += int(cached["support_observation_count"])
        triangulated = cached["triangulation"]
        if not bool(triangulated["success"]):
            continue
        points_3d.append(
            np.asarray(triangulated["point"], dtype=np.float64).reshape(3)
        )
        points_2d.append(target_point)
        point_ids.append(int(point_id))
        support_inlier_count += int(triangulated["inlier_count"])
        triangulation_rows.append(
            {
                "point_id": int(point_id),
                "support_inlier_count": int(triangulated["inlier_count"]),
                "support_observation_count": int(
                    triangulated["observation_count"]
                ),
                "support_reprojection_median_px": float(
                    triangulated["reprojection_median_px"]
                ),
                "support_reprojection_p90_px": float(
                    triangulated["reprojection_p90_px"]
                ),
                "max_triangulation_angle_deg": float(
                    triangulated["max_triangulation_angle_deg"]
                ),
            }
        )

    object_points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    minimum = max(4, int(cfg.min_pnp_inliers))
    base = {
        "triangulated_count": int(len(object_points)),
        "candidate_target_observation_count": int(len(target_observations)),
        "support_observation_count": int(support_observation_count),
        "support_observation_inlier_count": int(support_inlier_count),
        "support_observation_rejected_count": int(
            support_observation_count - support_inlier_count
        ),
        "triangulation_rows": triangulation_rows,
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "object_points": object_points,
        "image_points": image_points,
    }
    if len(object_points) < minimum:
        return {
            **base,
            "success": False,
            "reason": "insufficient_triangulated_points",
            "inlier_count": 0,
        }

    cv2.setRNGSeed(int(cfg.rng_seed))
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        np.asarray(target_template.K, dtype=np.float64),
        np.asarray(target_template.D, dtype=np.float64).reshape(-1),
        iterationsCount=int(cfg.pnp_iterations),
        reprojectionError=float(cfg.pnp_reprojection_error_px),
        confidence=float(cfg.pnp_confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_indices = (
        np.asarray(inliers, dtype=np.int64).reshape(-1)
        if success and inliers is not None
        else np.empty(0, dtype=np.int64)
    )
    if not success or len(inlier_indices) < minimum:
        return {
            **base,
            "success": False,
            "reason": "pnp_ransac_failed_or_too_few_inliers",
            "inlier_count": int(len(inlier_indices)),
        }

    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_points[inlier_indices],
        image_points[inlier_indices],
        np.asarray(target_template.K, dtype=np.float64),
        np.asarray(target_template.D, dtype=np.float64).reshape(-1),
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera = replace(
        target_template,
        R=np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        T=translation,
    )
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        np.asarray(target_template.K, dtype=np.float64),
        np.asarray(target_template.D, dtype=np.float64).reshape(-1),
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    inlier_errors = errors[inlier_indices]
    return {
        **base,
        "success": True,
        "reason": "ok",
        "camera": camera,
        "inlier_count": int(len(inlier_indices)),
        "inlier_ratio": float(len(inlier_indices) / len(object_points)),
        "inlier_indices": inlier_indices,
        "inlier_point_ids": np.asarray(point_ids, dtype=np.int64)[inlier_indices],
        "reprojection_mean_px": float(np.mean(inlier_errors)),
        "reprojection_median_px": float(np.median(inlier_errors)),
        "reprojection_p90_px": float(np.percentile(inlier_errors, 90.0)),
    }


def _score_triangulated_pose(
    *,
    object_points: np.ndarray,
    image_points: np.ndarray,
    target_camera: CameraParams,
    diagnostics: Mapping[str, Any],
    config: SameTimePnPConfig,
) -> dict[str, Any]:
    threshold = float(config.pnp_reprojection_error_px)
    minimum = max(4, int(config.min_pnp_inliers))
    base = dict(diagnostics)
    if len(object_points) < minimum:
        return {
            **base,
            "success": False,
            "reason": "insufficient_triangulated_points",
            "inlier_count": 0,
        }

    errors: list[float] = []
    truncated_errors: list[float] = []
    for point, target_point in zip(object_points, image_points):
        projected, depth = _project(target_camera, point)
        if depth <= 1e-8 or not np.isfinite(projected).all():
            error = float("inf")
        else:
            error = float(np.linalg.norm(projected - target_point))
        if np.isfinite(error):
            errors.append(error)
        truncated_errors.append(min(error, threshold) if np.isfinite(error) else threshold)

    error_array = np.asarray(errors, dtype=np.float64)
    truncated_array = np.asarray(truncated_errors, dtype=np.float64)
    inlier_count = int(np.count_nonzero(error_array <= threshold))
    raw_statistics = (
        {
            "reprojection_mean_px": float(np.mean(error_array)),
            "reprojection_median_px": float(np.median(error_array)),
            "reprojection_p90_px": float(np.percentile(error_array, 90.0)),
        }
        if len(error_array) > 0
        else {
            "reprojection_mean_px": None,
            "reprojection_median_px": None,
            "reprojection_p90_px": None,
        }
    )
    return {
        **base,
        "success": True,
        "reason": "ok",
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / len(truncated_array)),
        **raw_statistics,
        "truncated_reprojection_mean_px": float(np.mean(truncated_array)),
        "truncated_reprojection_values_px": truncated_array.tolist(),
    }


def score_same_time_multiview_poses(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    target_cameras: Mapping[str, CameraParams],
    config: SameTimePnPConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Triangulate once, then score fixed poses on common correspondences."""

    cfg = config or SameTimePnPConfig()
    threshold = float(cfg.pnp_reprojection_error_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("pnp_reprojection_error_px must be positive and finite")
    object_points, image_points, diagnostics = _triangulate_same_time_score_points(
        support_cameras=support_cameras,
        support_observations=support_observations,
        target_observations=target_observations,
        config=cfg,
    )
    return {
        str(name): _score_triangulated_pose(
            object_points=object_points,
            image_points=image_points,
            target_camera=camera,
            diagnostics=diagnostics,
            config=cfg,
        )
        for name, camera in target_cameras.items()
    }


def score_same_time_multiview_pose(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    target_camera: CameraParams,
    config: SameTimePnPConfig | None = None,
) -> dict[str, Any]:
    """Score one fixed pose on common same-time multiview correspondences."""

    return score_same_time_multiview_poses(
        support_cameras=support_cameras,
        support_observations=support_observations,
        target_observations=target_observations,
        target_cameras={"target": target_camera},
        config=config,
    )["target"]


def solve_same_time_multiview_pnp(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    target_template: CameraParams,
    config: SameTimePnPConfig | None = None,
) -> dict[str, Any]:
    """Triangulate same-time support tracks, then estimate the target full pose."""

    cfg = config or SameTimePnPConfig()
    points_3d = []
    points_2d = []
    point_ids = []
    triangulation_rows = []
    support_observation_count = 0
    support_inlier_count = 0
    for point_id in sorted(set(support_observations) & set(target_observations)):
        target_point = np.asarray(target_observations[point_id], dtype=np.float64).reshape(2)
        if not np.isfinite(target_point).all():
            continue
        observations = []
        for view, point in support_observations[point_id].items():
            normalized_view = str(view)
            if normalized_view not in support_cameras:
                continue
            value = np.asarray(point, dtype=np.float64).reshape(2)
            if not np.isfinite(value).all():
                continue
            observations.append(
                (normalized_view, support_cameras[normalized_view], value)
            )
        support_observation_count += len(observations)
        triangulated = _triangulate_robust(observations=observations, config=cfg)
        if not triangulated["success"]:
            continue
        points_3d.append(triangulated["point"])
        points_2d.append(target_point)
        point_ids.append(int(point_id))
        support_inlier_count += int(triangulated["inlier_count"])
        triangulation_rows.append(
            {
                "point_id": int(point_id),
                "support_inlier_count": int(triangulated["inlier_count"]),
                "support_observation_count": int(triangulated["observation_count"]),
                "support_reprojection_median_px": float(
                    triangulated["reprojection_median_px"]
                ),
                "support_reprojection_p90_px": float(
                    triangulated["reprojection_p90_px"]
                ),
                "max_triangulation_angle_deg": float(
                    triangulated["max_triangulation_angle_deg"]
                ),
            }
        )

    object_points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    minimum = max(4, int(cfg.min_pnp_inliers))
    base = {
        "triangulated_count": int(len(object_points)),
        "candidate_target_observation_count": int(len(target_observations)),
        "support_observation_count": int(support_observation_count),
        "support_observation_inlier_count": int(support_inlier_count),
        "support_observation_rejected_count": int(
            support_observation_count - support_inlier_count
        ),
        "triangulation_rows": triangulation_rows,
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "object_points": object_points,
        "image_points": image_points,
    }
    if len(object_points) < minimum:
        return {
            **base,
            "success": False,
            "reason": "insufficient_triangulated_points",
            "inlier_count": 0,
        }

    cv2.setRNGSeed(int(cfg.rng_seed))
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        np.asarray(target_template.K, dtype=np.float64),
        np.asarray(target_template.D, dtype=np.float64).reshape(-1),
        iterationsCount=int(cfg.pnp_iterations),
        reprojectionError=float(cfg.pnp_reprojection_error_px),
        confidence=float(cfg.pnp_confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_indices = (
        np.asarray(inliers, dtype=np.int64).reshape(-1)
        if success and inliers is not None
        else np.empty(0, dtype=np.int64)
    )
    if not success or len(inlier_indices) < minimum:
        return {
            **base,
            "success": False,
            "reason": "pnp_ransac_failed_or_too_few_inliers",
            "inlier_count": int(len(inlier_indices)),
        }

    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_points[inlier_indices],
        image_points[inlier_indices],
        np.asarray(target_template.K, dtype=np.float64),
        np.asarray(target_template.D, dtype=np.float64).reshape(-1),
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera = replace(
        target_template,
        R=np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        T=translation,
    )
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        np.asarray(target_template.K, dtype=np.float64),
        np.asarray(target_template.D, dtype=np.float64).reshape(-1),
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    inlier_errors = errors[inlier_indices]
    return {
        **base,
        "success": True,
        "reason": "ok",
        "camera": camera,
        "inlier_count": int(len(inlier_indices)),
        "inlier_ratio": float(len(inlier_indices) / len(object_points)),
        "inlier_indices": inlier_indices,
        "inlier_point_ids": np.asarray(point_ids, dtype=np.int64)[inlier_indices],
        "reprojection_mean_px": float(np.mean(inlier_errors)),
        "reprojection_median_px": float(np.median(inlier_errors)),
        "reprojection_p90_px": float(np.percentile(inlier_errors, 90.0)),
    }


def crossfit_same_time_multiview_inliers(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    target_template: CameraParams,
    config: SameTimePnPConfig | None = None,
    num_folds: int,
    validation_fold: int,
) -> dict[str, Any]:
    """Select disjoint fit/verify static tracks with a fit-only PnP model.

    PnP RANSAC is estimated only from the fit point IDs. Its fixed pose then
    classifies independently triangulated verification points. The verify fold
    never affects the PnP hypothesis or a downstream repair candidate.
    """

    cfg = config or SameTimePnPConfig()
    folds = int(num_folds)
    heldout = int(validation_fold)
    if folds < 2 or not 0 <= heldout < folds:
        raise ValueError("validation fold must lie inside a multi-fold split")

    point_ids = np.asarray(
        sorted(set(support_observations) & set(target_observations)),
        dtype=np.int64,
    )
    groups = stable_point_id_validation_groups(point_ids)
    verify_mask = np.mod(groups, folds) == heldout
    fit_ids = point_ids[~verify_mask]
    verify_ids = point_ids[verify_mask]
    fit_targets = {int(point_id): target_observations[int(point_id)] for point_id in fit_ids}
    verify_targets = {
        int(point_id): target_observations[int(point_id)] for point_id in verify_ids
    }

    fit = solve_same_time_multiview_pnp(
        support_cameras=support_cameras,
        support_observations=support_observations,
        target_observations=fit_targets,
        target_template=target_template,
        config=cfg,
    )
    if not bool(fit.get("success", False)):
        return {
            "success": False,
            "reason": f"fit_pnp_{fit.get('reason', 'failed')}",
            "fit_candidate_count": int(len(fit_ids)),
            "verify_candidate_count": int(len(verify_ids)),
            "fit_inlier_count": int(fit.get("inlier_count", 0)),
            "verify_inlier_count": 0,
        }

    verify = triangulate_same_time_multiview_points(
        support_cameras=support_cameras,
        support_observations=support_observations,
        target_observations=verify_targets,
        config=cfg,
    )
    if not bool(verify.get("success", False)):
        return {
            "success": False,
            "reason": f"verify_triangulation_{verify.get('reason', 'failed')}",
            "fit_candidate_count": int(len(fit_ids)),
            "verify_candidate_count": int(len(verify_ids)),
            "fit_inlier_count": int(fit["inlier_count"]),
            "verify_inlier_count": 0,
        }

    verify_object_points = np.asarray(verify["object_points"], dtype=np.float64)
    verify_image_points = np.asarray(verify["image_points"], dtype=np.float64)
    verify_errors = []
    for point, image_point in zip(verify_object_points, verify_image_points):
        projected, depth = _project(fit["camera"], point)
        error = float(np.linalg.norm(projected - image_point))
        verify_errors.append(error if depth > 1e-8 else float("inf"))
    verify_errors_array = np.asarray(verify_errors, dtype=np.float64)
    verify_inlier_mask = (
        np.isfinite(verify_errors_array)
        & (verify_errors_array <= float(cfg.pnp_reprojection_error_px))
    )
    minimum_verify = 4
    verify_inlier_count = int(np.count_nonzero(verify_inlier_mask))
    if verify_inlier_count < minimum_verify:
        return {
            "success": False,
            "reason": "too_few_fit_model_verify_inliers",
            "fit_candidate_count": int(len(fit_ids)),
            "verify_candidate_count": int(len(verify_ids)),
            "fit_inlier_count": int(fit["inlier_count"]),
            "verify_inlier_count": verify_inlier_count,
        }

    fit_indices = np.asarray(fit["inlier_indices"], dtype=np.int64)
    fit_point_ids = np.asarray(fit["point_ids"], dtype=np.int64)[fit_indices]
    verify_point_ids = np.asarray(verify["point_ids"], dtype=np.int64)[verify_inlier_mask]
    if set(fit_point_ids.tolist()) & set(verify_point_ids.tolist()):
        raise RuntimeError("cross-fit PnP point-ID folds overlap")

    selected_verify_errors = verify_errors_array[verify_inlier_mask]
    return {
        "success": True,
        "reason": "ok",
        "camera": fit["camera"],
        "object_points": np.concatenate(
            [
                np.asarray(fit["object_points"], dtype=np.float64)[fit_indices],
                verify_object_points[verify_inlier_mask],
            ],
            axis=0,
        ),
        "image_points": np.concatenate(
            [
                np.asarray(fit["image_points"], dtype=np.float64)[fit_indices],
                verify_image_points[verify_inlier_mask],
            ],
            axis=0,
        ),
        "point_ids": np.concatenate([fit_point_ids, verify_point_ids], axis=0),
        "fit_point_ids": fit_point_ids,
        "verify_point_ids": verify_point_ids,
        "fit_candidate_count": int(len(fit_ids)),
        "verify_candidate_count": int(len(verify_ids)),
        "fit_inlier_count": int(len(fit_point_ids)),
        "verify_inlier_count": verify_inlier_count,
        "verify_reprojection_mean_px": float(np.mean(selected_verify_errors)),
        "verify_reprojection_median_px": float(np.median(selected_verify_errors)),
        "verify_reprojection_p90_px": float(
            np.percentile(selected_verify_errors, 90.0)
        ),
        "fairness": {
            "pnp_hypothesis_uses_verify_points": False,
            "fit_verify_point_ids_disjoint": True,
        },
    }
