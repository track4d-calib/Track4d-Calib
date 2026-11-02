from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def extract_hloc_raw_correspondences(
    *,
    localization_record: dict[str, Any],
    points3d_by_id: Any,
) -> dict[str, Any]:
    """Extract raw HLoc 2D-3D matches without reusing a full-PnP inlier mask."""

    image_points = np.asarray(
        localization_record["keypoints_query"],
        dtype=np.float64,
    ).reshape(-1, 2)
    point_ids = np.asarray(
        localization_record["points3D_ids"],
        dtype=np.int64,
    ).reshape(-1)
    if len(image_points) != len(point_ids):
        raise ValueError("HLoc keypoints and point IDs have different lengths")
    selected_points = []
    for point_id in point_ids:
        if int(point_id) not in points3d_by_id:
            raise KeyError(f"HLoc references missing point3D id {int(point_id)}")
        point = points3d_by_id[int(point_id)]
        selected_points.append(
            np.asarray(getattr(point, "xyz", point), dtype=np.float64).reshape(3)
        )
    points3d = np.stack(selected_points) if selected_points else np.empty((0, 3))
    camera = localization_record["PnP_ret"]["camera"]
    intrinsic = np.asarray(
        camera.calibration_matrix(),
        dtype=np.float64,
    ).reshape(3, 3)
    camera_params = np.asarray(getattr(camera, "params", []), dtype=np.float64)
    if not all(
        np.isfinite(value).all()
        for value in (points3d, image_points, intrinsic, camera_params)
    ):
        raise ValueError("HLoc raw correspondence geometry contains non-finite values")
    return {
        "point_ids": point_ids,
        "points3d": points3d,
        "image_points": image_points,
        "intrinsic": intrinsic,
        "width": int(camera.width),
        "height": int(camera.height),
        "camera_model": str(camera.model),
        "camera_model_name": str(getattr(camera.model, "name", camera.model)),
        "camera_params": camera_params,
    }


def extract_hloc_pnp_inliers(
    *,
    localization_record: dict[str, Any],
    points3d_by_id: Any,
) -> dict[str, np.ndarray]:
    """Extract ordered inlier geometry from one HLoc localization log record."""

    pnp = localization_record["PnP_ret"]
    image_points = np.asarray(
        localization_record["keypoints_query"],
        dtype=np.float64,
    ).reshape(-1, 2)
    point_ids = np.asarray(localization_record["points3D_ids"], dtype=np.int64).reshape(-1)
    inlier_mask = np.asarray(pnp["inlier_mask"], dtype=bool).reshape(-1)
    if len(image_points) != len(point_ids) or len(point_ids) != len(inlier_mask):
        raise ValueError("HLoc keypoints, point IDs, and inlier mask have different lengths")

    selected_ids = point_ids[inlier_mask]
    selected_points = []
    for point_id in selected_ids:
        if int(point_id) not in points3d_by_id:
            raise KeyError(f"HLoc references missing point3D id {int(point_id)}")
        point = points3d_by_id[int(point_id)]
        selected_points.append(
            np.asarray(getattr(point, "xyz", point), dtype=np.float64).reshape(3)
        )
    points3d = np.stack(selected_points) if selected_points else np.empty((0, 3))

    pose = pnp["cam_from_world"]
    camera = pnp["camera"]
    rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(pose.translation, dtype=np.float64).reshape(3)
    intrinsic = np.asarray(camera.calibration_matrix(), dtype=np.float64).reshape(3, 3)
    if not all(
        np.isfinite(value).all()
        for value in (points3d, image_points[inlier_mask], rotation, translation, intrinsic)
    ):
        raise ValueError("HLoc PnP inlier geometry contains non-finite values")
    return {
        "point_ids": selected_ids,
        "points3d": points3d,
        "image_points": image_points[inlier_mask],
        "rotation": rotation,
        "translation": translation,
        "intrinsic": intrinsic,
    }


def fuse_tracking_rotation_weight(
    *,
    pnp_variance: float,
    tracking_variance: float,
) -> float:
    """Return the MAP weight of a tracking measurement against a zero PnP prior."""

    pnp = float(pnp_variance)
    tracking = float(tracking_variance)
    if not np.isfinite(pnp) or not np.isfinite(tracking):
        raise ValueError("rotation variances must be finite")
    if pnp < 0.0 or tracking < 0.0:
        raise ValueError("rotation variances must be non-negative")
    total = pnp + tracking
    if total <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(pnp / total, 0.0, 1.0))


def _sample_covariance(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    if len(values) < 2:
        return np.zeros((3, 3), dtype=np.float64)
    covariance = np.asarray(np.cov(values, rowvar=False, ddof=1), dtype=np.float64)
    return 0.5 * (covariance + covariance.T)


def _second_moment(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    if len(values) == 0:
        return np.zeros((3, 3), dtype=np.float64)
    moment = values.T @ values / float(len(values))
    return 0.5 * (moment + moment.T)


def tracking_model_rotation_variance(
    *,
    common_vectors: np.ndarray,
    per_view_vectors: np.ndarray,
) -> dict[str, Any]:
    """Estimate shared-rotation uncertainty from seed and cross-view disagreement."""

    common = np.asarray(common_vectors, dtype=np.float64)
    per_view = np.asarray(per_view_vectors, dtype=np.float64)
    if common.ndim != 2 or common.shape[1] != 3:
        raise ValueError("common_vectors must have shape (seeds, 3)")
    if per_view.ndim != 3 or per_view.shape[0] != common.shape[0] or per_view.shape[2] != 3:
        raise ValueError("per_view_vectors must have shape (seeds, views, 3)")
    if not np.isfinite(common).all():
        raise ValueError("common tracking rotation vectors must be finite")
    finite_coordinates = np.isfinite(per_view)
    valid_rows = finite_coordinates.all(axis=2)
    partial_rows = finite_coordinates.any(axis=2) & ~valid_rows
    if np.any(partial_rows):
        raise ValueError("missing per-view tracking vectors must use full NaN rows")
    if int(np.sum(valid_rows)) < 2:
        raise ValueError("at least two valid seed-view tracking vectors are required")

    seed_covariance = _sample_covariance(common)
    residuals = per_view - common[:, None, :]
    # The common model must pay for repeatable per-view bias as well as random
    # disagreement. A centered sample covariance would hide such model bias.
    cross_view_covariance = _second_moment(residuals[valid_rows])
    covariance = seed_covariance + cross_view_covariance
    seed_variance = max(float(np.trace(seed_covariance) / 3.0), 0.0)
    cross_view_variance = max(float(np.trace(cross_view_covariance) / 3.0), 0.0)
    variance = max(float(np.trace(covariance) / 3.0), 0.0)
    return {
        "variance": variance,
        "rotation_std_deg": float(np.sqrt(variance) * 180.0 / np.pi),
        "seed_variance": seed_variance,
        "cross_view_variance": cross_view_variance,
        "covariance": covariance,
        "seed_covariance": seed_covariance,
        "cross_view_covariance": cross_view_covariance,
        "num_seeds": int(common.shape[0]),
        "num_views": int(per_view.shape[1]),
        "valid_seed_view_rows": int(np.sum(valid_rows)),
        "missing_seed_view_rows": int(valid_rows.size - np.sum(valid_rows)),
    }


def tracking_model_rotation_variance_by_view(
    *,
    common_vectors: np.ndarray,
    per_view_vectors: np.ndarray,
) -> dict[str, Any]:
    """Estimate common-rotation model uncertainty independently for each view."""

    common = np.asarray(common_vectors, dtype=np.float64)
    per_view = np.asarray(per_view_vectors, dtype=np.float64)
    if common.ndim != 2 or common.shape[1] != 3:
        raise ValueError("common_vectors must have shape (seeds, 3)")
    if per_view.ndim != 3 or per_view.shape[0] != common.shape[0] or per_view.shape[2] != 3:
        raise ValueError("per_view_vectors must have shape (seeds, views, 3)")
    if len(common) < 2:
        raise ValueError("at least two common tracking seeds are required")
    if not np.isfinite(common).all():
        raise ValueError("common tracking rotation vectors must be finite")
    finite_coordinates = np.isfinite(per_view)
    valid_rows = finite_coordinates.all(axis=2)
    partial_rows = finite_coordinates.any(axis=2) & ~valid_rows
    if np.any(partial_rows):
        raise ValueError("missing per-view tracking vectors must use full NaN rows")

    seed_covariance = _sample_covariance(common)
    seed_variance = max(float(np.trace(seed_covariance) / 3.0), 0.0)
    rows = []
    for view_index in range(per_view.shape[1]):
        valid = valid_rows[:, view_index]
        valid_count = int(np.sum(valid))
        if valid_count < 2:
            rows.append(
                {
                    "available": False,
                    "reason": "insufficient_seed_support",
                    "valid_seeds": valid_count,
                    "missing_seeds": int(len(common) - valid_count),
                }
            )
            continue

        residuals = per_view[valid, view_index] - common[valid]
        model_mismatch = _second_moment(residuals)
        covariance = seed_covariance + model_mismatch
        model_mismatch_variance = max(
            float(np.trace(model_mismatch) / 3.0),
            0.0,
        )
        variance = max(float(np.trace(covariance) / 3.0), 0.0)
        bias = np.mean(residuals, axis=0)
        rows.append(
            {
                "available": True,
                "variance": variance,
                "rotation_std_deg": float(np.sqrt(variance) * 180.0 / np.pi),
                "seed_variance": seed_variance,
                "model_mismatch_variance": model_mismatch_variance,
                "model_bias_deg": float(np.linalg.norm(bias) * 180.0 / np.pi),
                "covariance": covariance,
                "seed_covariance": seed_covariance,
                "model_mismatch_second_moment": model_mismatch,
                "valid_seeds": valid_count,
                "missing_seeds": int(len(common) - valid_count),
            }
        )
    return {
        "views": rows,
        "num_seeds": int(len(common)),
        "num_views": int(per_view.shape[1]),
    }


def _project_with_pose_perturbation(
    *,
    points3d: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    center: np.ndarray,
    perturbation: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(perturbation, dtype=np.float64).reshape(6)
    world_rotation, _ = cv2.Rodrigues(delta[:3].reshape(3, 1))
    corrected_rotation = rotation @ world_rotation
    corrected_center = center + delta[3:]
    camera_points = (
        corrected_rotation @ (points3d - corrected_center).T
    ).T
    pixels_homogeneous = (intrinsic @ camera_points.T).T
    return pixels_homogeneous[:, :2] / pixels_homogeneous[:, 2:3]


def estimate_pnp_rotation_variance(
    *,
    points3d: np.ndarray,
    image_points: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    """Marginalize camera-center uncertainty from a PnP reprojection Hessian."""

    xyz = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    camera_intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    camera_rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    camera_translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if len(xyz) != len(pixels):
        raise ValueError("points3d and image_points must have the same length")
    if len(xyz) < 6:
        raise ValueError("at least six PnP inliers are required")
    if not all(
        np.isfinite(value).all()
        for value in (xyz, pixels, camera_intrinsic, camera_rotation, camera_translation)
    ):
        raise ValueError("PnP covariance inputs must be finite")

    center = -camera_rotation.T @ camera_translation
    zero = np.zeros(6, dtype=np.float64)
    projected = _project_with_pose_perturbation(
        points3d=xyz,
        intrinsic=camera_intrinsic,
        rotation=camera_rotation,
        center=center,
        perturbation=zero,
    )
    if not np.isfinite(projected).all():
        raise ValueError("PnP projection produced non-finite pixels")
    residuals_2d = projected - pixels
    residuals = residuals_2d.reshape(-1)

    epsilon = 1e-6
    jacobian = np.empty((residuals.size, 6), dtype=np.float64)
    for index in range(6):
        offset = np.zeros(6, dtype=np.float64)
        offset[index] = epsilon
        positive = _project_with_pose_perturbation(
            points3d=xyz,
            intrinsic=camera_intrinsic,
            rotation=camera_rotation,
            center=center,
            perturbation=offset,
        )
        negative = _project_with_pose_perturbation(
            points3d=xyz,
            intrinsic=camera_intrinsic,
            rotation=camera_rotation,
            center=center,
            perturbation=-offset,
        )
        jacobian[:, index] = ((positive - negative) / (2.0 * epsilon)).reshape(-1)

    radial_errors = np.linalg.norm(residuals_2d, axis=1)
    median_error = float(np.median(radial_errors))
    noise_sigma_px = float(
        1.4826 * np.median(np.abs(radial_errors - median_error))
    )
    noise_sigma_px = max(noise_sigma_px, np.finfo(np.float64).eps)
    normal = jacobian.T @ jacobian
    normal_rank = int(np.linalg.matrix_rank(normal))
    normal_condition = float(np.linalg.cond(normal))
    covariance = noise_sigma_px**2 * np.linalg.pinv(
        normal,
        rcond=1e-12,
        hermitian=True,
    )
    rotation_covariance = 0.5 * (covariance[:3, :3] + covariance[:3, :3].T)
    variance = max(float(np.trace(rotation_covariance) / 3.0), 0.0)
    return {
        "variance": variance,
        "rotation_std_deg": float(np.sqrt(variance) * 180.0 / np.pi),
        "rotation_covariance": rotation_covariance,
        "full_covariance": covariance,
        "noise_sigma_px": noise_sigma_px,
        "reprojection_mean_px": float(np.mean(radial_errors)),
        "reprojection_median_px": median_error,
        "normal_rank": normal_rank,
        "normal_condition": normal_condition,
        "num_points": int(len(xyz)),
    }


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) @ np.asarray(
        right, dtype=np.float64
    ).T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _fit_pnp_subset(
    *,
    points3d: np.ndarray,
    image_points: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    reprojection_error_px: float,
    min_inliers: int,
    iterations_count: int,
    confidence: float,
    rng_seed: int,
) -> dict[str, Any]:
    minimum = max(4, int(min_inliers))
    if len(points3d) < minimum:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "point_count": int(len(points3d)),
            "inlier_count": 0,
        }
    cv2.setRNGSeed(int(rng_seed))
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        np.asarray(points3d, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        np.asarray(intrinsic, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64).reshape(-1),
        iterationsCount=int(iterations_count),
        reprojectionError=float(reprojection_error_px),
        confidence=float(confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_indices = (
        np.asarray(inliers, dtype=np.int64).reshape(-1)
        if success and inliers is not None
        else np.empty(0, dtype=np.int64)
    )
    if not success or len(inlier_indices) < minimum:
        return {
            "success": False,
            "reason": "pnp_ransac_failed_or_too_few_inliers",
            "point_count": int(len(points3d)),
            "inlier_count": int(len(inlier_indices)),
        }
    rotation_vector, translation = cv2.solvePnPRefineLM(
        np.asarray(points3d, dtype=np.float64)[inlier_indices],
        np.asarray(image_points, dtype=np.float64)[inlier_indices],
        np.asarray(intrinsic, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64).reshape(-1),
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    return {
        "success": True,
        "reason": "ok",
        "point_count": int(len(points3d)),
        "inlier_count": int(len(inlier_indices)),
        "inlier_ratio": float(len(inlier_indices) / len(points3d)),
        "rotation": np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        "translation": translation,
        "center": -np.asarray(rotation, dtype=np.float64).T @ translation,
    }


def _project_pose(
    *,
    points3d: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_vector, _ = cv2.Rodrigues(
        np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    )
    projected, _ = cv2.projectPoints(
        np.asarray(points3d, dtype=np.float64),
        rotation_vector,
        np.asarray(translation, dtype=np.float64).reshape(3),
        np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        np.asarray(distortion, dtype=np.float64).reshape(-1),
    )
    camera_points = (
        np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        @ np.asarray(points3d, dtype=np.float64).T
    ).T + np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return projected.reshape(-1, 2), camera_points[:, 2] > 1e-8


def balanced_point_id_folds(
    *,
    point_ids: np.ndarray,
    num_folds: int,
) -> np.ndarray:
    """Assign stable point IDs to deterministic, exactly balanced folds."""

    ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("point_ids must be unique")
    if np.any(ids < 0):
        raise ValueError("point_ids must be non-negative")

    mask = (1 << 64) - 1

    def mix(value: int) -> int:
        mixed = int(value) & mask
        mixed = (mixed + 0x9E3779B97F4A7C15) & mask
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
        return (mixed ^ (mixed >> 31)) & mask

    order = sorted(
        range(len(ids)),
        key=lambda index: (mix(int(ids[index])), int(ids[index])),
    )
    assignment = np.empty(len(ids), dtype=np.int64)
    for rank, index in enumerate(order):
        assignment[index] = int(rank % folds)
    return assignment


def estimate_pnp_jackknife_stability(
    *,
    point_ids: np.ndarray,
    points3d: np.ndarray,
    image_points: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    reference_rotation: np.ndarray | None = None,
    reference_translation: np.ndarray | None = None,
    num_folds: int = 3,
    reprojection_error_px: float = 4.0,
    min_inliers: int = 12,
    iterations_count: int = 2000,
    confidence: float = 0.999,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Measure PnP pose instability under deterministic leave-one-fold-out fits.

    Fold membership uses a balanced deterministic ordering of stable point IDs.
    The diagnostic therefore needs no ground truth, trajectory prior, or
    dataset-specific threshold.
    """

    ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    xyz = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    camera_intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    camera_distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    if len(ids) != len(xyz) or len(xyz) != len(pixels):
        raise ValueError("point_ids, points3d, and image_points must align")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("point_ids must be unique")
    if np.any(ids < 0):
        raise ValueError("point_ids must be non-negative")
    if not all(
        np.isfinite(value).all()
        for value in (xyz, pixels, camera_intrinsic, camera_distortion)
    ):
        raise ValueError("PnP jackknife inputs must be finite")
    if (reference_rotation is None) != (reference_translation is None):
        raise ValueError(
            "reference_rotation and reference_translation must be provided together"
        )
    reference_r = (
        None
        if reference_rotation is None
        else np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)
    )
    reference_t = (
        None
        if reference_translation is None
        else np.asarray(reference_translation, dtype=np.float64).reshape(3)
    )
    if reference_r is not None and not all(
        np.isfinite(value).all() for value in (reference_r, reference_t)
    ):
        raise ValueError("reference PnP pose must be finite")

    fold_ids = balanced_point_id_folds(point_ids=ids, num_folds=folds)
    fold_rows: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    for validation_fold in range(folds):
        fit_mask = fold_ids != validation_fold
        result = _fit_pnp_subset(
            points3d=xyz[fit_mask],
            image_points=pixels[fit_mask],
            intrinsic=camera_intrinsic,
            distortion=camera_distortion,
            reprojection_error_px=float(reprojection_error_px),
            min_inliers=int(min_inliers),
            iterations_count=int(iterations_count),
            confidence=float(confidence),
            rng_seed=int(rng_seed) + validation_fold,
        )
        row = {
            "validation_fold": int(validation_fold),
            "fit_point_count": int(np.count_nonzero(fit_mask)),
            "validation_point_count": int(np.count_nonzero(~fit_mask)),
            "fit_success": bool(result["success"]),
            "fit_reason": str(result["reason"]),
            "fit_inlier_count": int(result["inlier_count"]),
        }
        if result["success"]:
            row["fit_inlier_ratio"] = float(result["inlier_ratio"])
            hypotheses.append(
                {
                    "validation_fold": int(validation_fold),
                    "rotation": result["rotation"],
                    "translation": result["translation"],
                    "center": result["center"],
                }
            )
        fold_rows.append(row)

    if len(hypotheses) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_successful_folds",
            "num_folds": folds,
            "successful_fold_count": int(len(hypotheses)),
            "folds": fold_rows,
        }

    projections = {
        int(row["validation_fold"]): _project_pose(
            points3d=xyz,
            intrinsic=camera_intrinsic,
            distortion=camera_distortion,
            rotation=row["rotation"],
            translation=row["translation"],
        )
        for row in hypotheses
    }
    pairwise_rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(hypotheses):
        first_fold = int(first["validation_fold"])
        first_pixels, first_front = projections[first_fold]
        for second in hypotheses[first_index + 1 :]:
            second_fold = int(second["validation_fold"])
            second_pixels, second_front = projections[second_fold]
            valid = (
                first_front
                & second_front
                & np.isfinite(first_pixels).all(axis=1)
                & np.isfinite(second_pixels).all(axis=1)
            )
            disagreement = np.linalg.norm(
                first_pixels[valid] - second_pixels[valid], axis=1
            )
            pairwise_rows.append(
                {
                    "validation_fold_a": first_fold,
                    "validation_fold_b": second_fold,
                    "rotation_deg": _rotation_distance_deg(
                        first["rotation"], second["rotation"]
                    ),
                    "center_distance": float(
                        np.linalg.norm(first["center"] - second["center"])
                    ),
                    "projection_point_count": int(len(disagreement)),
                    "projection_median_px": (
                        float(np.median(disagreement))
                        if len(disagreement)
                        else float("inf")
                    ),
                    "projection_p90_px": (
                        float(np.percentile(disagreement, 90.0))
                        if len(disagreement)
                        else float("inf")
                    ),
                }
            )

    rotation_values = np.asarray(
        [row["rotation_deg"] for row in pairwise_rows], dtype=np.float64
    )
    center_values = np.asarray(
        [row["center_distance"] for row in pairwise_rows], dtype=np.float64
    )
    projection_values = np.asarray(
        [row["projection_median_px"] for row in pairwise_rows], dtype=np.float64
    )
    output: dict[str, Any] = {
        "available": True,
        "reason": "ok",
        "fold_policy": "balanced_stable_point_id_hash_rank_leave_one_fold_out",
        "num_folds": folds,
        "successful_fold_count": int(len(hypotheses)),
        "folds": fold_rows,
        "pairwise": pairwise_rows,
        "pairwise_rotation_median_deg": float(np.median(rotation_values)),
        "pairwise_rotation_max_deg": float(np.max(rotation_values)),
        "pairwise_center_median": float(np.median(center_values)),
        "pairwise_center_max": float(np.max(center_values)),
        "pairwise_projection_median_px": float(np.median(projection_values)),
        "pairwise_projection_max_median_px": float(np.max(projection_values)),
    }
    if reference_r is not None and reference_t is not None:
        reference_center = -reference_r.T @ reference_t
        reference_rotation_values = np.asarray(
            [
                _rotation_distance_deg(row["rotation"], reference_r)
                for row in hypotheses
            ],
            dtype=np.float64,
        )
        reference_center_values = np.asarray(
            [
                np.linalg.norm(row["center"] - reference_center)
                for row in hypotheses
            ],
            dtype=np.float64,
        )
        output.update(
            {
                "reference_rotation_median_deg": float(
                    np.median(reference_rotation_values)
                ),
                "reference_rotation_max_deg": float(
                    np.max(reference_rotation_values)
                ),
                "reference_center_median": float(
                    np.median(reference_center_values)
                ),
                "reference_center_max": float(np.max(reference_center_values)),
            }
        )
    return output
