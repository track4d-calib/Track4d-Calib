from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from track4d_calib.cameras import CameraParams
from track4d_calib.track_crossview_verify import (
    CrossViewTrackVerifierConfig,
    combine_focus_camera_comparisons,
    compare_focus_cameras,
    split_point_bank_for_validation,
)


def _project_camera_points(
    camera: CameraParams,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera_xyz = (np.asarray(camera.R, dtype=np.float64) @ xyz.T).T + np.asarray(
        camera.T,
        dtype=np.float64,
    ).reshape(1, 3)
    rotation_vector, _ = cv2.Rodrigues(
        np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
    )
    projected, _ = cv2.projectPoints(
        xyz,
        rotation_vector,
        np.asarray(camera.T, dtype=np.float64).reshape(3),
        np.asarray(camera.K, dtype=np.float64).reshape(3, 3),
        np.asarray(camera.D, dtype=np.float64).reshape(-1),
    )
    return projected.reshape(-1, 2), camera_xyz[:, 2] > 1e-8


def _summarize_crossfit_camera_consistency(
    *,
    successful_candidates: Sequence[Mapping[str, Any]],
    points: np.ndarray,
) -> dict[str, Any]:
    """Measure fold-camera agreement without using held-out scores or GT."""

    candidates = sorted(
        successful_candidates,
        key=lambda row: int(row["validation_fold"]),
    )
    if len(candidates) < 2:
        raise ValueError("at least two fold cameras need common valid projection support")
    finite_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite_points = finite_points[np.isfinite(finite_points).all(axis=1)]
    if len(finite_points) == 0:
        raise ValueError("fold cameras need common valid projection support")
    projected = {
        int(row["validation_fold"]): _project_camera_points(
            row["fit_result"]["camera"],
            finite_points,
        )
        for row in candidates
    }
    pairwise: list[dict[str, Any]] = []
    medoid_cost = {
        int(row["validation_fold"]): 0.0 for row in candidates
    }
    for first_index, first in enumerate(candidates):
        first_fold = int(first["validation_fold"])
        first_uv, first_front = projected[first_fold]
        for second in candidates[first_index + 1 :]:
            second_fold = int(second["validation_fold"])
            second_uv, second_front = projected[second_fold]
            valid = (
                first_front
                & second_front
                & np.isfinite(first_uv).all(axis=1)
                & np.isfinite(second_uv).all(axis=1)
            )
            errors = np.linalg.norm(first_uv[valid] - second_uv[valid], axis=1)
            if len(errors) == 0 or not np.isfinite(errors).all():
                raise ValueError(
                    "fold cameras do not share common valid projection support"
                )
            median_px = float(np.median(errors))
            p90_px = float(np.percentile(errors, 90.0))
            pairwise.append(
                {
                    "validation_fold_a": first_fold,
                    "validation_fold_b": second_fold,
                    "point_count": int(len(errors)),
                    "projection_median_px": median_px,
                    "projection_p90_px": p90_px,
                }
            )
            medoid_cost[first_fold] += median_px
            medoid_cost[second_fold] += median_px
    medoid_fold = min(medoid_cost, key=lambda fold: (medoid_cost[fold], fold))
    pairwise_medians = [row["projection_median_px"] for row in pairwise]
    return {
        "policy": "fold_camera_medoid_by_pairwise_bank_projection_median",
        "successful_fold_count": int(len(candidates)),
        "medoid_validation_fold": int(medoid_fold),
        "medoid_projection_cost_px": float(medoid_cost[medoid_fold]),
        "max_pairwise_projection_median_px": float(max(pairwise_medians)),
        "pairwise_projection_disagreement": pairwise,
    }


def predict_focus_center_from_rig_offsets(
    *,
    anchor_cameras: Mapping[str, CameraParams],
    current_cameras: Mapping[str, CameraParams],
    source_views: Sequence[str],
    focus_view: str,
) -> np.ndarray:
    """Predict one rig center from anchor-relative offsets of the other views."""

    focus = str(focus_view)
    if focus not in anchor_cameras:
        raise KeyError(f"focus view {focus!r} is absent from anchor cameras")
    rig_rotations = []
    normalized_sources = []
    for source_view in source_views:
        source = str(source_view)
        if source == focus:
            continue
        if source not in anchor_cameras or source not in current_cameras:
            raise KeyError(f"source view {source!r} is absent from rig cameras")
        rotation = (
            np.asarray(current_cameras[source].R, dtype=np.float64).T
            @ np.asarray(anchor_cameras[source].R, dtype=np.float64)
        )
        if not np.isfinite(rotation).all():
            raise ValueError("rig rotations must be finite")
        rig_rotations.append(rotation)
        normalized_sources.append(source)
    if not rig_rotations:
        raise ValueError("at least one non-focus source view is required")

    rotation_costs = []
    for candidate in rig_rotations:
        cost = 0.0
        for other in rig_rotations:
            delta = candidate @ other.T
            cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
            cost += float(np.arccos(cosine))
        rotation_costs.append(cost)
    rig_rotation = rig_rotations[int(np.argmin(rotation_costs))]

    predictions = []
    for source in normalized_sources:
        anchor_offset = (
            np.asarray(anchor_cameras[focus].camera_center, dtype=np.float64)
            - np.asarray(anchor_cameras[source].camera_center, dtype=np.float64)
        )
        predictions.append(
            np.asarray(current_cameras[source].camera_center, dtype=np.float64)
            + rig_rotation @ anchor_offset
        )
    values = np.stack(predictions)
    if not np.isfinite(values).all():
        raise ValueError("rig center predictions must be finite")
    return np.median(values, axis=0)


def solve_focus_track_pnp(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_view: str,
    camera_template: CameraParams,
    reprojection_error_px: float = 4.0,
    min_inliers: int = 12,
    iterations_count: int = 1000,
    confidence: float = 0.999,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Estimate one focus camera from a fixed static-context track point bank."""

    views = [str(view) for view in view_ids]
    focus = str(focus_view)
    if focus not in views:
        raise KeyError(f"focus view {focus!r} is absent from track bundle")
    focus_index = views.index(focus)
    tracks = np.asarray(tracks_frame, dtype=np.float64)
    visibility = np.asarray(visibility_frame, dtype=bool)
    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    xyz = np.asarray(point_bank.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    if len(point_ids) != len(xyz):
        raise ValueError("point bank point_ids and xyz have different lengths")

    object_points = []
    image_points = []
    paired_ids = []
    for point_id, point in zip(point_ids.tolist(), xyz):
        if point_id < 0 or point_id >= tracks.shape[1]:
            continue
        if not bool(visibility[focus_index, point_id]):
            continue
        uv = np.asarray(tracks[focus_index, point_id], dtype=np.float64).reshape(2)
        if not np.isfinite(point).all() or not np.isfinite(uv).all():
            continue
        object_points.append(point)
        image_points.append(uv)
        paired_ids.append(int(point_id))

    object_array = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_array = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    minimum = max(4, int(min_inliers))
    if len(object_array) < minimum:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "paired_count": int(len(object_array)),
            "inlier_count": 0,
        }

    cv2.setRNGSeed(int(rng_seed))
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_array,
        image_array,
        np.asarray(camera_template.K, dtype=np.float64),
        np.asarray(camera_template.D, dtype=np.float64).reshape(-1),
        iterationsCount=int(iterations_count),
        reprojectionError=float(reprojection_error_px),
        confidence=float(confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_indices = (
        np.asarray(inliers, dtype=np.int64).reshape(-1)
        if success and inliers is not None
        else np.empty((0,), dtype=np.int64)
    )
    if not success or len(inlier_indices) < minimum:
        return {
            "success": False,
            "reason": "pnp_ransac_failed_or_too_few_inliers",
            "paired_count": int(len(object_array)),
            "inlier_count": int(len(inlier_indices)),
        }

    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_array[inlier_indices],
        image_array[inlier_indices],
        np.asarray(camera_template.K, dtype=np.float64),
        np.asarray(camera_template.D, dtype=np.float64).reshape(-1),
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    candidate = replace(
        camera_template,
        R=np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        T=translation,
    )
    projected, _ = cv2.projectPoints(
        object_array,
        rotation_vector,
        translation,
        np.asarray(camera_template.K, dtype=np.float64),
        np.asarray(camera_template.D, dtype=np.float64).reshape(-1),
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_array, axis=1)
    return {
        "success": True,
        "reason": "ok",
        "camera": candidate,
        "paired_count": int(len(object_array)),
        "inlier_count": int(len(inlier_indices)),
        "inlier_ratio": float(len(inlier_indices) / len(object_array)),
        "reprojection_mean_px": float(np.mean(errors)),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_p90_px": float(np.percentile(errors, 90.0)),
        "point_ids": np.asarray(paired_ids, dtype=np.int64),
        "inlier_indices": inlier_indices,
    }


def solve_focus_track_rotation(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_view: str,
    camera_template: CameraParams,
    fixed_center: np.ndarray,
    reprojection_error_px: float = 4.0,
    min_inliers: int = 12,
    max_evaluations: int = 100,
    ransac_prefilter: bool = False,
    direction_ransac_prefilter: bool = False,
    iterations_count: int = 1000,
    confidence: float = 0.999,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Refine only track-supported rotation around an independently predicted center."""

    views = [str(view) for view in view_ids]
    focus = str(focus_view)
    if focus not in views:
        raise KeyError(f"focus view {focus!r} is absent from track bundle")
    focus_index = views.index(focus)
    tracks = np.asarray(tracks_frame, dtype=np.float64)
    visibility = np.asarray(visibility_frame, dtype=bool)
    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    xyz = np.asarray(point_bank.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    if len(point_ids) != len(xyz):
        raise ValueError("point bank point_ids and xyz have different lengths")

    valid = (
        (point_ids >= 0)
        & (point_ids < tracks.shape[1])
        & np.isfinite(xyz).all(axis=1)
    )
    safe_ids = point_ids[valid]
    safe_xyz = xyz[valid]
    visible = visibility[focus_index, safe_ids]
    image_points = np.asarray(tracks[focus_index, safe_ids], dtype=np.float64)
    finite = visible & np.isfinite(image_points).all(axis=1)
    paired_ids = safe_ids[finite]
    object_points = safe_xyz[finite]
    image_points = image_points[finite]
    minimum = max(4, int(min_inliers))
    if len(object_points) < minimum:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "paired_count": int(len(object_points)),
            "inlier_count": 0,
        }

    center = np.asarray(fixed_center, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all():
        raise ValueError("fixed_center must be finite")

    if bool(ransac_prefilter) and bool(direction_ransac_prefilter):
        raise ValueError("select only one fixed-center RANSAC prefilter")

    prefilter_indices = np.arange(len(object_points), dtype=np.int64)
    prefilter_model = "none"
    optimization_base_rotation = np.asarray(
        camera_template.R, dtype=np.float64
    ).reshape(3, 3)
    if bool(direction_ransac_prefilter):
        normalized = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2),
            np.asarray(camera_template.K, dtype=np.float64),
            np.asarray(camera_template.D, dtype=np.float64).reshape(-1),
        ).reshape(-1, 2)
        camera_rays = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        camera_rays /= np.linalg.norm(camera_rays, axis=1, keepdims=True)
        world_rays = object_points - center.reshape(1, 3)
        world_norms = np.linalg.norm(world_rays, axis=1)
        valid_rays = np.isfinite(world_rays).all(axis=1) & (world_norms > 1e-10)
        world_rays[valid_rays] /= world_norms[valid_rays, None]

        def fit_direction_rotation(indices: np.ndarray) -> np.ndarray | None:
            selected_world = world_rays[indices]
            selected_camera = camera_rays[indices]
            if (
                len(indices) < 3
                or not np.isfinite(selected_world).all()
                or not np.isfinite(selected_camera).all()
            ):
                return None
            covariance = selected_world.T @ selected_camera
            left, _singular, right_t = np.linalg.svd(covariance)
            correction = np.eye(3, dtype=np.float64)
            correction[-1, -1] = np.sign(
                np.linalg.det(right_t.T @ left.T)
            )
            rotation = right_t.T @ correction @ left.T
            if not np.isfinite(rotation).all() or np.linalg.det(rotation) < 0.0:
                return None
            return rotation

        def direction_errors(rotation: np.ndarray) -> np.ndarray:
            translation = -rotation @ center
            rotation_vector, _ = cv2.Rodrigues(rotation)
            projected, _ = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation,
                np.asarray(camera_template.K, dtype=np.float64),
                np.asarray(camera_template.D, dtype=np.float64).reshape(-1),
            )
            camera_xyz = (rotation @ object_points.T).T + translation.reshape(1, 3)
            errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            errors[camera_xyz[:, 2] <= 1e-8] = np.inf
            return errors

        rng = np.random.default_rng(int(rng_seed))
        valid_indices = np.flatnonzero(valid_rays)
        best_rotation: np.ndarray | None = None
        best_inliers = np.empty((0,), dtype=np.int64)
        best_score: tuple[int, float] | None = None
        if len(valid_indices) >= 3:
            for _iteration in range(max(1, int(iterations_count))):
                sample = rng.choice(valid_indices, size=3, replace=False)
                rotation = fit_direction_rotation(sample)
                if rotation is None:
                    continue
                errors = direction_errors(rotation)
                inliers = np.flatnonzero(
                    np.isfinite(errors)
                    & (errors <= float(reprojection_error_px))
                ).astype(np.int64)
                if len(inliers) < 3:
                    continue
                score = (int(len(inliers)), -float(np.median(errors[inliers])))
                if best_score is None or score > best_score:
                    best_score = score
                    best_rotation = rotation
                    best_inliers = inliers
        if best_rotation is None or len(best_inliers) < minimum:
            return {
                "success": False,
                "reason": "direction_rotation_ransac_failed_or_too_few_inliers",
                "paired_count": int(len(object_points)),
                "inlier_count": int(len(best_inliers)),
                "ransac_prefilter_inlier_count": int(len(best_inliers)),
                "ransac_prefilter_model": "fixed_center_direction_rotation",
            }
        for _iteration in range(2):
            refined = fit_direction_rotation(best_inliers)
            if refined is None:
                break
            errors = direction_errors(refined)
            refined_inliers = np.flatnonzero(
                np.isfinite(errors)
                & (errors <= float(reprojection_error_px))
            ).astype(np.int64)
            if len(refined_inliers) < minimum:
                break
            best_rotation = refined
            best_inliers = refined_inliers
        prefilter_indices = best_inliers
        optimization_base_rotation = best_rotation
        prefilter_model = "fixed_center_direction_rotation"
    elif bool(ransac_prefilter):
        cv2.setRNGSeed(int(rng_seed))
        success, _rotation_vector, _translation, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            np.asarray(camera_template.K, dtype=np.float64),
            np.asarray(camera_template.D, dtype=np.float64).reshape(-1),
            iterationsCount=int(iterations_count),
            reprojectionError=float(reprojection_error_px),
            confidence=float(confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
        prefilter_indices = (
            np.asarray(inliers, dtype=np.int64).reshape(-1)
            if success and inliers is not None
            else np.empty((0,), dtype=np.int64)
        )
        if not success or len(prefilter_indices) < minimum:
            return {
                "success": False,
                "reason": "rotation_ransac_prefilter_failed_or_too_few_inliers",
                "paired_count": int(len(object_points)),
                "inlier_count": int(len(prefilter_indices)),
                "ransac_prefilter_inlier_count": int(len(prefilter_indices)),
                "ransac_prefilter_model": "free_6dof_pnp",
            }
        prefilter_model = "free_6dof_pnp"
    fit_object_points = object_points[prefilter_indices]
    fit_image_points = image_points[prefilter_indices]

    def project(delta_vector: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        correction, _ = cv2.Rodrigues(
            np.asarray(delta_vector, dtype=np.float64).reshape(3, 1)
        )
        rotation = optimization_base_rotation @ correction
        translation = -rotation @ center
        rotation_vector, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            xyz,
            rotation_vector,
            translation,
            np.asarray(camera_template.K, dtype=np.float64),
            np.asarray(camera_template.D, dtype=np.float64).reshape(-1),
        )
        return projected.reshape(-1, 2)

    def residuals(delta_vector: np.ndarray) -> np.ndarray:
        return (
            project(delta_vector, fit_object_points) - fit_image_points
        ).reshape(-1)

    solution = least_squares(
        residuals,
        np.zeros(3, dtype=np.float64),
        loss="huber",
        f_scale=float(reprojection_error_px),
        max_nfev=int(max_evaluations),
        x_scale="jac",
    )
    errors = np.linalg.norm(
        project(solution.x, object_points) - image_points,
        axis=1,
    )
    inlier_indices = np.flatnonzero(errors <= float(reprojection_error_px)).astype(np.int64)
    if not bool(solution.success) or len(inlier_indices) < minimum:
        return {
            "success": False,
            "reason": "rotation_refine_failed_or_too_few_inliers",
            "paired_count": int(len(object_points)),
            "inlier_count": int(len(inlier_indices)),
        }
    correction, _ = cv2.Rodrigues(np.asarray(solution.x, dtype=np.float64).reshape(3, 1))
    rotation = optimization_base_rotation @ correction
    candidate = replace(camera_template, R=rotation, T=-rotation @ center)
    return {
        "success": True,
        "reason": "ok",
        "camera": candidate,
        "paired_count": int(len(object_points)),
        "inlier_count": int(len(inlier_indices)),
        "inlier_ratio": float(len(inlier_indices) / len(object_points)),
        "reprojection_mean_px": float(np.mean(errors)),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_p90_px": float(np.percentile(errors, 90.0)),
        "point_ids": paired_ids,
        "inlier_indices": inlier_indices,
        "rotation_correction_deg": float(np.linalg.norm(solution.x) * 180.0 / np.pi),
        "optimizer_evaluations": int(solution.nfev),
        "ransac_prefilter_enabled": bool(
            ransac_prefilter or direction_ransac_prefilter
        ),
        "ransac_prefilter_inlier_count": int(len(prefilter_indices)),
        "ransac_prefilter_model": prefilter_model,
    }


def crossfit_focus_track_pnp(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_view: str,
    base_camera: CameraParams,
    camera_template: CameraParams,
    verifier_config: CrossViewTrackVerifierConfig,
    reprojection_error_px: float = 4.0,
    min_inliers: int = 12,
    num_folds: int = 3,
    iterations_count: int = 1000,
    confidence: float = 0.999,
    rng_seed: int = 0,
    fixed_center: np.ndarray | None = None,
    fixed_center_ransac_prefilter: bool = False,
    fixed_center_direction_ransac_prefilter: bool = False,
) -> dict[str, Any]:
    """Cross-fit track PnP and score only on observations excluded from successful fits."""

    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    full_minimum = max(4, int(min_inliers))
    fit_minimum = max(4, int(math.ceil(full_minimum * (folds - 1) / folds)))
    required_successful_folds = int(folds // 2 + 1)
    fold_rows: list[dict[str, Any]] = []
    heldout_comparisons: list[dict[str, Any]] = []
    successful_candidates: list[dict[str, Any]] = []

    def solve(point_subset: Mapping[str, Any], *, minimum: int, seed: int) -> dict[str, Any]:
        if fixed_center is not None:
            return solve_focus_track_rotation(
                point_bank=point_subset,
                tracks_frame=tracks_frame,
                visibility_frame=visibility_frame,
                view_ids=view_ids,
                focus_view=focus_view,
                camera_template=camera_template,
                fixed_center=fixed_center,
                reprojection_error_px=reprojection_error_px,
                min_inliers=minimum,
                ransac_prefilter=bool(fixed_center_ransac_prefilter),
                direction_ransac_prefilter=bool(
                    fixed_center_direction_ransac_prefilter
                ),
                iterations_count=iterations_count,
                confidence=confidence,
                rng_seed=seed,
            )
        return solve_focus_track_pnp(
            point_bank=point_subset,
            tracks_frame=tracks_frame,
            visibility_frame=visibility_frame,
            view_ids=view_ids,
            focus_view=focus_view,
            camera_template=camera_template,
            reprojection_error_px=reprojection_error_px,
            min_inliers=minimum,
            iterations_count=iterations_count,
            confidence=confidence,
            rng_seed=seed,
        )

    for validation_fold in range(folds):
        split = split_point_bank_for_validation(
            point_bank,
            num_folds=folds,
            validation_fold=validation_fold,
        )
        fit_result = solve(
            split["fit"],
            minimum=fit_minimum,
            seed=int(rng_seed) + validation_fold,
        )
        fold_row = {
            "validation_fold": int(validation_fold),
            "fit_point_count": int(split["fit_count"]),
            "validation_point_count": int(split["validation_count"]),
            "fit_min_inliers": int(fit_minimum),
            "fit_success": bool(fit_result["success"]),
            "fit_inlier_count": int(fit_result["inlier_count"]),
            "fit_reason": str(fit_result["reason"]),
        }
        fold_rows.append(fold_row)
        if not fit_result["success"]:
            fold_row["validation_paired_count"] = 0
            fold_row["validation_success"] = False
            continue
        heldout = compare_focus_cameras(
            point_bank=split["validation"],
            tracks_frame=tracks_frame,
            visibility_frame=visibility_frame,
            view_ids=view_ids,
            focus_view=focus_view,
            base_camera=base_camera,
            candidate_camera=fit_result["camera"],
            config=verifier_config,
        )
        fold_row["validation_paired_count"] = int(heldout["paired_count"])
        if int(heldout["paired_count"]) == 0:
            fold_row["validation_success"] = False
            continue
        fold_row["validation_success"] = True
        heldout_comparisons.append(heldout)
        successful_candidates.append(
            {
                "validation_fold": int(validation_fold),
                "fit_result": fit_result,
                "heldout_comparison": heldout,
            }
        )

    successful_fold_count = int(len(heldout_comparisons))
    if successful_fold_count < required_successful_folds:
        return {
            "success": False,
            "reason": "crossfit_insufficient_successful_folds",
            "folds": fold_rows,
            "required_successful_fold_count": required_successful_folds,
            "successful_fold_count": successful_fold_count,
            "full_refit": {
                "success": False,
                "reason": "not_run",
                "inlier_count": 0,
            },
        }

    aggregate_oof_comparison = combine_focus_camera_comparisons(
        heldout_comparisons,
        config=verifier_config,
    )
    point_ids = np.asarray(aggregate_oof_comparison["point_ids"], dtype=np.int64)
    if len(np.unique(point_ids)) != len(point_ids):
        raise RuntimeError("cross-fit validation folds are not disjoint")

    full_refit = solve(
        point_bank,
        minimum=full_minimum,
        seed=int(rng_seed) + folds,
    )
    try:
        camera_consistency = _summarize_crossfit_camera_consistency(
            successful_candidates=successful_candidates,
            points=np.asarray(point_bank.get("xyz", []), dtype=np.float64),
        )
    except ValueError as exc:
        return {
            "success": False,
            "reason": "crossfit_camera_consistency_unavailable",
            "camera_consistency_error": str(exc),
            "folds": fold_rows,
            "required_successful_fold_count": required_successful_folds,
            "successful_fold_count": successful_fold_count,
            "full_refit": full_refit,
        }
    chosen = next(
        row
        for row in successful_candidates
        if int(row["validation_fold"])
        == int(camera_consistency["medoid_validation_fold"])
    )
    commit_fit = chosen["fit_result"]
    comparison = chosen["heldout_comparison"]
    return {
        "success": True,
        "reason": "ok",
        "camera": commit_fit["camera"],
        "camera_source": "crossfit_projection_medoid_fold_fit",
        "commit_validation_fold": int(chosen["validation_fold"]),
        "folds": fold_rows,
        "required_successful_fold_count": required_successful_folds,
        "successful_fold_count": successful_fold_count,
        "crossfit_camera_consistency": camera_consistency,
        "comparison": comparison,
        "aggregate_oof_comparison": aggregate_oof_comparison,
        "full_refit": full_refit,
        "paired_count": int(commit_fit["paired_count"]),
        "inlier_count": int(commit_fit["inlier_count"]),
        "inlier_ratio": float(commit_fit["inlier_ratio"]),
        "reprojection_mean_px": float(commit_fit["reprojection_mean_px"]),
        "reprojection_median_px": float(commit_fit["reprojection_median_px"]),
        "reprojection_p90_px": float(commit_fit["reprojection_p90_px"]),
        "solver_model": (
            "rig_center_track_rotation" if fixed_center is not None else "full_track_pnp"
        ),
    }


def _subset_point_bank(
    point_bank: Mapping[str, Any], mask: np.ndarray
) -> dict[str, Any]:
    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if len(selected) != len(point_ids):
        raise ValueError("point-bank subset mask has the wrong length")
    output: dict[str, Any] = {}
    for key, value in point_bank.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == len(point_ids)
        ):
            output[key] = value[selected]
        else:
            output[key] = value
    output["num_points_valid"] = int(np.count_nonzero(selected))
    return output


def _score_absolute_focus_camera(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_view: str,
    camera: CameraParams,
    inlier_px: float,
) -> dict[str, Any]:
    views = [str(view) for view in view_ids]
    focus = str(focus_view)
    if focus not in views:
        raise KeyError(f"focus view {focus!r} is absent from track bundle")
    focus_index = views.index(focus)
    tracks = np.asarray(tracks_frame, dtype=np.float64)
    visibility = np.asarray(visibility_frame, dtype=bool)
    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    xyz = np.asarray(point_bank.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    if len(point_ids) != len(xyz):
        raise ValueError("point bank point_ids and xyz have different lengths")
    in_range = (point_ids >= 0) & (point_ids < tracks.shape[1])
    safe_ids = point_ids[in_range]
    safe_xyz = xyz[in_range]
    observed = np.asarray(tracks[focus_index, safe_ids], dtype=np.float64)
    projected, in_front = _project_camera_points(camera, safe_xyz)
    valid = (
        visibility[focus_index, safe_ids]
        & np.isfinite(safe_xyz).all(axis=1)
        & np.isfinite(observed).all(axis=1)
        & np.isfinite(projected).all(axis=1)
        & in_front
    )
    errors = np.linalg.norm(projected[valid] - observed[valid], axis=1)
    valid_ids = safe_ids[valid]
    if len(errors) == 0:
        return {
            "paired_count": 0,
            "inlier_count": 0,
            "point_ids": valid_ids,
            "errors_px": errors,
            "reprojection_mean_px": float("inf"),
            "reprojection_median_px": float("inf"),
            "reprojection_p90_px": float("inf"),
            "inlier_ratio": 0.0,
        }
    return {
        "paired_count": int(len(errors)),
        "inlier_count": int(np.count_nonzero(errors <= float(inlier_px))),
        "point_ids": valid_ids,
        "errors_px": errors,
        "reprojection_mean_px": float(np.mean(errors)),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_p90_px": float(np.percentile(errors, 90.0)),
        "inlier_ratio": float(np.mean(errors <= float(inlier_px))),
    }


def crossfit_missing_focus_track_pnp(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_view: str,
    camera_template: CameraParams,
    verifier_config: CrossViewTrackVerifierConfig,
    base_camera: CameraParams | None = None,
    reprojection_error_px: float = 4.0,
    min_inliers: int = 12,
    iterations_count: int = 1000,
    confidence: float = 0.999,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Recover a missing pose with seed-exclusive absolute track verification."""

    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    group_ids = np.asarray(
        point_bank.get("validation_group_ids", []), dtype=np.int64
    ).reshape(-1)
    if len(point_ids) == 0 or len(group_ids) != len(point_ids):
        raise ValueError(
            "missing-pose crossfit requires one validation_group_id per point"
        )
    groups = sorted(int(value) for value in np.unique(group_ids).tolist())
    if len(groups) < 2:
        raise ValueError("missing-pose crossfit requires at least two seed groups")
    required_successful_folds = int(len(groups) // 2 + 1)
    minimum = max(4, int(min_inliers))
    threshold = float(verifier_config.focus_inlier_px)
    fold_rows: list[dict[str, Any]] = []
    successful_candidates: list[dict[str, Any]] = []
    accepted_scores: list[dict[str, Any]] = []
    accepted_comparisons: list[dict[str, Any]] = []
    for fold_index, validation_group in enumerate(groups):
        validation_mask = group_ids == int(validation_group)
        fit_mask = ~validation_mask
        fit_bank = _subset_point_bank(point_bank, fit_mask)
        validation_bank = _subset_point_bank(point_bank, validation_mask)
        fit = solve_focus_track_pnp(
            point_bank=fit_bank,
            tracks_frame=tracks_frame,
            visibility_frame=visibility_frame,
            view_ids=view_ids,
            focus_view=focus_view,
            camera_template=camera_template,
            reprojection_error_px=float(reprojection_error_px),
            min_inliers=minimum,
            iterations_count=int(iterations_count),
            confidence=float(confidence),
            rng_seed=int(rng_seed) + fold_index,
        )
        score = (
            _score_absolute_focus_camera(
                point_bank=validation_bank,
                tracks_frame=tracks_frame,
                visibility_frame=visibility_frame,
                view_ids=view_ids,
                focus_view=focus_view,
                camera=fit["camera"],
                inlier_px=threshold,
            )
            if bool(fit.get("success", False))
            else {
                "paired_count": 0,
                "inlier_count": 0,
                "point_ids": np.empty((0,), dtype=np.int64),
                "errors_px": np.empty((0,), dtype=np.float64),
                "reprojection_mean_px": float("inf"),
                "reprojection_median_px": float("inf"),
                "reprojection_p90_px": float("inf"),
                "inlier_ratio": 0.0,
            }
        )
        validation_accepted = bool(
            fit.get("success", False)
            and int(score["inlier_count"]) >= minimum
        )
        comparison = (
            compare_focus_cameras(
                point_bank=validation_bank,
                tracks_frame=tracks_frame,
                visibility_frame=visibility_frame,
                view_ids=view_ids,
                focus_view=focus_view,
                base_camera=base_camera,
                candidate_camera=fit["camera"],
                config=verifier_config,
            )
            if base_camera is not None and bool(fit.get("success", False))
            else None
        )
        fold_row = {
            "validation_fold": int(fold_index),
            "validation_group": int(validation_group),
            "fit_point_count": int(np.count_nonzero(fit_mask)),
            "validation_point_count": int(np.count_nonzero(validation_mask)),
            "fit_point_ids": point_ids[fit_mask].tolist(),
            "validation_point_ids": point_ids[validation_mask].tolist(),
            "fit_validation_disjoint": bool(
                not set(point_ids[fit_mask].tolist())
                & set(point_ids[validation_mask].tolist())
            ),
            "fit_success": bool(fit.get("success", False)),
            "fit_reason": str(fit.get("reason", "unknown")),
            "fit_inlier_count": int(fit.get("inlier_count", 0)),
            "absolute_validation_paired_count": int(score["paired_count"]),
            "absolute_validation_inlier_count": int(score["inlier_count"]),
            "absolute_validation_reprojection_mean_px": float(
                score["reprojection_mean_px"]
            ),
            "absolute_validation_reprojection_median_px": float(
                score["reprojection_median_px"]
            ),
            "absolute_validation_reprojection_p90_px": float(
                score["reprojection_p90_px"]
            ),
            "absolute_validation_inlier_ratio": float(score["inlier_ratio"]),
            "absolute_validation_accepted": validation_accepted,
        }
        if comparison is not None:
            fold_row.update(
                {
                    "base_candidate_validation_paired_count": int(
                        comparison["paired_count"]
                    ),
                    "base_candidate_validation_gain_mean_px": float(
                        comparison["gain_mean_px"]
                    ),
                    "base_candidate_validation_gain_median_px": float(
                        comparison["gain_median_px"]
                    ),
                    "base_candidate_validation_gain_p90_px": float(
                        comparison["gain_p90_px"]
                    ),
                }
            )
        fold_rows.append(fold_row)
        if validation_accepted:
            successful_candidates.append(
                {
                    "validation_fold": int(fold_index),
                    "validation_group": int(validation_group),
                    "fit_result": fit,
                    "absolute_score": score,
                    "base_candidate_comparison": comparison,
                }
            )
            accepted_scores.append(score)
            if comparison is not None and int(comparison["paired_count"]) > 0:
                accepted_comparisons.append(comparison)

    successful_fold_count = len(successful_candidates)
    full_refit = solve_focus_track_pnp(
        point_bank=point_bank,
        tracks_frame=tracks_frame,
        visibility_frame=visibility_frame,
        view_ids=view_ids,
        focus_view=focus_view,
        camera_template=camera_template,
        reprojection_error_px=float(reprojection_error_px),
        min_inliers=minimum,
        iterations_count=int(iterations_count),
        confidence=float(confidence),
        rng_seed=int(rng_seed) + len(groups),
    )
    if successful_fold_count < required_successful_folds:
        return {
            "success": False,
            "reason": "crossfit_insufficient_absolute_validation_folds",
            "folds": fold_rows,
            "required_successful_fold_count": required_successful_folds,
            "successful_fold_count": successful_fold_count,
            "full_refit": full_refit,
        }
    try:
        consistency = _summarize_crossfit_camera_consistency(
            successful_candidates=successful_candidates,
            points=np.asarray(point_bank.get("xyz", []), dtype=np.float64),
        )
    except ValueError as exc:
        return {
            "success": False,
            "reason": "crossfit_camera_consistency_unavailable",
            "camera_consistency_error": str(exc),
            "folds": fold_rows,
            "required_successful_fold_count": required_successful_folds,
            "successful_fold_count": successful_fold_count,
            "full_refit": full_refit,
        }
    if float(consistency["max_pairwise_projection_median_px"]) > threshold + 1e-12:
        return {
            "success": False,
            "reason": "crossfit_camera_projection_disagreement",
            "folds": fold_rows,
            "required_successful_fold_count": required_successful_folds,
            "successful_fold_count": successful_fold_count,
            "crossfit_camera_consistency": consistency,
            "full_refit": full_refit,
        }

    chosen = next(
        row
        for row in successful_candidates
        if int(row["validation_fold"])
        == int(consistency["medoid_validation_fold"])
    )
    errors = np.concatenate(
        [np.asarray(score["errors_px"], dtype=np.float64) for score in accepted_scores]
    )
    validation_ids = np.concatenate(
        [np.asarray(score["point_ids"], dtype=np.int64) for score in accepted_scores]
    )
    if len(np.unique(validation_ids)) != len(validation_ids):
        raise RuntimeError("seed-exclusive validation point IDs overlap")
    aggregate = {
        "paired_count": int(len(errors)),
        "point_ids": validation_ids,
        "reprojection_mean_px": float(np.mean(errors)),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_p90_px": float(np.percentile(errors, 90.0)),
        "inlier_ratio": float(np.mean(errors <= threshold)),
    }
    aggregate_comparison = (
        combine_focus_camera_comparisons(
            accepted_comparisons,
            config=verifier_config,
        )
        if base_camera is not None
        and len(accepted_comparisons) == len(successful_candidates)
        else None
    )
    commit_fit = chosen["fit_result"]
    return {
        "success": True,
        "reason": "seed_exclusive_absolute_geometry_verified",
        "camera": commit_fit["camera"],
        "camera_source": "seed_crossfit_projection_medoid_fold_fit",
        "commit_validation_fold": int(chosen["validation_fold"]),
        "commit_validation_group": int(chosen["validation_group"]),
        "folds": fold_rows,
        "required_successful_fold_count": required_successful_folds,
        "successful_fold_count": successful_fold_count,
        "crossfit_camera_consistency": consistency,
        "aggregate_oof_absolute_score": aggregate,
        "aggregate_oof_base_candidate_comparison": aggregate_comparison,
        "commit_fold_base_candidate_comparison": chosen[
            "base_candidate_comparison"
        ],
        "full_refit": full_refit,
        "paired_count": int(commit_fit["paired_count"]),
        "inlier_count": int(commit_fit["inlier_count"]),
        "inlier_ratio": float(commit_fit["inlier_ratio"]),
        "reprojection_mean_px": float(commit_fit["reprojection_mean_px"]),
        "reprojection_median_px": float(commit_fit["reprojection_median_px"]),
        "reprojection_p90_px": float(commit_fit["reprojection_p90_px"]),
        "solver_model": "full_track_pnp_missing_base",
    }
