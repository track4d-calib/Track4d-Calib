from __future__ import annotations

from dataclasses import replace
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from track4d_calib.cameras import CameraParams


def _fit_rotation(source_rays: np.ndarray, target_rays: np.ndarray) -> np.ndarray | None:
    source = np.asarray(source_rays, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_rays, dtype=np.float64).reshape(-1, 3)
    if len(source) < 3 or len(source) != len(target):
        return None
    covariance = source.T @ target
    left, singular, right_t = np.linalg.svd(covariance)
    if float(singular[1]) <= 1e-10:
        return None
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.sign(np.linalg.det(right_t.T @ left.T))
    rotation = right_t.T @ correction @ left.T
    if not np.isfinite(rotation).all() or np.linalg.det(rotation) <= 0.0:
        return None
    return rotation


def _fit_rotation_batch(
    source_rays: np.ndarray, target_rays: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit many three-ray Wahba hypotheses in one deterministic batch."""

    source = np.asarray(source_rays, dtype=np.float64)
    target = np.asarray(target_rays, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (3, 3) or target.shape != source.shape:
        raise ValueError("batched ray samples must have shape (H,3,3)")
    covariance = np.einsum("hni,hnj->hij", source, target)
    left, singular, right_t = np.linalg.svd(covariance)
    right = np.swapaxes(right_t, -1, -2)
    left_t = np.swapaxes(left, -1, -2)
    determinant = np.linalg.det(right @ left_t)
    correction = np.broadcast_to(np.eye(3, dtype=np.float64), covariance.shape).copy()
    correction[:, -1, -1] = np.where(determinant < 0.0, -1.0, 1.0)
    rotations = right @ correction @ left_t
    valid = (
        np.isfinite(rotations).all(axis=(1, 2))
        & np.isfinite(singular).all(axis=1)
        & (singular[:, 1] > 1e-10)
        & (np.linalg.det(rotations) > 0.0)
    )
    return rotations, valid


def _project_errors_batch(
    *,
    rotations: np.ndarray,
    centered_points: np.ndarray,
    image_points: np.ndarray,
    camera: CameraParams,
) -> np.ndarray:
    """Project fixed-center hypotheses with OpenCV's common 5-coefficient model."""

    rotation_array = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    points = np.asarray(centered_points, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    distortion = np.asarray(camera.D, dtype=np.float64).reshape(-1)
    if len(distortion) > 5 and np.any(np.abs(distortion[5:]) > 0.0):
        raise ValueError("batched fixed-center PnP supports up to five distortion terms")
    coefficients = np.zeros(5, dtype=np.float64)
    coefficients[: min(5, len(distortion))] = distortion[:5]
    k1, k2, p1, p2, k3 = coefficients

    camera_points = np.einsum("hij,nj->hni", rotation_array, points)
    depth = camera_points[..., 2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, 1.0)
    x = camera_points[..., 0] / safe_depth
    y = camera_points[..., 1] / safe_depth
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    distorted_y = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    intrinsic = np.asarray(camera.K, dtype=np.float64).reshape(3, 3)
    homogeneous = np.stack(
        [distorted_x, distorted_y, np.ones_like(distorted_x)], axis=-1
    )
    projected = np.einsum("ij,hnj->hni", intrinsic, homogeneous)
    projected = projected[..., :2] / projected[..., 2:3]
    errors = np.linalg.norm(projected - pixels[None], axis=-1)
    invalid = (
        ~np.isfinite(camera_points).all(axis=-1)
        | ~np.isfinite(projected).all(axis=-1)
        | (depth <= 1e-8)
    )
    errors[invalid] = np.inf
    return errors


def _refine_rotation_pixels(
    *,
    rotation: np.ndarray,
    centered_points: np.ndarray,
    image_points: np.ndarray,
    camera: CameraParams,
) -> np.ndarray | None:
    """Refine SO(3) in pixel space while keeping the camera center fixed."""

    points = np.asarray(centered_points, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or len(points) != len(pixels):
        return None
    initial = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0].reshape(3)

    def residuals(rotation_vector: np.ndarray) -> np.ndarray:
        candidate = cv2.Rodrigues(
            np.asarray(rotation_vector, dtype=np.float64).reshape(3)
        )[0]
        camera_points = (candidate @ points.T).T
        projected, _ = cv2.projectPoints(
            camera_points,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera.K,
            camera.D,
        )
        residual = projected.reshape(-1, 2) - pixels
        invalid = (
            ~np.isfinite(camera_points).all(axis=1)
            | (camera_points[:, 2] <= 1e-8)
            | ~np.isfinite(residual).all(axis=1)
        )
        residual[invalid] = 1e6
        return residual.reshape(-1)

    optimized = least_squares(residuals, initial, method="lm")
    if not bool(optimized.success) or not np.isfinite(optimized.x).all():
        return None
    refined = cv2.Rodrigues(np.asarray(optimized.x, dtype=np.float64))[0]
    if not np.isfinite(refined).all() or np.linalg.det(refined) <= 0.0:
        return None
    return refined


def solve_fixed_center_pnp(
    *,
    object_points: np.ndarray,
    image_points: np.ndarray,
    target_template: CameraParams,
    fixed_center: np.ndarray,
    reprojection_error_px: float,
    min_inliers: int,
    iterations_count: int,
    rng_seed: int,
) -> dict[str, Any]:
    """Estimate only camera rotation while holding its world center fixed."""

    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(fixed_center, dtype=np.float64).reshape(-1)
    threshold = float(reprojection_error_px)
    minimum = max(4, int(min_inliers))
    if len(points) != len(pixels):
        raise ValueError("object and image point counts differ")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("fixed_center must be a finite 3-vector")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("reprojection_error_px must be finite and positive")
    if int(iterations_count) <= 0:
        raise ValueError("iterations_count must be positive")

    world_rays = points - center[None]
    world_norms = np.linalg.norm(world_rays, axis=1)
    normalized_pixels = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), target_template.K, target_template.D
    ).reshape(-1, 2)
    camera_rays = np.column_stack(
        [normalized_pixels, np.ones(len(normalized_pixels), dtype=np.float64)]
    )
    camera_norms = np.linalg.norm(camera_rays, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(pixels).all(axis=1)
        & np.isfinite(world_rays).all(axis=1)
        & (world_norms > 1e-10)
        & np.isfinite(camera_rays).all(axis=1)
        & (camera_norms > 1e-10)
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < minimum:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "candidate_count": int(len(valid_indices)),
            "required_inliers": minimum,
        }
    points = points[valid]
    pixels = pixels[valid]
    world_rays = world_rays[valid] / world_norms[valid, None]
    camera_rays = camera_rays[valid] / camera_norms[valid, None]

    centered_points = points - center[None]

    def errors_for(rotation: np.ndarray) -> np.ndarray:
        return _project_errors_batch(
            rotations=np.asarray(rotation, dtype=np.float64).reshape(1, 3, 3),
            centered_points=centered_points,
            image_points=pixels,
            camera=target_template,
        )[0]

    def score(errors: np.ndarray) -> tuple[int, float, float]:
        inliers = errors <= threshold
        count = int(np.count_nonzero(inliers))
        if count == 0:
            return 0, float("-inf"), float("-inf")
        values = errors[inliers]
        return count, -float(np.median(values)), -float(np.mean(values))

    rng = np.random.default_rng(int(rng_seed))
    samples = np.stack(
        [
            rng.choice(len(points), size=3, replace=False)
            for _iteration in range(int(iterations_count))
        ]
    )
    rotations, valid_hypotheses = _fit_rotation_batch(
        world_rays[samples], camera_rays[samples]
    )
    hypothesis_errors = _project_errors_batch(
        rotations=rotations,
        centered_points=centered_points,
        image_points=pixels,
        camera=target_template,
    )
    hypothesis_errors[~valid_hypotheses] = np.inf
    inlier_mask = hypothesis_errors <= threshold
    counts = np.count_nonzero(inlier_mask, axis=1)
    ranked_errors = np.sort(
        np.where(inlier_mask, hypothesis_errors, np.inf), axis=1
    )
    row_indices = np.arange(len(ranked_errors))
    lower_indices = np.maximum((counts - 1) // 2, 0)
    upper_indices = np.maximum(counts // 2, 0)
    medians = 0.5 * (
        ranked_errors[row_indices, lower_indices]
        + ranked_errors[row_indices, upper_indices]
    )
    means = np.divide(
        np.sum(np.where(inlier_mask, hypothesis_errors, 0.0), axis=1),
        counts,
        out=np.full(len(counts), np.inf, dtype=np.float64),
        where=counts > 0,
    )
    ordering = np.lexsort((means, medians, -counts))
    best_index = int(ordering[0])
    best_rotation = rotations[best_index] if valid_hypotheses[best_index] else None
    best_errors = hypothesis_errors[best_index] if best_rotation is not None else None
    best_score = score(best_errors) if best_errors is not None else None
    if best_rotation is None or best_errors is None:
        return {
            "success": False,
            "reason": "rotation_ransac_failed",
            "candidate_count": int(len(points)),
            "required_inliers": minimum,
        }

    for _iteration in range(3):
        inliers = np.flatnonzero(best_errors <= threshold)
        refined = _fit_rotation(world_rays[inliers], camera_rays[inliers])
        if refined is None:
            break
        refined_errors = errors_for(refined)
        refined_score = score(refined_errors)
        if refined_score < best_score:
            break
        best_rotation = refined
        best_errors = refined_errors
        best_score = refined_score

    inliers = np.flatnonzero(best_errors <= threshold)
    pixel_refinement_accepted = False
    pixel_refined = _refine_rotation_pixels(
        rotation=best_rotation,
        centered_points=centered_points[inliers],
        image_points=pixels[inliers],
        camera=target_template,
    )
    if pixel_refined is not None:
        pixel_errors = errors_for(pixel_refined)
        pixel_score = score(pixel_errors)
        if pixel_score >= best_score:
            best_rotation = pixel_refined
            best_errors = pixel_errors
            best_score = pixel_score
            pixel_refinement_accepted = True

    inliers = np.flatnonzero(best_errors <= threshold)
    if len(inliers) < minimum:
        return {
            "success": False,
            "reason": "insufficient_inliers",
            "candidate_count": int(len(points)),
            "inlier_count": int(len(inliers)),
            "required_inliers": minimum,
            "reprojection_median_px": float(np.median(best_errors)),
        }
    camera = replace(
        target_template,
        R=best_rotation,
        T=-best_rotation @ center,
    )
    finite = np.isfinite(best_errors)
    return {
        "success": True,
        "reason": "ok",
        "camera": camera,
        "candidate_count": int(len(points)),
        "inlier_count": int(len(inliers)),
        "inlier_ratio": float(len(inliers) / len(points)),
        "inlier_indices": valid_indices[inliers],
        "reprojection_median_px": float(np.median(best_errors[finite])),
        "reprojection_p90_px": float(np.percentile(best_errors[finite], 90.0)),
        "inlier_reprojection_median_px": float(np.median(best_errors[inliers])),
        "fixed_center": center.copy(),
        "pixel_refinement": "least_squares_lm_on_ransac_inliers",
        "pixel_refinement_accepted": pixel_refinement_accepted,
    }


def _bounded_center_delta(raw: np.ndarray, max_distance: float) -> np.ndarray:
    """Map an unconstrained 3-vector into a Euclidean center trust region."""

    value = np.asarray(raw, dtype=np.float64).reshape(3)
    radius = float(max_distance)
    if not np.isfinite(value).all() or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("center delta and max distance must be finite and positive")
    magnitude = float(np.linalg.norm(value))
    if magnitude <= np.finfo(np.float64).eps:
        return value.copy()
    return value * (radius * np.tanh(magnitude / radius) / magnitude)


def solve_near_fixed_center_pnp(
    *,
    object_points: np.ndarray,
    image_points: np.ndarray,
    target_template: CameraParams,
    fixed_center: np.ndarray,
    center_radius_m: float,
    reprojection_error_px: float,
    min_inliers: int,
    iterations_count: int,
    rng_seed: int,
    max_nfev: int = 100,
) -> dict[str, Any]:
    """Fit rotation plus a centimetre-scale center correction.

    A fixed-center RANSAC supplies the rotation initialization.  The final
    least-squares solve can move the center only inside a Euclidean ball around
    ``fixed_center``; this prevents the free-PnP translation from compensating
    for a bad rotation while still allowing a physically plausible tiny rig
    adjustment.  The center bound is enforced by a smooth tanh map rather than
    clipping individual coordinates.
    """

    radius = float(center_radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("center_radius_m must be finite and positive")
    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(fixed_center, dtype=np.float64).reshape(-1)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("fixed_center must be a finite 3-vector")
    if len(points) != len(pixels):
        raise ValueError("object and image point counts differ")

    fixed = solve_fixed_center_pnp(
        object_points=points,
        image_points=pixels,
        target_template=target_template,
        fixed_center=center,
        reprojection_error_px=float(reprojection_error_px),
        min_inliers=int(min_inliers),
        iterations_count=int(iterations_count),
        rng_seed=int(rng_seed),
    )
    if not bool(fixed.get("success", False)):
        return {
            **fixed,
            "reason": "near_fixed_center_initial_rotation_failed",
            "center_radius_m": radius,
        }

    # Optimize on the RANSAC inlier subset.  Running the local solve on all
    # correspondences lets the many outliers in these sparse folds dominate
    # the tiny center derivative and frequently exhaust the trust-region
    # iteration budget before making a useful update.
    inlier_indices = np.asarray(
        fixed.get("inlier_indices", np.arange(len(points), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
    inlier_indices = inlier_indices[
        (inlier_indices >= 0) & (inlier_indices < len(points))
    ]
    if len(inlier_indices) < max(4, int(min_inliers)):
        return {
            **fixed,
            "success": False,
            "reason": "near_fixed_center_insufficient_initial_inliers",
            "center_radius_m": radius,
        }
    fit_points = points[inlier_indices]
    fit_pixels = pixels[inlier_indices]

    initial_rotation = np.asarray(fixed["camera"].R, dtype=np.float64)
    initial_delta = np.zeros(3, dtype=np.float64)
    initial_rotation_vector = cv2.Rodrigues(initial_rotation)[0].reshape(3)
    threshold = float(reprojection_error_px)

    def materialize(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_rotation = cv2.Rodrigues(
            np.asarray(parameters[:3], dtype=np.float64).reshape(3)
        )[0]
        rotation = delta_rotation @ initial_rotation
        delta_center = _bounded_center_delta(parameters[3:6], radius)
        return rotation, center + delta_center

    def residuals(parameters: np.ndarray) -> np.ndarray:
        rotation, current_center = materialize(parameters)
        camera_points = (rotation @ (fit_points - current_center[None]).T).T
        projected, _ = cv2.projectPoints(
            camera_points,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.asarray(target_template.K, dtype=np.float64),
            np.asarray(target_template.D, dtype=np.float64).reshape(-1),
        )
        delta = projected.reshape(-1, 2) - fit_pixels
        invalid = (
            ~np.isfinite(camera_points).all(axis=1)
            | (camera_points[:, 2] <= 1e-8)
            | ~np.isfinite(delta).all(axis=1)
        )
        delta[invalid] = threshold
        norms = np.linalg.norm(delta, axis=1)
        weights = np.sqrt(np.minimum(1.0, threshold / np.maximum(norms, 1e-12)))
        weights[invalid] = 1.0
        return (delta * weights[:, None]).reshape(-1)

    initial = np.concatenate([initial_delta, np.zeros(3, dtype=np.float64)])
    try:
        optimized = least_squares(
            residuals,
            initial,
            loss="huber",
            f_scale=threshold,
            method="trf",
            max_nfev=max(int(max_nfev), 250),
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
    except Exception as error:  # pragma: no cover - defensive runtime guard
        return {
            **fixed,
            "success": False,
            "reason": "near_fixed_center_optimization_exception",
            "error": repr(error),
            "center_radius_m": radius,
        }
    if not np.isfinite(optimized.x).all():
        return {
            **fixed,
            "success": False,
            "reason": "near_fixed_center_optimization_failed",
            "center_radius_m": radius,
            "optimizer_message": str(optimized.message),
        }

    rotation, current_center = materialize(np.asarray(optimized.x, dtype=np.float64))
    camera = replace(
        target_template,
        R=np.asarray(rotation, dtype=np.float64),
        T=-np.asarray(rotation, dtype=np.float64) @ current_center,
    )
    errors = _project_errors_batch(
        rotations=np.asarray(rotation, dtype=np.float64).reshape(1, 3, 3),
        centered_points=fit_points - current_center[None],
        image_points=fit_pixels,
        camera=target_template,
    )[0]
    finite = np.isfinite(errors)
    inliers = finite & (errors <= threshold)
    if int(np.count_nonzero(inliers)) < max(4, int(min_inliers)):
        return {
            **fixed,
            "success": False,
            "reason": "near_fixed_center_insufficient_inliers",
            "center_radius_m": radius,
            "candidate_inlier_count": int(np.count_nonzero(inliers)),
        }
    return {
        "success": True,
        "reason": "ok",
        "camera": camera,
        "candidate_count": int(len(points)),
        "inlier_count": int(np.count_nonzero(inliers)),
        "inlier_ratio": float(np.count_nonzero(inliers) / len(points)),
        "reprojection_median_px": float(np.median(errors[finite])) if np.any(finite) else None,
        "reprojection_p90_px": float(np.percentile(errors[finite], 90.0)) if np.any(finite) else None,
        "inlier_reprojection_median_px": float(np.median(errors[inliers])),
        "fixed_center": center.copy(),
        "optimized_center": current_center.copy(),
        "center_delta_m": float(np.linalg.norm(current_center - center)),
        "center_radius_m": radius,
        "rotation_delta_deg_from_fixed": float(
            np.degrees(
                np.linalg.norm(
                    cv2.Rodrigues(np.asarray(rotation) @ initial_rotation.T)[0]
                )
            )
        ),
        "optimizer_nfev": int(optimized.nfev),
        "optimizer_success": bool(optimized.success),
        "optimizer_cost": float(optimized.cost),
        "initial_rotation_vector": initial_rotation_vector,
    }
