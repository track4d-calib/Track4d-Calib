from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    solve_same_time_multiview_pnp,
)
from track4d_calib.hloc_temporal_track_repair import (
    solve_tracked_map_pnp,
    verify_cross_seed_candidate,
)
from track4d_calib.fixed_center_pnp import solve_fixed_center_pnp


def select_boundary_seed_frames(
    *,
    successful_frames: Sequence[int],
    band_frames: Sequence[int],
    max_seed_banks: int = 4,
) -> list[int]:
    """Select nearby successful frames while alternating temporal sides."""

    successful = sorted({int(frame) for frame in successful_frames})
    band = sorted({int(frame) for frame in band_frames})
    budget = int(max_seed_banks)
    if not band:
        raise ValueError("band_frames must not be empty")
    if budget < 2:
        raise ValueError("max_seed_banks must allow independent verification")
    if any(frame in set(successful) for frame in band):
        raise ValueError("band_frames must not be marked successful")

    left = [frame for frame in reversed(successful) if frame < band[0]]
    right = [frame for frame in successful if frame > band[-1]]
    selected: list[int] = []
    depth = 0
    while len(selected) < budget and (depth < len(left) or depth < len(right)):
        if depth < len(left):
            selected.append(left[depth])
            if len(selected) == budget:
                break
        if depth < len(right):
            selected.append(right[depth])
        depth += 1
    return selected[:budget]


def _solver_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "success",
        "reason",
        "triangulated_count",
        "candidate_target_observation_count",
        "support_observation_count",
        "support_observation_inlier_count",
        "support_observation_rejected_count",
        "inlier_count",
        "inlier_ratio",
        "reprojection_mean_px",
        "reprojection_median_px",
        "reprojection_p90_px",
    )
    return {key: result[key] for key in keys if key in result}


def _matches_expected(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return bool(
            isinstance(actual, (int, float, np.number))
            and math.isclose(float(actual), expected, rel_tol=1e-4, abs_tol=1e-3)
        )
    return actual == expected


def recover_static_pnp_inlier_seed_bank(
    *,
    seed_frame: int,
    target_view: str,
    support_cameras: Mapping[str, Any],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    target_template: Any,
    config: SameTimePnPConfig,
    expected_solver: Mapping[str, Any] | None = None,
    seed_point_policy: str = "pnp_inliers",
) -> dict[str, Any]:
    """Replay one successful Static+PnP node and expose its aligned inlier map."""

    result = solve_same_time_multiview_pnp(
        support_cameras=support_cameras,
        support_observations=support_observations,
        target_observations=target_observations,
        target_template=target_template,
        config=config,
    )
    if not bool(result.get("success")):
        raise ValueError(
            f"Static+PnP seed replay failed at {target_view}@{int(seed_frame)}: "
            f"{result.get('reason')}"
        )
    if expected_solver is not None:
        mismatches = {
            str(key): {"expected": value, "actual": result.get(key)}
            for key, value in expected_solver.items()
            if key not in result or not _matches_expected(result[key], value)
        }
        if mismatches:
            raise ValueError(f"Static+PnP seed replay mismatch: {mismatches}")

    policy = str(seed_point_policy)
    if policy not in {"pnp_inliers", "triangulated_static_candidates"}:
        raise ValueError(f"unsupported seed_point_policy: {policy}")
    point_ids = np.asarray(result["point_ids"], dtype=np.int64).reshape(-1)
    object_points = np.asarray(result["object_points"], dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(result["image_points"], dtype=np.float64).reshape(-1, 2)
    inlier_indices = np.asarray(result["inlier_indices"], dtype=np.int64).reshape(-1)
    if not (len(point_ids) == len(object_points) == len(image_points)):
        raise ValueError("Static+PnP seed replay returned misaligned geometry arrays")
    if np.any(inlier_indices < 0) or np.any(inlier_indices >= len(point_ids)):
        raise ValueError("Static+PnP seed replay returned invalid inlier indices")
    if not np.array_equal(
        point_ids[inlier_indices],
        np.asarray(result["inlier_point_ids"], dtype=np.int64).reshape(-1),
    ):
        raise ValueError("Static+PnP seed replay returned misaligned inlier point IDs")

    selected_indices = (
        inlier_indices
        if policy == "pnp_inliers"
        else np.arange(len(point_ids), dtype=np.int64)
    )
    return {
        "seed_frame": int(seed_frame),
        "target_view": str(target_view),
        "seed_point_policy": policy,
        "source_point_ids": point_ids[selected_indices],
        "points3d": object_points[selected_indices],
        "query_points": image_points[selected_indices],
        "camera": result["camera"],
        "solver": _solver_diagnostics(result),
    }


def _tracked_pnp_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "success",
        "reason",
        "point_count",
        "candidate_count",
        "inlier_count",
        "inlier_ratio",
        "reprojection_mean_px",
        "reprojection_median_px",
        "reprojection_p90_px",
        "inlier_reprojection_median_px",
        "fixed_center",
        "pixel_refinement_accepted",
    )
    return {key: result[key] for key in keys if key in result}


def _solve_track_pose(
    *,
    pose_model: str,
    points3d: np.ndarray,
    image_points: np.ndarray,
    visible: np.ndarray,
    camera_template: Any,
    fixed_center: np.ndarray | None,
    reprojection_error_px: float,
    min_inliers: int,
    iterations_count: int,
    rng_seed: int,
) -> dict[str, Any]:
    visible = np.asarray(visible, dtype=bool).reshape(-1)
    points = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    valid = visible & np.isfinite(points).all(axis=1) & np.isfinite(pixels).all(axis=1)
    if pose_model == "full_pnp":
        return solve_tracked_map_pnp(
            points3d=points,
            image_points=pixels,
            visible=valid,
            camera_template=camera_template,
            reprojection_error_px=float(reprojection_error_px),
            min_inliers=int(min_inliers),
            rng_seed=int(rng_seed),
        )
    if pose_model != "fixed_center_rotation":
        raise ValueError(f"unsupported track pose model: {pose_model}")
    if fixed_center is None:
        raise ValueError("fixed_center is required for fixed_center_rotation")
    result = solve_fixed_center_pnp(
        object_points=points[valid],
        image_points=pixels[valid],
        target_template=camera_template,
        fixed_center=np.asarray(fixed_center, dtype=np.float64),
        reprojection_error_px=float(reprojection_error_px),
        min_inliers=int(min_inliers),
        iterations_count=int(iterations_count),
        rng_seed=int(rng_seed),
    )
    return {**result, "point_count": int(np.count_nonzero(valid))}


def _normalized_track_bank(bank: Mapping[str, Any]) -> dict[str, Any]:
    point_ids = np.asarray(bank["global_point_ids"], dtype=np.int64).reshape(-1)
    points3d = np.asarray(bank["points3d"], dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(bank["image_points"], dtype=np.float64).reshape(-1, 2)
    visible = np.asarray(bank["visible"], dtype=bool).reshape(-1)
    if not (len(point_ids) == len(points3d) == len(image_points) == len(visible)):
        raise ValueError("track seed-bank arrays must have equal length")
    if len(np.unique(point_ids)) != len(point_ids):
        raise ValueError("global_point_ids must be unique within each seed bank")
    return {
        "seed_frame": int(bank["seed_frame"]),
        "global_point_ids": point_ids,
        "points3d": points3d,
        "image_points": image_points,
        "visible": visible,
    }


def fit_cross_seed_track_candidate(
    *,
    banks: Sequence[Mapping[str, Any]],
    base_camera: Any,
    camera_template: Any,
    reprojection_error_px: float,
    min_inliers: int,
    rng_seed: int = 0,
    pose_model: str = "full_pnp",
    fixed_center: np.ndarray | None = None,
    iterations_count: int = 1000,
) -> dict[str, Any]:
    """Fit tracking PnP and require improvement on every held-out seed bank."""

    if pose_model not in {"full_pnp", "fixed_center_rotation"}:
        raise ValueError(f"unsupported track pose model: {pose_model}")
    if pose_model == "fixed_center_rotation" and fixed_center is None:
        raise ValueError("fixed_center is required for fixed_center_rotation")
    if int(iterations_count) <= 0:
        raise ValueError("iterations_count must be positive")

    normalized = [_normalized_track_bank(bank) for bank in banks]
    if len(normalized) < 2:
        return {
            "success": False,
            "reason": "insufficient_independent_seed_banks",
            "folds": [],
        }
    all_ids = np.concatenate([bank["global_point_ids"] for bank in normalized])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("global_point_ids must be unique across seed banks")

    folds: list[dict[str, Any]] = []
    minimum_verify_points = max(6, int(min_inliers) // 2)
    for heldout_index, heldout in enumerate(normalized):
        heldout_valid = (
            heldout["visible"]
            & np.isfinite(heldout["points3d"]).all(axis=1)
            & np.isfinite(heldout["image_points"]).all(axis=1)
        )
        if int(np.count_nonzero(heldout_valid)) < minimum_verify_points:
            folds.append(
                {
                    "heldout_seed_frame": int(heldout["seed_frame"]),
                    "fit_seed_frames": [],
                    "status": "abstain",
                    "fit": {"success": False, "reason": "verification_bank_not_visible"},
                    "verification": {
                        "accepted": False,
                        "reason": "insufficient_independent_seed_points",
                        "min_points": minimum_verify_points,
                        "exclusive_point_count": int(np.count_nonzero(heldout_valid)),
                    },
                }
            )
            continue
        fit_banks = [
            bank for index, bank in enumerate(normalized) if index != heldout_index
        ]
        fit_ids = np.concatenate([bank["global_point_ids"] for bank in fit_banks])
        fit_points = np.concatenate([bank["points3d"] for bank in fit_banks], axis=0)
        fit_pixels = np.concatenate([bank["image_points"] for bank in fit_banks], axis=0)
        fit_visible = np.concatenate([bank["visible"] for bank in fit_banks])
        fit_result = _solve_track_pose(
            pose_model=pose_model,
            points3d=fit_points,
            image_points=fit_pixels,
            visible=fit_visible,
            camera_template=camera_template,
            fixed_center=fixed_center,
            reprojection_error_px=float(reprojection_error_px),
            min_inliers=int(min_inliers),
            iterations_count=int(iterations_count),
            rng_seed=int(rng_seed),
        )
        if bool(fit_result.get("success")):
            verification = verify_cross_seed_candidate(
                base_camera=base_camera,
                candidate_camera=fit_result["camera"],
                fit_point_ids=fit_ids,
                verify_point_ids=heldout["global_point_ids"],
                verify_points3d=heldout["points3d"],
                verify_image_points=heldout["image_points"],
                verify_visible=heldout["visible"],
                min_points=minimum_verify_points,
            )
        else:
            verification = {
                "accepted": False,
                "reason": "cross_seed_fit_failed",
                "min_points": minimum_verify_points,
            }
        folds.append(
            {
                "heldout_seed_frame": int(heldout["seed_frame"]),
                "fit_seed_frames": [int(bank["seed_frame"]) for bank in fit_banks],
                "status": "accept" if verification["accepted"] else "reject",
                "fit": _tracked_pnp_diagnostics(fit_result),
                "verification": verification,
            }
        )

    pooled_points = np.concatenate([bank["points3d"] for bank in normalized], axis=0)
    pooled_pixels = np.concatenate([bank["image_points"] for bank in normalized], axis=0)
    pooled_visible = np.concatenate([bank["visible"] for bank in normalized])
    final = _solve_track_pose(
        pose_model=pose_model,
        points3d=pooled_points,
        image_points=pooled_pixels,
        visible=pooled_visible,
        camera_template=camera_template,
        fixed_center=fixed_center,
        reprojection_error_px=float(reprojection_error_px),
        min_inliers=int(min_inliers),
        iterations_count=int(iterations_count),
        rng_seed=int(rng_seed),
    )
    eligible_fold_count = int(sum(row["status"] != "abstain" for row in folds))
    accepted_fold_count = int(sum(row["status"] == "accept" for row in folds))
    abstained_fold_count = int(sum(row["status"] == "abstain" for row in folds))
    consensus = bool(
        eligible_fold_count >= 2
        and accepted_fold_count > eligible_fold_count / 2.0
    )
    success = bool(consensus and final.get("success"))
    return {
        "success": success,
        "reason": (
            "ok"
            if success
            else "cross_seed_consensus_failed"
            if not consensus
            else "pooled_candidate_fit_failed"
        ),
        "seed_frames": [int(bank["seed_frame"]) for bank in normalized],
        "pose_model": pose_model,
        "eligible_fold_count": eligible_fold_count,
        "accepted_fold_count": accepted_fold_count,
        "abstained_fold_count": abstained_fold_count,
        "folds": folds,
        "final": _tracked_pnp_diagnostics(final),
        **({"camera": final["camera"]} if final.get("success") else {}),
    }
