from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import t as student_t_distribution

from track4d_calib.cameras import CameraParams
from track4d_calib.track_crossview_verify import stable_point_id_validation_groups
from track4d_calib.track_pnp_repair import (
    solve_focus_track_pnp,
    solve_focus_track_rotation,
)


def _geometric_median(points: np.ndarray, *, max_iterations: int = 64) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("points must have shape (N, 3) with N > 0")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")
    estimate = np.median(values, axis=0)
    for _iteration in range(int(max_iterations)):
        distances = np.linalg.norm(values - estimate.reshape(1, 3), axis=1)
        coincident = distances < 1e-12
        if np.any(coincident):
            return values[np.flatnonzero(coincident)[0]].copy()
        weights = 1.0 / distances
        updated = np.sum(values * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(updated - estimate) < 1e-12:
            return updated
        estimate = updated
    return estimate


def _prepare_observations(
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[int] = set()
    for payload in observations:
        frame = int(payload["frame"])
        if frame in seen:
            raise ValueError(f"duplicate frame observation: {frame}")
        seen.add(frame)
        if frame not in initial_cameras:
            raise ValueError(f"missing initial camera for frame {frame}")
        xyz = np.asarray(payload["xyz"], dtype=np.float64)
        image_points = np.asarray(payload["image_points"], dtype=np.float64)
        point_ids = np.asarray(payload["point_ids"], dtype=np.int64).reshape(-1)
        validation_group_ids = np.asarray(
            payload.get("validation_group_ids", point_ids), dtype=np.int64
        ).reshape(-1)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        if image_points.shape != (len(xyz), 2):
            raise ValueError("image_points must have shape (N, 2)")
        if len(point_ids) != len(xyz):
            raise ValueError("point_ids and xyz must have equal lengths")
        if len(validation_group_ids) != len(xyz):
            raise ValueError("validation_group_ids and xyz must have equal lengths")
        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(image_points).all(axis=1)
        prepared.append(
            {
                "frame": frame,
                "point_ids": point_ids[finite],
                "validation_group_ids": validation_group_ids[finite],
                "xyz": xyz[finite],
                "image_points": image_points[finite],
            }
        )
    return sorted(prepared, key=lambda row: int(row["frame"]))


def split_joint_ptz_track_groups(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
) -> dict[str, Any]:
    """Assign persistent landmark groups once to fit, confirm, and final roles."""

    prepared = _prepare_observations(observations, initial_cameras)
    xyz_by_point: dict[int, np.ndarray] = {}
    group_by_point: dict[int, int] = {}
    for row in prepared:
        for point_id, group_id, xyz in zip(
            row["point_ids"], row["validation_group_ids"], row["xyz"]
        ):
            key = int(point_id)
            group = int(group_id)
            point = np.asarray(xyz, dtype=np.float64).reshape(3)
            if key in xyz_by_point and not np.array_equal(xyz_by_point[key], point):
                raise ValueError("persistent point ID changed XYZ across frames")
            if key in group_by_point and group_by_point[key] != group:
                raise ValueError("persistent point ID changed validation group")
            xyz_by_point[key] = point.copy()
            group_by_point[key] = group

    role_names = ("fit", "confirm", "final")
    role_masks = (
        lambda buckets: buckets < 2,
        lambda buckets: buckets == 2,
        lambda buckets: buckets == 3,
    )
    result: dict[str, Any] = {role: [] for role in role_names}
    role_groups: dict[str, set[int]] = {role: set() for role in role_names}
    for source in prepared:
        groups = np.asarray(source["validation_group_ids"], dtype=np.int64)
        assignments = np.mod(stable_point_id_validation_groups(groups), 4)
        for role_index, role in enumerate(role_names):
            mask = role_masks[role_index](assignments)
            result[role].append(
                {
                    "frame": int(source["frame"]),
                    "point_ids": np.asarray(source["point_ids"])[mask],
                    "validation_group_ids": groups[mask],
                    "xyz": np.asarray(source["xyz"], dtype=np.float64)[mask],
                    "image_points": np.asarray(
                        source["image_points"], dtype=np.float64
                    )[mask],
                }
            )
            role_groups[role].update(int(value) for value in groups[mask])
    group_disjoint = all(
        not (role_groups[first] & role_groups[second])
        for first_index, first in enumerate(role_names)
        for second in role_names[first_index + 1 :]
    )
    result["group_disjoint"] = bool(group_disjoint)
    result["role_group_ids"] = {
        role: np.asarray(sorted(groups), dtype=np.int64)
        for role, groups in role_groups.items()
    }
    result["policy"] = (
        "stable_physical_landmark_group_splitmix64_fit2_confirm1_final1"
    )
    return result


def _solve_rotations(
    *,
    observations: Sequence[Mapping[str, Any]],
    cameras: Mapping[int, CameraParams],
    center: np.ndarray,
    reprojection_error_px: float,
    min_inliers_per_frame: int,
    rng_seed: int,
) -> dict[int, dict[str, Any]]:
    solved: dict[int, dict[str, Any]] = {}
    for row in observations:
        frame = int(row["frame"])
        xyz = np.asarray(row["xyz"], dtype=np.float64)
        image_points = np.asarray(row["image_points"], dtype=np.float64)
        count = len(xyz)
        if count < int(min_inliers_per_frame):
            continue
        template = cameras[frame]
        local_ids = np.arange(count, dtype=np.int64)
        result = solve_focus_track_rotation(
            point_bank={"point_ids": local_ids, "xyz": xyz},
            tracks_frame=image_points.reshape(1, count, 2),
            visibility_frame=np.ones((1, count), dtype=bool),
            view_ids=[template.view_id],
            focus_view=template.view_id,
            camera_template=template,
            fixed_center=np.asarray(center, dtype=np.float64),
            reprojection_error_px=float(reprojection_error_px),
            min_inliers=int(min_inliers_per_frame),
            max_evaluations=100,
            direction_ransac_prefilter=True,
            iterations_count=512,
            confidence=0.999,
            rng_seed=int(rng_seed) + frame,
        )
        if bool(result.get("success", False)):
            solved[frame] = result
    return solved


def _solve_bootstrap_poses(
    *,
    observations: Sequence[Mapping[str, Any]],
    cameras: Mapping[int, CameraParams],
    reprojection_error_px: float,
    min_inliers_per_frame: int,
    rng_seed: int,
) -> dict[int, dict[str, Any]]:
    solved: dict[int, dict[str, Any]] = {}
    for row in observations:
        frame = int(row["frame"])
        xyz = np.asarray(row["xyz"], dtype=np.float64)
        image_points = np.asarray(row["image_points"], dtype=np.float64)
        count = len(xyz)
        if count < int(min_inliers_per_frame):
            continue
        local_ids = np.arange(count, dtype=np.int64)
        result = solve_focus_track_pnp(
            point_bank={"point_ids": local_ids, "xyz": xyz},
            tracks_frame=image_points.reshape(1, count, 2),
            visibility_frame=np.ones((1, count), dtype=bool),
            view_ids=[cameras[frame].view_id],
            focus_view=cameras[frame].view_id,
            camera_template=cameras[frame],
            reprojection_error_px=float(reprojection_error_px),
            min_inliers=int(min_inliers_per_frame),
            iterations_count=1000,
            confidence=0.999,
            rng_seed=int(rng_seed) + frame,
        )
        if bool(result.get("success", False)):
            solved[frame] = result
    return solved


def _project_with_fixed_rotation(
    *,
    xyz: np.ndarray,
    camera: CameraParams,
    rotation: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    target_center = np.asarray(center, dtype=np.float64).reshape(3)
    translation = -matrix @ target_center
    rotation_vector, _ = cv2.Rodrigues(matrix)
    projected, _ = cv2.projectPoints(
        points,
        rotation_vector,
        translation,
        np.asarray(camera.K, dtype=np.float64),
        np.asarray(camera.D, dtype=np.float64).reshape(-1),
    )
    depths = (matrix @ points.T).T[:, 2] + translation[2]
    return projected.reshape(-1, 2), depths


def _weighted_joint_reprojection_residuals(
    *,
    delta: np.ndarray,
    depths: np.ndarray,
    reprojection_error_px: float,
    frame_weight: float,
) -> np.ndarray:
    residual = np.asarray(delta, dtype=np.float64).reshape(-1, 2)
    point_depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    if len(residual) != len(point_depths):
        raise ValueError("delta and depths must have equal lengths")
    threshold = float(reprojection_error_px)
    weight = float(frame_weight)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("reprojection_error_px must be finite and positive")
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("frame_weight must be finite and positive")

    norms = np.linalg.norm(residual, axis=1)
    valid = (
        np.isfinite(residual).all(axis=1)
        & np.isfinite(norms)
        & np.isfinite(point_depths)
        & (point_depths > 1e-8)
    )
    robust_weights = np.sqrt(
        np.minimum(1.0, threshold / np.maximum(norms, 1e-12))
    )
    output = np.zeros_like(residual)
    output[valid] = residual[valid] * robust_weights[valid, None]
    output[~valid] = threshold / np.sqrt(2.0)
    return (output * weight).reshape(-1)


def _joint_center_rotation_jacobian_sparsity(
    rows: Sequence[Mapping[str, Any]],
) -> csr_matrix:
    """Return the exact block sparsity for shared-center rotation refinement.

    Every reprojection residual depends on the three shared-center parameters
    and only the three rotation parameters of its own frame.  Supplying this
    structure avoids finite-differencing all frame rotations for every
    residual while leaving the residual function and objective unchanged.
    """

    residual_count = sum(
        2 * len(np.asarray(row["xyz"]).reshape(-1, 3)) for row in rows
    )
    parameter_count = 3 + 3 * len(rows)
    sparsity = lil_matrix(
        (residual_count, parameter_count), dtype=np.int8
    )
    residual_start = 0
    for row_index, row in enumerate(rows):
        row_residual_count = 2 * len(
            np.asarray(row["xyz"]).reshape(-1, 3)
        )
        residual_stop = residual_start + row_residual_count
        sparsity[residual_start:residual_stop, :3] = 1
        rotation_start = 3 + 3 * row_index
        sparsity[
            residual_start:residual_stop,
            rotation_start : rotation_start + 3,
        ] = 1
        residual_start = residual_stop
    return sparsity.tocsr()


def _refine_joint_center_rotations(
    *,
    center: np.ndarray,
    observations_by_frame: Mapping[int, Mapping[str, Any]],
    cameras: Mapping[int, CameraParams],
    bootstrap_results: Mapping[int, Mapping[str, Any]],
    reprojection_error_px: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for frame, result in sorted(bootstrap_results.items()):
        source = observations_by_frame[int(frame)]
        indices = np.asarray(
            result.get("optimization_indices", result.get("inlier_indices", [])),
            dtype=np.int64,
        ).reshape(-1)
        rows.append(
            {
                "frame": int(frame),
                "xyz": np.asarray(source["xyz"], dtype=np.float64)[indices],
                "image_points": np.asarray(source["image_points"], dtype=np.float64)[
                    indices
                ],
                "rotation_vector": cv2.Rodrigues(
                    np.asarray(result["camera"].R, dtype=np.float64)
                )[0].reshape(3),
            }
        )

    threshold = float(reprojection_error_px)
    initial = np.concatenate(
        [
            np.asarray(center, dtype=np.float64).reshape(3),
            *[np.asarray(row["rotation_vector"]).reshape(3) for row in rows],
        ]
    )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        candidate_center = np.asarray(parameters[:3], dtype=np.float64)
        output: list[np.ndarray] = []
        for row_index, row in enumerate(rows):
            rotation, _ = cv2.Rodrigues(
                np.asarray(
                    parameters[3 + 3 * row_index : 6 + 3 * row_index],
                    dtype=np.float64,
                ).reshape(3, 1)
            )
            predicted, depths = _project_with_fixed_rotation(
                xyz=row["xyz"],
                camera=cameras[int(row["frame"])],
                rotation=rotation,
                center=candidate_center,
            )
            delta = predicted - row["image_points"]
            frame_weight = 1.0 / np.sqrt(max(1, len(delta)))
            output.append(
                _weighted_joint_reprojection_residuals(
                    delta=delta,
                    depths=depths,
                    reprojection_error_px=threshold,
                    frame_weight=frame_weight,
                )
            )
        return np.concatenate(output)

    solution = least_squares(
        residuals,
        initial,
        loss="linear",
        max_nfev=200,
        x_scale="jac",
        jac_sparsity=_joint_center_rotation_jacobian_sparsity(rows),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    return {
        "success": bool(solution.success) and np.all(np.isfinite(solution.x)),
        "center": np.asarray(solution.x[:3], dtype=np.float64),
        "rotations": {
            int(row["frame"]): cv2.Rodrigues(
                np.asarray(
                    solution.x[3 + 3 * index : 6 + 3 * index],
                    dtype=np.float64,
                ).reshape(3, 1)
            )[0]
            for index, row in enumerate(rows)
        },
        "cost": float(solution.cost),
        "evaluations": int(solution.nfev),
    }


def solve_fixed_rotation_ptz_shared_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    fixed_rotations: Mapping[int, np.ndarray],
    reprojection_error_px: float = 4.0,
    min_points_per_frame: int = 12,
    min_successful_frames: int = 3,
) -> dict[str, Any]:
    """Estimate one PTZ center while keeping every supplied rotation fixed."""

    threshold = float(reprojection_error_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("reprojection_error_px must be finite and positive")
    if int(min_points_per_frame) < 4:
        raise ValueError("min_points_per_frame must be at least four")
    if int(min_successful_frames) < 2:
        raise ValueError("min_successful_frames must be at least two")

    prepared = _prepare_observations(observations, initial_cameras)
    supported = [
        row for row in prepared if len(row["xyz"]) >= int(min_points_per_frame)
    ]
    if len(supported) < int(min_successful_frames):
        return {
            "success": False,
            "reason": "insufficient_supported_frames",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
        }

    rotations: dict[int, np.ndarray] = {}
    for frame, camera in initial_cameras.items():
        rotation = np.asarray(
            fixed_rotations.get(int(frame), camera.R), dtype=np.float64
        )
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"invalid fixed rotation for frame {frame}")
        rotations[int(frame)] = rotation.copy()

    center_initial = _geometric_median(
        np.asarray(
            [initial_cameras[int(row["frame"])].camera_center for row in supported],
            dtype=np.float64,
        )
    )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        center = np.asarray(parameters, dtype=np.float64).reshape(3)
        output: list[np.ndarray] = []
        for row in supported:
            frame = int(row["frame"])
            predicted, depths = _project_with_fixed_rotation(
                xyz=np.asarray(row["xyz"], dtype=np.float64),
                camera=initial_cameras[frame],
                rotation=rotations[frame],
                center=center,
            )
            delta = predicted - np.asarray(row["image_points"], dtype=np.float64)
            frame_weight = 1.0 / np.sqrt(max(1, len(delta)))
            output.append(
                _weighted_joint_reprojection_residuals(
                    delta=delta,
                    depths=depths,
                    reprojection_error_px=threshold,
                    frame_weight=frame_weight,
                )
            )
        return np.concatenate(output)

    solution = least_squares(
        residuals,
        center_initial,
        loss="linear",
        max_nfev=200,
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not bool(solution.success) or not np.all(np.isfinite(solution.x)):
        return {
            "success": False,
            "reason": "shared_center_optimization_failed",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
            "center_initial": center_initial,
        }

    center = np.asarray(solution.x, dtype=np.float64).reshape(3)
    cameras = {
        int(frame): replace(camera, R=rotations[int(frame)], T=-rotations[int(frame)] @ center)
        for frame, camera in initial_cameras.items()
    }
    return {
        "success": True,
        "reason": "ok",
        "center_initial": center_initial,
        "center": center,
        "cameras": cameras,
        "input_frame_count": int(len(prepared)),
        "successful_frame_count": int(len(supported)),
        "successful_frames": [int(row["frame"]) for row in supported],
        "frame_point_counts": {
            int(row["frame"]): int(len(row["xyz"])) for row in supported
        },
        "frame_balancing": "equal_robust_cost_per_supported_frame",
        "cost": float(solution.cost),
        "evaluations": int(solution.nfev),
    }


def solve_fixed_center_ptz_rotations(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    fixed_center: np.ndarray,
    reprojection_error_px: float = 4.0,
    min_inliers_per_frame: int = 12,
    min_successful_frames: int = 3,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Estimate per-frame PTZ rotations while preserving one supplied center."""

    threshold = float(reprojection_error_px)
    minimum = int(min_inliers_per_frame)
    required_frames = int(min_successful_frames)
    center = np.asarray(fixed_center, dtype=np.float64).reshape(3)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("reprojection_error_px must be finite and positive")
    if minimum < 4:
        raise ValueError("min_inliers_per_frame must be at least four")
    if required_frames < 2:
        raise ValueError("min_successful_frames must be at least two")
    if not np.all(np.isfinite(center)):
        raise ValueError("fixed_center must be finite")

    prepared = _prepare_observations(observations, initial_cameras)
    solved = _solve_rotations(
        observations=prepared,
        cameras=initial_cameras,
        center=center,
        reprojection_error_px=threshold,
        min_inliers_per_frame=minimum,
        rng_seed=int(rng_seed),
    )
    if len(solved) < required_frames:
        return {
            "success": False,
            "reason": "insufficient_fixed_center_rotation_frames",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(solved)),
            "successful_frames": sorted(int(frame) for frame in solved),
            "center": center,
        }

    cameras: dict[int, CameraParams] = {}
    for frame, camera in initial_cameras.items():
        rotation = np.asarray(
            solved[int(frame)]["camera"].R
            if int(frame) in solved
            else camera.R,
            dtype=np.float64,
        )
        cameras[int(frame)] = replace(
            camera,
            R=rotation,
            T=-rotation @ center,
        )
    return {
        "success": True,
        "reason": "ok",
        "center": center,
        "cameras": cameras,
        "input_frame_count": int(len(prepared)),
        "successful_frame_count": int(len(solved)),
        "successful_frames": sorted(int(frame) for frame in solved),
        "frame_inlier_counts": {
            int(frame): int(result["inlier_count"])
            for frame, result in solved.items()
        },
        "frame_reprojection_median_px": {
            int(frame): float(result["reprojection_median_px"])
            for frame, result in solved.items()
        },
        "sparse_frame_policy": (
            "retain_initial_rotation_with_supplied_fixed_center"
        ),
    }


def solve_temporal_joint_ptz_shared_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    temporal_rotations: Mapping[int, np.ndarray],
    temporal_track_observations: Sequence[Mapping[str, Any]] = (),
    reprojection_error_px: float = 4.0,
    temporal_error_px: float = 1.5,
    min_points_per_frame: int = 12,
    min_successful_frames: int = 3,
) -> dict[str, Any]:
    """Jointly fit PTZ rotations and center with adjacent track-rotation factors."""

    map_threshold = float(reprojection_error_px)
    temporal_threshold = float(temporal_error_px)
    if not np.isfinite(map_threshold) or map_threshold <= 0.0:
        raise ValueError("reprojection_error_px must be finite and positive")
    if not np.isfinite(temporal_threshold) or temporal_threshold <= 0.0:
        raise ValueError("temporal_error_px must be finite and positive")
    if int(min_points_per_frame) < 4:
        raise ValueError("min_points_per_frame must be at least four")
    if int(min_successful_frames) < 2:
        raise ValueError("min_successful_frames must be at least two")

    prepared = _prepare_observations(observations, initial_cameras)
    supported = [
        row for row in prepared if len(row["xyz"]) >= int(min_points_per_frame)
    ]
    if len(supported) < int(min_successful_frames):
        return {
            "success": False,
            "reason": "insufficient_supported_frames",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
        }

    frames = sorted(int(frame) for frame in initial_cameras)
    frame_index = {frame: index for index, frame in enumerate(frames)}
    edges: list[tuple[int, int, np.ndarray]] = []
    for current, value in sorted(temporal_rotations.items()):
        current_frame = int(current)
        previous_frame = current_frame - 1
        if previous_frame not in frame_index or current_frame not in frame_index:
            continue
        rotation = np.asarray(value, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"invalid temporal rotation for frame {current_frame}")
        edges.append((previous_frame, current_frame, rotation.copy()))
    if not edges:
        return {
            "success": False,
            "reason": "no_valid_temporal_edges",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
        }

    edge_keys = {(previous, current) for previous, current, _value in edges}
    temporal_track_rows: list[dict[str, Any]] = []
    for payload in temporal_track_observations:
        previous = int(payload["previous_frame"])
        current = int(payload["current_frame"])
        if (previous, current) not in edge_keys:
            continue
        source_points = np.asarray(payload["source_points"], dtype=np.float64)
        target_points = np.asarray(payload["target_points"], dtype=np.float64)
        if source_points.ndim != 2 or source_points.shape[1] != 2:
            raise ValueError("temporal source_points must have shape (N, 2)")
        if target_points.shape != source_points.shape:
            raise ValueError("temporal target_points must match source_points")
        finite = np.isfinite(source_points).all(axis=1) & np.isfinite(
            target_points
        ).all(axis=1)
        source_points = source_points[finite]
        target_points = target_points[finite]
        if len(source_points) < 4:
            continue
        source_camera = initial_cameras[previous]
        source_rays = cv2.undistortPoints(
            source_points.reshape(-1, 1, 2),
            np.asarray(source_camera.K, dtype=np.float64),
            np.asarray(source_camera.D, dtype=np.float64).reshape(-1),
        ).reshape(-1, 2)
        temporal_track_rows.append(
            {
                "previous_frame": previous,
                "current_frame": current,
                "source_rays": np.column_stack(
                    [source_rays, np.ones(len(source_rays), dtype=np.float64)]
                ),
                "target_points": target_points,
            }
        )

    bootstrap = solve_joint_ptz_shared_center(
        observations=prepared,
        initial_cameras=initial_cameras,
        reprojection_error_px=map_threshold,
        min_inliers_per_frame=int(min_points_per_frame),
        min_successful_frames=int(min_successful_frames),
        rng_seed=0,
    )
    if not bool(bootstrap.get("success", False)):
        return {
            "success": False,
            "reason": "map_bootstrap_failed",
            "map_bootstrap_reason": str(bootstrap.get("reason", "unknown")),
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
            "temporal_edge_count": int(len(edges)),
        }
    center_initial = np.asarray(bootstrap["center"], dtype=np.float64).reshape(3)
    rotation_vectors = [
        cv2.Rodrigues(np.asarray(bootstrap["cameras"][frame].R, dtype=np.float64))[0]
        .reshape(3)
        .copy()
        for frame in frames
    ]
    initial = np.concatenate([center_initial, *rotation_vectors])

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        center = np.asarray(parameters[:3], dtype=np.float64).reshape(3)
        rotations = {
            frame: cv2.Rodrigues(
                np.asarray(
                    parameters[3 + 3 * index : 6 + 3 * index],
                    dtype=np.float64,
                ).reshape(3, 1)
            )[0]
            for frame, index in frame_index.items()
        }
        return center, rotations

    def residuals(parameters: np.ndarray) -> np.ndarray:
        center, rotations = unpack(parameters)
        output: list[np.ndarray] = []
        for row in supported:
            frame = int(row["frame"])
            predicted, depths = _project_with_fixed_rotation(
                xyz=np.asarray(row["xyz"], dtype=np.float64),
                camera=initial_cameras[frame],
                rotation=rotations[frame],
                center=center,
            )
            delta = predicted - np.asarray(row["image_points"], dtype=np.float64)
            norms = np.linalg.norm(delta, axis=1)
            invalid = ~np.isfinite(norms) | (depths <= 1e-8)
            weights = np.sqrt(
                np.minimum(1.0, map_threshold / np.maximum(norms, 1e-12))
            )
            safe_delta = np.asarray(delta, dtype=np.float64).copy()
            safe_delta[invalid] = map_threshold
            weights[invalid] = 1.0
            frame_weight = 1.0 / np.sqrt(max(1, len(delta)))
            output.append(
                (safe_delta * weights[:, None] * frame_weight).reshape(-1)
            )
        if temporal_track_rows:
            for row in temporal_track_rows:
                previous = int(row["previous_frame"])
                current = int(row["current_frame"])
                relative = rotations[current] @ rotations[previous].T
                rotated_rays = (
                    relative @ np.asarray(row["source_rays"], dtype=np.float64).T
                ).T
                target_camera = initial_cameras[current]
                predicted, _ = cv2.projectPoints(
                    rotated_rays,
                    np.zeros(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    np.asarray(target_camera.K, dtype=np.float64),
                    np.asarray(target_camera.D, dtype=np.float64).reshape(-1),
                )
                delta = predicted.reshape(-1, 2) - np.asarray(
                    row["target_points"], dtype=np.float64
                )
                norms = np.linalg.norm(delta, axis=1)
                invalid = ~np.isfinite(norms) | (rotated_rays[:, 2] <= 1e-8)
                weights = np.sqrt(
                    np.minimum(
                        1.0,
                        temporal_threshold / np.maximum(norms, 1e-12),
                    )
                )
                safe_delta = np.asarray(delta, dtype=np.float64).copy()
                safe_delta[invalid] = temporal_threshold
                weights[invalid] = 1.0
                group_weight = 1.0 / np.sqrt(max(1, len(delta)))
                output.append(
                    (safe_delta * weights[:, None] * group_weight).reshape(-1)
                )
        else:
            for previous, current, measured in edges:
                predicted = rotations[current] @ rotations[previous].T
                error_rotation = predicted @ measured.T
                error_vector = cv2.Rodrigues(error_rotation)[0].reshape(3)
                focal = 0.25 * (
                    float(initial_cameras[previous].K[0, 0])
                    + float(initial_cameras[previous].K[1, 1])
                    + float(initial_cameras[current].K[0, 0])
                    + float(initial_cameras[current].K[1, 1])
                )
                error_pixels = error_vector * focal
                norm_pixels = float(np.linalg.norm(error_pixels))
                weight = np.sqrt(
                    min(1.0, temporal_threshold / max(norm_pixels, 1e-12))
                )
                output.append(error_pixels * weight / np.sqrt(3.0))
        return np.concatenate(output)

    solution = least_squares(
        residuals,
        initial,
        loss="linear",
        max_nfev=200,
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not bool(solution.success) or not np.all(np.isfinite(solution.x)):
        return {
            "success": False,
            "reason": "temporal_joint_optimization_failed",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
            "temporal_edge_count": int(len(edges)),
            "center_initial": center_initial,
        }

    center, rotations = unpack(solution.x)
    cameras = {
        frame: replace(
            initial_cameras[frame],
            R=rotations[frame],
            T=-rotations[frame] @ center,
        )
        for frame in frames
    }
    return {
        "success": True,
        "reason": "ok",
        "center_initial": center_initial,
        "map_bootstrap_center": np.asarray(bootstrap["center"], dtype=np.float64),
        "center": center,
        "cameras": cameras,
        "input_frame_count": int(len(prepared)),
        "successful_frame_count": int(len(supported)),
        "successful_frames": [int(row["frame"]) for row in supported],
        "temporal_edge_count": int(len(edges)),
        "temporal_track_group_count": int(len(temporal_track_rows)),
        "temporal_frames": [int(current) for _previous, current, _value in edges],
        "frame_balancing": "equal_robust_cost_per_supported_frame",
        "temporal_weight_policy": (
            "equal_cost_per_track_seed_edge"
            if temporal_track_rows
            else "focal_pixel_equivalent"
        ),
        "cost": float(solution.cost),
        "evaluations": int(solution.nfev),
    }


def solve_bounded_rotation_ptz_median_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    tracking_rotations: Mapping[int, np.ndarray],
    temporal_track_observations: Sequence[Mapping[str, Any]] = (),
    reprojection_error_px: float = 4.0,
    temporal_error_px: float = 1.5,
    min_points_per_frame: int = 12,
    min_successful_frames: int = 3,
    fixed_center: np.ndarray | None = None,
) -> dict[str, Any]:
    """Blend bounded tracking repairs while fixing one robust PTZ center."""

    map_threshold = float(reprojection_error_px)
    temporal_threshold = float(temporal_error_px)
    if not np.isfinite(map_threshold) or map_threshold <= 0.0:
        raise ValueError("reprojection_error_px must be finite and positive")
    if not np.isfinite(temporal_threshold) or temporal_threshold <= 0.0:
        raise ValueError("temporal_error_px must be finite and positive")
    if int(min_points_per_frame) < 4:
        raise ValueError("min_points_per_frame must be at least four")
    if int(min_successful_frames) < 2:
        raise ValueError("min_successful_frames must be at least two")

    prepared = _prepare_observations(observations, initial_cameras)
    supported = [
        row for row in prepared if len(row["xyz"]) >= int(min_points_per_frame)
    ]
    if len(supported) < int(min_successful_frames):
        return {
            "success": False,
            "reason": "insufficient_supported_frames",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
        }

    repair_frames = sorted(
        int(frame) for frame in tracking_rotations if int(frame) in initial_cameras
    )
    if not repair_frames:
        return {
            "success": False,
            "reason": "no_tracking_rotation_candidates",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
        }
    repair_index = {frame: index for index, frame in enumerate(repair_frames)}
    base_rotations = {
        int(frame): np.asarray(camera.R, dtype=np.float64).copy()
        for frame, camera in initial_cameras.items()
    }
    delta_vectors: dict[int, np.ndarray] = {}
    for frame in repair_frames:
        target = np.asarray(tracking_rotations[frame], dtype=np.float64)
        if target.shape != (3, 3) or not np.all(np.isfinite(target)):
            raise ValueError(f"invalid tracking rotation for frame {frame}")
        delta = target @ base_rotations[frame].T
        delta_vectors[frame] = cv2.Rodrigues(delta)[0].reshape(3)

    if fixed_center is None:
        center = _geometric_median(
            np.asarray(
                [
                    initial_cameras[int(row["frame"])].camera_center
                    for row in prepared
                ],
                dtype=np.float64,
            )
        )
        center_policy = "geometric_median_static_pnp_centers"
    else:
        center = np.asarray(fixed_center, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(center)):
            raise ValueError("fixed_center must be finite")
        center_policy = "caller_supplied_fixed_center"
    temporal_rows: list[dict[str, Any]] = []
    for payload in temporal_track_observations:
        previous = int(payload["previous_frame"])
        current = int(payload["current_frame"])
        if previous not in initial_cameras or current not in initial_cameras:
            continue
        source_points = np.asarray(payload["source_points"], dtype=np.float64)
        target_points = np.asarray(payload["target_points"], dtype=np.float64)
        if source_points.ndim != 2 or source_points.shape[1] != 2:
            raise ValueError("temporal source_points must have shape (N, 2)")
        if target_points.shape != source_points.shape:
            raise ValueError("temporal target_points must match source_points")
        finite = np.isfinite(source_points).all(axis=1) & np.isfinite(
            target_points
        ).all(axis=1)
        source_points = source_points[finite]
        target_points = target_points[finite]
        if len(source_points) < 4:
            continue
        source_camera = initial_cameras[previous]
        source_rays = cv2.undistortPoints(
            source_points.reshape(-1, 1, 2),
            np.asarray(source_camera.K, dtype=np.float64),
            np.asarray(source_camera.D, dtype=np.float64).reshape(-1),
        ).reshape(-1, 2)
        temporal_rows.append(
            {
                "previous_frame": previous,
                "current_frame": current,
                "source_rays": np.column_stack(
                    [source_rays, np.ones(len(source_rays), dtype=np.float64)]
                ),
                "target_points": target_points,
            }
        )

    def rotations_for(blends: np.ndarray) -> dict[int, np.ndarray]:
        rotations = dict(base_rotations)
        for frame, index in repair_index.items():
            correction = cv2.Rodrigues(
                np.asarray(delta_vectors[frame] * blends[index], dtype=np.float64)
            )[0]
            rotations[frame] = correction @ base_rotations[frame]
        return rotations

    def residuals(blends: np.ndarray) -> np.ndarray:
        rotations = rotations_for(blends)
        output: list[np.ndarray] = []
        for row in supported:
            frame = int(row["frame"])
            predicted, depths = _project_with_fixed_rotation(
                xyz=np.asarray(row["xyz"], dtype=np.float64),
                camera=initial_cameras[frame],
                rotation=rotations[frame],
                center=center,
            )
            delta = predicted - np.asarray(row["image_points"], dtype=np.float64)
            norms = np.linalg.norm(delta, axis=1)
            invalid = ~np.isfinite(norms) | (depths <= 1e-8)
            weights = np.sqrt(
                np.minimum(1.0, map_threshold / np.maximum(norms, 1e-12))
            )
            safe_delta = np.asarray(delta, dtype=np.float64).copy()
            safe_delta[invalid] = map_threshold
            weights[invalid] = 1.0
            output.append(
                (
                    safe_delta
                    * weights[:, None]
                    / np.sqrt(max(1, len(delta)))
                ).reshape(-1)
            )
        for row in temporal_rows:
            previous = int(row["previous_frame"])
            current = int(row["current_frame"])
            relative = rotations[current] @ rotations[previous].T
            rotated_rays = (
                relative @ np.asarray(row["source_rays"], dtype=np.float64).T
            ).T
            target_camera = initial_cameras[current]
            predicted, _ = cv2.projectPoints(
                rotated_rays,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.asarray(target_camera.K, dtype=np.float64),
                np.asarray(target_camera.D, dtype=np.float64).reshape(-1),
            )
            delta = predicted.reshape(-1, 2) - np.asarray(
                row["target_points"], dtype=np.float64
            )
            norms = np.linalg.norm(delta, axis=1)
            invalid = ~np.isfinite(norms) | (rotated_rays[:, 2] <= 1e-8)
            weights = np.sqrt(
                np.minimum(
                    1.0,
                    temporal_threshold / np.maximum(norms, 1e-12),
                )
            )
            safe_delta = np.asarray(delta, dtype=np.float64).copy()
            safe_delta[invalid] = temporal_threshold
            weights[invalid] = 1.0
            output.append(
                (
                    safe_delta
                    * weights[:, None]
                    / np.sqrt(max(1, len(delta)))
                ).reshape(-1)
            )
        return np.concatenate(output)

    solution = least_squares(
        residuals,
        np.full(len(repair_frames), 0.5, dtype=np.float64),
        bounds=(
            np.zeros(len(repair_frames), dtype=np.float64),
            np.ones(len(repair_frames), dtype=np.float64),
        ),
        loss="linear",
        max_nfev=200,
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not bool(solution.success) or not np.all(np.isfinite(solution.x)):
        return {
            "success": False,
            "reason": "bounded_rotation_optimization_failed",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(supported)),
        }

    rotations = rotations_for(np.asarray(solution.x, dtype=np.float64))
    cameras = {
        int(frame): replace(camera, R=rotations[int(frame)], T=-rotations[int(frame)] @ center)
        for frame, camera in initial_cameras.items()
    }
    return {
        "success": True,
        "reason": "ok",
        "center": center,
        "center_policy": center_policy,
        "cameras": cameras,
        "rotation_blends": {
            frame: float(solution.x[index]) for frame, index in repair_index.items()
        },
        "repair_frames": repair_frames,
        "temporal_track_group_count": int(len(temporal_rows)),
        "frame_balancing": "equal_robust_cost_per_supported_frame",
        "temporal_weight_policy": "equal_cost_per_track_seed_edge",
        "cost": float(solution.cost),
        "evaluations": int(solution.nfev),
    }


def solve_joint_ptz_shared_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    reprojection_error_px: float = 4.0,
    min_inliers_per_frame: int = 12,
    min_successful_frames: int = 3,
    rng_seed: int = 0,
    allow_initial_camera_bootstrap: bool = False,
    allow_joint_refined_rotation_fallback: bool = False,
) -> dict[str, Any]:
    """Jointly estimate one PTZ center and one track-supported rotation per frame.

    The defaults require independent PnP and final rotation support. The
    opt-ins may supplement sparse PnP initialization with supplied cameras and
    sparse final rotation support with rotations produced by joint refinement.
    """

    threshold = float(reprojection_error_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("reprojection_error_px must be finite and positive")
    if int(min_inliers_per_frame) < 4:
        raise ValueError("min_inliers_per_frame must be at least four")
    if int(min_successful_frames) < 2:
        raise ValueError("min_successful_frames must be at least two")
    prepared = _prepare_observations(observations, initial_cameras)
    if len(prepared) < int(min_successful_frames):
        return {
            "success": False,
            "reason": "insufficient_input_frames",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": 0,
        }

    cameras = {int(frame): camera for frame, camera in initial_cameras.items()}
    input_center = _geometric_median(
        np.asarray(
            [cameras[int(row["frame"])].camera_center for row in prepared],
            dtype=np.float64,
        )
    )
    observations_by_frame = {int(row["frame"]): row for row in prepared}
    bootstrap_results = _solve_bootstrap_poses(
        observations=prepared,
        cameras=cameras,
        reprojection_error_px=threshold,
        min_inliers_per_frame=int(min_inliers_per_frame),
        rng_seed=int(rng_seed),
    )
    pnp_bootstrap_frame_count = len(bootstrap_results)
    initial_camera_bootstrap_frames: list[int] = []
    if (
        len(bootstrap_results) < int(min_successful_frames)
        and bool(allow_initial_camera_bootstrap)
    ):
        for row in prepared:
            frame = int(row["frame"])
            point_count = len(np.asarray(row["xyz"]).reshape(-1, 3))
            if (
                frame in bootstrap_results
                or point_count < int(min_inliers_per_frame)
            ):
                continue
            bootstrap_results[frame] = {
                "camera": cameras[frame],
                "optimization_indices": np.arange(
                    point_count, dtype=np.int64
                ),
            }
            initial_camera_bootstrap_frames.append(frame)
    bootstrap_audit = (
        {
            "bootstrap_policy": (
                "pnp_then_supplied_initial_camera_fallback"
            ),
            "bootstrap_pnp_frame_count": int(pnp_bootstrap_frame_count),
            "initial_camera_bootstrap_frame_count": int(
                len(initial_camera_bootstrap_frames)
            ),
            "initial_camera_bootstrap_frames": sorted(
                initial_camera_bootstrap_frames
            ),
        }
        if bool(allow_initial_camera_bootstrap)
        else {}
    )
    if len(bootstrap_results) < int(min_successful_frames):
        return {
            "success": False,
            "reason": "insufficient_bootstrap_pnp_frames",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(bootstrap_results)),
            "center_initial": input_center,
            **bootstrap_audit,
        }
    bootstrap_center = _geometric_median(
        np.asarray(
            [result["camera"].camera_center for result in bootstrap_results.values()],
            dtype=np.float64,
        )
    )
    refined = _refine_joint_center_rotations(
        center=bootstrap_center,
        observations_by_frame=observations_by_frame,
        cameras=cameras,
        bootstrap_results=bootstrap_results,
        reprojection_error_px=threshold,
    )
    if not bool(refined["success"]):
        return {
            "success": False,
            "reason": "joint_center_rotation_optimization_failed",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(bootstrap_results)),
            "center_initial": input_center,
            "center_bootstrap": bootstrap_center,
            **bootstrap_audit,
        }
    center = np.asarray(refined["center"], dtype=np.float64)
    centered_cameras: dict[int, CameraParams] = {}
    for frame, camera in cameras.items():
        rotation = np.asarray(
            refined["rotations"].get(frame, camera.R), dtype=np.float64
        )
        centered_cameras[frame] = replace(
            camera,
            R=rotation,
            T=-rotation @ center,
        )
    cameras = centered_cameras
    history = [
        {
            "iteration": 0,
            "successful_frame_count": int(len(bootstrap_results)),
            "center": center.copy(),
            "center_shift_from_bootstrap": float(
                np.linalg.norm(center - bootstrap_center)
            ),
            "joint_cost": float(refined["cost"]),
            "joint_optimizer_evaluations": int(refined["evaluations"]),
        }
    ]

    rotation_results = _solve_rotations(
        observations=prepared,
        cameras=cameras,
        center=center,
        reprojection_error_px=threshold,
        min_inliers_per_frame=int(min_inliers_per_frame),
        rng_seed=int(rng_seed) + 7919,
    )
    independent_rotation_frames = sorted(int(frame) for frame in rotation_results)
    joint_refined_frames = {
        int(frame) for frame in refined.get("rotations", {})
    }
    joint_refined_rotation_fallback_frames = sorted(
        joint_refined_frames - set(independent_rotation_frames)
    )
    final_rotation_supported_frames = sorted(
        joint_refined_frames | set(independent_rotation_frames)
    )
    final_rotation_audit = (
        {
            "final_rotation_policy": (
                "independent_rotation_then_joint_refined_fallback"
            ),
            "independent_rotation_frame_count": int(
                len(independent_rotation_frames)
            ),
            "independent_rotation_frames": independent_rotation_frames,
            "joint_refined_rotation_fallback_frame_count": int(
                len(joint_refined_rotation_fallback_frames)
            ),
            "joint_refined_rotation_fallback_frames": (
                joint_refined_rotation_fallback_frames
            ),
            "final_rotation_supported_frame_count": int(
                len(final_rotation_supported_frames)
            ),
            "final_rotation_supported_frames": final_rotation_supported_frames,
        }
        if bool(allow_joint_refined_rotation_fallback)
        else {}
    )
    insufficient_final_support = len(rotation_results) < int(
        min_successful_frames
    ) and (
        not bool(allow_joint_refined_rotation_fallback)
        or len(final_rotation_supported_frames) < int(min_successful_frames)
    )
    if insufficient_final_support:
        return {
            "success": False,
            "reason": "insufficient_final_rotation_supported_frames",
            "input_frame_count": int(len(prepared)),
            "successful_frame_count": int(len(rotation_results)),
            "center_initial": input_center,
            "center_bootstrap": bootstrap_center,
            "center": center,
            "history": history,
            **bootstrap_audit,
            **final_rotation_audit,
        }

    output_cameras = dict(cameras)
    for frame, result in rotation_results.items():
        rotation = np.asarray(result["camera"].R, dtype=np.float64)
        output_cameras[int(frame)] = replace(
            cameras[int(frame)],
            R=rotation,
            T=-rotation @ center,
        )
    return {
        "success": True,
        "reason": "ok",
        "center_initial": input_center,
        "center_bootstrap": bootstrap_center,
        "center": center,
        "cameras": output_cameras,
        "input_frame_count": int(len(prepared)),
        "successful_frame_count": int(len(rotation_results)),
        "successful_frames": sorted(int(frame) for frame in rotation_results),
        "frame_inlier_counts": {
            int(frame): int(result["inlier_count"])
            for frame, result in rotation_results.items()
        },
        "frame_reprojection_median_px": {
            int(frame): float(result["reprojection_median_px"])
            for frame, result in rotation_results.items()
        },
        "frame_balancing": "equal_robust_cost_per_successful_frame",
        "history": history,
        **bootstrap_audit,
        **final_rotation_audit,
    }


def _split_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    num_folds: int,
    validation_fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    fit_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    disjoint = True
    for source in observations:
        point_ids = np.asarray(source["point_ids"], dtype=np.int64).reshape(-1)
        groups = stable_point_id_validation_groups(point_ids)
        validation_mask = np.mod(groups, int(num_folds)) == int(validation_fold)
        fit_mask = ~validation_mask

        def subset(mask: np.ndarray) -> dict[str, Any]:
            return {
                "frame": int(source["frame"]),
                "point_ids": point_ids[mask],
                "xyz": np.asarray(source["xyz"], dtype=np.float64)[mask],
                "image_points": np.asarray(source["image_points"], dtype=np.float64)[
                    mask
                ],
            }

        fit = subset(fit_mask)
        validation = subset(validation_mask)
        fit_rows.append(fit)
        validation_rows.append(validation)
        disjoint = disjoint and not bool(
            set(fit["point_ids"].tolist()) & set(validation["point_ids"].tolist())
        )
    return fit_rows, validation_rows, bool(disjoint)


def _camera_reprojection_errors(
    *,
    camera: CameraParams,
    xyz: np.ndarray,
    image_points: np.ndarray,
) -> np.ndarray:
    predicted, depths = _project_with_fixed_rotation(
        xyz=np.asarray(xyz, dtype=np.float64),
        camera=camera,
        rotation=camera.R,
        center=camera.camera_center,
    )
    errors = np.linalg.norm(
        predicted - np.asarray(image_points, dtype=np.float64), axis=1
    )
    errors[(depths <= 1e-8) | ~np.isfinite(errors)] = np.inf
    return errors


def _summarize_reprojection_errors(
    errors: Sequence[float] | np.ndarray,
    *,
    inlier_px: float,
) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "count": 0,
            "mean_px": float("inf"),
            "median_px": float("inf"),
            "p90_px": float("inf"),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
        }
    inliers = finite <= float(inlier_px)
    return {
        "count": int(len(finite)),
        "mean_px": float(np.mean(finite)),
        "median_px": float(np.median(finite)),
        "p90_px": float(np.percentile(finite, 90.0)),
        "inlier_count": int(np.count_nonzero(inliers)),
        "inlier_ratio": float(np.mean(inliers)),
    }


def _crossfit_model_projection_distance(
    *,
    first: Mapping[int, CameraParams],
    second: Mapping[int, CameraParams],
    observations: Sequence[Mapping[str, Any]],
) -> float:
    errors: list[np.ndarray] = []
    for row in observations:
        frame = int(row["frame"])
        if frame not in first or frame not in second:
            continue
        xyz = np.asarray(row["xyz"], dtype=np.float64)
        first_uv, first_depth = _project_with_fixed_rotation(
            xyz=xyz,
            camera=first[frame],
            rotation=first[frame].R,
            center=first[frame].camera_center,
        )
        second_uv, second_depth = _project_with_fixed_rotation(
            xyz=xyz,
            camera=second[frame],
            rotation=second[frame].R,
            center=second[frame].camera_center,
        )
        valid = (
            (first_depth > 1e-8)
            & (second_depth > 1e-8)
            & np.isfinite(first_uv).all(axis=1)
            & np.isfinite(second_uv).all(axis=1)
        )
        if np.any(valid):
            errors.append(np.linalg.norm(first_uv[valid] - second_uv[valid], axis=1))
    if not errors:
        return float("inf")
    return float(np.median(np.concatenate(errors)))


def _crossfit_candidate_solver(
    *,
    prepared: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    reprojection_error_px: float,
    num_folds: int,
    fit_solver: Any,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    for validation_fold in range(int(num_folds)):
        fit_rows, validation_rows, disjoint = _split_observations(
            prepared,
            num_folds=int(num_folds),
            validation_fold=validation_fold,
        )
        fit = fit_solver(fit_rows)
        row: dict[str, Any] = {
            "validation_fold": int(validation_fold),
            "fit_success": bool(fit.get("success", False)),
            "fit_reason": str(fit.get("reason", "unknown")),
            "fit_validation_disjoint": bool(disjoint),
            "fit_rows": [
                {
                    "frame": int(item["frame"]),
                    "point_ids": np.asarray(item["point_ids"], dtype=np.int64),
                }
                for item in fit_rows
            ],
            "validation_rows": [
                {
                    "frame": int(item["frame"]),
                    "point_ids": np.asarray(item["point_ids"], dtype=np.int64),
                }
                for item in validation_rows
            ],
        }
        fold_rows.append(row)
        if not bool(fit.get("success", False)):
            continue

        base_errors: list[np.ndarray] = []
        candidate_errors: list[np.ndarray] = []
        for validation in validation_rows:
            frame = int(validation["frame"])
            if frame not in initial_cameras or frame not in fit["cameras"]:
                continue
            if len(np.asarray(validation["xyz"]).reshape(-1, 3)) == 0:
                continue
            base_errors.append(
                _camera_reprojection_errors(
                    camera=initial_cameras[frame],
                    xyz=validation["xyz"],
                    image_points=validation["image_points"],
                )
            )
            candidate_errors.append(
                _camera_reprojection_errors(
                    camera=fit["cameras"][frame],
                    xyz=validation["xyz"],
                    image_points=validation["image_points"],
                )
            )
        if not base_errors:
            row["fit_success"] = False
            row["fit_reason"] = "empty_validation_support"
            continue
        row["base_errors_px"] = np.concatenate(base_errors)
        row["candidate_errors_px"] = np.concatenate(candidate_errors)
        row["base"] = _summarize_reprojection_errors(
            row["base_errors_px"], inlier_px=float(reprojection_error_px)
        )
        row["candidate"] = _summarize_reprojection_errors(
            row["candidate_errors_px"], inlier_px=float(reprojection_error_px)
        )
        successful.append({"fold": validation_fold, "fit": fit, "row": row})

    required = int(num_folds) // 2 + 1
    if len(successful) < required:
        return {
            "success": False,
            "reason": "crossfit_insufficient_successful_folds",
            "folds": fold_rows,
            "successful_fold_count": int(len(successful)),
            "required_successful_fold_count": int(required),
        }

    medoid_costs = {int(item["fold"]): 0.0 for item in successful}
    pairwise: list[dict[str, Any]] = []
    for first_index, first in enumerate(successful):
        for second in successful[first_index + 1 :]:
            distance = _crossfit_model_projection_distance(
                first=first["fit"]["cameras"],
                second=second["fit"]["cameras"],
                observations=prepared,
            )
            pairwise.append(
                {
                    "fold_a": int(first["fold"]),
                    "fold_b": int(second["fold"]),
                    "projection_median_px": float(distance),
                }
            )
            medoid_costs[int(first["fold"])] += float(distance)
            medoid_costs[int(second["fold"])] += float(distance)
    medoid_fold = min(medoid_costs, key=lambda fold: (medoid_costs[fold], fold))
    selected = next(item for item in successful if int(item["fold"]) == medoid_fold)
    base_errors = np.concatenate(
        [np.asarray(item["row"]["base_errors_px"]) for item in successful]
    )
    candidate_errors = np.concatenate(
        [np.asarray(item["row"]["candidate_errors_px"]) for item in successful]
    )
    return {
        "success": True,
        "reason": "ok",
        "cameras": selected["fit"]["cameras"],
        "center": selected["fit"]["center"],
        "selected_fit": selected["fit"],
        "selected_validation_fold": int(medoid_fold),
        "selected_model_heldout": {
            "base_errors_px": np.asarray(selected["row"]["base_errors_px"]),
            "candidate_errors_px": np.asarray(
                selected["row"]["candidate_errors_px"]
            ),
            "base": selected["row"]["base"],
            "candidate": selected["row"]["candidate"],
            "validation_fold": int(medoid_fold),
        },
        "successful_fold_count": int(len(successful)),
        "required_successful_fold_count": int(required),
        "folds": fold_rows,
        "aggregate_oof": {
            "base": _summarize_reprojection_errors(
                base_errors, inlier_px=float(reprojection_error_px)
            ),
            "candidate": _summarize_reprojection_errors(
                candidate_errors, inlier_px=float(reprojection_error_px)
            ),
            "base_errors_px": base_errors,
            "candidate_errors_px": candidate_errors,
        },
        "crossfit_consistency": {
            "policy": "fold_model_medoid_by_band_projection_median",
            "medoid_fold": int(medoid_fold),
            "medoid_cost_px": float(medoid_costs[medoid_fold]),
            "max_pairwise_projection_median_px": float(
                max(row["projection_median_px"] for row in pairwise)
            ),
            "pairwise": pairwise,
        },
        "fit_validation_split": "stable_track_id_splitmix64_modulo",
    }


def crossfit_bounded_rotation_ptz_median_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    tracking_rotations: Mapping[int, np.ndarray],
    temporal_track_observations: Sequence[Mapping[str, Any]] = (),
    reprojection_error_px: float = 4.0,
    temporal_error_px: float = 1.5,
    min_points_per_frame: int = 12,
    min_successful_frames: int = 3,
    num_folds: int = 3,
    fixed_center: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cross-fit bounded track repair and score only held-out map IDs."""

    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    prepared = _prepare_observations(observations, initial_cameras)
    fit_minimum = max(
        4,
        int(np.ceil(int(min_points_per_frame) * (folds - 1) / folds)),
    )

    result = _crossfit_candidate_solver(
        prepared=prepared,
        initial_cameras=initial_cameras,
        reprojection_error_px=float(reprojection_error_px),
        num_folds=folds,
        fit_solver=lambda fit_rows: solve_bounded_rotation_ptz_median_center(
            observations=fit_rows,
            initial_cameras=initial_cameras,
            tracking_rotations=tracking_rotations,
            temporal_track_observations=temporal_track_observations,
            reprojection_error_px=float(reprojection_error_px),
            temporal_error_px=float(temporal_error_px),
            min_points_per_frame=fit_minimum,
            min_successful_frames=int(min_successful_frames),
            fixed_center=fixed_center,
        ),
    )
    if not bool(result.get("success", False)):
        return result
    selected_fit = result.pop("selected_fit")
    result["rotation_blends"] = selected_fit["rotation_blends"]
    result["center_policy"] = selected_fit["center_policy"]
    result["temporal_track_policy"] = "fit_only_not_heldout_score"
    return result


def crossfit_fixed_center_ptz_rotations(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    fixed_center: np.ndarray,
    reprojection_error_px: float = 4.0,
    min_inliers_per_frame: int = 12,
    min_successful_frames: int = 3,
    num_folds: int = 3,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Cross-fit fixed-center frame rotations on disjoint persistent track IDs."""

    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    center = np.asarray(fixed_center, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center)):
        raise ValueError("fixed_center must be finite")
    prepared = _prepare_observations(observations, initial_cameras)
    fit_minimum = max(
        4,
        int(np.ceil(int(min_inliers_per_frame) * (folds - 1) / folds)),
    )
    result = _crossfit_candidate_solver(
        prepared=prepared,
        initial_cameras=initial_cameras,
        reprojection_error_px=float(reprojection_error_px),
        num_folds=folds,
        fit_solver=lambda fit_rows: solve_fixed_center_ptz_rotations(
            observations=fit_rows,
            initial_cameras=initial_cameras,
            fixed_center=center,
            reprojection_error_px=float(reprojection_error_px),
            min_inliers_per_frame=fit_minimum,
            min_successful_frames=int(min_successful_frames),
            rng_seed=int(rng_seed),
        ),
    )
    if not bool(result.get("success", False)):
        return result
    selected_fit = result.pop("selected_fit")
    np.testing.assert_allclose(
        np.asarray(selected_fit["center"], dtype=np.float64),
        center,
        atol=0.0,
        rtol=0.0,
    )
    result.update(
        {
            "center": center,
            "selected_successful_frames": list(
                selected_fit["successful_frames"]
            ),
            "selected_frame_inlier_counts": dict(
                selected_fit["frame_inlier_counts"]
            ),
            "temporal_track_policy": "fit_only_not_heldout_score",
            "model": "fixed_global_center_independent_frame_rotations",
        }
    )
    return result


def crossfit_fixed_rotation_ptz_shared_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    fixed_rotations: Mapping[int, np.ndarray],
    reprojection_error_px: float = 4.0,
    min_points_per_frame: int = 12,
    min_successful_frames: int = 3,
    num_folds: int = 3,
) -> dict[str, Any]:
    """Cross-fit a fixed-rotation PTZ center on disjoint track IDs."""

    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    prepared = _prepare_observations(observations, initial_cameras)
    fit_minimum = max(
        4,
        int(np.ceil(int(min_points_per_frame) * (folds - 1) / folds)),
    )
    fold_rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    for validation_fold in range(folds):
        fit_rows, validation_rows, disjoint = _split_observations(
            prepared,
            num_folds=folds,
            validation_fold=validation_fold,
        )
        fit = solve_fixed_rotation_ptz_shared_center(
            observations=fit_rows,
            initial_cameras=initial_cameras,
            fixed_rotations=fixed_rotations,
            reprojection_error_px=float(reprojection_error_px),
            min_points_per_frame=fit_minimum,
            min_successful_frames=int(min_successful_frames),
        )
        row: dict[str, Any] = {
            "validation_fold": int(validation_fold),
            "fit_success": bool(fit.get("success", False)),
            "fit_reason": str(fit.get("reason", "unknown")),
            "fit_validation_disjoint": bool(disjoint),
            "fit_rows": [
                {
                    "frame": int(item["frame"]),
                    "point_ids": np.asarray(item["point_ids"], dtype=np.int64),
                }
                for item in fit_rows
            ],
            "validation_rows": [
                {
                    "frame": int(item["frame"]),
                    "point_ids": np.asarray(item["point_ids"], dtype=np.int64),
                }
                for item in validation_rows
            ],
        }
        fold_rows.append(row)
        if not bool(fit.get("success", False)):
            continue

        base_errors: list[np.ndarray] = []
        candidate_errors: list[np.ndarray] = []
        for validation in validation_rows:
            frame = int(validation["frame"])
            if frame not in initial_cameras or frame not in fit["cameras"]:
                continue
            if len(np.asarray(validation["xyz"]).reshape(-1, 3)) == 0:
                continue
            base_errors.append(
                _camera_reprojection_errors(
                    camera=initial_cameras[frame],
                    xyz=validation["xyz"],
                    image_points=validation["image_points"],
                )
            )
            candidate_errors.append(
                _camera_reprojection_errors(
                    camera=fit["cameras"][frame],
                    xyz=validation["xyz"],
                    image_points=validation["image_points"],
                )
            )
        if not base_errors:
            row["fit_success"] = False
            row["fit_reason"] = "empty_validation_support"
            continue
        row["base_errors_px"] = np.concatenate(base_errors)
        row["candidate_errors_px"] = np.concatenate(candidate_errors)
        row["base"] = _summarize_reprojection_errors(
            row["base_errors_px"], inlier_px=float(reprojection_error_px)
        )
        row["candidate"] = _summarize_reprojection_errors(
            row["candidate_errors_px"], inlier_px=float(reprojection_error_px)
        )
        successful.append({"fold": validation_fold, "fit": fit, "row": row})

    required = folds // 2 + 1
    if len(successful) < required:
        return {
            "success": False,
            "reason": "crossfit_insufficient_successful_folds",
            "folds": fold_rows,
            "successful_fold_count": int(len(successful)),
            "required_successful_fold_count": int(required),
        }

    medoid_costs = {int(item["fold"]): 0.0 for item in successful}
    pairwise: list[dict[str, Any]] = []
    for first_index, first in enumerate(successful):
        for second in successful[first_index + 1 :]:
            distance = _crossfit_model_projection_distance(
                first=first["fit"]["cameras"],
                second=second["fit"]["cameras"],
                observations=prepared,
            )
            pairwise.append(
                {
                    "fold_a": int(first["fold"]),
                    "fold_b": int(second["fold"]),
                    "projection_median_px": float(distance),
                }
            )
            medoid_costs[int(first["fold"])] += float(distance)
            medoid_costs[int(second["fold"])] += float(distance)
    medoid_fold = min(medoid_costs, key=lambda fold: (medoid_costs[fold], fold))
    selected = next(item for item in successful if int(item["fold"]) == medoid_fold)
    base_errors = np.concatenate(
        [np.asarray(item["row"]["base_errors_px"]) for item in successful]
    )
    candidate_errors = np.concatenate(
        [np.asarray(item["row"]["candidate_errors_px"]) for item in successful]
    )
    selected_heldout = {
        "base_errors_px": np.asarray(selected["row"]["base_errors_px"]),
        "candidate_errors_px": np.asarray(
            selected["row"]["candidate_errors_px"]
        ),
        "base": selected["row"]["base"],
        "candidate": selected["row"]["candidate"],
        "validation_fold": int(medoid_fold),
    }
    return {
        "success": True,
        "reason": "ok",
        "cameras": selected["fit"]["cameras"],
        "center": selected["fit"]["center"],
        "selected_validation_fold": int(medoid_fold),
        "selected_model_heldout": selected_heldout,
        "successful_fold_count": int(len(successful)),
        "required_successful_fold_count": int(required),
        "folds": fold_rows,
        "aggregate_oof": {
            "base": _summarize_reprojection_errors(
                base_errors, inlier_px=float(reprojection_error_px)
            ),
            "candidate": _summarize_reprojection_errors(
                candidate_errors, inlier_px=float(reprojection_error_px)
            ),
            "base_errors_px": base_errors,
            "candidate_errors_px": candidate_errors,
        },
        "crossfit_consistency": {
            "policy": "fold_model_medoid_by_band_projection_median",
            "medoid_fold": int(medoid_fold),
            "medoid_cost_px": float(medoid_costs[medoid_fold]),
            "max_pairwise_projection_median_px": float(
                max(row["projection_median_px"] for row in pairwise)
            ),
            "pairwise": pairwise,
        },
        "fit_validation_split": "stable_track_id_splitmix64_modulo",
        "rotation_policy": "fixed_input_rotation_per_frame",
    }


def crossfit_temporal_joint_ptz_shared_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    temporal_rotations: Mapping[int, np.ndarray],
    temporal_track_observations: Sequence[Mapping[str, Any]] = (),
    reprojection_error_px: float = 4.0,
    temporal_error_px: float = 1.5,
    min_points_per_frame: int = 12,
    min_successful_frames: int = 3,
    num_folds: int = 3,
) -> dict[str, Any]:
    """Cross-fit temporal-track PTZ BA and score only held-out map IDs."""

    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    prepared = _prepare_observations(observations, initial_cameras)
    fit_minimum = max(
        4,
        int(np.ceil(int(min_points_per_frame) * (folds - 1) / folds)),
    )
    fold_rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    for validation_fold in range(folds):
        fit_rows, validation_rows, disjoint = _split_observations(
            prepared,
            num_folds=folds,
            validation_fold=validation_fold,
        )
        fit = solve_temporal_joint_ptz_shared_center(
            observations=fit_rows,
            initial_cameras=initial_cameras,
            temporal_rotations=temporal_rotations,
            temporal_track_observations=temporal_track_observations,
            reprojection_error_px=float(reprojection_error_px),
            temporal_error_px=float(temporal_error_px),
            min_points_per_frame=fit_minimum,
            min_successful_frames=int(min_successful_frames),
        )
        row: dict[str, Any] = {
            "validation_fold": int(validation_fold),
            "fit_success": bool(fit.get("success", False)),
            "fit_reason": str(fit.get("reason", "unknown")),
            "fit_validation_disjoint": bool(disjoint),
            "fit_rows": [
                {
                    "frame": int(item["frame"]),
                    "point_ids": np.asarray(item["point_ids"], dtype=np.int64),
                }
                for item in fit_rows
            ],
            "validation_rows": [
                {
                    "frame": int(item["frame"]),
                    "point_ids": np.asarray(item["point_ids"], dtype=np.int64),
                }
                for item in validation_rows
            ],
        }
        fold_rows.append(row)
        if not bool(fit.get("success", False)):
            continue

        base_errors: list[np.ndarray] = []
        candidate_errors: list[np.ndarray] = []
        for validation in validation_rows:
            frame = int(validation["frame"])
            if frame not in initial_cameras or frame not in fit["cameras"]:
                continue
            if len(np.asarray(validation["xyz"]).reshape(-1, 3)) == 0:
                continue
            base_errors.append(
                _camera_reprojection_errors(
                    camera=initial_cameras[frame],
                    xyz=validation["xyz"],
                    image_points=validation["image_points"],
                )
            )
            candidate_errors.append(
                _camera_reprojection_errors(
                    camera=fit["cameras"][frame],
                    xyz=validation["xyz"],
                    image_points=validation["image_points"],
                )
            )
        if not base_errors:
            row["fit_success"] = False
            row["fit_reason"] = "empty_validation_support"
            continue
        row["base_errors_px"] = np.concatenate(base_errors)
        row["candidate_errors_px"] = np.concatenate(candidate_errors)
        row["base"] = _summarize_reprojection_errors(
            row["base_errors_px"], inlier_px=float(reprojection_error_px)
        )
        row["candidate"] = _summarize_reprojection_errors(
            row["candidate_errors_px"], inlier_px=float(reprojection_error_px)
        )
        successful.append({"fold": validation_fold, "fit": fit, "row": row})

    required = folds // 2 + 1
    if len(successful) < required:
        return {
            "success": False,
            "reason": "crossfit_insufficient_successful_folds",
            "folds": fold_rows,
            "successful_fold_count": int(len(successful)),
            "required_successful_fold_count": int(required),
        }

    medoid_costs = {int(item["fold"]): 0.0 for item in successful}
    pairwise: list[dict[str, Any]] = []
    for first_index, first in enumerate(successful):
        for second in successful[first_index + 1 :]:
            distance = _crossfit_model_projection_distance(
                first=first["fit"]["cameras"],
                second=second["fit"]["cameras"],
                observations=prepared,
            )
            pairwise.append(
                {
                    "fold_a": int(first["fold"]),
                    "fold_b": int(second["fold"]),
                    "projection_median_px": float(distance),
                }
            )
            medoid_costs[int(first["fold"])] += float(distance)
            medoid_costs[int(second["fold"])] += float(distance)
    medoid_fold = min(medoid_costs, key=lambda fold: (medoid_costs[fold], fold))
    selected = next(item for item in successful if int(item["fold"]) == medoid_fold)
    base_errors = np.concatenate(
        [np.asarray(item["row"]["base_errors_px"]) for item in successful]
    )
    candidate_errors = np.concatenate(
        [np.asarray(item["row"]["candidate_errors_px"]) for item in successful]
    )
    return {
        "success": True,
        "reason": "ok",
        "cameras": selected["fit"]["cameras"],
        "center": selected["fit"]["center"],
        "selected_validation_fold": int(medoid_fold),
        "selected_model_heldout": {
            "base_errors_px": np.asarray(selected["row"]["base_errors_px"]),
            "candidate_errors_px": np.asarray(
                selected["row"]["candidate_errors_px"]
            ),
            "base": selected["row"]["base"],
            "candidate": selected["row"]["candidate"],
            "validation_fold": int(medoid_fold),
        },
        "successful_fold_count": int(len(successful)),
        "required_successful_fold_count": int(required),
        "folds": fold_rows,
        "aggregate_oof": {
            "base": _summarize_reprojection_errors(
                base_errors, inlier_px=float(reprojection_error_px)
            ),
            "candidate": _summarize_reprojection_errors(
                candidate_errors, inlier_px=float(reprojection_error_px)
            ),
            "base_errors_px": base_errors,
            "candidate_errors_px": candidate_errors,
        },
        "crossfit_consistency": {
            "policy": "fold_model_medoid_by_band_projection_median",
            "medoid_fold": int(medoid_fold),
            "medoid_cost_px": float(medoid_costs[medoid_fold]),
            "max_pairwise_projection_median_px": float(
                max(row["projection_median_px"] for row in pairwise)
            ),
            "pairwise": pairwise,
        },
        "fit_validation_split": "stable_track_id_splitmix64_modulo",
        "temporal_track_policy": "fit_only_not_heldout_score",
    }


def crossfit_joint_ptz_shared_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    reprojection_error_px: float = 4.0,
    min_inliers_per_frame: int = 12,
    min_successful_frames: int = 3,
    num_folds: int = 3,
    rng_seed: int = 0,
    allow_initial_camera_bootstrap: bool = False,
    allow_joint_refined_rotation_fallback: bool = False,
) -> dict[str, Any]:
    """Fit shared-center PTZ bundles and score only on held-out track IDs."""

    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    prepared = _prepare_observations(observations, initial_cameras)
    fit_minimum = max(
        4,
        int(np.ceil(int(min_inliers_per_frame) * (folds - 1) / folds)),
    )
    fold_rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    for validation_fold in range(folds):
        fit_rows, validation_rows, disjoint = _split_observations(
            prepared,
            num_folds=folds,
            validation_fold=validation_fold,
        )
        fit = solve_joint_ptz_shared_center(
            observations=fit_rows,
            initial_cameras=initial_cameras,
            reprojection_error_px=float(reprojection_error_px),
            min_inliers_per_frame=fit_minimum,
            min_successful_frames=int(min_successful_frames),
            rng_seed=int(rng_seed) + 1009 * validation_fold,
            allow_initial_camera_bootstrap=bool(
                allow_initial_camera_bootstrap
            ),
            allow_joint_refined_rotation_fallback=bool(
                allow_joint_refined_rotation_fallback
            ),
        )
        row: dict[str, Any] = {
            "validation_fold": int(validation_fold),
            "fit_success": bool(fit.get("success", False)),
            "fit_reason": str(fit.get("reason", "unknown")),
            "fit_validation_disjoint": bool(disjoint),
            "fit_rows": [
                {
                    "frame": int(item["frame"]),
                    "point_ids": np.asarray(item["point_ids"], dtype=np.int64),
                }
                for item in fit_rows
            ],
            "validation_rows": [
                {
                    "frame": int(item["frame"]),
                    "point_ids": np.asarray(item["point_ids"], dtype=np.int64),
                }
                for item in validation_rows
            ],
        }
        if bool(allow_initial_camera_bootstrap):
            row.update(
                {
                    "fit_bootstrap_policy": str(
                        fit.get("bootstrap_policy", "not_reached")
                    ),
                    "fit_bootstrap_pnp_frame_count": int(
                        fit.get("bootstrap_pnp_frame_count", 0)
                    ),
                    "fit_initial_camera_bootstrap_frame_count": int(
                        fit.get("initial_camera_bootstrap_frame_count", 0)
                    ),
                    "fit_initial_camera_bootstrap_frames": list(
                        fit.get("initial_camera_bootstrap_frames", [])
                    ),
                }
            )
        if bool(allow_joint_refined_rotation_fallback):
            row.update(
                {
                    "fit_final_rotation_policy": str(
                        fit.get("final_rotation_policy", "not_reached")
                    ),
                    "fit_independent_rotation_frame_count": int(
                        fit.get("independent_rotation_frame_count", 0)
                    ),
                    "fit_independent_rotation_frames": list(
                        fit.get("independent_rotation_frames", [])
                    ),
                    "fit_joint_refined_rotation_fallback_frame_count": int(
                        fit.get(
                            "joint_refined_rotation_fallback_frame_count", 0
                        )
                    ),
                    "fit_joint_refined_rotation_fallback_frames": list(
                        fit.get("joint_refined_rotation_fallback_frames", [])
                    ),
                    "fit_final_rotation_supported_frame_count": int(
                        fit.get("final_rotation_supported_frame_count", 0)
                    ),
                }
            )
        fold_rows.append(row)
        if not bool(fit.get("success", False)):
            continue
        base_errors: list[np.ndarray] = []
        candidate_errors: list[np.ndarray] = []
        for validation in validation_rows:
            frame = int(validation["frame"])
            if frame not in initial_cameras or frame not in fit["cameras"]:
                continue
            if len(np.asarray(validation["xyz"]).reshape(-1, 3)) == 0:
                continue
            base_errors.append(
                _camera_reprojection_errors(
                    camera=initial_cameras[frame],
                    xyz=validation["xyz"],
                    image_points=validation["image_points"],
                )
            )
            candidate_errors.append(
                _camera_reprojection_errors(
                    camera=fit["cameras"][frame],
                    xyz=validation["xyz"],
                    image_points=validation["image_points"],
                )
            )
        if not base_errors:
            row["fit_success"] = False
            row["fit_reason"] = "empty_validation_support"
            continue
        row["base_errors_px"] = np.concatenate(base_errors)
        row["candidate_errors_px"] = np.concatenate(candidate_errors)
        row["base"] = _summarize_reprojection_errors(
            row["base_errors_px"], inlier_px=float(reprojection_error_px)
        )
        row["candidate"] = _summarize_reprojection_errors(
            row["candidate_errors_px"], inlier_px=float(reprojection_error_px)
        )
        successful.append({"fold": validation_fold, "fit": fit, "row": row})

    required = folds // 2 + 1
    if len(successful) < required:
        return {
            "success": False,
            "reason": "crossfit_insufficient_successful_folds",
            "folds": fold_rows,
            "successful_fold_count": int(len(successful)),
            "required_successful_fold_count": int(required),
            **(
                {"initial_camera_bootstrap_opt_in": True}
                if bool(allow_initial_camera_bootstrap)
                else {}
            ),
            **(
                {"joint_refined_rotation_fallback_opt_in": True}
                if bool(allow_joint_refined_rotation_fallback)
                else {}
            ),
        }

    medoid_costs = {int(item["fold"]): 0.0 for item in successful}
    pairwise: list[dict[str, Any]] = []
    for first_index, first in enumerate(successful):
        for second in successful[first_index + 1 :]:
            distance = _crossfit_model_projection_distance(
                first=first["fit"]["cameras"],
                second=second["fit"]["cameras"],
                observations=prepared,
            )
            pairwise.append(
                {
                    "fold_a": int(first["fold"]),
                    "fold_b": int(second["fold"]),
                    "projection_median_px": float(distance),
                }
            )
            medoid_costs[int(first["fold"])] += float(distance)
            medoid_costs[int(second["fold"])] += float(distance)
    medoid_fold = min(medoid_costs, key=lambda fold: (medoid_costs[fold], fold))
    selected = next(item for item in successful if int(item["fold"]) == medoid_fold)
    base_errors = np.concatenate(
        [np.asarray(item["row"]["base_errors_px"]) for item in successful]
    )
    candidate_errors = np.concatenate(
        [np.asarray(item["row"]["candidate_errors_px"]) for item in successful]
    )
    aggregate = {
        "base": _summarize_reprojection_errors(
            base_errors, inlier_px=float(reprojection_error_px)
        ),
        "candidate": _summarize_reprojection_errors(
            candidate_errors, inlier_px=float(reprojection_error_px)
        ),
        "base_errors_px": base_errors,
        "candidate_errors_px": candidate_errors,
    }
    return {
        "success": True,
        "reason": "ok",
        "cameras": selected["fit"]["cameras"],
        "center": selected["fit"]["center"],
        "selected_validation_fold": int(medoid_fold),
        "successful_fold_count": int(len(successful)),
        "required_successful_fold_count": int(required),
        "folds": fold_rows,
        "aggregate_oof": aggregate,
        "crossfit_consistency": {
            "policy": "fold_model_medoid_by_band_projection_median",
            "medoid_fold": int(medoid_fold),
            "medoid_cost_px": float(medoid_costs[medoid_fold]),
            "max_pairwise_projection_median_px": float(
                max(row["projection_median_px"] for row in pairwise)
            ),
            "pairwise": pairwise,
        },
        "fit_validation_split": "stable_track_id_splitmix64_modulo",
        **(
            {"initial_camera_bootstrap_opt_in": True}
            if bool(allow_initial_camera_bootstrap)
            else {}
        ),
        **(
            {"joint_refined_rotation_fallback_opt_in": True}
            if bool(allow_joint_refined_rotation_fallback)
            else {}
        ),
    }


def decide_joint_ptz_band_commit(
    result: Mapping[str, Any],
    *,
    focus_inlier_px: float,
) -> dict[str, Any]:
    """Commit only when held-out band geometry improves without regression."""

    threshold = float(focus_inlier_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    if not bool(result.get("success", False)):
        return {"commit": False, "reason": "crossfit_band_failed"}
    consistency = float(
        result.get("crossfit_consistency", {}).get(
            "max_pairwise_projection_median_px", float("inf")
        )
    )
    if not np.isfinite(consistency) or consistency > threshold:
        return {"commit": False, "reason": "crossfit_band_inconsistent"}
    aggregate = result.get("aggregate_oof", {})
    base = aggregate.get("base", {})
    candidate = aggregate.get("candidate", {})
    names = ("mean_px", "median_px", "p90_px", "inlier_ratio")
    try:
        base_values = {name: float(base[name]) for name in names}
        candidate_values = {name: float(candidate[name]) for name in names}
    except (KeyError, TypeError, ValueError):
        return {"commit": False, "reason": "heldout_band_score_unavailable"}
    if not all(
        np.isfinite(value)
        for value in [*base_values.values(), *candidate_values.values()]
    ):
        return {"commit": False, "reason": "heldout_band_score_unavailable"}
    pareto = (
        candidate_values["median_px"] < base_values["median_px"]
        and candidate_values["mean_px"] <= base_values["mean_px"]
        and candidate_values["p90_px"] <= base_values["p90_px"]
        and candidate_values["inlier_ratio"] >= base_values["inlier_ratio"]
    )
    return {
        "commit": bool(pareto),
        "reason": (
            "heldout_band_pareto_improved"
            if pareto
            else "heldout_band_not_pareto"
        ),
    }


def decide_joint_ptz_band_model_commit(
    result: Mapping[str, Any],
    *,
    focus_inlier_px: float,
    frame_count: int,
) -> dict[str, Any]:
    """Commit only when the fixed candidate lowers held-out robust cost."""

    threshold = float(focus_inlier_px)
    frames = int(frame_count)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    if frames < 2:
        raise ValueError("frame_count must be at least two")
    if not bool(result.get("success", False)):
        return {"commit": False, "reason": "crossfit_band_failed"}
    consistency = float(
        result.get("crossfit_consistency", {}).get(
            "max_pairwise_projection_median_px", float("inf")
        )
    )
    if not np.isfinite(consistency) or consistency > threshold:
        return {"commit": False, "reason": "crossfit_band_inconsistent"}
    selected_fold = result.get("selected_validation_fold")
    if selected_fold is None:
        score_row = result.get("aggregate_oof", {})
        score_source = "aggregate_oof_legacy"
    else:
        selected_rows = [
            row
            for row in result.get("folds", [])
            if int(row.get("validation_fold", -1)) == int(selected_fold)
            and bool(row.get("fit_success", False))
        ]
        if len(selected_rows) != 1:
            return {
                "commit": False,
                "reason": "selected_model_heldout_score_unavailable",
            }
        score_row = selected_rows[0]
        score_source = "selected_model_heldout_fold"
    base = np.asarray(score_row.get("base_errors_px", []), dtype=np.float64).reshape(
        -1
    )
    candidate = np.asarray(
        score_row.get("candidate_errors_px", []), dtype=np.float64
    ).reshape(-1)
    if (
        len(base) == 0
        or base.shape != candidate.shape
        or np.any(base[np.isfinite(base)] < 0.0)
        or np.any(candidate[np.isfinite(candidate)] < 0.0)
    ):
        return {"commit": False, "reason": "heldout_band_score_unavailable"}

    base_finite = np.isfinite(base)
    candidate_finite = np.isfinite(candidate)
    comparable = base_finite | candidate_finite
    if not np.any(comparable):
        return {"commit": False, "reason": "heldout_band_score_unavailable"}
    jointly_invalid = ~base_finite & ~candidate_finite
    base_only_invalid = ~base_finite & candidate_finite
    candidate_only_invalid = base_finite & ~candidate_finite
    base = np.where(base_finite, base, threshold)[comparable]
    candidate = np.where(candidate_finite, candidate, threshold)[comparable]

    def robust_cost(errors: np.ndarray) -> float:
        return float(np.mean(np.minimum(errors, threshold) ** 2))

    base_cost = robust_cost(base)
    candidate_cost = robust_cost(candidate)
    improved = candidate_cost < base_cost
    return {
        "commit": bool(improved),
        "reason": (
            "heldout_robust_cost_improved"
            if improved
            else "heldout_robust_cost_not_improved"
        ),
        "residual_count": int(len(base)),
        "raw_residual_count": int(len(comparable)),
        "jointly_invalid_residual_count": int(np.count_nonzero(jointly_invalid)),
        "base_only_invalid_residual_count": int(
            np.count_nonzero(base_only_invalid)
        ),
        "candidate_only_invalid_residual_count": int(
            np.count_nonzero(candidate_only_invalid)
        ),
        "invalid_residual_policy": (
            "drop_jointly_invalid_and_cap_one_sided_invalid"
        ),
        "frame_count": int(frames),
        "base_robust_cost": float(base_cost),
        "candidate_robust_cost": float(candidate_cost),
        "robust_cost_gain": float(base_cost - candidate_cost),
        "max_crossfit_projection_median_px": float(consistency),
        "score_source": score_source,
        **(
            {"selected_validation_fold": int(selected_fold)}
            if selected_fold is not None
            else {}
        ),
    }


def decide_selected_joint_ptz_frame_pareto_commit(
    result: Mapping[str, Any],
    *,
    focus_inlier_px: float,
    min_inliers_per_frame: int,
) -> dict[str, Any]:
    """Verify one selected shared-center model on its untouched track fold."""

    threshold = float(focus_inlier_px)
    minimum = int(min_inliers_per_frame)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    if minimum < 1:
        raise ValueError("min_inliers_per_frame must be positive")
    if not bool(result.get("success", False)):
        return {"commit": False, "reason": "crossfit_band_failed"}
    consistency = float(
        result.get("crossfit_consistency", {}).get(
            "max_pairwise_projection_median_px", float("inf")
        )
    )
    if not np.isfinite(consistency) or consistency > threshold:
        return {"commit": False, "reason": "crossfit_band_inconsistent"}

    selected_fold = result.get("selected_validation_fold")
    selected_rows = [
        row
        for row in result.get("folds", [])
        if selected_fold is not None
        and int(row.get("validation_fold", -1)) == int(selected_fold)
        and bool(row.get("fit_success", False))
    ]
    if len(selected_rows) != 1:
        return {
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
        }
    selected = selected_rows[0]
    base = np.asarray(selected.get("base_errors_px", []), dtype=np.float64).reshape(
        -1
    )
    candidate = np.asarray(
        selected.get("candidate_errors_px", []), dtype=np.float64
    ).reshape(-1)
    validation_rows = list(selected.get("validation_rows", []))
    expected = sum(
        len(np.asarray(row.get("point_ids", []), dtype=np.int64).reshape(-1))
        for row in validation_rows
    )
    if len(base) == 0 or base.shape != candidate.shape or len(base) != expected:
        return {
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
        }

    frame_rows: list[dict[str, Any]] = []
    support_failed: list[int] = []
    regressed: list[int] = []
    improved: list[int] = []
    offset = 0
    for validation in validation_rows:
        frame = int(validation["frame"])
        count = len(
            np.asarray(validation.get("point_ids", []), dtype=np.int64).reshape(-1)
        )
        base_frame = base[offset : offset + count]
        candidate_frame = candidate[offset : offset + count]
        offset += count
        base_finite = np.isfinite(base_frame)
        candidate_finite = np.isfinite(candidate_frame)
        comparable = base_finite | candidate_finite
        base_positive = int(np.count_nonzero(base_finite))
        candidate_positive = int(np.count_nonzero(candidate_finite))
        base_inliers = int(
            np.count_nonzero(base_finite & (base_frame <= threshold))
        )
        candidate_inliers = int(
            np.count_nonzero(candidate_finite & (candidate_frame <= threshold))
        )
        if np.any(comparable):
            base_capped = np.minimum(
                np.where(base_finite, base_frame, threshold)[comparable],
                threshold,
            )
            candidate_capped = np.minimum(
                np.where(candidate_finite, candidate_frame, threshold)[comparable],
                threshold,
            )
            base_cost = float(np.mean(base_capped**2))
            candidate_cost = float(np.mean(candidate_capped**2))
        else:
            base_cost = float("inf")
            candidate_cost = float("inf")
        if np.isfinite(base_cost) and np.isfinite(candidate_cost):
            scale = max(1.0, abs(base_cost), abs(candidate_cost))
            tolerance = 64.0 * np.finfo(np.float64).eps * scale
            cost_nonregression = bool(candidate_cost <= base_cost + tolerance)
            strict_cost_improvement = bool(
                candidate_cost < base_cost - tolerance
            )
        else:
            tolerance = 0.0
            cost_nonregression = bool(candidate_cost <= base_cost)
            strict_cost_improvement = False
        absolute_support = bool(
            candidate_positive >= base_positive
            and candidate_inliers >= base_inliers
            and candidate_inliers >= minimum
        )
        strict = bool(
            candidate_positive > base_positive
            or candidate_inliers > base_inliers
            or strict_cost_improvement
        )
        if not absolute_support:
            support_failed.append(frame)
        elif not cost_nonregression:
            regressed.append(frame)
        elif strict:
            improved.append(frame)
        frame_rows.append(
            {
                "frame": frame,
                "heldout_point_count": int(count),
                "base_positive_depth_count": base_positive,
                "candidate_positive_depth_count": candidate_positive,
                "base_inlier_count": base_inliers,
                "candidate_inlier_count": candidate_inliers,
                "base_truncated_cost": base_cost,
                "candidate_truncated_cost": candidate_cost,
                "absolute_support_nonregression": absolute_support,
                "truncated_cost_nonregression": cost_nonregression,
                "strictly_improved": strict,
            }
        )
    if support_failed:
        reason = "heldout_frame_absolute_support_failed"
    elif regressed:
        reason = "heldout_frame_robust_cost_regressed"
    elif not improved:
        reason = "heldout_band_not_strictly_improved"
    else:
        reason = "heldout_every_frame_nonregressing_band_improved"
    return {
        "commit": bool(not support_failed and not regressed and improved),
        "reason": reason,
        "selected_validation_fold": int(selected_fold),
        "max_crossfit_projection_median_px": consistency,
        "focus_inlier_px": threshold,
        "min_inliers_per_frame": minimum,
        "support_failed_frames": support_failed,
        "regressed_frames": regressed,
        "strictly_improved_frames": improved,
        "frames": frame_rows,
    }


def decide_selected_fixed_center_temporal_band_commit(
    result: Mapping[str, Any],
    *,
    focus_inlier_px: float,
    min_inliers_per_frame: int,
) -> dict[str, Any]:
    """Atomically commit a fixed-center band only when heldout geometry is safe."""

    frame_decision = decide_selected_joint_ptz_frame_pareto_commit(
        result,
        focus_inlier_px=float(focus_inlier_px),
        min_inliers_per_frame=int(min_inliers_per_frame),
    )
    if not bool(result.get("success", False)):
        return {**frame_decision, "band_pareto": {}}
    selected_fold = result.get("selected_validation_fold")
    rows = [
        row
        for row in result.get("folds", [])
        if selected_fold is not None
        and int(row.get("validation_fold", -1)) == int(selected_fold)
        and bool(row.get("fit_success", False))
    ]
    if len(rows) != 1:
        return {
            **frame_decision,
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
            "band_pareto": {},
        }
    base = np.asarray(rows[0].get("base_errors_px", []), dtype=np.float64).reshape(-1)
    candidate = np.asarray(
        rows[0].get("candidate_errors_px", []), dtype=np.float64
    ).reshape(-1)
    if len(base) == 0 or base.shape != candidate.shape:
        return {
            **frame_decision,
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
            "band_pareto": {},
        }

    threshold = float(focus_inlier_px)
    base_finite = np.isfinite(base)
    candidate_finite = np.isfinite(candidate)
    comparable = base_finite | candidate_finite
    base_capped = np.minimum(
        np.where(base_finite, base, threshold)[comparable], threshold
    )
    candidate_capped = np.minimum(
        np.where(candidate_finite, candidate, threshold)[comparable], threshold
    )
    if len(base_capped) == 0:
        return {
            **frame_decision,
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
            "band_pareto": {},
        }

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "clipped_mean_px": float(np.mean(values)),
            "clipped_median_px": float(np.median(values)),
            "clipped_p90_px": float(np.percentile(values, 90.0)),
        }

    base_summary = summarize(base_capped)
    candidate_summary = summarize(candidate_capped)
    base_positive = int(np.count_nonzero(base_finite))
    candidate_positive = int(np.count_nonzero(candidate_finite))
    base_inliers = int(np.count_nonzero(base_finite & (base <= threshold)))
    candidate_inliers = int(
        np.count_nonzero(candidate_finite & (candidate <= threshold))
    )
    checks = {
        "clipped_mean_not_worse": (
            candidate_summary["clipped_mean_px"]
            <= base_summary["clipped_mean_px"]
        ),
        "clipped_median_not_worse": (
            candidate_summary["clipped_median_px"]
            <= base_summary["clipped_median_px"]
        ),
        "clipped_p90_not_worse": (
            candidate_summary["clipped_p90_px"]
            <= base_summary["clipped_p90_px"]
        ),
        "positive_depth_support_not_lower": candidate_positive >= base_positive,
        "inlier_support_not_lower": candidate_inliers >= base_inliers,
    }
    strict = bool(
        candidate_summary["clipped_mean_px"] < base_summary["clipped_mean_px"]
        or candidate_summary["clipped_median_px"]
        < base_summary["clipped_median_px"]
        or candidate_summary["clipped_p90_px"] < base_summary["clipped_p90_px"]
        or candidate_positive > base_positive
        or candidate_inliers > base_inliers
    )
    band_safe = bool(all(checks.values()) and strict)
    commit = bool(frame_decision.get("commit", False) and band_safe)
    if not bool(frame_decision.get("commit", False)):
        reason = str(frame_decision.get("reason", "heldout_frame_check_failed"))
    elif not band_safe:
        reason = "selected_heldout_band_not_pareto"
    else:
        reason = "selected_heldout_temporal_band_pareto_improved"
    return {
        **frame_decision,
        "commit": commit,
        "reason": reason,
        "band_pareto": {
            "base": base_summary,
            "candidate": candidate_summary,
            "base_positive_depth_count": base_positive,
            "candidate_positive_depth_count": candidate_positive,
            "base_inlier_count": base_inliers,
            "candidate_inlier_count": candidate_inliers,
            "checks": checks,
            "strictly_improved": strict,
        },
    }


def decide_joint_ptz_frame_cluster_confidence_commit(
    result: Mapping[str, Any],
    *,
    focus_inlier_px: float,
    frame_count: int,
    expected_frames: Sequence[int],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Commit the selected cross-fit model only with frame-level evidence."""

    threshold = float(focus_inlier_px)
    frames = int(frame_count)
    confidence = float(confidence_level)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    if frames < 2:
        raise ValueError("frame_count must be at least two")
    expected_frame_ids = [int(frame) for frame in expected_frames]
    if len(expected_frame_ids) != frames or len(set(expected_frame_ids)) != frames:
        raise ValueError("expected_frames must contain frame_count unique IDs")
    if not np.isfinite(confidence) or not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between 0.5 and 1")
    if not bool(result.get("success", False)):
        return {"commit": False, "reason": "crossfit_band_failed"}

    consistency = float(
        result.get("crossfit_consistency", {}).get(
            "max_pairwise_projection_median_px", float("inf")
        )
    )
    if not np.isfinite(consistency) or consistency > threshold:
        return {"commit": False, "reason": "crossfit_band_inconsistent"}

    selected_fold = result.get("selected_validation_fold")
    if selected_fold is None:
        return {
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
        }
    medoid_fold = result.get("crossfit_consistency", {}).get("medoid_fold")
    if medoid_fold is None or int(selected_fold) != int(medoid_fold):
        return {
            "commit": False,
            "reason": "selected_model_medoid_mismatch",
        }
    selected_rows = [
        row
        for row in result.get("folds", [])
        if int(row.get("validation_fold", -1)) == int(selected_fold)
        and bool(row.get("fit_success", False))
    ]
    if len(selected_rows) != 1:
        return {
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
        }
    score_row = selected_rows[0]
    base_raw = np.asarray(
        score_row.get("base_errors_px", []), dtype=np.float64
    ).reshape(-1)
    candidate_raw = np.asarray(
        score_row.get("candidate_errors_px", []), dtype=np.float64
    ).reshape(-1)
    if (
        len(base_raw) == 0
        or base_raw.shape != candidate_raw.shape
        or np.any(base_raw[np.isfinite(base_raw)] < 0.0)
        or np.any(candidate_raw[np.isfinite(candidate_raw)] < 0.0)
    ):
        return {"commit": False, "reason": "heldout_band_score_unavailable"}

    validation_rows = list(score_row.get("validation_rows", []))
    if len(validation_rows) != frames:
        return {
            "commit": False,
            "reason": "incomplete_frame_cluster_score",
            "frame_cluster_count": int(len(validation_rows)),
            "frame_count": frames,
        }
    row_frames = [int(row["frame"]) for row in validation_rows]
    if len(set(row_frames)) != frames:
        return {
            "commit": False,
            "reason": "incomplete_frame_cluster_score",
            "frame_cluster_count": int(len(set(row_frames))),
            "frame_count": frames,
        }
    if set(row_frames) != set(expected_frame_ids):
        return {
            "commit": False,
            "reason": "frame_cluster_identity_mismatch",
            "scored_frames": sorted(set(row_frames)),
            "expected_frames": sorted(expected_frame_ids),
        }

    frame_gain_rows: list[tuple[int, float]] = []
    pooled_base: list[np.ndarray] = []
    pooled_candidate: list[np.ndarray] = []
    offset = 0
    jointly_invalid_count = 0
    base_only_invalid_count = 0
    candidate_only_invalid_count = 0
    for row in validation_rows:
        count = len(np.asarray(row.get("point_ids", [])).reshape(-1))
        stop = offset + count
        if count <= 0 or stop > len(base_raw):
            return {
                "commit": False,
                "reason": "incomplete_frame_cluster_score",
                "frame_cluster_count": int(len(frame_gain_rows)),
                "frame_count": frames,
            }
        base_slice = base_raw[offset:stop]
        candidate_slice = candidate_raw[offset:stop]
        offset = stop
        base_finite = np.isfinite(base_slice)
        candidate_finite = np.isfinite(candidate_slice)
        comparable = base_finite | candidate_finite
        jointly_invalid_count += int(np.count_nonzero(~base_finite & ~candidate_finite))
        base_only_invalid_count += int(np.count_nonzero(~base_finite & candidate_finite))
        candidate_only_invalid_count += int(
            np.count_nonzero(base_finite & ~candidate_finite)
        )
        if not np.any(comparable):
            return {
                "commit": False,
                "reason": "incomplete_frame_cluster_score",
                "frame_cluster_count": int(len(frame_gain_rows)),
                "frame_count": frames,
            }
        base = np.where(base_finite, base_slice, threshold)[comparable]
        candidate = np.where(candidate_finite, candidate_slice, threshold)[
            comparable
        ]
        base_costs = np.minimum(base, threshold) ** 2
        candidate_costs = np.minimum(candidate, threshold) ** 2
        frame_gain_rows.append(
            (int(row["frame"]), float(np.mean(base_costs - candidate_costs)))
        )
        pooled_base.append(base_costs)
        pooled_candidate.append(candidate_costs)
    if offset != len(base_raw):
        return {
            "commit": False,
            "reason": "incomplete_frame_cluster_score",
            "frame_cluster_count": int(len(frame_gain_rows)),
            "frame_count": frames,
        }

    gain_by_frame = {frame: gain for frame, gain in frame_gain_rows}
    ordered_frames = list(expected_frame_ids)
    gains = np.asarray(
        [gain_by_frame[frame] for frame in ordered_frames], dtype=np.float64
    )
    mean_gain = float(np.mean(gains))
    bandwidth = min(
        len(gains) - 1,
        int(np.floor(4.0 * (len(gains) / 100.0) ** (2.0 / 9.0))),
    )
    centered_gains = gains - mean_gain
    long_run_variance = float(np.mean(centered_gains**2))
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        autocovariance = float(
            np.sum(centered_gains[lag:] * centered_gains[:-lag]) / len(gains)
        )
        long_run_variance += 2.0 * weight * autocovariance
    finite_sample_correction = float(len(gains) / (len(gains) - 1))
    corrected_long_run_variance = max(
        0.0, finite_sample_correction * long_run_variance
    )
    standard_error = float(
        np.sqrt(corrected_long_run_variance / len(gains))
    )
    critical_value = float(
        student_t_distribution.ppf(confidence, df=len(gains) - 1)
    )
    lower_bound = float(mean_gain - critical_value * standard_error)
    commit = bool(lower_bound > 0.0)
    base_cost = float(np.mean(np.concatenate(pooled_base)))
    candidate_cost = float(np.mean(np.concatenate(pooled_candidate)))
    return {
        "commit": commit,
        "reason": (
            "selected_model_frame_hac_confidence_positive"
            if commit
            else "selected_model_frame_hac_confidence_not_positive"
        ),
        "score_source": "selected_model_heldout_fold",
        "selected_validation_fold": int(selected_fold),
        "confidence_level": confidence,
        "frame_count": frames,
        "expected_frames": sorted(expected_frame_ids),
        "scored_frame_order": ordered_frames,
        "frame_cluster_count": int(len(gains)),
        "frame_positive_count": int(np.count_nonzero(gains > 0.0)),
        "frame_robust_cost_gains": gains.tolist(),
        "frame_robust_cost_gain_mean": mean_gain,
        "frame_robust_cost_gain_standard_error": standard_error,
        "frame_robust_cost_gain_lower_bound": lower_bound,
        "variance_estimator": "bartlett_newey_west",
        "hac_kernel": "bartlett",
        "hac_bandwidth": bandwidth,
        "hac_bandwidth_rule": "floor(4*(n/100)^(2/9))",
        "hac_long_run_variance": long_run_variance,
        "hac_corrected_long_run_variance": corrected_long_run_variance,
        "finite_sample_correction": finite_sample_correction,
        "student_t_critical_value": critical_value,
        "pooled_residual_count": int(sum(len(values) for values in pooled_base)),
        "base_robust_cost": base_cost,
        "candidate_robust_cost": candidate_cost,
        "pooled_robust_cost_gain": float(base_cost - candidate_cost),
        "jointly_invalid_residual_count": jointly_invalid_count,
        "base_only_invalid_residual_count": base_only_invalid_count,
        "candidate_only_invalid_residual_count": candidate_only_invalid_count,
        "invalid_residual_policy": (
            "drop_jointly_invalid_and_cap_one_sided_invalid"
        ),
        "max_crossfit_projection_median_px": consistency,
    }


def decide_joint_ptz_frame_hac_pareto_commit(
    result: Mapping[str, Any],
    *,
    focus_inlier_px: float,
    expected_frames: Sequence[int],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Accept a component only with HAC-positive gain and pooled Pareto safety."""

    frames = [int(frame) for frame in expected_frames]
    confidence = decide_joint_ptz_frame_cluster_confidence_commit(
        result,
        focus_inlier_px=float(focus_inlier_px),
        frame_count=len(frames),
        expected_frames=frames,
        confidence_level=float(confidence_level),
    )
    if "base_robust_cost" not in confidence:
        return confidence

    selected_fold = int(confidence["selected_validation_fold"])
    selected_rows = [
        row
        for row in result.get("folds", [])
        if int(row.get("validation_fold", -1)) == selected_fold
        and bool(row.get("fit_success", False))
    ]
    if len(selected_rows) != 1:
        return {
            **confidence,
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
        }
    row = selected_rows[0]
    base_raw = np.asarray(row.get("base_errors_px", []), dtype=np.float64).reshape(-1)
    candidate_raw = np.asarray(
        row.get("candidate_errors_px", []), dtype=np.float64
    ).reshape(-1)
    threshold = float(focus_inlier_px)
    base_finite = np.isfinite(base_raw)
    candidate_finite = np.isfinite(candidate_raw)
    comparable = base_finite | candidate_finite
    if not np.any(comparable) or base_raw.shape != candidate_raw.shape:
        return {
            **confidence,
            "commit": False,
            "reason": "selected_model_heldout_score_unavailable",
        }
    base = np.minimum(
        np.where(base_finite, base_raw, threshold)[comparable], threshold
    )
    candidate = np.minimum(
        np.where(candidate_finite, candidate_raw, threshold)[comparable], threshold
    )
    base_positive = int(np.count_nonzero(base_finite))
    candidate_positive = int(np.count_nonzero(candidate_finite))
    base_inliers = int(np.count_nonzero(base_finite & (base_raw <= threshold)))
    candidate_inliers = int(
        np.count_nonzero(candidate_finite & (candidate_raw <= threshold))
    )
    checks = {
        "hac_gain_lower_bound_positive": bool(confidence.get("commit", False)),
        "clipped_median_not_worse": bool(
            np.median(candidate) <= np.median(base)
        ),
        "clipped_p90_not_worse": bool(
            np.percentile(candidate, 90.0) <= np.percentile(base, 90.0)
        ),
        "clipped_worst_not_worse": bool(np.max(candidate) <= np.max(base)),
        "positive_depth_support_not_lower": candidate_positive >= base_positive,
        "inlier_support_not_lower": candidate_inliers >= base_inliers,
    }
    commit = bool(all(checks.values()))
    return {
        **confidence,
        "commit": commit,
        "reason": (
            "selected_model_frame_hac_and_pooled_pareto_improved"
            if commit
            else "selected_model_frame_hac_or_pooled_pareto_failed"
        ),
        "component_pareto": {
            "checks": checks,
            "base_clipped_median_px": float(np.median(base)),
            "candidate_clipped_median_px": float(np.median(candidate)),
            "base_clipped_p90_px": float(np.percentile(base, 90.0)),
            "candidate_clipped_p90_px": float(np.percentile(candidate, 90.0)),
            "base_clipped_worst_px": float(np.max(base)),
            "candidate_clipped_worst_px": float(np.max(candidate)),
            "base_positive_depth_count": base_positive,
            "candidate_positive_depth_count": candidate_positive,
            "base_inlier_count": base_inliers,
            "candidate_inlier_count": candidate_inliers,
        },
    }


def decide_joint_ptz_pooled_band_model_commit(
    results: Sequence[Mapping[str, Any]],
    *,
    focus_inlier_px: float,
    frame_count: int,
) -> dict[str, Any]:
    """Commit an angular band when its equal-member held-out cost improves."""

    members = list(results)
    if len(members) < 2:
        raise ValueError("pooled band must contain at least two members")

    scores: list[dict[str, Any]] = []
    for index, result in enumerate(members):
        score = decide_joint_ptz_band_model_commit(
            result,
            focus_inlier_px=focus_inlier_px,
            frame_count=frame_count,
        )
        if "base_robust_cost" not in score:
            reason = str(score.get("reason", "score_unavailable")).removeprefix(
                "crossfit_band_"
            )
            return {
                "commit": False,
                "reason": f"pooled_band_member_{index}_crossfit_{reason}",
            }
        scores.append(score)

    base_cost = float(np.mean([score["base_robust_cost"] for score in scores]))
    candidate_cost = float(
        np.mean([score["candidate_robust_cost"] for score in scores])
    )
    gain = base_cost - candidate_cost
    improved = candidate_cost < base_cost
    return {
        "commit": bool(improved),
        "reason": (
            "pooled_heldout_robust_cost_improved"
            if improved
            else "pooled_heldout_robust_cost_not_improved"
        ),
        "member_count": int(len(scores)),
        "pooling": "equal_member_mean",
        "base_robust_cost": base_cost,
        "candidate_robust_cost": candidate_cost,
        "pooled_robust_cost_gain": float(gain),
        "member_robust_cost_gains": [
            float(score["robust_cost_gain"]) for score in scores
        ],
        "residual_count": int(sum(score["residual_count"] for score in scores)),
        "raw_residual_count": int(
            sum(score["raw_residual_count"] for score in scores)
        ),
        "jointly_invalid_residual_count": int(
            sum(score["jointly_invalid_residual_count"] for score in scores)
        ),
        "base_only_invalid_residual_count": int(
            sum(score["base_only_invalid_residual_count"] for score in scores)
        ),
        "candidate_only_invalid_residual_count": int(
            sum(score["candidate_only_invalid_residual_count"] for score in scores)
        ),
        "invalid_residual_policy": (
            "drop_jointly_invalid_and_cap_one_sided_invalid"
        ),
        "frame_count": int(frame_count),
        "max_crossfit_projection_median_px": float(
            max(score["max_crossfit_projection_median_px"] for score in scores)
        ),
    }


def decide_joint_ptz_center_plausibility_commit(
    result: Mapping[str, Any],
    *,
    initial_cameras: Mapping[int, CameraParams],
    focus_inlier_px: float,
) -> dict[str, Any]:
    """Accept a stable track repair only within the base PTZ jitter scale."""

    threshold = float(focus_inlier_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    if not bool(result.get("success", False)):
        return {"commit": False, "reason": "crossfit_band_failed"}
    consistency = float(
        result.get("crossfit_consistency", {}).get(
            "max_pairwise_projection_median_px", float("inf")
        )
    )
    if not np.isfinite(consistency) or consistency > threshold:
        return {"commit": False, "reason": "crossfit_band_inconsistent"}
    if len(initial_cameras) < 2:
        return {"commit": False, "reason": "base_center_scale_unavailable"}

    base_centers = np.asarray(
        [
            initial_cameras[frame].camera_center
            for frame in sorted(initial_cameras)
        ],
        dtype=np.float64,
    )
    candidate_center = np.asarray(
        result.get("center", np.full(3, np.nan)), dtype=np.float64
    ).reshape(-1)
    if (
        base_centers.shape != (len(initial_cameras), 3)
        or candidate_center.shape != (3,)
        or not np.isfinite(base_centers).all()
        or not np.isfinite(candidate_center).all()
    ):
        return {"commit": False, "reason": "base_center_scale_unavailable"}

    base_center = _geometric_median(base_centers)
    scatter = np.linalg.norm(base_centers - base_center.reshape(1, 3), axis=1)
    median_scatter = float(np.median(scatter))
    center_shift = float(np.linalg.norm(candidate_center - base_center))
    if median_scatter > 0.0:
        ratio = center_shift / median_scatter
    else:
        ratio = 0.0 if center_shift == 0.0 else float("inf")
    plausible = center_shift <= median_scatter
    return {
        "commit": bool(plausible),
        "reason": (
            "track_center_shift_within_base_jitter"
            if plausible
            else "track_center_shift_exceeds_base_jitter"
        ),
        "base_center": base_center,
        "candidate_center": candidate_center,
        "candidate_center_shift": center_shift,
        "base_center_median_scatter": median_scatter,
        "center_shift_to_scatter_ratio": float(ratio),
        "max_crossfit_projection_median_px": float(consistency),
    }


def _merge_joint_ptz_observation_roles(
    *roles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[Mapping[str, Any]]] = {}
    for role in roles:
        for row in role:
            by_frame.setdefault(int(row["frame"]), []).append(row)
    merged: list[dict[str, Any]] = []
    for frame, rows in sorted(by_frame.items()):
        merged.append(
            {
                "frame": frame,
                "point_ids": np.concatenate(
                    [np.asarray(row["point_ids"], dtype=np.int64) for row in rows]
                ),
                "validation_group_ids": np.concatenate(
                    [
                        np.asarray(row["validation_group_ids"], dtype=np.int64)
                        for row in rows
                    ]
                ),
                "xyz": np.concatenate(
                    [np.asarray(row["xyz"], dtype=np.float64) for row in rows],
                    axis=0,
                ),
                "image_points": np.concatenate(
                    [
                        np.asarray(row["image_points"], dtype=np.float64)
                        for row in rows
                    ],
                    axis=0,
                ),
            }
        )
    return merged


def _verify_exact_joint_ptz_candidate(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    candidate_cameras: Mapping[int, CameraParams],
    reprojection_error_px: float,
    min_inliers_per_frame: int,
    verification_source: str,
    verification_policy: str = "strict_frame_pareto",
) -> dict[str, Any]:
    majority_policy = verification_policy == "frame_hac_majority_pooled_pareto"
    all_frames = [int(row["frame"]) for row in observations]
    if len(set(all_frames)) != len(all_frames):
        return {
            "commit": False,
            "reason": "exact_candidate_frame_identity_not_unique",
            "verification_source": verification_source,
            "verification_policy": verification_policy,
        }
    validation_rows: list[dict[str, Any]] = []
    base_errors: list[np.ndarray] = []
    candidate_errors: list[np.ndarray] = []
    unscored_frames: list[int] = []
    for row in observations:
        frame = int(row["frame"])
        if frame not in initial_cameras or frame not in candidate_cameras:
            return {
                "commit": False,
                "reason": "exact_candidate_frame_missing",
                "verification_source": verification_source,
            }
        point_ids = np.asarray(row["point_ids"], dtype=np.int64).reshape(-1)
        xyz = np.asarray(row["xyz"], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(row["image_points"], dtype=np.float64).reshape(
            -1, 2
        )
        if majority_policy and len(point_ids) == 0:
            unscored_frames.append(frame)
            continue
        validation_rows.append({"frame": frame, "point_ids": point_ids})
        base_errors.append(
            _camera_reprojection_errors(
                camera=initial_cameras[frame],
                xyz=xyz,
                image_points=image_points,
            )
        )
        candidate_errors.append(
            _camera_reprojection_errors(
                camera=candidate_cameras[frame],
                xyz=xyz,
                image_points=image_points,
            )
        )
    scored_frames = [int(row["frame"]) for row in validation_rows]
    minimum_scored_frame_count = len(all_frames) // 2 + 1
    coverage = {
        "all_frame_count": int(len(all_frames)),
        "scored_frame_count": int(len(scored_frames)),
        "unscored_frame_count": int(len(unscored_frames)),
        "scored_frames": scored_frames,
        "unscored_frames": unscored_frames,
    }
    if majority_policy:
        coverage["minimum_scored_frame_count"] = int(minimum_scored_frame_count)
        if len(scored_frames) < minimum_scored_frame_count:
            return {
                "commit": False,
                "reason": "exact_candidate_majority_frame_coverage_failed",
                "verification_source": verification_source,
                "verification_policy": verification_policy,
                **coverage,
            }
    if not base_errors:
        return {
            "commit": False,
            "reason": "exact_candidate_validation_empty",
            "verification_source": verification_source,
            "verification_policy": verification_policy,
            **coverage,
        }
    pseudo_crossfit_result = {
        "success": True,
        "selected_validation_fold": 0,
        "folds": [
            {
                "validation_fold": 0,
                "fit_success": True,
                "validation_rows": validation_rows,
                "base_errors_px": np.concatenate(base_errors),
                "candidate_errors_px": np.concatenate(candidate_errors),
            }
        ],
        "crossfit_consistency": {
            "max_pairwise_projection_median_px": 0.0,
            "medoid_fold": 0,
        },
    }
    if verification_policy == "strict_frame_pareto":
        decision = decide_selected_joint_ptz_frame_pareto_commit(
            pseudo_crossfit_result,
            focus_inlier_px=float(reprojection_error_px),
            min_inliers_per_frame=int(min_inliers_per_frame),
        )
    elif verification_policy == "frame_hac_pooled_pareto":
        decision = decide_joint_ptz_frame_hac_pareto_commit(
            pseudo_crossfit_result,
            focus_inlier_px=float(reprojection_error_px),
            expected_frames=[int(row["frame"]) for row in validation_rows],
        )
    elif majority_policy:
        decision = decide_joint_ptz_frame_hac_pareto_commit(
            pseudo_crossfit_result,
            focus_inlier_px=float(reprojection_error_px),
            expected_frames=scored_frames,
        )
    elif verification_policy == "window_component_pooled_pareto":
        decision = decide_window_component_pooled_pareto_commit(
            pseudo_crossfit_result,
            focus_inlier_px=float(reprojection_error_px),
            expected_frames=scored_frames,
        )
    else:
        raise ValueError(f"unsupported exact verification policy: {verification_policy}")
    return {
        **decision,
        "verification_source": verification_source,
        "verification_policy": verification_policy,
        **coverage,
    }


def decide_window_component_pooled_pareto_commit(
    result: Mapping[str, Any],
    *,
    focus_inlier_px: float,
    expected_frames: Sequence[int],
) -> dict[str, Any]:
    """Verify a short component with pooled heldout geometry and paired frames."""

    threshold = float(focus_inlier_px)
    frames = [int(frame) for frame in expected_frames]
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("expected_frames must contain unique frame IDs")
    if not bool(result.get("success", False)):
        return {"commit": False, "reason": "crossfit_component_failed"}

    selected_fold = result.get("selected_validation_fold")
    selected = [
        row
        for row in result.get("folds", [])
        if selected_fold is not None
        and int(row.get("validation_fold", -1)) == int(selected_fold)
        and bool(row.get("fit_success", False))
    ]
    if len(selected) != 1:
        return {"commit": False, "reason": "component_score_unavailable"}
    row = selected[0]
    validation_rows = list(row.get("validation_rows", []))
    if [int(item["frame"]) for item in validation_rows] != frames:
        return {"commit": False, "reason": "component_frame_identity_mismatch"}
    base_raw = np.asarray(row.get("base_errors_px", []), dtype=np.float64).reshape(-1)
    candidate_raw = np.asarray(
        row.get("candidate_errors_px", []), dtype=np.float64
    ).reshape(-1)
    if len(base_raw) == 0 or base_raw.shape != candidate_raw.shape:
        return {"commit": False, "reason": "component_score_unavailable"}

    pooled_base: list[np.ndarray] = []
    pooled_candidate: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    offset = 0
    for validation in validation_rows:
        count = len(np.asarray(validation.get("point_ids", [])).reshape(-1))
        stop = offset + count
        if count <= 0 or stop > len(base_raw):
            return {"commit": False, "reason": "component_score_unavailable"}
        base_slice = base_raw[offset:stop]
        candidate_slice = candidate_raw[offset:stop]
        offset = stop
        base_finite = np.isfinite(base_slice)
        candidate_finite = np.isfinite(candidate_slice)
        comparable = base_finite | candidate_finite
        if not np.any(comparable):
            return {"commit": False, "reason": "component_score_unavailable"}
        base = np.minimum(
            np.where(base_finite, base_slice, threshold)[comparable], threshold
        )
        candidate = np.minimum(
            np.where(candidate_finite, candidate_slice, threshold)[comparable],
            threshold,
        )
        base_mean = float(np.mean(base))
        candidate_mean = float(np.mean(candidate))
        frame_rows.append(
            {
                "frame": int(validation["frame"]),
                "base_clipped_mean_px": base_mean,
                "candidate_clipped_mean_px": candidate_mean,
                "clipped_mean_gain_px": float(base_mean - candidate_mean),
            }
        )
        pooled_base.append(base)
        pooled_candidate.append(candidate)
    if offset != len(base_raw):
        return {"commit": False, "reason": "component_score_unavailable"}

    base = np.concatenate(pooled_base)
    candidate = np.concatenate(pooled_candidate)
    gains = np.asarray(
        [item["clipped_mean_gain_px"] for item in frame_rows], dtype=np.float64
    )
    scale = max(1.0, float(np.max(np.abs(gains))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    wins = int(np.count_nonzero(gains > tolerance))
    losses = int(np.count_nonzero(gains < -tolerance))
    ties = int(len(gains) - wins - losses)
    base_positive = int(np.count_nonzero(np.isfinite(base_raw)))
    candidate_positive = int(np.count_nonzero(np.isfinite(candidate_raw)))
    base_inliers = int(np.count_nonzero(np.isfinite(base_raw) & (base_raw <= threshold)))
    candidate_inliers = int(
        np.count_nonzero(np.isfinite(candidate_raw) & (candidate_raw <= threshold))
    )
    base_summary = {
        "clipped_mean_px": float(np.mean(base)),
        "clipped_median_px": float(np.median(base)),
        "clipped_p90_px": float(np.percentile(base, 90.0)),
    }
    candidate_summary = {
        "clipped_mean_px": float(np.mean(candidate)),
        "clipped_median_px": float(np.median(candidate)),
        "clipped_p90_px": float(np.percentile(candidate, 90.0)),
    }
    worst_frame_regression = float(max(0.0, -float(np.min(gains))))
    checks = {
        "paired_wins_exceed_losses": wins > losses,
        "clipped_mean_improved": (
            candidate_summary["clipped_mean_px"]
            < base_summary["clipped_mean_px"] - tolerance
        ),
        "clipped_median_not_worse": (
            candidate_summary["clipped_median_px"]
            <= base_summary["clipped_median_px"] + tolerance
        ),
        "clipped_p90_not_worse": (
            candidate_summary["clipped_p90_px"]
            <= base_summary["clipped_p90_px"] + tolerance
        ),
        "positive_depth_support_not_lower": candidate_positive >= base_positive,
        "inlier_support_not_lower": candidate_inliers >= base_inliers,
        "worst_frame_regression_bounded": worst_frame_regression <= threshold,
    }
    commit = bool(all(checks.values()))
    return {
        "commit": commit,
        "reason": (
            "window_component_pooled_pareto_improved"
            if commit
            else "window_component_pooled_pareto_failed"
        ),
        "expected_frames": frames,
        "selected_validation_fold": int(selected_fold),
        "base": base_summary,
        "candidate": candidate_summary,
        "base_positive_depth_count": base_positive,
        "candidate_positive_depth_count": candidate_positive,
        "base_inlier_count": base_inliers,
        "candidate_inlier_count": candidate_inliers,
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "worst_frame_regression_px": worst_frame_regression,
        "worst_frame_regression_bound_px": threshold,
        "checks": checks,
        "frames": frame_rows,
    }


def fit_confirm_refit_verify_joint_ptz_shared_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    reprojection_error_px: float = 4.0,
    min_inliers_per_frame: int = 12,
    min_successful_frames: int = 3,
    rng_seed: int = 0,
    verification_policy: str = "strict_frame_pareto",
    candidate_center_policy: str = "joint_fit",
    fixed_center_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit on A, confirm on B, refit on A+B, and verify exactly on C."""

    if candidate_center_policy not in {
        "joint_fit",
        "static_pnp_median_fixed",
        "static_pnp_median_bounded_rotation",
    }:
        raise ValueError(
            f"unsupported candidate center policy: {candidate_center_policy}"
        )
    if fixed_center_override is not None and candidate_center_policy == "joint_fit":
        raise ValueError("fixed_center_override requires a fixed-center policy")
    fixed_center = (
        np.asarray(fixed_center_override, dtype=np.float64).reshape(3)
        if fixed_center_override is not None
        else _geometric_median(
            np.asarray(
                [camera.camera_center for camera in initial_cameras.values()],
                dtype=np.float64,
            )
        )
        if candidate_center_policy
        in {"static_pnp_median_fixed", "static_pnp_median_bounded_rotation"}
        else None
    )

    def solve_candidate(
        candidate_observations: Sequence[Mapping[str, Any]],
        candidate_initial_cameras: Mapping[int, CameraParams],
    ) -> dict[str, Any]:
        common = {
            "observations": candidate_observations,
            "initial_cameras": candidate_initial_cameras,
            "reprojection_error_px": float(reprojection_error_px),
            "min_inliers_per_frame": int(min_inliers_per_frame),
            "min_successful_frames": int(min_successful_frames),
            "rng_seed": int(rng_seed),
        }
        if fixed_center is not None:
            target = solve_fixed_center_ptz_rotations(
                **common,
                fixed_center=fixed_center,
            )
            if (
                candidate_center_policy != "static_pnp_median_bounded_rotation"
                or not bool(target.get("success", False))
            ):
                return target
            target_frames = [int(frame) for frame in target["successful_frames"]]
            bounded = solve_bounded_rotation_ptz_median_center(
                observations=candidate_observations,
                initial_cameras=candidate_initial_cameras,
                tracking_rotations={
                    frame: np.asarray(target["cameras"][frame].R, dtype=np.float64)
                    for frame in target_frames
                },
                reprojection_error_px=float(reprojection_error_px),
                min_points_per_frame=int(min_inliers_per_frame),
                min_successful_frames=int(min_successful_frames),
                fixed_center=fixed_center,
            )
            return {
                **bounded,
                "target_rotation_solver": "fixed_center_ptz_rotations",
                "target_rotation_successful_frames": target_frames,
            }
        return solve_joint_ptz_shared_center(**common)

    split = split_joint_ptz_track_groups(
        observations=observations,
        initial_cameras=initial_cameras,
    )
    split_diagnostics = {
        "policy": split["policy"],
        "group_disjoint": bool(split["group_disjoint"]),
        "role_group_ids": split["role_group_ids"],
        "rows": {
            role: [
                {
                    "frame": int(row["frame"]),
                    "point_ids": np.asarray(row["point_ids"], dtype=np.int64),
                    "validation_group_ids": np.asarray(
                        row["validation_group_ids"], dtype=np.int64
                    ),
                }
                for row in split[role]
            ]
            for role in ("fit", "confirm", "final")
        },
    }
    if not bool(split["group_disjoint"]):
        return {
            "success": False,
            "reason": "track_group_split_not_disjoint",
            "split": split_diagnostics,
        }

    fit = solve_candidate(split["fit"], initial_cameras)
    fit_diagnostics = {key: value for key, value in fit.items() if key != "cameras"}
    if not bool(fit.get("success", False)):
        return {
            "success": False,
            "reason": "fit_fold_candidate_failed",
            "split": split_diagnostics,
            "fit": fit_diagnostics,
        }

    confirmation = _verify_exact_joint_ptz_candidate(
        observations=split["confirm"],
        initial_cameras=initial_cameras,
        candidate_cameras=fit["cameras"],
        reprojection_error_px=float(reprojection_error_px),
        min_inliers_per_frame=int(min_inliers_per_frame),
        verification_source="confirm_fold_exact_fit_candidate",
        verification_policy=str(verification_policy),
    )
    if not bool(confirmation.get("commit", False)):
        return {
            "success": False,
            "reason": "confirm_fold_rejected_candidate",
            "split": split_diagnostics,
            "fit": fit_diagnostics,
            "confirmation": confirmation,
        }

    refit_observations = _merge_joint_ptz_observation_roles(
        split["fit"], split["confirm"]
    )
    refit = solve_candidate(refit_observations, fit["cameras"])
    refit_diagnostics = {
        key: value for key, value in refit.items() if key != "cameras"
    }
    if not bool(refit.get("success", False)):
        return {
            "success": False,
            "reason": "fit_confirm_refit_failed",
            "split": split_diagnostics,
            "fit": fit_diagnostics,
            "confirmation": confirmation,
            "refit": refit_diagnostics,
        }

    final_verification = _verify_exact_joint_ptz_candidate(
        observations=split["final"],
        initial_cameras=initial_cameras,
        candidate_cameras=refit["cameras"],
        reprojection_error_px=float(reprojection_error_px),
        min_inliers_per_frame=int(min_inliers_per_frame),
        verification_source="final_fold_exact_refit_candidate",
        verification_policy=str(verification_policy),
    )
    return {
        "success": True,
        "reason": "ok",
        "cameras": refit["cameras"],
        "center": refit["center"],
        "candidate_center_policy": candidate_center_policy,
        "fixed_center": fixed_center,
        "fixed_center_override_used": fixed_center_override is not None,
        "split": split_diagnostics,
        "fit": fit_diagnostics,
        "confirmation": confirmation,
        "refit": refit_diagnostics,
        "final_verification": final_verification,
        "final_fold_used_for_candidate": False,
        "verification_source_independence": "track_group_disjoint_same_tracker_source",
    }
