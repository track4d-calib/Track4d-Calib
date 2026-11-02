from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from track4d_calib.cameras import CameraParams


@dataclass(frozen=True)
class DirectRotationBAConfig:
    max_rotation_correction_deg: float = 5.0
    max_nfev: int = 10
    robust_loss: str = "huber"
    robust_scale_px: float = 3.0
    reprojection_weight: float = 1.0
    epipolar_weight: float = 1.0
    x_scale: str | float = 1.0
    convergence_tolerance: float = 1e-8


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def bounded_rotation_residual(
    unconstrained: np.ndarray, *, max_angle_rad: float
) -> np.ndarray:
    """Map an unconstrained vector smoothly into a rotation-angle trust region."""

    raw = np.asarray(unconstrained, dtype=np.float64).reshape(3)
    maximum = float(max_angle_rad)
    if not np.isfinite(raw).all() or not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("rotation residual and max angle must be finite")
    magnitude = float(np.linalg.norm(raw))
    if magnitude <= np.finfo(np.float64).eps:
        return raw.copy()
    return raw * (maximum * np.tanh(magnitude / maximum) / magnitude)


def unconstrained_from_bounded_rotation(
    bounded: np.ndarray, *, max_angle_rad: float
) -> np.ndarray:
    """Invert the smooth trust-region map for a bounded initialization."""

    value = np.asarray(bounded, dtype=np.float64).reshape(3)
    maximum = float(max_angle_rad)
    magnitude = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("rotation residual and max angle must be finite")
    if magnitude <= np.finfo(np.float64).eps:
        return value.copy()
    clipped = min(magnitude / maximum, 1.0 - 1e-6)
    raw_magnitude = maximum * np.arctanh(clipped)
    return value * (raw_magnitude / magnitude)


def bounded_center_residual(
    unconstrained: np.ndarray, *, max_distance: float | None
) -> np.ndarray:
    """Map a center correction into a finite, differentiable trust region."""

    if max_distance is None:
        value = np.asarray(unconstrained, dtype=np.float64).reshape(3)
        if not np.isfinite(value).all():
            raise ValueError("center residual must be finite")
        return value.copy()
    return bounded_rotation_residual(
        unconstrained, max_angle_rad=float(max_distance)
    )


def unconstrained_from_bounded_center(
    bounded: np.ndarray, *, max_distance: float | None
) -> np.ndarray:
    """Invert the bounded center map for a center initialization."""

    if max_distance is None:
        return bounded_center_residual(bounded, max_distance=None)
    return unconstrained_from_bounded_rotation(
        bounded, max_angle_rad=float(max_distance)
    )


def materialize_direct_rotation(
    *,
    base_camera: CameraParams,
    fixed_center: np.ndarray,
    unconstrained_residual: np.ndarray,
    max_angle_rad: float,
) -> CameraParams:
    residual = bounded_rotation_residual(
        unconstrained_residual, max_angle_rad=max_angle_rad
    )
    rotation = Rotation.from_rotvec(residual).as_matrix() @ np.asarray(
        base_camera.R, dtype=np.float64
    )
    center = np.asarray(fixed_center, dtype=np.float64).reshape(3)
    return replace(base_camera, R=rotation, T=-rotation @ center)


def signed_reprojection_residuals(
    camera: CameraParams,
    *,
    xyz: np.ndarray,
    image_points: np.ndarray,
) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    observed = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] != observed.shape[0]:
        raise ValueError("xyz and image_points must have equal length")
    camera_points = (np.asarray(camera.R) @ points.T).T + np.asarray(camera.T)
    projected = (np.asarray(camera.K) @ camera_points.T).T
    depth = projected[:, 2]
    safe_depth = np.where(
        np.abs(depth) >= 1e-9,
        depth,
        np.where(depth < 0.0, -1e-9, 1e-9),
    )
    pixels = projected[:, :2] / safe_depth[:, None]
    residual = pixels - observed
    return residual.reshape(-1)


def signed_epipolar_residuals(
    *,
    left_camera: CameraParams,
    right_camera: CameraParams,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left_points, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(right_points, dtype=np.float64).reshape(-1, 2)
    if left.shape != right.shape:
        raise ValueError("left and right points must have equal shape")
    relative_rotation = np.asarray(right_camera.R) @ np.asarray(left_camera.R).T
    relative_translation = np.asarray(right_camera.T) - relative_rotation @ np.asarray(
        left_camera.T
    )
    essential = _skew(relative_translation) @ relative_rotation
    fundamental = (
        np.linalg.inv(np.asarray(right_camera.K)).T
        @ essential
        @ np.linalg.inv(np.asarray(left_camera.K))
    )
    left_h = np.column_stack([left, np.ones(len(left))])
    right_h = np.column_stack([right, np.ones(len(right))])
    right_lines = (fundamental @ left_h.T).T
    left_lines = (fundamental.T @ right_h.T).T
    numerator = np.sum(right_h * right_lines, axis=1)
    denominator = np.sqrt(
        np.square(right_lines[:, 0])
        + np.square(right_lines[:, 1])
        + np.square(left_lines[:, 0])
        + np.square(left_lines[:, 1])
        + 1e-12
    )
    return numerator / denominator


def _normalized_confidence(
    nodes: Sequence[Hashable], confidence: Mapping[Hashable, float]
) -> dict[Hashable, float]:
    values = np.asarray(
        [max(float(confidence.get(node, 0.0)), 0.0) for node in nodes],
        dtype=np.float64,
    )
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return {
        node: float(value / max(scale, np.finfo(np.float64).eps))
        for node, value in zip(nodes, values)
    }


def solve_direct_rotation_component(
    *,
    nodes: Sequence[Hashable],
    base_cameras: Mapping[Hashable, CameraParams],
    fixed_centers: Mapping[Hashable, np.ndarray],
    reprojection_observations: Mapping[Hashable, Mapping[str, np.ndarray]],
    epipolar_observations: Sequence[Mapping[str, Any]],
    confidence: Mapping[Hashable, float],
    config: DirectRotationBAConfig,
    initial_rotation_residuals: Mapping[Hashable, np.ndarray] | None = None,
    center_group_by_node: Mapping[Hashable, Hashable] | None = None,
    center_trust_radius_by_group: Mapping[Hashable, float | None] | None = None,
    initial_center_residuals: Mapping[Hashable, np.ndarray] | None = None,
) -> dict[str, Any]:
    ordered = list(nodes)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("nodes must be non-empty and unique")
    if set(ordered) - set(base_cameras) or set(ordered) - set(fixed_centers):
        raise ValueError("every node needs a base camera and fixed center")
    maximum = np.deg2rad(float(config.max_rotation_correction_deg))
    if maximum <= 0.0 or int(config.max_nfev) <= 0:
        raise ValueError("invalid direct rotation BA config")
    residual_weights = np.asarray(
        [config.reprojection_weight, config.epipolar_weight], dtype=np.float64
    )
    if (
        not np.isfinite(residual_weights).all()
        or np.any(residual_weights < 0.0)
        or not np.any(residual_weights > 0.0)
    ):
        raise ValueError("direct BA observation weights must be finite, non-negative, and not all zero")
    x_scale = config.x_scale
    if isinstance(x_scale, str):
        if x_scale != "jac":
            raise ValueError("direct BA x_scale string must be 'jac'")
    elif not np.isfinite(float(x_scale)) or float(x_scale) <= 0.0:
        raise ValueError("direct BA x_scale must be positive and finite")
    tolerance = float(config.convergence_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("direct BA convergence tolerance must be positive and finite")
    reprojection_scale = float(np.sqrt(config.reprojection_weight))
    epipolar_scale = float(np.sqrt(config.epipolar_weight))
    node_index = {node: index for index, node in enumerate(ordered)}
    weights = _normalized_confidence(ordered, confidence)

    center_groups: list[Hashable] = []
    center_group_index: dict[Hashable, int] = {}
    center_anchor: dict[Hashable, np.ndarray] = {}
    center_radius: dict[Hashable, float | None] = {}
    if center_group_by_node is not None:
        if set(ordered) - set(center_group_by_node):
            raise ValueError("every node needs a center group")
        if center_trust_radius_by_group is None:
            raise ValueError("center trust radii are required with center groups")
        for node in ordered:
            group = center_group_by_node[node]
            if group not in center_group_index:
                center_group_index[group] = len(center_groups)
                center_groups.append(group)
                center_anchor[group] = np.asarray(
                    fixed_centers[node], dtype=np.float64
                ).reshape(3)
            elif not np.allclose(
                center_anchor[group],
                np.asarray(fixed_centers[node], dtype=np.float64).reshape(3),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError("nodes in one center group need one anchor center")
        if set(center_groups) - set(center_trust_radius_by_group):
            raise ValueError("every center group needs a trust radius")
        for group in center_groups:
            if center_trust_radius_by_group[group] is None:
                center_radius[group] = None
                continue
            radius = float(center_trust_radius_by_group[group])
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("center trust radii must be finite and positive")
            center_radius[group] = radius
    elif center_trust_radius_by_group is not None or initial_center_residuals is not None:
        raise ValueError("center residual inputs require center groups")

    rotation_parameter_count = 3 * len(ordered)
    parameter_count = rotation_parameter_count + 3 * len(center_groups)

    def dependency_columns(
        dependency_nodes: Sequence[Hashable],
    ) -> tuple[tuple[int, int], ...]:
        columns = {
            (3 * node_index[node], 3) for node in dependency_nodes
        }
        for node in dependency_nodes:
            if center_group_by_node is None:
                continue
            group = center_group_by_node[node]
            columns.add(
                (rotation_parameter_count + 3 * center_group_index[group], 3)
            )
        return tuple(sorted(columns))

    def epipolar_side(
        edge: Mapping[str, Any], side: str
    ) -> tuple[Hashable | None, CameraParams | None]:
        node_present = side in edge
        fixed_key = f"{side}_camera"
        camera_present = fixed_key in edge
        if node_present == camera_present:
            raise ValueError(
                f"epipolar {side} must specify exactly one node or fixed camera"
            )
        if camera_present:
            camera = edge[fixed_key]
            if not isinstance(camera, CameraParams):
                raise ValueError(f"epipolar {fixed_key} must be CameraParams")
            return None, camera
        node = edge[side]
        if node not in node_index:
            raise ValueError(f"epipolar {side} node is absent from component")
        return node, None

    dependencies: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for node in ordered:
        observation = reprojection_observations.get(node)
        count = 0 if observation is None else len(np.asarray(observation["xyz"]))
        if count:
            dependencies.append((2 * count, dependency_columns((node,))))
    for edge in epipolar_observations:
        count = len(np.asarray(edge["left_points"]))
        if count:
            left_node, _ = epipolar_side(edge, "left")
            right_node, _ = epipolar_side(edge, "right")
            variable_nodes = tuple(
                node for node in (left_node, right_node) if node is not None
            )
            if not variable_nodes:
                raise ValueError("epipolar edge must depend on an optimized node")
            dependencies.append(
                (count, dependency_columns(variable_nodes))
            )
    for node in ordered:
        dependencies.append((3, ((3 * node_index[node], 3),)))
    for group in center_groups:
        if center_radius[group] is None:
            continue
        dependencies.append(
            (
                3,
                ((rotation_parameter_count + 3 * center_group_index[group], 3),),
            )
        )
    residual_count = sum(count for count, _ in dependencies)
    sparsity = lil_matrix((residual_count, parameter_count), dtype=np.int8)
    row_start = 0
    for count, columns in dependencies:
        for column, width in columns:
            sparsity[row_start : row_start + count, column : column + width] = 1
        row_start += count

    def unpack(
        values: np.ndarray,
    ) -> tuple[np.ndarray, dict[Hashable, np.ndarray], dict[Hashable, np.ndarray]]:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        rotation_raw = array[:rotation_parameter_count].reshape(len(ordered), 3)
        center_raw = array[rotation_parameter_count:].reshape(len(center_groups), 3)
        center_corrections = {
            group: bounded_center_residual(
                center_raw[index], max_distance=center_radius[group]
            )
            for index, group in enumerate(center_groups)
        }
        centers = {
            group: center_anchor[group] + center_corrections[group]
            for group in center_groups
        }
        return rotation_raw, center_raw, centers

    def materialize(values: np.ndarray) -> dict[Hashable, CameraParams]:
        rotation_raw, _, centers = unpack(values)
        return {
            node: materialize_direct_rotation(
                base_camera=base_cameras[node],
                fixed_center=(
                    fixed_centers[node]
                    if center_group_by_node is None
                    else centers[center_group_by_node[node]]
                ),
                unconstrained_residual=rotation_raw[index],
                max_angle_rad=maximum,
            )
            for index, node in enumerate(ordered)
        }

    def residuals(values: np.ndarray) -> np.ndarray:
        cameras = materialize(values)
        parts: list[np.ndarray] = []
        for node in ordered:
            observation = reprojection_observations.get(node)
            if observation is None or not len(np.asarray(observation["xyz"])):
                continue
            parts.append(
                reprojection_scale * signed_reprojection_residuals(
                    cameras[node],
                    xyz=observation["xyz"],
                    image_points=observation["image_points"],
                )
            )
        for edge in epipolar_observations:
            if not len(np.asarray(edge["left_points"])):
                continue
            left_node, left_fixed = epipolar_side(edge, "left")
            right_node, right_fixed = epipolar_side(edge, "right")
            parts.append(
                epipolar_scale * signed_epipolar_residuals(
                    left_camera=(
                        left_fixed
                        if left_fixed is not None
                        else cameras[left_node]
                    ),
                    right_camera=(
                        right_fixed
                        if right_fixed is not None
                        else cameras[right_node]
                    ),
                    left_points=edge["left_points"],
                    right_points=edge["right_points"],
                )
            )
        rotation_raw, center_raw, _ = unpack(values)
        for index, node in enumerate(ordered):
            parts.append(
                np.sqrt(weights[node]) * rotation_raw[index] / maximum
            )
        for index, group in enumerate(center_groups):
            if center_radius[group] is None:
                continue  # No hidden pull toward a fixed center in free mode.
            group_weights = [
                weights[node]
                for node in ordered
                if center_group_by_node is not None
                and center_group_by_node[node] == group
            ]
            parts.append(
                np.sqrt(float(np.mean(group_weights)))
                * center_raw[index]
                / center_radius[group]
            )
        return np.concatenate(parts)

    initial_map = initial_rotation_residuals or {}
    initial_parts = [
        np.concatenate(
            [
            unconstrained_from_bounded_rotation(
                np.asarray(initial_map.get(node, np.zeros(3)), dtype=np.float64),
                max_angle_rad=maximum,
            )
            for node in ordered
            ]
        )
    ]
    initial_center_map = initial_center_residuals or {}
    if center_groups:
        initial_parts.append(
            np.concatenate(
                [
                    unconstrained_from_bounded_center(
                        np.asarray(
                            initial_center_map.get(group, np.zeros(3)),
                            dtype=np.float64,
                        ),
                        max_distance=center_radius[group],
                    )
                    for group in center_groups
                ]
            )
        )
    initial = np.concatenate(initial_parts)
    initial_residual = residuals(initial)
    optimized = least_squares(
        residuals,
        initial,
        jac_sparsity=sparsity.tocsr(),
        max_nfev=int(config.max_nfev),
        loss=str(config.robust_loss),
        f_scale=float(config.robust_scale_px),
        method="trf",
        x_scale=x_scale,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
    )
    values = np.asarray(optimized.x, dtype=np.float64)
    cameras = materialize(values)
    raw, center_raw, optimized_centers = unpack(values)
    bounded = np.asarray(
        [bounded_rotation_residual(value, max_angle_rad=maximum) for value in raw]
    )
    center_corrections = {
        group: optimized_centers[group] - center_anchor[group]
        for group in center_groups
    }
    finite_solution = bool(np.isfinite(values).all())
    return {
        "success": bool(optimized.success and finite_solution),
        "finite_solution": finite_solution,
        "cameras": cameras,
        "unconstrained_residuals": {
            node: raw[index] for index, node in enumerate(ordered)
        },
        "unconstrained_center_residuals": {
            group: center_raw[index]
            for index, group in enumerate(center_groups)
        },
        "optimized_centers": optimized_centers,
        "center_correction_norm": {
            group: float(np.linalg.norm(center_corrections[group]))
            for group in center_groups
        },
        "rotation_correction_deg": {
            node: float(np.degrees(np.linalg.norm(bounded[index])))
            for index, node in enumerate(ordered)
        },
        "least_squares": {
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "nfev": int(optimized.nfev),
            "cost": float(optimized.cost),
            "optimality": float(optimized.optimality),
            "initial_squared_cost": float(0.5 * np.sum(np.square(initial_residual))),
            "message": str(optimized.message),
            "x_scale": x_scale,
            "convergence_tolerance": tolerance,
        },
        "node_count": len(ordered),
        "center_group_count": len(center_groups),
        "residual_count": int(residual_count),
        "jacobian_nonzeros": int(sparsity.nnz),
    }
