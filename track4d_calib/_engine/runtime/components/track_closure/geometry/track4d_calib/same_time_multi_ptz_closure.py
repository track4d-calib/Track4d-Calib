from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import binomtest

from track4d_calib.cameras import CameraParams
from track4d_calib.track_crossview_verify import stable_point_id_validation_groups


SPATIAL_POINT_STRIDE = 1_000_000


def candidate_fit_residuals(
    errors: np.ndarray,
    *,
    mode: str,
    threshold_px: float,
) -> np.ndarray:
    """Prepare candidate-fit residuals without changing verifier thresholds."""

    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    threshold = float(threshold_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_px must be positive and finite")
    finite = np.where(np.isfinite(values), values, threshold)
    if mode == "clipped":
        return np.minimum(finite, threshold)
    if mode == "huber_raw":
        return finite
    raise ValueError(f"unsupported candidate fit residual mode: {mode}")


def split_frame_local_track_ids(
    *, frame: int, point_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Split frame-local identities once into fit, confirm, and final roles."""

    points = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    if len(np.unique(points)) != len(points) or np.any(points < 0):
        raise ValueError("point_ids must be unique and non-negative")
    if np.any(points >= SPATIAL_POINT_STRIDE):
        raise ValueError("point ID exceeds the frame-local namespace")
    identities = (int(frame) + 1) * SPATIAL_POINT_STRIDE + points
    buckets = np.mod(stable_point_id_validation_groups(identities), 4)
    return {
        "fit": points[buckets < 2],
        "confirm": points[buckets == 2],
        "final": points[buckets == 3],
    }


def target_components_from_window_visibility(
    *,
    visibility_by_frame: Sequence[np.ndarray],
    view_ids: Sequence[str],
    target_views: Sequence[str],
    min_shared_points: int = 8,
) -> dict[str, Any]:
    """Build deterministic PTZ components from public same-time shared IDs."""

    views = [str(view) for view in view_ids]
    targets = list(dict.fromkeys(str(view) for view in target_views))
    minimum = int(min_shared_points)
    if not targets or set(targets) - set(views):
        raise ValueError("target views must be a non-empty subset of view_ids")
    if minimum <= 0:
        raise ValueError("min_shared_points must be positive")
    index = {view: views.index(view) for view in targets}
    edge_counts: dict[tuple[str, str], list[int]] = {}
    supported_edges: list[tuple[str, str]] = []
    for left_index, left in enumerate(targets):
        for right in targets[left_index + 1 :]:
            counts: list[int] = []
            for raw_visibility in visibility_by_frame:
                visibility = np.asarray(raw_visibility, dtype=bool)
                if visibility.ndim != 2 or visibility.shape[0] != len(views):
                    raise ValueError("each visibility array must have shape [view, point]")
                counts.append(
                    int(
                        np.count_nonzero(
                            visibility[index[left]] & visibility[index[right]]
                        )
                    )
                )
            edge_counts[(left, right)] = counts
            if any(count >= minimum for count in counts):
                supported_edges.append((left, right))

    adjacency = {view: set() for view in targets}
    for left, right in supported_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(targets)
    components: list[list[str]] = []
    while remaining:
        seed = min(remaining, key=targets.index)
        remaining.remove(seed)
        queue = [seed]
        component: list[str] = []
        while queue:
            current = queue.pop(0)
            component.append(current)
            for neighbor in targets:
                if neighbor in remaining and neighbor in adjacency[current]:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append([view for view in targets if view in set(component)])
    return {
        "components": components,
        "supported_edges": [list(edge) for edge in supported_edges],
        "shared_count_by_edge": {
            f"{left}:{right}": counts
            for (left, right), counts in edge_counts.items()
        },
        "min_shared_points": minimum,
    }


def materialize_fixed_center_rotation_path(
    *,
    base_camera: CameraParams,
    proposal_rotation: np.ndarray,
    fixed_center: np.ndarray,
    alpha: float,
) -> CameraParams:
    """Interpolate only rotation while enforcing one physical PTZ center."""

    proposal = np.asarray(proposal_rotation, dtype=np.float64).reshape(3, 3)
    center = np.asarray(fixed_center, dtype=np.float64).reshape(3)
    weight = float(alpha)
    if not np.isfinite(proposal).all() or not np.isfinite(center).all():
        raise ValueError("proposal rotation and fixed center must be finite")
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    base = np.asarray(base_camera.R, dtype=np.float64).reshape(3, 3)
    residual = Rotation.from_matrix(proposal @ base.T).as_rotvec()
    rotation = Rotation.from_rotvec(weight * residual).as_matrix() @ base
    return replace(base_camera, R=rotation, T=-rotation @ center)


def materialize_candidate_pose_path(
    *,
    base_camera: CameraParams,
    proposal_rotation: np.ndarray,
    fixed_center: np.ndarray,
    alpha: float,
    center_path_mode: str,
) -> CameraParams:
    """Materialize a bounded pose path, optionally preserving alpha-zero fallback."""

    if center_path_mode == "fixed":
        return materialize_fixed_center_rotation_path(
            base_camera=base_camera,
            proposal_rotation=proposal_rotation,
            fixed_center=fixed_center,
            alpha=alpha,
        )
    if center_path_mode != "interpolate":
        raise ValueError(f"unsupported center path mode: {center_path_mode}")
    weight = float(alpha)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    endpoint = materialize_fixed_center_rotation_path(
        base_camera=base_camera,
        proposal_rotation=proposal_rotation,
        fixed_center=fixed_center,
        alpha=weight,
    )
    center = (
        (1.0 - weight) * np.asarray(base_camera.camera_center, dtype=np.float64)
        + weight * np.asarray(fixed_center, dtype=np.float64)
    )
    return replace(endpoint, T=-endpoint.R @ center)


def pooled_pareto_score(
    *,
    base_errors_by_node: Mapping[str, np.ndarray],
    candidate_errors_by_node: Mapping[str, np.ndarray],
    threshold_px: float,
) -> dict[str, Any]:
    """Apply the DYNAMIC_TRACK pooled Pareto policy to a generic spatial component."""

    threshold = float(threshold_px)
    keys = list(base_errors_by_node)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_px must be positive and finite")
    if not keys or set(keys) != set(candidate_errors_by_node):
        return {"commit": False, "reason": "spatial_score_unavailable"}

    base_pooled: list[np.ndarray] = []
    candidate_pooled: list[np.ndarray] = []
    node_rows: list[dict[str, Any]] = []
    base_positive = 0
    candidate_positive = 0
    base_inliers = 0
    candidate_inliers = 0
    for key in keys:
        base_raw = np.asarray(base_errors_by_node[key], dtype=np.float64).reshape(-1)
        candidate_raw = np.asarray(
            candidate_errors_by_node[key], dtype=np.float64
        ).reshape(-1)
        if len(base_raw) == 0 or base_raw.shape != candidate_raw.shape:
            return {"commit": False, "reason": "spatial_score_unavailable"}
        base_finite = np.isfinite(base_raw)
        candidate_finite = np.isfinite(candidate_raw)
        comparable = base_finite | candidate_finite
        if not np.any(comparable):
            return {"commit": False, "reason": "spatial_score_unavailable"}
        base = np.minimum(
            np.where(base_finite, base_raw, threshold)[comparable], threshold
        )
        candidate = np.minimum(
            np.where(candidate_finite, candidate_raw, threshold)[comparable],
            threshold,
        )
        gain = float(np.mean(base) - np.mean(candidate))
        node_rows.append(
            {
                "node": str(key),
                "count": int(len(base)),
                "base_clipped_mean_px": float(np.mean(base)),
                "candidate_clipped_mean_px": float(np.mean(candidate)),
                "clipped_mean_gain_px": gain,
            }
        )
        base_pooled.append(base)
        candidate_pooled.append(candidate)
        base_positive += int(np.count_nonzero(base_finite))
        candidate_positive += int(np.count_nonzero(candidate_finite))
        base_inliers += int(np.count_nonzero(base_finite & (base_raw <= threshold)))
        candidate_inliers += int(
            np.count_nonzero(candidate_finite & (candidate_raw <= threshold))
        )

    base = np.concatenate(base_pooled)
    candidate = np.concatenate(candidate_pooled)
    gains = np.asarray(
        [float(row["clipped_mean_gain_px"]) for row in node_rows], dtype=np.float64
    )
    scale = max(1.0, float(np.max(np.abs(gains))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    wins = int(np.count_nonzero(gains > tolerance))
    losses = int(np.count_nonzero(gains < -tolerance))
    ties = int(len(gains) - wins - losses)
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
    worst_regression = float(max(0.0, -float(np.min(gains))))
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
        "worst_node_regression_bounded": worst_regression <= threshold,
    }
    commit = bool(all(checks.values()))
    return {
        "commit": commit,
        "reason": (
            "spatial_component_pooled_pareto_improved"
            if commit
            else "spatial_component_pooled_pareto_failed"
        ),
        "base": base_summary,
        "candidate": candidate_summary,
        "base_positive_depth_count": int(base_positive),
        "candidate_positive_depth_count": int(candidate_positive),
        "base_inlier_count": int(base_inliers),
        "candidate_inlier_count": int(candidate_inliers),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "worst_node_regression_px": worst_regression,
        "checks": checks,
        "nodes": node_rows,
    }


def pooled_noninferiority_guard(
    *,
    base_errors_by_node: Mapping[str, np.ndarray],
    candidate_errors_by_node: Mapping[str, np.ndarray],
    threshold_px: float,
    min_observations: int = 12,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Reject a candidate only when paired heldout evidence shows degradation."""

    threshold = float(threshold_px)
    minimum = int(min_observations)
    alpha = float(significance_level)
    keys = list(base_errors_by_node)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_px must be positive and finite")
    if minimum <= 0:
        raise ValueError("min_observations must be positive")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("significance_level must lie in (0, 1)")
    if not keys or set(keys) != set(candidate_errors_by_node):
        return {"guard": False, "reason": "noninferiority_evidence_unavailable"}

    base_pooled: list[np.ndarray] = []
    candidate_pooled: list[np.ndarray] = []
    raw_differences: list[np.ndarray] = []
    node_rows: list[dict[str, Any]] = []
    base_positive = 0
    candidate_positive = 0
    skipped_empty_nodes = 0
    for key in keys:
        base_raw = np.asarray(base_errors_by_node[key], dtype=np.float64).reshape(-1)
        candidate_raw = np.asarray(
            candidate_errors_by_node[key], dtype=np.float64
        ).reshape(-1)
        if base_raw.shape != candidate_raw.shape:
            return {"guard": False, "reason": "noninferiority_shape_mismatch"}
        if not len(base_raw):
            skipped_empty_nodes += 1
            continue
        base_finite = np.isfinite(base_raw)
        candidate_finite = np.isfinite(candidate_raw)
        comparable = base_finite | candidate_finite
        if not np.any(comparable):
            skipped_empty_nodes += 1
            continue
        base_for_sign = np.where(base_finite, base_raw, threshold * 2.0)[comparable]
        candidate_for_sign = np.where(
            candidate_finite, candidate_raw, threshold * 2.0
        )[comparable]
        base = np.minimum(base_for_sign, threshold)
        candidate = np.minimum(candidate_for_sign, threshold)
        base_pooled.append(base)
        candidate_pooled.append(candidate)
        raw_differences.append(candidate_for_sign - base_for_sign)
        base_positive += int(np.count_nonzero(base_finite))
        candidate_positive += int(np.count_nonzero(candidate_finite))
        node_rows.append(
            {
                "node": str(key),
                "count": int(len(base)),
                "base_clipped_mean_px": float(np.mean(base)),
                "candidate_clipped_mean_px": float(np.mean(candidate)),
            }
        )

    if not base_pooled:
        return {"guard": False, "reason": "noninferiority_evidence_unavailable"}
    base = np.concatenate(base_pooled)
    candidate = np.concatenate(candidate_pooled)
    differences = np.concatenate(raw_differences)
    finite_scale = np.abs(differences[np.isfinite(differences)])
    scale = max(1.0, float(np.max(finite_scale)) if len(finite_scale) else 1.0)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    wins = int(np.count_nonzero(differences < -tolerance))
    losses = int(np.count_nonzero(differences > tolerance))
    ties = int(len(differences) - wins - losses)
    informative = wins + losses
    p_worse = (
        float(
            binomtest(
                losses,
                n=informative,
                p=0.5,
                alternative="greater",
            ).pvalue
        )
        if informative
        else 1.0
    )
    checks = {
        "minimum_observations": int(len(differences)) >= minimum,
        "minimum_informative_pairs": informative >= minimum,
        "no_significant_paired_degradation": p_worse >= alpha,
        "positive_depth_support_not_lower": candidate_positive >= base_positive,
    }
    guard = bool(all(checks.values()))
    return {
        "guard": guard,
        "reason": (
            "paired_heldout_noninferiority_guard_passed"
            if guard
            else "paired_heldout_noninferiority_guard_failed"
        ),
        "base": {
            "clipped_mean_px": float(np.mean(base)),
            "clipped_median_px": float(np.median(base)),
            "clipped_p90_px": float(np.percentile(base, 90.0)),
        },
        "candidate": {
            "clipped_mean_px": float(np.mean(candidate)),
            "clipped_median_px": float(np.median(candidate)),
            "clipped_p90_px": float(np.percentile(candidate, 90.0)),
        },
        "observation_count": int(len(differences)),
        "informative_pair_count": int(informative),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "one_sided_degradation_pvalue": p_worse,
        "significance_level": alpha,
        "base_positive_depth_count": int(base_positive),
        "candidate_positive_depth_count": int(candidate_positive),
        "skipped_empty_nodes": int(skipped_empty_nodes),
        "checks": checks,
        "nodes": node_rows,
    }
