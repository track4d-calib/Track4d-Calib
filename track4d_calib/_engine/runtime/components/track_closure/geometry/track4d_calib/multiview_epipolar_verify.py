from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping

import numpy as np


def build_same_time_match_plan(
    *,
    candidate_nodes: set[tuple[int, str]],
    support_views: list[str],
    available_image_names: set[str],
) -> dict[str, Any]:
    """Plan deterministic target-to-support matching from available features."""

    normalized_support_views = sorted({str(view) for view in support_views})
    available = {str(name) for name in available_image_names}
    names_by_stem: dict[str, list[str]] = {}
    for name in sorted(available):
        names_by_stem.setdefault(PurePosixPath(name).stem, []).append(name)

    def resolve(stem: str) -> str | None:
        matches = names_by_stem.get(str(stem), [])
        if len(matches) > 1:
            raise ValueError(f"ambiguous feature stem {stem!r}: {matches}")
        return matches[0] if matches else None

    valid_nodes: list[tuple[int, str]] = []
    skipped_nodes: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []
    target_names_by_node: dict[tuple[int, str], str] = {}
    support_names_by_node: dict[tuple[int, str], dict[str, str]] = {}
    for frame, view in sorted(candidate_nodes, key=lambda node: (node[0], node[1])):
        frame = int(frame)
        view = str(view)
        target_name = resolve(f"q_f{frame:06d}_view{view}")
        if target_name is None:
            skipped_nodes.append(
                {
                    "frame": frame,
                    "view": view,
                    "reason": "target_features_missing",
                }
            )
            continue
        support_names = {}
        for support_view in normalized_support_views:
            if support_view == view:
                continue
            support_name = resolve(f"q_f{frame:06d}_view{support_view}")
            if support_name is not None:
                support_names[support_view] = support_name
        if not support_names:
            skipped_nodes.append(
                {
                    "frame": frame,
                    "view": view,
                    "reason": "support_features_missing",
                }
            )
            continue
        node = (frame, view)
        valid_nodes.append(node)
        target_names_by_node[node] = target_name
        support_names_by_node[node] = support_names
        pairs.extend((target_name, name) for name in support_names.values())
    return {
        "valid_nodes": valid_nodes,
        "skipped_nodes": skipped_nodes,
        "pairs": sorted(set(pairs)),
        "target_names_by_node": target_names_by_node,
        "support_names_by_node": support_names_by_node,
    }


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def fundamental_from_cameras(
    *,
    target_camera: Any,
    support_camera: Any,
) -> np.ndarray:
    """Build the target-to-support fundamental matrix from two calibrated poses."""

    for camera in (target_camera, support_camera):
        distortion = np.asarray(camera.D, dtype=np.float64).reshape(-1)
        if distortion.size and float(np.max(np.abs(distortion))) > 1e-12:
            raise ValueError("epipolar verification expects undistorted image points")
    target_rotation = np.asarray(target_camera.R, dtype=np.float64).reshape(3, 3)
    target_translation = np.asarray(target_camera.T, dtype=np.float64).reshape(3)
    support_rotation = np.asarray(support_camera.R, dtype=np.float64).reshape(3, 3)
    support_translation = np.asarray(support_camera.T, dtype=np.float64).reshape(3)
    relative_rotation = support_rotation @ target_rotation.T
    relative_translation = (
        support_translation - relative_rotation @ target_translation
    )
    essential = _skew(relative_translation) @ relative_rotation
    target_intrinsic_inverse = np.linalg.inv(
        np.asarray(target_camera.K, dtype=np.float64).reshape(3, 3)
    )
    support_intrinsic_inverse = np.linalg.inv(
        np.asarray(support_camera.K, dtype=np.float64).reshape(3, 3)
    )
    fundamental = (
        support_intrinsic_inverse.T @ essential @ target_intrinsic_inverse
    )
    norm = float(np.linalg.norm(fundamental))
    if norm <= 1e-15:
        raise ValueError("camera pair has a degenerate zero-baseline fundamental matrix")
    return fundamental / norm


def sampson_distances_px(
    *,
    target_camera: Any,
    support_camera: Any,
    target_points: np.ndarray,
    support_points: np.ndarray,
) -> np.ndarray:
    """Return first-order point-to-epipolar distances in pixel units."""

    target_points = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    support_points = np.asarray(support_points, dtype=np.float64).reshape(-1, 2)
    if len(target_points) != len(support_points):
        raise ValueError("target and support point counts differ")
    if not len(target_points):
        return np.empty((0,), dtype=np.float64)
    valid = np.isfinite(target_points).all(axis=1) & np.isfinite(support_points).all(axis=1)
    target_h = np.column_stack(
        [target_points[valid], np.ones(int(np.sum(valid)), dtype=np.float64)]
    )
    support_h = np.column_stack(
        [support_points[valid], np.ones(int(np.sum(valid)), dtype=np.float64)]
    )
    fundamental = fundamental_from_cameras(
        target_camera=target_camera,
        support_camera=support_camera,
    )
    lines_support = target_h @ fundamental.T
    lines_target = support_h @ fundamental
    numerator = np.sum(support_h * lines_support, axis=1) ** 2
    denominator = (
        lines_support[:, 0] ** 2
        + lines_support[:, 1] ** 2
        + lines_target[:, 0] ** 2
        + lines_target[:, 1] ** 2
    )
    return np.sqrt(numerator / np.maximum(denominator, 1e-18))


def _summarize(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mean_px": float("inf"),
            "median_px": float("inf"),
            "p90_px": float("inf"),
        }
    return {
        "count": int(len(values)),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90.0)),
    }


def robust_epipolar_noise_threshold(pair_medians_px: np.ndarray) -> dict[str, float | int]:
    """Estimate a fixed robust 3-sigma epipolar noise ceiling from support pairs."""

    values = np.asarray(pair_medians_px, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "median_px": float("inf"),
            "robust_sigma_px": float("inf"),
            "threshold_px": float("inf"),
        }
    median = float(np.median(values))
    robust_sigma = float(1.4826 * np.median(np.abs(values - median)))
    return {
        "count": int(len(values)),
        "median_px": median,
        "robust_sigma_px": robust_sigma,
        "threshold_px": float(median + 3.0 * max(robust_sigma, 1e-6)),
    }


def evaluate_same_time_epipolar_candidate(
    *,
    base_target_camera: Any,
    candidate_target_camera: Any,
    support_cameras: Mapping[str, Any],
    matches_by_view: Mapping[str, Mapping[str, np.ndarray]],
    min_matches: int,
) -> dict[str, Any]:
    """Compare one target pose on fixed same-time cross-view correspondences."""

    rows: list[dict[str, Any]] = []
    base_pair_medians: list[float] = []
    candidate_pair_medians: list[float] = []
    for view in sorted(set(support_cameras) & set(matches_by_view)):
        target_points = np.asarray(
            matches_by_view[view]["target_points"],
            dtype=np.float64,
        ).reshape(-1, 2)
        support_points = np.asarray(
            matches_by_view[view]["support_points"],
            dtype=np.float64,
        ).reshape(-1, 2)
        if len(target_points) < int(min_matches):
            rows.append(
                {
                    "support_view": str(view),
                    "match_count": int(len(target_points)),
                    "valid": False,
                    "reason": "insufficient_matches",
                }
            )
            continue
        base = _summarize(
            sampson_distances_px(
                target_camera=base_target_camera,
                support_camera=support_cameras[view],
                target_points=target_points,
                support_points=support_points,
            )
        )
        candidate = _summarize(
            sampson_distances_px(
                target_camera=candidate_target_camera,
                support_camera=support_cameras[view],
                target_points=target_points,
                support_points=support_points,
            )
        )
        base_median = float(base["median_px"])
        candidate_median = float(candidate["median_px"])
        base_p90 = float(base["p90_px"])
        candidate_p90 = float(candidate["p90_px"])
        base_pair_medians.append(base_median)
        candidate_pair_medians.append(candidate_median)
        rows.append(
            {
                "support_view": str(view),
                "match_count": int(len(target_points)),
                "valid": True,
                "base": base,
                "candidate": candidate,
                "median_gain_px": float(base_median - candidate_median),
                "improved": bool(candidate_median < base_median - 1e-12),
                "regressed": bool(candidate_median > base_median + 1e-12),
                "p90_gain_px": float(base_p90 - candidate_p90),
                "p90_improved": bool(candidate_p90 < base_p90 - 1e-12),
                "p90_regressed": bool(candidate_p90 > base_p90 + 1e-12),
            }
        )

    valid_rows = [row for row in rows if bool(row.get("valid", False))]
    base_aggregate = (
        float(np.median(base_pair_medians)) if base_pair_medians else float("inf")
    )
    candidate_aggregate = (
        float(np.median(candidate_pair_medians))
        if candidate_pair_medians
        else float("inf")
    )
    improved_count = int(sum(bool(row["improved"]) for row in valid_rows))
    regressed_count = int(sum(bool(row["regressed"]) for row in valid_rows))
    p90_improved_count = int(sum(bool(row["p90_improved"]) for row in valid_rows))
    p90_regressed_count = int(sum(bool(row["p90_regressed"]) for row in valid_rows))
    uniform_median_and_tail_gain = bool(
        valid_rows
        and improved_count == len(valid_rows)
        and p90_improved_count == len(valid_rows)
    )
    accepted = bool(
        valid_rows
        and candidate_aggregate < base_aggregate - 1e-12
        and improved_count > regressed_count
    )
    return {
        "policy": "fixed_same_time_matches_equal_support_view_median_and_majority_gain",
        "accepted": accepted,
        "reason": (
            "epipolar_median_and_majority_gain"
            if accepted
            else "no_valid_support_pairs"
            if not valid_rows
            else "epipolar_median_or_majority_not_improved"
        ),
        "min_matches": int(min_matches),
        "valid_pair_count": int(len(valid_rows)),
        "improved_pair_count": improved_count,
        "regressed_pair_count": regressed_count,
        "p90_improved_pair_count": p90_improved_count,
        "p90_regressed_pair_count": p90_regressed_count,
        "uniform_median_and_p90_gain": uniform_median_and_tail_gain,
        "base_pair_median_px": base_aggregate,
        "candidate_pair_median_px": candidate_aggregate,
        "pair_median_gain_px": float(base_aggregate - candidate_aggregate),
        "rows": rows,
    }


def evaluate_static_target_epipolar_band(
    *,
    base_target_cameras: Mapping[str, Any],
    candidate_target_cameras: Mapping[str, Any],
    static_cameras: Mapping[str, Any],
    matches_by_target: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Any]:
    """Verify an atomic target-camera band on independent static-target matches.

    The fixed correspondences are scored under both camera hypotheses.  No
    absolute pixel threshold is used: the candidate must improve pooled,
    target-balanced, and static-support-balanced paired evidence.
    """

    target_views = sorted(str(view) for view in base_target_cameras)
    static_views = sorted(str(view) for view in static_cameras)
    if not target_views or not static_views:
        raise ValueError("target and static camera rosters must be non-empty")
    if set(target_views) != {str(view) for view in candidate_target_cameras}:
        raise ValueError("base and candidate target camera rosters differ")

    expected_pairs = {
        (target_view, static_view)
        for target_view in target_views
        for static_view in static_views
    }
    available_pairs = {
        (str(target_view), str(static_view))
        for target_view, support_matches in matches_by_target.items()
        for static_view in support_matches
    }
    missing_pairs = sorted(expected_pairs - available_pairs)
    if missing_pairs:
        return {
            "policy": "paired_dense_static_target_atomic_band_v1",
            "accepted": False,
            "reason": "incomplete_fixed_evidence",
            "target_views": target_views,
            "static_views": static_views,
            "expected_pair_count": int(len(expected_pairs)),
            "pair_count": int(len(expected_pairs) - len(missing_pairs)),
            "missing_pairs": [list(pair) for pair in missing_pairs],
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    pooled_base: list[np.ndarray] = []
    pooled_candidate: list[np.ndarray] = []
    pooled_log_gain: list[np.ndarray] = []
    pair_gain_by_target: dict[str, list[float]] = {
        view: [] for view in target_views
    }
    pair_gain_by_support: dict[str, list[float]] = {
        view: [] for view in static_views
    }
    for target_view in target_views:
        for static_view in static_views:
            pair_matches = matches_by_target[target_view][static_view]
            target_points = np.asarray(
                pair_matches["target_points"], dtype=np.float64
            ).reshape(-1, 2)
            support_points = np.asarray(
                pair_matches["support_points"], dtype=np.float64
            ).reshape(-1, 2)
            if len(target_points) != len(support_points):
                raise ValueError(
                    f"unaligned fixed evidence for {target_view} and {static_view}"
                )
            if not len(target_points):
                return {
                    "policy": "paired_dense_static_target_atomic_band_v1",
                    "accepted": False,
                    "reason": "empty_fixed_evidence",
                    "target_views": target_views,
                    "static_views": static_views,
                    "expected_pair_count": int(len(expected_pairs)),
                    "pair_count": int(len(rows)),
                    "empty_pair": [target_view, static_view],
                    "rows": rows,
                }
            base_errors = sampson_distances_px(
                target_camera=base_target_cameras[target_view],
                support_camera=static_cameras[static_view],
                target_points=target_points,
                support_points=support_points,
            )
            candidate_errors = sampson_distances_px(
                target_camera=candidate_target_cameras[target_view],
                support_camera=static_cameras[static_view],
                target_points=target_points,
                support_points=support_points,
            )
            if len(base_errors) != len(target_points) or len(candidate_errors) != len(
                target_points
            ):
                raise ValueError(
                    f"non-finite fixed evidence for {target_view} and {static_view}"
                )
            log_gain = np.log1p(base_errors) - np.log1p(candidate_errors)
            pair_median_gain = float(np.median(log_gain))
            pair_mean_gain = float(np.mean(log_gain))
            pair_gain_by_target[target_view].append(pair_median_gain)
            pair_gain_by_support[static_view].append(pair_median_gain)
            pooled_base.append(base_errors)
            pooled_candidate.append(candidate_errors)
            pooled_log_gain.append(log_gain)
            rows.append(
                {
                    "target_view": target_view,
                    "static_view": static_view,
                    "match_count": int(len(target_points)),
                    "base": _summarize(base_errors),
                    "candidate": _summarize(candidate_errors),
                    "mean_log_gain": pair_mean_gain,
                    "median_log_gain": pair_median_gain,
                }
            )

    base_values = np.concatenate(pooled_base)
    candidate_values = np.concatenate(pooled_candidate)
    gain_values = np.concatenate(pooled_log_gain)
    pooled_mean_gain = float(np.mean(gain_values))
    pooled_median_gain = float(np.median(gain_values))
    target_gains = {
        view: float(np.median(values))
        for view, values in pair_gain_by_target.items()
    }
    support_gains = {
        view: float(np.median(values))
        for view, values in pair_gain_by_support.items()
    }
    target_balanced_gain = float(np.median(list(target_gains.values())))
    support_balanced_gain = float(np.median(list(support_gains.values())))
    epsilon = 1e-12
    checks = {
        "pooled_mean_improved": pooled_mean_gain > epsilon,
        "pooled_median_improved": pooled_median_gain > epsilon,
        "target_balanced_improved": target_balanced_gain > epsilon,
        "support_balanced_improved": support_balanced_gain > epsilon,
    }
    accepted = bool(all(checks.values()))
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "paired_dense_static_target_atomic_band_v1",
        "accepted": accepted,
        "reason": (
            "independent_static_target_geometry_improved"
            if accepted
            else "independent_static_target_geometry_not_improved"
        ),
        "target_views": target_views,
        "static_views": static_views,
        "expected_pair_count": int(len(expected_pairs)),
        "pair_count": int(len(rows)),
        "match_count": int(len(base_values)),
        "pooled": {
            "base": _summarize(base_values),
            "candidate": _summarize(candidate_values),
            "mean_log_gain": pooled_mean_gain,
            "median_log_gain": pooled_median_gain,
        },
        "target_balanced": {
            "median_pair_gain_by_target": target_gains,
            "median_target_pair_gain": target_balanced_gain,
        },
        "support_balanced": {
            "median_pair_gain_by_support": support_gains,
            "median_support_pair_gain": support_balanced_gain,
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "new_tuned_numeric_thresholds": 0,
        "new_objective_weights": 0,
        "rows": rows,
    }
