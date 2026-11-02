from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_pooled_errors(
    values: Sequence[float],
    *,
    inlier_px: float,
) -> dict[str, float | int]:
    errors = [float(value) for value in values]
    if not errors:
        raise ValueError("reprojection errors must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in errors):
        raise ValueError("reprojection errors must be finite and non-negative")
    threshold = float(inlier_px)
    return {
        "count": len(errors),
        "median_px": _percentile(errors, 50.0),
        "raw_p90_px": _percentile(errors, 90.0),
        "raw_max_px": max(errors),
        "inlier_ratio": sum(value <= threshold for value in errors)
        / len(errors),
        "truncated_mean_px": sum(min(value, threshold) for value in errors)
        / len(errors),
    }


def _robust_pose_model_bic(
    values: Sequence[float],
    *,
    clip_px: float,
    parameter_count: int,
) -> float:
    errors = [float(value) for value in values]
    if not errors:
        raise ValueError("BIC requires at least one reprojection error")
    scalar_observation_count = 2 * len(errors)
    robust_rss = sum(min(value, float(clip_px)) ** 2 for value in errors)
    robust_rss = max(robust_rss, 1e-12)
    return (
        scalar_observation_count
        * math.log(robust_rss / scalar_observation_count)
        + int(parameter_count) * math.log(scalar_observation_count)
    )


def rank_shared_center_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        source_frame = int(value["source_frame"])
        validated_frames = int(value["validated_frame_count"])
        total_inliers = int(value["total_inlier_count"])
        median_px = float(value["median_reprojection_px"])
        if validated_frames < 0 or total_inliers < 0:
            raise ValueError("center support counts must be non-negative")
        if not math.isfinite(median_px) or median_px < 0.0:
            raise ValueError("center reprojection score must be finite and non-negative")
        value.update(
            {
                "source_frame": source_frame,
                "validated_frame_count": validated_frames,
                "total_inlier_count": total_inliers,
                "median_reprojection_px": median_px,
            }
        )
        normalized.append(value)
    normalized.sort(
        key=lambda row: (
            -int(row["validated_frame_count"]),
            -int(row["total_inlier_count"]),
            float(row["median_reprojection_px"]),
            int(row["source_frame"]),
        )
    )
    return [dict(row, rank=index + 1) for index, row in enumerate(normalized)]


def robust_geometric_median_center(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fuse track-validated center candidates without choosing one frame."""

    centers = np.asarray(
        [np.asarray(row["center"], dtype=np.float64).reshape(3) for row in rows],
        dtype=np.float64,
    ).reshape(-1, 3)
    if not len(centers):
        raise ValueError("center rows must be non-empty")
    if not np.isfinite(centers).all():
        raise ValueError("center rows must contain finite 3D centers")

    estimate = np.median(centers, axis=0)
    for _iteration in range(128):
        distances = np.linalg.norm(centers - estimate[None], axis=1)
        coincident = np.flatnonzero(distances <= 1e-12)
        if len(coincident):
            next_estimate = centers[int(coincident[0])]
        else:
            weights = 1.0 / distances
            next_estimate = np.sum(centers * weights[:, None], axis=0) / np.sum(
                weights
            )
        if float(np.linalg.norm(next_estimate - estimate)) <= 1e-12:
            estimate = next_estimate
            break
        estimate = next_estimate

    residuals = np.linalg.norm(centers - estimate[None], axis=1)
    return {
        "center": estimate.tolist(),
        "candidate_count": int(len(centers)),
        "median_residual": float(np.median(residuals)),
        "max_residual": float(np.max(residuals)),
        "estimator": "weiszfeld_geometric_median_of_track_validated_centers",
    }


def rank_consensus_center_candidates(
    primary_rows: Sequence[Mapping[str, Any]],
    auxiliary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate two geometry-only center rankings without score calibration."""

    def index_rows(
        rows: Sequence[Mapping[str, Any]],
        *,
        name: str,
    ) -> dict[int, dict[str, Any]]:
        indexed: dict[int, dict[str, Any]] = {}
        ranks: set[int] = set()
        for value in rows:
            row = dict(value)
            frame = int(row["source_frame"])
            rank = int(row["rank"])
            center = tuple(float(item) for item in row["center"])
            if len(center) != 3 or not all(math.isfinite(item) for item in center):
                raise ValueError(f"{name} center payload must have three finite values")
            if rank <= 0:
                raise ValueError(f"{name} center rank must be positive")
            if frame in indexed or rank in ranks:
                raise ValueError(f"{name} center ranking contains duplicates")
            row.update({"source_frame": frame, "rank": rank, "center": list(center)})
            indexed[frame] = row
            ranks.add(rank)
        return indexed

    primary = index_rows(primary_rows, name="primary")
    auxiliary = index_rows(auxiliary_rows, name="auxiliary")
    common = sorted(set(primary) & set(auxiliary))
    if not common:
        raise ValueError("center rankings have no common candidates")

    consensus: list[dict[str, Any]] = []
    for frame in common:
        first = primary[frame]
        second = auxiliary[frame]
        if first["center"] != second["center"]:
            raise ValueError("center payload differs between observation sources")
        primary_rank = int(first["rank"])
        auxiliary_rank = int(second["rank"])
        consensus.append(
            {
                **first,
                "primary_rank": primary_rank,
                "auxiliary_rank": auxiliary_rank,
                "rank_sum": primary_rank + auxiliary_rank,
                "worst_rank": max(primary_rank, auxiliary_rank),
            }
        )
    consensus.sort(
        key=lambda row: (
            int(row["rank_sum"]),
            int(row["worst_rank"]),
            int(row["primary_rank"]),
            int(row["auxiliary_rank"]),
            int(row["source_frame"]),
        )
    )
    return [
        dict(row, consensus_rank=index + 1)
        for index, row in enumerate(consensus)
    ]


def decide_fixed_center_commit(
    *,
    base_available: bool,
    verification: Mapping[str, Any],
    focus_inlier_px: float,
) -> dict[str, Any]:
    threshold = float(focus_inlier_px)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    if not bool(verification.get("success", False)):
        return {"commit": False, "reason": "candidate_crossfit_failed"}

    comparison = verification.get("aggregate_oof_comparison", {})
    base = comparison.get("base", {})
    candidate = comparison.get("candidate", {})
    consistency = verification.get("crossfit_camera_consistency", {})
    values = {
        "base_median": float(base.get("reproj_median_px", float("nan"))),
        "base_p90": float(base.get("reproj_p90_px", float("nan"))),
        "base_inlier": float(base.get("inlier_ratio", float("nan"))),
        "candidate_median": float(
            candidate.get("reproj_median_px", float("nan"))
        ),
        "candidate_p90": float(candidate.get("reproj_p90_px", float("nan"))),
        "candidate_inlier": float(candidate.get("inlier_ratio", float("nan"))),
        "consistency": float(
            consistency.get("max_pairwise_projection_median_px", float("nan"))
        ),
    }
    required = [
        values["candidate_median"],
        values["candidate_p90"],
        values["candidate_inlier"],
        values["consistency"],
    ]
    if base_available:
        required.extend(
            [values["base_median"], values["base_p90"], values["base_inlier"]]
        )
    if not all(math.isfinite(value) for value in required):
        return {"commit": False, "reason": "nonfinite_geometry_score"}
    if values["consistency"] > threshold:
        return {"commit": False, "reason": "crossfit_camera_inconsistent"}
    multiseed = verification.get("multiseed_verification", {})
    if not bool(multiseed.get("verified", False)):
        return {"commit": False, "reason": "insufficient_multiseed_support"}
    minimum_oof = int(verification.get("minimum_oof_inliers", 0))
    if int(multiseed.get("oof_inlier_count", 0)) < minimum_oof:
        return {"commit": False, "reason": "insufficient_oof_inliers"}

    if not base_available:
        absolute_ok = values["candidate_median"] <= threshold
        return {
            "commit": bool(absolute_ok),
            "reason": (
                "missing_base_geometry_verified"
                if absolute_ok
                else "missing_base_absolute_geometry_failed"
            ),
        }

    pareto = (
        values["candidate_median"] < values["base_median"]
        and values["candidate_p90"] <= values["base_p90"]
        and values["candidate_inlier"] >= values["base_inlier"]
    )
    return {
        "commit": bool(pareto),
        "reason": "base_pareto_improved" if pareto else "base_not_pareto_improved",
    }


def decide_fixed_center_band_commit(
    *,
    rows: Sequence[Mapping[str, Any]],
    focus_inlier_px: float,
    minimum_oof_inliers: int,
) -> dict[str, Any]:
    """Verify one shared-center candidate on pooled OOF track residuals."""
    threshold = float(focus_inlier_px)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    minimum_support = int(minimum_oof_inliers)
    if minimum_support <= 0:
        raise ValueError("minimum_oof_inliers must be positive")
    if len(rows) < 2:
        return {"commit": False, "reason": "band_requires_multiple_nodes"}

    views = {str(row.get("view", "")) for row in rows}
    if len(views) != 1:
        return {"commit": False, "reason": "band_mixed_views"}
    frames = [int(row["frame"]) for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError("band rows must have unique frames")

    base_errors: list[float] = []
    candidate_errors_on_base: list[float] = []
    candidate_errors_all: list[float] = []
    missing_base_node_count = 0
    base_node_count = 0
    missing_candidate_median_max = 0.0
    max_consistency = 0.0
    for row in rows:
        crossfit = row.get("crossfit", {})
        if not bool(crossfit.get("success", False)):
            return {"commit": False, "reason": "band_candidate_crossfit_failed"}
        comparison = crossfit.get("aggregate_oof_comparison", {})
        count = int(comparison.get("paired_count", 0))
        if count <= 0:
            return {"commit": False, "reason": "band_empty_oof_support"}

        candidate_payload = comparison.get("candidate_errors_px")
        if candidate_payload is None:
            return {
                "commit": False,
                "reason": "band_missing_paired_error_payload",
            }
        try:
            candidate_values = [float(value) for value in candidate_payload]
            candidate_summary = _summarize_pooled_errors(
                candidate_values,
                inlier_px=threshold,
            )
        except (TypeError, ValueError):
            return {"commit": False, "reason": "band_nonfinite_geometry_score"}
        if len(candidate_values) != count:
            return {
                "commit": False,
                "reason": "band_error_payload_count_mismatch",
            }
        candidate_errors_all.extend(candidate_values)

        consistency = float(
            crossfit.get("crossfit_camera_consistency", {}).get(
                "max_pairwise_projection_median_px", float("nan")
            )
        )
        if not math.isfinite(consistency):
            return {"commit": False, "reason": "band_nonfinite_geometry_score"}
        if consistency > threshold:
            return {
                "commit": False,
                "reason": "band_crossfit_camera_inconsistent",
            }
        max_consistency = max(max_consistency, consistency)

        multiseed = row.get("multiseed_verification", {})
        if not bool(multiseed.get("verified", False)):
            return {
                "commit": False,
                "reason": "band_insufficient_multiseed_support",
            }
        if int(multiseed.get("oof_inlier_count", 0)) < minimum_support:
            return {"commit": False, "reason": "band_insufficient_oof_inliers"}
        candidate_inlier_count = sum(value <= threshold for value in candidate_values)
        if candidate_inlier_count < minimum_support:
            return {
                "commit": False,
                "reason": "band_insufficient_error_payload_inliers",
            }

        base_available = bool(row.get("base_available", False))
        if not base_available:
            missing_base_node_count += 1
            missing_candidate_median_max = max(
                missing_candidate_median_max,
                float(candidate_summary["median_px"]),
            )
            continue

        base_node_count += 1

        base_payload = comparison.get("base_errors_px")
        if base_payload is None:
            return {
                "commit": False,
                "reason": "band_missing_paired_error_payload",
            }
        try:
            base_values = [float(value) for value in base_payload]
            _summarize_pooled_errors(base_values, inlier_px=threshold)
        except (TypeError, ValueError):
            return {"commit": False, "reason": "band_nonfinite_geometry_score"}
        if len(base_values) != count or len(base_values) != len(candidate_values):
            return {
                "commit": False,
                "reason": "band_error_payload_count_mismatch",
            }
        base_errors.extend(base_values)
        candidate_errors_on_base.extend(candidate_values)

    candidate_all = _summarize_pooled_errors(
        candidate_errors_all,
        inlier_px=threshold,
    )
    if not base_errors:
        absolute_ok = float(candidate_all["median_px"]) <= threshold
        return {
            "commit": bool(absolute_ok),
            "reason": (
                "band_missing_base_geometry_verified"
                if absolute_ok
                else "band_missing_base_absolute_geometry_failed"
            ),
            "view": next(iter(views)),
            "frames": sorted(frames),
            "node_count": len(rows),
            "paired_count": 0,
            "total_paired_count": len(candidate_errors_all),
            "missing_base_node_count": missing_base_node_count,
            "missing_candidate_median_max_px": missing_candidate_median_max,
            "aggregate": {
                "candidate_all_median_px": candidate_all["median_px"],
                "candidate_all_raw_p90_px": candidate_all["raw_p90_px"],
                "candidate_all_inlier_ratio": candidate_all["inlier_ratio"],
                "candidate_all_truncated_mean_px": candidate_all[
                    "truncated_mean_px"
                ],
                "max_crossfit_camera_consistency_px": max_consistency,
            },
        }

    base_summary = _summarize_pooled_errors(base_errors, inlier_px=threshold)
    candidate_summary = _summarize_pooled_errors(
        candidate_errors_on_base,
        inlier_px=threshold,
    )
    aggregate = {
        "base_median_px": base_summary["median_px"],
        "candidate_median_px": candidate_summary["median_px"],
        "base_raw_p90_px": base_summary["raw_p90_px"],
        "candidate_raw_p90_px": candidate_summary["raw_p90_px"],
        "base_inlier_ratio": base_summary["inlier_ratio"],
        "candidate_inlier_ratio": candidate_summary["inlier_ratio"],
        "base_truncated_mean_px": base_summary["truncated_mean_px"],
        "candidate_truncated_mean_px": candidate_summary[
            "truncated_mean_px"
        ],
        "candidate_all_median_px": candidate_all["median_px"],
        "candidate_all_raw_p90_px": candidate_all["raw_p90_px"],
        "candidate_all_inlier_ratio": candidate_all["inlier_ratio"],
        "candidate_all_truncated_mean_px": candidate_all[
            "truncated_mean_px"
        ],
        "max_crossfit_camera_consistency_px": max_consistency,
    }
    base_inlier_count = sum(value <= threshold for value in base_errors)
    candidate_inlier_count = sum(
        value <= threshold for value in candidate_errors_on_base
    )
    aggregate["base_inlier_count"] = base_inlier_count
    aggregate["candidate_inlier_count"] = candidate_inlier_count
    aggregate["base_free_pose_parameter_count"] = 6 * base_node_count
    aggregate["candidate_shared_center_parameter_count"] = (
        3 * base_node_count + 3
    )
    aggregate["bic_scalar_observation_count"] = 2 * len(base_errors)
    aggregate["base_free_pose_bic"] = _robust_pose_model_bic(
        base_errors,
        clip_px=threshold,
        parameter_count=aggregate["base_free_pose_parameter_count"],
    )
    aggregate["candidate_shared_center_bic"] = _robust_pose_model_bic(
        candidate_errors_on_base,
        clip_px=threshold,
        parameter_count=aggregate["candidate_shared_center_parameter_count"],
    )
    consensus_improved = (
        candidate_inlier_count > base_inlier_count
        or (
            candidate_inlier_count == base_inlier_count
            and aggregate["candidate_median_px"]
            < aggregate["base_median_px"]
        )
    )
    shared_center_bic_improved = (
        aggregate["candidate_shared_center_bic"]
        < aggregate["base_free_pose_bic"]
    )
    commit = consensus_improved or shared_center_bic_improved
    if consensus_improved:
        reason = (
            "band_consensus_improved_with_missing_base"
            if missing_base_node_count
            else "band_consensus_improved"
        )
    elif shared_center_bic_improved:
        reason = (
            "band_shared_center_bic_improved_with_missing_base"
            if missing_base_node_count
            else "band_shared_center_bic_improved"
        )
    else:
        reason = "band_consensus_not_improved"
    return {
        "commit": bool(commit),
        "reason": reason,
        "view": next(iter(views)),
        "frames": sorted(frames),
        "node_count": len(rows),
        "paired_count": len(base_errors),
        "total_paired_count": len(candidate_errors_all),
        "missing_base_node_count": missing_base_node_count,
        "missing_candidate_median_max_px": missing_candidate_median_max,
        "aggregate": aggregate,
    }


def decide_frame_balanced_fixed_center_band_commit(
    *,
    rows: Sequence[Mapping[str, Any]],
    focus_inlier_px: float,
    minimum_oof_inliers: int,
) -> dict[str, Any]:
    """Commit a band only when equal-weight frame evidence improves."""

    threshold = float(focus_inlier_px)
    minimum_support = int(minimum_oof_inliers)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    if minimum_support <= 0:
        raise ValueError("minimum_oof_inliers must be positive")
    if len(rows) < 2:
        return {"commit": False, "reason": "frame_balanced_band_requires_multiple_nodes"}
    views = {str(row.get("view", "")) for row in rows}
    if len(views) != 1:
        return {"commit": False, "reason": "frame_balanced_band_mixed_views"}
    frames = [int(row["frame"]) for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError("frame-balanced band rows must have unique frames")

    gains: list[float] = []
    max_consistency = 0.0
    missing_base_frames: list[int] = []
    for row in rows:
        frame = int(row["frame"])
        crossfit = row.get("crossfit", {})
        if not bool(crossfit.get("success", False)):
            return {
                "commit": False,
                "reason": "frame_balanced_candidate_crossfit_failed",
                "frame": frame,
            }
        comparison = crossfit.get("aggregate_oof_comparison", {})
        count = int(comparison.get("paired_count", 0))
        candidate_payload = comparison.get("candidate_errors_px")
        if count <= 0 or candidate_payload is None:
            return {
                "commit": False,
                "reason": "frame_balanced_missing_paired_error_payload",
                "frame": frame,
            }
        try:
            candidate = [float(value) for value in candidate_payload]
        except (TypeError, ValueError):
            return {
                "commit": False,
                "reason": "frame_balanced_nonfinite_geometry_score",
                "frame": frame,
            }
        if len(candidate) != count or not all(math.isfinite(value) for value in candidate):
            return {
                "commit": False,
                "reason": "frame_balanced_error_payload_count_mismatch",
                "frame": frame,
            }
        consistency = float(
            crossfit.get("crossfit_camera_consistency", {}).get(
                "max_pairwise_projection_median_px", math.nan
            )
        )
        if not math.isfinite(consistency) or consistency > threshold:
            return {
                "commit": False,
                "reason": "frame_balanced_crossfit_camera_inconsistent",
                "frame": frame,
            }
        max_consistency = max(max_consistency, consistency)
        multiseed = row.get("multiseed_verification", {})
        if (
            not bool(multiseed.get("verified", False))
            or int(multiseed.get("oof_inlier_count", 0)) < minimum_support
            or sum(value <= threshold for value in candidate) < minimum_support
        ):
            return {
                "commit": False,
                "reason": "frame_balanced_insufficient_oof_support",
                "frame": frame,
            }

        if not bool(row.get("base_available", False)):
            missing_base_frames.append(frame)
            if max(candidate) > threshold:
                return {
                    "commit": False,
                    "reason": "frame_balanced_missing_base_tail_failed",
                    "frame": frame,
                }
            continue

        base_payload = comparison.get("base_errors_px")
        if base_payload is None:
            return {
                "commit": False,
                "reason": "frame_balanced_missing_paired_error_payload",
                "frame": frame,
            }
        try:
            base = [float(value) for value in base_payload]
        except (TypeError, ValueError):
            return {
                "commit": False,
                "reason": "frame_balanced_nonfinite_geometry_score",
                "frame": frame,
            }
        if (
            len(base) != count
            or len(base) != len(candidate)
            or not all(math.isfinite(value) for value in base)
        ):
            return {
                "commit": False,
                "reason": "frame_balanced_error_payload_count_mismatch",
                "frame": frame,
            }
        base_loss = sum(min(value, threshold) ** 2 for value in base) / len(base)
        candidate_loss = sum(
            min(value, threshold) ** 2 for value in candidate
        ) / len(candidate)
        gains.append(float(base_loss - candidate_loss))

    if not gains:
        return {
            "commit": True,
            "reason": "frame_balanced_missing_base_geometry_verified",
            "view": next(iter(views)),
            "frames": sorted(frames),
            "aggregate": {
                "positive_frame_count": 0,
                "negative_frame_count": 0,
                "missing_base_frame_count": len(missing_base_frames),
                "max_crossfit_camera_consistency_px": max_consistency,
            },
        }
    positive = sum(value > 0.0 for value in gains)
    negative = sum(value < 0.0 for value in gains)
    mean_gain = float(sum(gains) / len(gains))
    commit = positive > negative and mean_gain > 0.0
    return {
        "commit": bool(commit),
        "reason": (
            "frame_balanced_gain_verified"
            if commit
            else "frame_balanced_gain_not_consistent"
        ),
        "view": next(iter(views)),
        "frames": sorted(frames),
        "aggregate": {
            "positive_frame_count": int(positive),
            "negative_frame_count": int(negative),
            "zero_frame_count": int(len(gains) - positive - negative),
            "missing_base_frame_count": int(len(missing_base_frames)),
            "mean_frame_clipped_squared_gain": mean_gain,
            "frame_clipped_squared_gains": gains,
            "max_crossfit_camera_consistency_px": max_consistency,
        },
    }


def decide_frame_balanced_fixed_center_band_commit(
    *,
    rows: Sequence[Mapping[str, Any]],
    focus_inlier_px: float,
    minimum_oof_inliers: int,
) -> dict[str, Any]:
    """Atomically verify a fixed-center band with equal weight per frame."""

    threshold = float(focus_inlier_px)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    minimum_support = int(minimum_oof_inliers)
    if minimum_support <= 0:
        raise ValueError("minimum_oof_inliers must be positive")
    if len(rows) < 2:
        return {
            "commit": False,
            "reason": "frame_balanced_band_requires_multiple_nodes",
        }

    views = {str(row.get("view", "")) for row in rows}
    if len(views) != 1:
        return {"commit": False, "reason": "frame_balanced_band_mixed_views"}
    frames = [int(row["frame"]) for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError("band rows must have unique frames")

    frame_diagnostics: list[dict[str, Any]] = []
    frame_gains: list[float] = []
    missing_base_node_count = 0
    max_consistency = 0.0
    for row in sorted(rows, key=lambda value: int(value["frame"])):
        frame = int(row["frame"])
        crossfit = row.get("crossfit", {})
        if not bool(crossfit.get("success", False)):
            return {
                "commit": False,
                "reason": "frame_balanced_candidate_crossfit_failed",
                "frame": frame,
            }
        comparison = crossfit.get("aggregate_oof_comparison", {})
        paired_count = int(comparison.get("paired_count", 0))
        if paired_count <= 0:
            return {
                "commit": False,
                "reason": "frame_balanced_empty_oof_support",
                "frame": frame,
            }

        candidate_payload = comparison.get("candidate_errors_px")
        if candidate_payload is None:
            return {
                "commit": False,
                "reason": "frame_balanced_missing_paired_error_payload",
                "frame": frame,
            }
        try:
            candidate_values = [float(value) for value in candidate_payload]
            candidate_summary = _summarize_pooled_errors(
                candidate_values,
                inlier_px=threshold,
            )
        except (TypeError, ValueError):
            return {
                "commit": False,
                "reason": "frame_balanced_nonfinite_geometry_score",
                "frame": frame,
            }
        if len(candidate_values) != paired_count:
            return {
                "commit": False,
                "reason": "frame_balanced_error_payload_count_mismatch",
                "frame": frame,
            }

        consistency = float(
            crossfit.get("crossfit_camera_consistency", {}).get(
                "max_pairwise_projection_median_px", float("nan")
            )
        )
        if not math.isfinite(consistency):
            return {
                "commit": False,
                "reason": "frame_balanced_nonfinite_geometry_score",
                "frame": frame,
            }
        if consistency > threshold:
            return {
                "commit": False,
                "reason": "frame_balanced_crossfit_camera_inconsistent",
                "frame": frame,
            }
        max_consistency = max(max_consistency, consistency)

        multiseed = row.get("multiseed_verification", {})
        if not bool(multiseed.get("verified", False)):
            return {
                "commit": False,
                "reason": "frame_balanced_insufficient_multiseed_support",
                "frame": frame,
            }
        if int(multiseed.get("oof_inlier_count", 0)) < minimum_support:
            return {
                "commit": False,
                "reason": "frame_balanced_insufficient_oof_inliers",
                "frame": frame,
            }
        candidate_inliers = sum(
            value <= threshold for value in candidate_values
        )
        if candidate_inliers < minimum_support:
            return {
                "commit": False,
                "reason": "frame_balanced_insufficient_error_payload_inliers",
                "frame": frame,
            }

        diagnostic: dict[str, Any] = {
            "frame": frame,
            "paired_count": paired_count,
            "base_available": bool(row.get("base_available", False)),
            "candidate_median_px": candidate_summary["median_px"],
            "candidate_raw_p90_px": candidate_summary["raw_p90_px"],
            "candidate_inlier_ratio": candidate_summary["inlier_ratio"],
            "crossfit_camera_consistency_px": consistency,
        }
        if not diagnostic["base_available"]:
            missing_base_node_count += 1
            if float(candidate_summary["median_px"]) > threshold:
                return {
                    "commit": False,
                    "reason": "frame_balanced_missing_base_absolute_failed",
                    "frame": frame,
                }
            if float(candidate_summary["raw_p90_px"]) > threshold:
                return {
                    "commit": False,
                    "reason": "frame_balanced_missing_base_tail_failed",
                    "frame": frame,
                }
            frame_diagnostics.append(diagnostic)
            continue

        base_payload = comparison.get("base_errors_px")
        if base_payload is None:
            return {
                "commit": False,
                "reason": "frame_balanced_missing_paired_error_payload",
                "frame": frame,
            }
        try:
            base_values = [float(value) for value in base_payload]
            base_summary = _summarize_pooled_errors(
                base_values,
                inlier_px=threshold,
            )
        except (TypeError, ValueError):
            return {
                "commit": False,
                "reason": "frame_balanced_nonfinite_geometry_score",
                "frame": frame,
            }
        if len(base_values) != paired_count or len(base_values) != len(
            candidate_values
        ):
            return {
                "commit": False,
                "reason": "frame_balanced_error_payload_count_mismatch",
                "frame": frame,
            }

        base_risk = sum(min(value, threshold) ** 2 for value in base_values) / len(
            base_values
        )
        candidate_risk = sum(
            min(value, threshold) ** 2 for value in candidate_values
        ) / len(candidate_values)
        gain = float(base_risk - candidate_risk)
        frame_gains.append(gain)
        diagnostic.update(
            {
                "base_median_px": base_summary["median_px"],
                "base_raw_p90_px": base_summary["raw_p90_px"],
                "base_inlier_ratio": base_summary["inlier_ratio"],
                "base_clipped_squared_risk": base_risk,
                "candidate_clipped_squared_risk": candidate_risk,
                "clipped_squared_gain": gain,
            }
        )
        frame_diagnostics.append(diagnostic)

    if not frame_gains:
        return {
            "commit": True,
            "reason": "frame_balanced_missing_base_verified",
            "view": next(iter(views)),
            "frames": sorted(frames),
            "node_count": len(rows),
            "missing_base_node_count": missing_base_node_count,
            "per_frame": frame_diagnostics,
            "aggregate": {
                "positive_frame_count": 0,
                "negative_frame_count": 0,
                "tie_frame_count": 0,
                "mean_frame_clipped_squared_gain": None,
                "median_frame_clipped_squared_gain": None,
                "max_crossfit_camera_consistency_px": max_consistency,
            },
        }

    tie_atol = 1e-12
    positive_count = sum(gain > tie_atol for gain in frame_gains)
    negative_count = sum(gain < -tie_atol for gain in frame_gains)
    tie_count = len(frame_gains) - positive_count - negative_count
    mean_gain = float(sum(frame_gains) / len(frame_gains))
    median_gain = float(_percentile(frame_gains, 50.0))
    commit = mean_gain > tie_atol and positive_count > negative_count
    return {
        "commit": bool(commit),
        "reason": (
            "frame_balanced_gain_verified"
            if commit
            else "frame_balanced_gain_not_consistent"
        ),
        "view": next(iter(views)),
        "frames": sorted(frames),
        "node_count": len(rows),
        "missing_base_node_count": missing_base_node_count,
        "per_frame": frame_diagnostics,
        "aggregate": {
            "positive_frame_count": positive_count,
            "negative_frame_count": negative_count,
            "tie_frame_count": tie_count,
            "mean_frame_clipped_squared_gain": mean_gain,
            "median_frame_clipped_squared_gain": median_gain,
            "max_crossfit_camera_consistency_px": max_consistency,
        },
    }


def choose_crosssource_band_candidate(
    *,
    temporal: Mapping[str, Any],
    independent: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose an already-verified band using fold-camera stability only."""

    decisions = {
        "temporal_tracks": temporal,
        "independent_same_frame": independent,
    }
    committed = [
        source
        for source, decision in decisions.items()
        if bool(decision.get("commit", False))
    ]
    scores: dict[str, float | None] = {}
    for source, decision in decisions.items():
        if source not in committed:
            scores[source] = None
            continue
        aggregate = decision.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise ValueError(f"{source} committed without aggregate geometry")
        score = float(aggregate.get("max_crossfit_camera_consistency_px", math.nan))
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"{source} has invalid cross-fit camera consistency")
        scores[source] = score

    if not committed:
        selected = "static_pnp"
        reason = "no_geometry_committed_source"
    elif len(committed) == 1:
        selected = committed[0]
        reason = "only_geometry_committed_source"
    else:
        selected = min(
            committed,
            key=lambda source: (float(scores[source]), source),
        )
        reason = "minimum_crossfit_camera_disagreement"
    return {
        "source": selected,
        "reason": reason,
        "geometry_committed": {
            source: source in committed for source in decisions
        },
        "max_crossfit_camera_consistency_px": scores,
        "new_numeric_hyperparameters": 0,
    }
