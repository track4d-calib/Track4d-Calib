#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import (
    CameraIntrinsicsLoader,
    CameraLoader,
    CameraParams,
    save_cameras,
)
from track4d_calib.multiview_epipolar_verify import sampson_distances_px
from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    score_same_time_multiview_poses,
    solve_same_time_multiview_pnp,
)
from scripts.geometry.run_track_closure import (
    build_node_observations,
    seed_point_ranges,
    verify_multiseed_inlier_support,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _camera_file_hashes(camera_root: Path) -> dict[str, str]:
    return {
        name: _sha256(camera_root / name)
        for name in ("intri.yml", "extri.yml")
    }


def _load_base_frame_cameras(
    *,
    camera_root: Path,
    static_views: list[str],
    target_views: list[str],
    allow_missing_targets: bool,
) -> tuple[dict[str, CameraParams], list[str]]:
    """Load a frame while treating only absent target extrinsics as recoverable."""

    loader = CameraLoader(camera_root)
    cameras = loader.load_cameras(static_views)
    missing_targets: list[str] = []
    for view in target_views:
        try:
            cameras[view] = loader.load_camera(view)
        except ValueError as error:
            if "not found in extrinsics file" not in str(error):
                raise
            if not allow_missing_targets:
                raise
            missing_targets.append(view)
    return cameras, missing_targets


def _verify_prediction_pose_lineage(
    *,
    predictions: Mapping[str, Any],
    frames: list[int],
    label: str,
) -> Path:
    pose_root = Path(str(predictions.get("pose_root", ""))).resolve()
    if not pose_root.is_dir():
        raise FileNotFoundError(f"{label} pose root is missing: {pose_root}")
    declared = predictions.get("output_camera_hashes")
    if not isinstance(declared, Mapping):
        raise ValueError(f"{label} prediction is missing output camera hashes")
    frame_names = [f"f{frame:06d}" for frame in frames]
    if set(str(name) for name in declared) != set(frame_names):
        raise ValueError(f"{label} output camera hash frame set differs")
    actual = {
        frame_name: _camera_file_hashes(
            pose_root / frame_name / "cameras_pred"
        )
        for frame_name in frame_names
    }
    normalized = {
        str(frame_name): {
            str(name): str(value) for name, value in hashes.items()
        }
        for frame_name, hashes in declared.items()
    }
    if actual != normalized:
        raise ValueError(f"{label} output camera hash mismatch")
    return pose_root


def _verify_base_pose_lineage(
    *,
    base_pose_root: Path,
    base_pose_method: str,
    base_prediction_attestation: Path | None,
    runtime_digest: str,
    runtime_scene: Path,
    frames: list[int],
    static_views: list[str],
    target_views: list[str],
    require_track_bundles: bool,
) -> dict[str, Any]:
    if base_pose_method != "static_map_same_frame_pnp":
        if base_prediction_attestation is not None:
            raise ValueError(
                "base prediction attestation is currently defined only for "
                "static_map_same_frame_pnp"
            )
        return {
            "attestation_path": None,
            "prediction_path": None,
            "prediction_sha256": None,
            "protocol": "external-static-pnp-lineage",
            "matcher_invocations": None,
            "fit_point_budget": None,
            "config": None,
            "track_root_by_frame": {},
            "prediction_rows": [],
            "camera_hashes": {
                f"f{frame:06d}": _camera_file_hashes(
                    base_pose_root / f"f{frame:06d}" / "cameras_pred"
                )
                for frame in frames
            },
        }

    if base_prediction_attestation is None:
        raise ValueError(
            "static_map_same_frame_pnp requires a pre-GT base prediction attestation"
        )
    attestation_path = Path(base_prediction_attestation).resolve()
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    prediction_path = Path(str(attestation.get("path", ""))).resolve()
    prediction_sha256 = _sha256(prediction_path)
    if prediction_sha256 != str(attestation.get("sha256", "")):
        raise ValueError("base prediction attestation hash does not match")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if bool(predictions.get("gt_used", True)) or bool(
        predictions.get("nvs_used", True)
    ):
        raise ValueError("base pose prediction must be pre-GT and pre-NVS")
    if str(predictions.get("runtime_tree_digest", "")) != str(runtime_digest):
        raise ValueError("base pose prediction uses a different runtime tree")
    if str(predictions.get("track_source_mode", "")) != "independent_per_frame_bundle":
        raise ValueError(
            "static_map_same_frame_pnp must use independent per-frame bundles"
        )
    if Path(str(predictions.get("pose_root", ""))).resolve() != base_pose_root:
        raise ValueError("base pose root does not match its prediction manifest")
    if [int(frame) for frame in predictions.get("frames", [])] != frames:
        raise ValueError("base pose frames do not match temporal closure frames")
    if [str(view) for view in predictions.get("target_views", [])] != target_views:
        raise ValueError("base pose target views do not match temporal closure targets")
    matcher_roots = [str(path) for path in predictions.get("track_roots", [])]
    if len(matcher_roots) != len(frames):
        raise ValueError(
            "static_map_same_frame_pnp must attest one matcher bundle per frame"
        )
    track_root_by_frame: dict[int, str] = {}
    if require_track_bundles:
        if [str(view) for view in predictions.get("static_views", [])] != static_views:
            raise ValueError("base matcher static views do not match runtime")
        expected_views = static_views + target_views
        expected_data_root = (runtime_scene / "images").resolve()
        for root_text in matcher_roots:
            root = Path(root_text).resolve()
            _verify_track_files(track_root=root, predictions=predictions)
            meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
            views = (root / "views.txt").read_text(encoding="utf-8").split()
            if views != expected_views:
                raise ValueError(
                    "base matcher bundle must contain exactly static plus train views"
                )
            frame = int(meta.get("start_frame", -1))
            if int(meta.get("num_frames", 0)) != 1 or frame not in frames:
                raise ValueError("base matcher bundle is not a declared one-frame input")
            if frame in track_root_by_frame:
                raise ValueError("duplicate base matcher bundle frame")
            if Path(str(meta.get("data_root", ""))).resolve() != expected_data_root:
                raise ValueError("base matcher bundle uses a different image root")
            track_root_by_frame[frame] = str(root)
        if set(track_root_by_frame) != set(frames):
            raise ValueError("base matcher bundles do not cover the frame set")

    declared_hashes = predictions.get("output_camera_hashes")
    if not isinstance(declared_hashes, Mapping):
        raise ValueError("base prediction manifest is missing camera hashes")
    expected_frame_names = [f"f{frame:06d}" for frame in frames]
    if set(str(name) for name in declared_hashes) != set(expected_frame_names):
        raise ValueError("base prediction camera hash frame set does not match")
    actual_hashes = {
        frame_name: _camera_file_hashes(
            base_pose_root / frame_name / "cameras_pred"
        )
        for frame_name in expected_frame_names
    }
    if actual_hashes != {
        str(frame_name): {str(name): str(value) for name, value in hashes.items()}
        for frame_name, hashes in declared_hashes.items()
    }:
        raise ValueError("base camera hash mismatch")
    return {
        "attestation_path": str(attestation_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "protocol": str(predictions.get("protocol", "")),
        "matcher_invocations": int(len(matcher_roots)),
        "fit_point_budget": int(predictions.get("fit_point_budget", 0)),
        "config": dict(predictions.get("config") or {}),
        "track_root_by_frame": track_root_by_frame,
        "prediction_rows": list(predictions.get("rows", [])),
        "camera_hashes": actual_hashes,
    }


def _config_from_prediction(predictions: Mapping[str, Any]) -> SameTimePnPConfig:
    values = dict(predictions.get("config") or {})
    allowed = set(SameTimePnPConfig.__dataclass_fields__)
    return SameTimePnPConfig(**{key: value for key, value in values.items() if key in allowed})


def validate_track_seed_builder(*, seed_builder: str, commit_scope: str) -> None:
    if seed_builder == "spsg_target_closure":
        return
    if commit_scope == "node" and seed_builder in {
        "spsg_target_star",
        "spsg_target_relay",
    }:
        return
    raise ValueError(
        "band geometry closure requires seed_builder=spsg_target_closure; "
        "node commits also support spsg_target_star and spsg_target_relay"
    )


def _verify_track_files(
    *,
    track_root: Path,
    predictions: Mapping[str, Any],
) -> dict[str, str]:
    declared_by_root = predictions.get("track_files_sha256")
    if not isinstance(declared_by_root, Mapping):
        raise ValueError("track prediction manifest is missing track file hashes")
    declared = declared_by_root.get(str(track_root.resolve()))
    if not isinstance(declared, Mapping):
        raise ValueError("track prediction manifest does not attest the selected track root")
    required = ["meta.json", "views.txt", "tracks.npy", "visibility.npy"]
    quality_path = track_root / "track_quality.npy"
    if quality_path.is_file():
        required.append("track_quality.npy")
    actual = {name: _sha256(track_root / name) for name in required}
    if set(str(name) for name in declared) != set(required):
        raise ValueError("attested track file set does not match the current bundle")
    for name, digest in actual.items():
        if str(declared.get(name, "")) != digest:
            raise ValueError(f"track file hash mismatch: {name}")
    return actual


def _score_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
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
        "truncated_reprojection_mean_px",
    ]
    return {key: result[key] for key in keys if key in result}


def decide_track_commit(
    *,
    solver_success: bool,
    multiseed_verified: bool,
    static_score: Mapping[str, Any],
    track_score: Mapping[str, Any],
    commit_policy: str = "mean",
) -> tuple[bool, str]:
    if commit_policy not in {"mean", "confidence95"}:
        raise ValueError(f"unsupported commit policy: {commit_policy}")
    if not bool(solver_success):
        return False, "track_solver_failed"
    if not bool(multiseed_verified):
        return False, "track_multiseed_verification_failed"
    scores_valid = bool(static_score.get("success") and track_score.get("success"))
    if not scores_valid:
        return False, "holdout_inconclusive_keep_base"
    if commit_policy == "confidence95":
        confidence = paired_reprojection_confidence(static_score, track_score)
        if not bool(confidence["valid"]):
            return False, "holdout_confidence_unavailable_keep_base"
        if float(confidence["gain_lower_bound_one_sided_95_px"]) > 0.0:
            return True, "heldout_confidence95_prefers_track"
        if float(confidence["gain_upper_bound_one_sided_95_px"]) < 0.0:
            return False, "heldout_confidence95_prefers_static"
        return False, "heldout_confidence95_inconclusive_keep_base"
    static_value = float(static_score["truncated_reprojection_mean_px"])
    track_value = float(track_score["truncated_reprojection_mean_px"])
    if track_value < static_value:
        return True, "heldout_geometry_prefers_track"
    return False, "heldout_geometry_prefers_static"


def paired_reprojection_confidence(
    static_score: Mapping[str, Any],
    track_score: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate a paired one-sided 95% lower bound for track improvement."""

    if not bool(static_score.get("success") and track_score.get("success")):
        return {"valid": False, "reason": "candidate_score_failed"}
    static_values = np.asarray(
        static_score.get("truncated_reprojection_values_px", []), dtype=np.float64
    ).reshape(-1)
    track_values = np.asarray(
        track_score.get("truncated_reprojection_values_px", []), dtype=np.float64
    ).reshape(-1)
    if len(static_values) != len(track_values) or len(static_values) < 2:
        return {"valid": False, "reason": "insufficient_paired_residuals"}
    if not np.isfinite(static_values).all() or not np.isfinite(track_values).all():
        return {"valid": False, "reason": "nonfinite_paired_residuals"}
    gain = static_values - track_values
    gain_mean = float(np.mean(gain))
    gain_std = float(np.std(gain, ddof=1))
    gain_standard_error = float(gain_std / np.sqrt(len(gain)))
    half_width = float(1.6448536269514722 * gain_standard_error)
    lower_bound = float(gain_mean - half_width)
    upper_bound = float(gain_mean + half_width)
    return {
        "valid": True,
        "reason": "ok",
        "count": int(len(gain)),
        "gain_mean_px": gain_mean,
        "gain_std_px": gain_std,
        "gain_standard_error_px": gain_standard_error,
        "gain_lower_bound_one_sided_95_px": lower_bound,
        "gain_upper_bound_one_sided_95_px": upper_bound,
    }


def crossfit_point_id_folds(
    *,
    point_ids: list[int] | tuple[int, ...],
    point_ranges: list[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    """Split seed banks into deterministic folds with globally balanced budgets."""

    ranges = [tuple(int(value) for value in row["point_range"]) for row in point_ranges]
    if any(start < 0 or stop < start for start, stop in ranges):
        raise ValueError("invalid temporal seed point range")
    ordered_ids: list[int] = []
    buckets: list[list[int]] = [[] for _range in ranges]
    seen: set[int] = set()
    for raw_point_id in point_ids:
        point_id = int(raw_point_id)
        if point_id in seen:
            raise ValueError("crossfit point IDs must be unique")
        seen.add(point_id)
        memberships = [
            index
            for index, (start, stop) in enumerate(ranges)
            if start <= point_id < stop
        ]
        if len(memberships) != 1:
            raise ValueError("each crossfit point must belong to exactly one seed bank")
        ordered_ids.append(point_id)
        buckets[memberships[0]].append(point_id)

    first_ids: set[int] = set()
    second_ids: set[int] = set()
    for bucket in buckets:
        start_with_first = len(first_ids) <= len(second_ids)
        for offset, point_id in enumerate(bucket):
            use_first = (offset % 2 == 0) == start_with_first
            (first_ids if use_first else second_ids).add(point_id)
    first = [point_id for point_id in ordered_ids if point_id in first_ids]
    second = [point_id for point_id in ordered_ids if point_id in second_ids]
    if abs(len(first) - len(second)) > 1:
        raise RuntimeError("crossfit fold construction failed to balance point budgets")
    return first, second


def stratified_crossfit_point_id_folds(
    *,
    point_ids: list[int] | tuple[int, ...],
    point_ranges: list[Mapping[str, Any]],
    strata_by_point_id: Mapping[int, Any],
) -> tuple[list[int], list[int]]:
    """Split each seed and target-visibility stratum as evenly as possible."""

    ranges = [tuple(int(value) for value in row["point_range"]) for row in point_ranges]
    if any(start < 0 or stop < start for start, stop in ranges):
        raise ValueError("invalid temporal seed point range")

    groups: dict[tuple[int, tuple[str, ...]], list[int]] = {}
    seen: set[int] = set()
    for raw_point_id in point_ids:
        point_id = int(raw_point_id)
        if point_id in seen:
            raise ValueError("crossfit point IDs must be unique")
        seen.add(point_id)
        memberships = [
            index
            for index, (start, stop) in enumerate(ranges)
            if start <= point_id < stop
        ]
        if len(memberships) != 1:
            raise ValueError("each crossfit point must belong to exactly one seed bank")
        if point_id not in strata_by_point_id:
            raise ValueError(f"missing crossfit stratum for point {point_id}")
        raw_stratum = strata_by_point_id[point_id]
        if isinstance(raw_stratum, (list, tuple, set, frozenset)):
            stratum = tuple(sorted(str(value) for value in raw_stratum))
        else:
            stratum = (str(raw_stratum),)
        groups.setdefault((memberships[0], stratum), []).append(point_id)

    first: list[int] = []
    second: list[int] = []
    next_start_by_range = [0 for _range in ranges]
    for (range_index, _stratum), group in sorted(groups.items()):
        start_fold = int(next_start_by_range[range_index])
        for offset, point_id in enumerate(sorted(group)):
            (first if (start_fold + offset) % 2 == 0 else second).append(point_id)
        if len(group) % 2:
            next_start_by_range[range_index] = 1 - start_fold
    return sorted(first), sorted(second)


def exact_holdout_point_id_split(
    *,
    point_ids: list[int] | tuple[int, ...],
    point_ranges: list[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    """Build a deterministic seed-balanced 2:1 fit/verification split."""

    ranges = [tuple(int(value) for value in row["point_range"]) for row in point_ranges]
    if not ranges or any(start < 0 or stop < start for start, stop in ranges):
        raise ValueError("invalid temporal seed point range")
    buckets: list[list[int]] = [[] for _range in ranges]
    seen: set[int] = set()
    for raw_point_id in point_ids:
        point_id = int(raw_point_id)
        if point_id in seen:
            raise ValueError("holdout point IDs must be unique")
        seen.add(point_id)
        memberships = [
            index
            for index, (start, stop) in enumerate(ranges)
            if start <= point_id < stop
        ]
        if len(memberships) != 1:
            raise ValueError("each holdout point must belong to exactly one seed bank")
        buckets[memberships[0]].append(point_id)

    target_fit_count = (2 * len(seen)) // 3
    fit_count_by_bucket = [(2 * len(bucket)) // 3 for bucket in buckets]
    unassigned = target_fit_count - sum(fit_count_by_bucket)
    allocation_order = sorted(
        range(len(buckets)),
        key=lambda index: (-((2 * len(buckets[index])) % 3), index),
    )
    for bucket_index in allocation_order[:unassigned]:
        fit_count_by_bucket[bucket_index] += 1

    fit_set = {
        point_id
        for bucket, fit_count in zip(buckets, fit_count_by_bucket)
        for point_id in bucket[:fit_count]
    }
    fit_ids = [int(point_id) for point_id in point_ids if int(point_id) in fit_set]
    verify_ids = [
        int(point_id) for point_id in point_ids if int(point_id) not in fit_set
    ]
    return fit_ids, verify_ids


def capped_exact_holdout_point_id_split(
    *,
    point_ids: list[int] | tuple[int, ...],
    point_ranges: list[Mapping[str, Any]],
    fit_point_budget: int,
) -> tuple[list[int], list[int]]:
    """Build a seed-balanced 2:1 holdout while treating fit budget as a cap."""

    budget = int(fit_point_budget)
    if budget <= 0:
        raise ValueError("fit_point_budget must be positive")
    ordered_ids = [int(point_id) for point_id in point_ids]
    fit_ids, _verify_ids = exact_holdout_point_id_split(
        point_ids=ordered_ids,
        point_ranges=point_ranges,
    )
    if len(fit_ids) > budget:
        fit_ids = balanced_refit_point_ids(
            point_ids=fit_ids,
            point_ranges=point_ranges,
            budget=budget,
        )
    fit_set = set(fit_ids)
    verify_ids = [point_id for point_id in ordered_ids if point_id not in fit_set]
    return fit_ids, verify_ids


def select_base_validation_point_ids(
    *,
    point_ids: list[int] | tuple[int, ...],
    upstream_fit_count: int,
    fit_point_budget: int,
) -> dict[str, Any]:
    """Select unseen matcher points, or conservatively reuse its fit points."""

    budget = int(fit_point_budget)
    if budget <= 0:
        raise ValueError("fit_point_budget must be positive")
    ordered_ids = [int(point_id) for point_id in point_ids]
    reconstructed_fit_count = min(len(ordered_ids), budget)
    if int(upstream_fit_count) != reconstructed_fit_count:
        return {
            "point_ids": [],
            "source": "static_map_same_frame_pnp_unavailable",
            "fit_reuse": False,
            "reason": "base_fit_prefix_does_not_match_attested_candidate",
        }
    unseen = ordered_ids[reconstructed_fit_count:]
    if unseen:
        return {
            "point_ids": unseen,
            "source": "static_map_same_frame_pnp_holdout",
            "fit_reuse": False,
            "reason": "unseen_holdout",
        }
    return {
        "point_ids": ordered_ids[:reconstructed_fit_count],
        "source": "static_map_same_frame_pnp_fit_reuse_conservative",
        "fit_reuse": True,
        "reason": "no_unseen_points_reuse_base_fit_conservatively",
    }


def prediction_row_base_pose_available(prediction_row: Mapping[str, Any]) -> bool:
    """Read pose availability from direct or temporally completed PnP rows."""

    selected_source = prediction_row.get("selected_source")
    if selected_source is None:
        return bool((prediction_row.get("solver") or {}).get("success", False))
    source = str(selected_source)
    if source == "temporal_interpolation":
        return isinstance(prediction_row.get("temporal_completion"), Mapping)
    solver_key = {
        "primary": "primary_solver",
        "rescue": "rescue_solver",
        "tertiary": "tertiary_solver",
    }.get(source)
    if solver_key is None:
        raise ValueError(f"unsupported completed Static+PnP source: {source}")
    return bool((prediction_row.get(solver_key) or {}).get("success", False))


def select_prediction_row_base_validation_point_ids(
    *,
    point_ids: list[int] | tuple[int, ...],
    prediction_row: Mapping[str, Any],
    fit_point_budget: int,
) -> dict[str, Any]:
    """Select validation points using the completed base pose's true lineage."""

    ordered_ids = [int(point_id) for point_id in point_ids]
    selected_source = prediction_row.get("selected_source")
    if selected_source is not None:
        source = str(selected_source)
        if source == "temporal_interpolation":
            return {
                "point_ids": ordered_ids,
                "source": "same_frame_static_map_independent_of_temporal_completion",
                "fit_reuse": False,
                "reason": "temporal_completion_used_no_same_frame_fit_points",
            }
        solver_key = {
            "primary": "primary_solver",
            "rescue": "rescue_solver",
            "tertiary": "tertiary_solver",
        }.get(source)
        if solver_key is None:
            raise ValueError(f"unsupported completed Static+PnP source: {source}")
        upstream_fit_count = int(
            (prediction_row.get(solver_key) or {}).get(
                "candidate_target_observation_count", -1
            )
        )
    else:
        upstream_fit_count = int(
            (prediction_row.get("observation") or {}).get("selected_count", -1)
        )
    return select_base_validation_point_ids(
        point_ids=ordered_ids,
        upstream_fit_count=upstream_fit_count,
        fit_point_budget=fit_point_budget,
    )


def score_target_pair_crossfit(
    *,
    left_observations: Mapping[int, np.ndarray],
    right_observations: Mapping[int, np.ndarray],
    static_left_camera: CameraParams,
    static_right_camera: CameraParams,
    candidate_left_cameras: list[CameraParams | None],
    candidate_right_cameras: list[CameraParams | None],
    fold_point_ids: list[list[int]] | tuple[list[int], list[int]],
    min_shared_points: int = 8,
    truncation_px: float = 4.0,
) -> dict[str, Any]:
    """Score a camera pair on the fold disjoint from candidate fitting."""

    if len(candidate_left_cameras) != 2 or len(candidate_right_cameras) != 2:
        raise ValueError("crossfit target-pair scoring requires two candidate cameras")
    if len(fold_point_ids) != 2:
        raise ValueError("crossfit target-pair scoring requires two point folds")
    minimum = max(8, int(min_shared_points))
    cap = float(truncation_px)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("truncation_px must be positive and finite")

    shared = set(int(point_id) for point_id in left_observations) & set(
        int(point_id) for point_id in right_observations
    )
    rows: list[dict[str, Any]] = []
    for fit_index in range(2):
        validation_index = 1 - fit_index
        validation_ids = [
            int(point_id)
            for point_id in fold_point_ids[validation_index]
            if int(point_id) in shared
        ]
        row: dict[str, Any] = {
            "fit_fold": int(fit_index),
            "validation_fold": int(validation_index),
            "shared_point_count": int(len(validation_ids)),
            "valid": False,
        }
        left_candidate = candidate_left_cameras[fit_index]
        right_candidate = candidate_right_cameras[fit_index]
        if len(validation_ids) < minimum:
            row["reason"] = "insufficient_shared_points"
            rows.append(row)
            continue
        if left_candidate is None or right_candidate is None:
            row["reason"] = "candidate_camera_missing"
            rows.append(row)
            continue
        left_points = np.asarray(
            [left_observations[point_id] for point_id in validation_ids],
            dtype=np.float64,
        )
        right_points = np.asarray(
            [right_observations[point_id] for point_id in validation_ids],
            dtype=np.float64,
        )
        static_errors = sampson_distances_px(
            target_camera=static_left_camera,
            support_camera=static_right_camera,
            target_points=left_points,
            support_points=right_points,
        )
        track_errors = sampson_distances_px(
            target_camera=left_candidate,
            support_camera=right_candidate,
            target_points=left_points,
            support_points=right_points,
        )
        if len(static_errors) != len(validation_ids) or len(track_errors) != len(
            validation_ids
        ):
            row["reason"] = "nonfinite_pair_observations"
            rows.append(row)
            continue
        static_values = np.minimum(static_errors, cap)
        track_values = np.minimum(track_errors, cap)
        static_mean = float(np.mean(static_values))
        track_mean = float(np.mean(track_values))
        row.update(
            {
                "valid": True,
                "reason": "ok",
                "static_mean_px": static_mean,
                "track_mean_px": track_mean,
                "gain_mean_px": float(static_mean - track_mean),
                "prefers_track": bool(track_mean < static_mean),
            }
        )
        rows.append(row)

    supported = all(row["shared_point_count"] >= minimum for row in rows)
    valid = bool(supported and all(row["valid"] for row in rows))
    return {
        "supported": bool(supported),
        "valid": valid,
        "min_shared_points": int(minimum),
        "truncation_px": cap,
        "shared_point_count": int(len(shared)),
        "fold_rows": rows,
        "both_directions_prefer_track": bool(
            valid and all(bool(row["prefers_track"]) for row in rows)
        ),
    }


def decide_target_band_commits(
    *,
    target_views: list[str],
    node_commits: Mapping[str, bool],
    pair_scores: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically accept connected target bands supported by shared tracks."""

    views = [str(view) for view in target_views]
    if len(views) != len(set(views)):
        raise ValueError("target band views must be unique")
    parent = {view: view for view in views}

    def find(view: str) -> str:
        while parent[view] != view:
            parent[view] = parent[parent[view]]
            view = parent[view]
        return view

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    normalized_scores: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_pair, score in pair_scores.items():
        left, right = (str(raw_pair[0]), str(raw_pair[1]))
        if left == right or left not in parent or right not in parent:
            raise ValueError(f"invalid target pair: {raw_pair}")
        pair = tuple(sorted((left, right)))
        normalized_scores[pair] = score
        if bool(score.get("supported", False)):
            union(*pair)

    components_by_root: dict[str, list[str]] = {}
    for view in views:
        components_by_root.setdefault(find(view), []).append(view)
    components = sorted(
        (sorted(component, key=views.index) for component in components_by_root.values()),
        key=lambda component: views.index(component[0]),
    )

    commit_by_view: dict[str, bool] = {}
    component_rows: list[dict[str, Any]] = []
    for component in components:
        component_set = set(component)
        edges = [
            (pair, score)
            for pair, score in normalized_scores.items()
            if bool(score.get("supported", False)) and set(pair) <= component_set
        ]
        nodes_prefer_track = all(bool(node_commits.get(view, False)) for view in component)
        edges_prefer_track = all(
            bool(score.get("valid", False))
            and bool(score.get("both_directions_prefer_track", False))
            for _pair, score in edges
        )
        commit = bool(nodes_prefer_track and edges_prefer_track)
        for view in component:
            commit_by_view[view] = commit
        component_rows.append(
            {
                "views": component,
                "supported_edges": [list(pair) for pair, _score in edges],
                "nodes_prefer_track": bool(nodes_prefer_track),
                "edges_prefer_track": bool(edges_prefer_track),
                "commit": commit,
            }
        )
    return {
        "commit_by_view": commit_by_view,
        "components": components,
        "component_rows": component_rows,
    }


def decide_target_edge_supported_commits(
    *,
    target_views: list[str],
    node_commits: Mapping[str, bool],
    pair_scores: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Commit only nodes incident to a verified positive cross-target edge."""

    views = [str(view) for view in target_views]
    if len(views) != len(set(views)):
        raise ValueError("target band views must be unique")
    parent = {view: view for view in views}

    def find(view: str) -> str:
        while parent[view] != view:
            parent[view] = parent[parent[view]]
            view = parent[view]
        return view

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    positive_edges: list[tuple[str, str]] = []
    for raw_pair, score in pair_scores.items():
        left, right = sorted((str(raw_pair[0]), str(raw_pair[1])))
        if left == right or left not in parent or right not in parent:
            raise ValueError(f"invalid target pair: {raw_pair}")
        positive = bool(
            node_commits.get(left, False)
            and node_commits.get(right, False)
            and score.get("supported", False)
            and score.get("valid", False)
            and score.get("both_directions_prefer_track", False)
        )
        if positive:
            positive_edges.append((left, right))
            union(left, right)

    incident = {view: False for view in views}
    for left, right in positive_edges:
        incident[left] = True
        incident[right] = True
    commit_by_view = {
        view: bool(node_commits.get(view, False) and incident[view])
        for view in views
    }
    if len(views) == 1:
        commit_by_view[views[0]] = bool(node_commits.get(views[0], False))

    components_by_root: dict[str, list[str]] = {}
    for view in views:
        components_by_root.setdefault(find(view), []).append(view)
    components = sorted(
        (sorted(component, key=views.index) for component in components_by_root.values()),
        key=lambda component: views.index(component[0]),
    )
    component_rows = [
        {
            "views": component,
            "positive_edges": [
                list(pair) for pair in positive_edges if set(pair) <= set(component)
            ],
            "commit": all(commit_by_view[view] for view in component),
        }
        for component in components
    ]
    return {
        "commit_by_view": commit_by_view,
        "components": components,
        "component_rows": component_rows,
        "positive_edges": [list(pair) for pair in positive_edges],
    }


def _observation_subset(
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    point_ids: list[int],
) -> tuple[dict[int, Mapping[str, np.ndarray]], dict[int, np.ndarray]]:
    return (
        {point_id: support_observations[point_id] for point_id in point_ids},
        {point_id: target_observations[point_id] for point_id in point_ids},
    )


def _exact_holdout_geometry_scores(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    static_camera: CameraParams,
    target_template: CameraParams,
    point_ranges: list[Mapping[str, Any]],
    config: SameTimePnPConfig,
) -> dict[str, Any]:
    """Fit once, score on unseen points, and return that exact fitted pose."""

    point_ids = list(target_observations)
    fit_ids, verify_ids = exact_holdout_point_id_split(
        point_ids=point_ids,
        point_ranges=point_ranges,
    )
    fit_support, fit_target = _observation_subset(
        support_observations,
        target_observations,
        fit_ids,
    )
    verify_support, verify_target = _observation_subset(
        support_observations,
        target_observations,
        verify_ids,
    )
    fit_result = solve_same_time_multiview_pnp(
        support_cameras=support_cameras,
        support_observations=fit_support,
        target_observations=fit_target,
        target_template=target_template,
        config=config,
    )
    fit_verification = (
        verify_multiseed_inlier_support(
            inlier_point_ids=np.asarray(fit_result["inlier_point_ids"]),
            point_ranges=point_ranges,
        )
        if bool(fit_result.get("success"))
        else {"verified": False, "reason": str(fit_result.get("reason"))}
    )
    if bool(fit_result.get("success")) and bool(fit_verification.get("verified")):
        scores = score_same_time_multiview_poses(
            support_cameras=support_cameras,
            support_observations=verify_support,
            target_observations=verify_target,
            target_cameras={
                "static": static_camera,
                "track": fit_result["camera"],
            },
            config=config,
        )
    else:
        scores = {
            "static": {"success": False, "reason": "exact_holdout_fit_failed"},
            "track": {"success": False, "reason": "exact_holdout_fit_failed"},
        }
    static_score = scores["static"]
    track_score = scores["track"]
    scores_valid = bool(static_score.get("success") and track_score.get("success"))
    static_value = static_score.get("truncated_reprojection_mean_px")
    track_value = track_score.get("truncated_reprojection_mean_px")
    prefers_track = bool(
        scores_valid
        and static_value is not None
        and track_value is not None
        and float(track_value) < float(static_value)
    )
    valid = bool(
        fit_result.get("success")
        and fit_verification.get("verified")
        and scores_valid
    )
    fit_hash = hashlib.sha256(
        np.asarray(fit_ids, dtype=np.int64).tobytes()
    ).hexdigest()
    verify_hash = hashlib.sha256(
        np.asarray(verify_ids, dtype=np.int64).tobytes()
    ).hexdigest()
    candidate = fit_result.get("camera") if valid else None
    return {
        "valid": valid,
        "reason": "ok" if valid else "exact_holdout_fit_or_score_failed",
        "point_count": int(len(point_ids)),
        "fold_point_counts": [int(len(fit_ids)), int(len(verify_ids))],
        "fold_point_ids_sha256": [fit_hash, verify_hash],
        "fold_rows": [
            {
                "fit_fold": 0,
                "fit_point_count": int(len(fit_ids)),
                "validation_point_count": int(len(verify_ids)),
                "solver_success": bool(fit_result.get("success")),
                "solver_reason": str(fit_result.get("reason")),
                "multiseed_verified": bool(fit_verification.get("verified")),
                "verification_reason": str(fit_verification.get("reason")),
                "validation_prefers_track": prefers_track,
                "static_score": _score_diagnostics(static_score),
                "track_score": _score_diagnostics(track_score),
            }
        ],
        "both_directions_prefer_track": prefers_track,
        "static_score": static_score,
        "track_score": track_score,
        "commit_candidate_protocol": "exact-holdout-quality-prefix-2to1-v4",
        "commit_candidate_fit_point_count": int(len(fit_ids)),
        "commit_candidate_multiseed_verified": bool(
            fit_verification.get("verified")
        ),
        "commit_candidate_verification_reason": str(
            fit_verification.get("reason")
        ),
        "commit_candidate_solver": _score_diagnostics(fit_result),
        "commit_camera": candidate,
        "_candidate_cameras": [candidate, candidate],
    }


def _dual_source_holdout_geometry_scores(
    *,
    support_cameras: Mapping[str, CameraParams],
    temporal_support_observations: Mapping[int, Mapping[str, np.ndarray]],
    temporal_target_observations: Mapping[int, np.ndarray],
    base_validation_support_observations: Mapping[
        int, Mapping[str, np.ndarray]
    ],
    base_validation_target_observations: Mapping[int, np.ndarray],
    base_validation_source: str = "static_map_same_frame_pnp",
    base_camera: CameraParams,
    target_template: CameraParams,
    point_ranges: list[Mapping[str, Any]],
    fit_point_budget: int,
    config: SameTimePnPConfig,
) -> dict[str, Any]:
    """Fit temporal repair once and require agreement from two unseen sources."""

    fit_budget = int(fit_point_budget)
    if fit_budget <= 0:
        raise ValueError("dual-source fit point budget must be positive")
    temporal_ids = [int(point_id) for point_id in temporal_target_observations]
    fit_ids, temporal_verify_ids = capped_exact_holdout_point_id_split(
        point_ids=temporal_ids,
        point_ranges=point_ranges,
        fit_point_budget=fit_budget,
    )
    base_verify_ids = [
        int(point_id) for point_id in base_validation_target_observations
    ]
    fit_support, fit_target = _observation_subset(
        temporal_support_observations,
        temporal_target_observations,
        fit_ids,
    )
    temporal_verify_support, temporal_verify_target = _observation_subset(
        temporal_support_observations,
        temporal_target_observations,
        temporal_verify_ids,
    )
    fit_result = solve_same_time_multiview_pnp(
        support_cameras=support_cameras,
        support_observations=fit_support,
        target_observations=fit_target,
        target_template=target_template,
        config=config,
    )
    fit_verification = (
        verify_multiseed_inlier_support(
            inlier_point_ids=np.asarray(fit_result["inlier_point_ids"]),
            point_ranges=point_ranges,
        )
        if bool(fit_result.get("success"))
        else {"verified": False, "reason": str(fit_result.get("reason"))}
    )
    candidate = (
        fit_result.get("camera")
        if bool(fit_result.get("success"))
        and bool(fit_verification.get("verified"))
        else None
    )

    validation_inputs = [
        (
            str(base_validation_source),
            base_validation_support_observations,
            base_validation_target_observations,
        ),
        (
            "temporal_track",
            temporal_verify_support,
            temporal_verify_target,
        ),
    ]
    validation_rows: list[dict[str, Any]] = []
    base_scores: list[Mapping[str, Any]] = []
    temporal_scores: list[Mapping[str, Any]] = []
    for source, verify_support, verify_target in validation_inputs:
        if isinstance(candidate, CameraParams):
            scores = score_same_time_multiview_poses(
                support_cameras=support_cameras,
                support_observations=verify_support,
                target_observations=verify_target,
                target_cameras={"base": base_camera, "temporal": candidate},
                config=config,
            )
        else:
            scores = {
                "base": {"success": False, "reason": "candidate_fit_failed"},
                "temporal": {
                    "success": False,
                    "reason": "candidate_fit_failed",
                },
            }
        base_score = scores["base"]
        temporal_score = scores["temporal"]
        scores_valid = bool(
            base_score.get("success") and temporal_score.get("success")
        )
        base_value = base_score.get("truncated_reprojection_mean_px")
        temporal_value = temporal_score.get("truncated_reprojection_mean_px")
        prefers_temporal = bool(
            scores_valid
            and base_value is not None
            and temporal_value is not None
            and float(temporal_value) < float(base_value)
        )
        base_scores.append(base_score)
        temporal_scores.append(temporal_score)
        validation_rows.append(
            {
                "validation_source": source,
                "validation_point_count": int(len(verify_target)),
                "validation_prefers_track": prefers_temporal,
                "base_score": _score_diagnostics(base_score),
                "temporal_score": _score_diagnostics(temporal_score),
            }
        )

    combined_base = _combined_crossfit_score(base_scores)
    combined_temporal = _combined_crossfit_score(temporal_scores)
    both_prefer_temporal = bool(
        all(row["validation_prefers_track"] for row in validation_rows)
    )
    valid = bool(
        isinstance(candidate, CameraParams)
        and fit_verification.get("verified")
        and combined_base.get("success")
        and combined_temporal.get("success")
    )
    point_id_groups = [fit_ids, base_verify_ids, temporal_verify_ids]
    return {
        "valid": valid,
        "reason": "ok" if valid else "dual_source_fit_or_score_failed",
        "point_count": int(len(temporal_ids)),
        "fold_point_counts": [int(len(ids)) for ids in point_id_groups],
        "fold_point_ids_sha256": [
            hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
            for ids in point_id_groups
        ],
        "fold_rows": validation_rows,
        "both_directions_prefer_track": both_prefer_temporal,
        "static_score": combined_base,
        "track_score": combined_temporal,
        "commit_candidate_protocol": (
            "dual-source-capped-fit-conservative-base-validation-v2"
        ),
        "commit_candidate_fit_point_count": int(len(fit_ids)),
        "commit_candidate_multiseed_verified": bool(
            fit_verification.get("verified")
        ),
        "commit_candidate_verification_reason": str(
            fit_verification.get("reason")
        ),
        "commit_candidate_solver": _score_diagnostics(fit_result),
        "commit_camera": candidate if valid else None,
        "_candidate_cameras": [candidate, candidate],
    }


def _missing_base_dual_source_geometry_scores(
    *,
    support_cameras: Mapping[str, CameraParams],
    temporal_support_observations: Mapping[int, Mapping[str, np.ndarray]],
    temporal_target_observations: Mapping[int, np.ndarray],
    base_validation_support_observations: Mapping[
        int, Mapping[str, np.ndarray]
    ],
    base_validation_target_observations: Mapping[int, np.ndarray],
    target_template: CameraParams,
    point_ranges: list[Mapping[str, Any]],
    fit_point_budget: int,
    config: SameTimePnPConfig,
) -> dict[str, Any]:
    """Repair a missing base pose only after two absolute geometry checks."""

    fit_budget = int(fit_point_budget)
    if fit_budget <= 0:
        raise ValueError("missing-base fit point budget must be positive")
    temporal_ids = [int(point_id) for point_id in temporal_target_observations]
    fit_ids = temporal_ids[:fit_budget]
    temporal_verify_ids = temporal_ids[fit_budget:]
    base_verify_ids = [
        int(point_id) for point_id in base_validation_target_observations
    ]
    fit_support, fit_target = _observation_subset(
        temporal_support_observations,
        temporal_target_observations,
        fit_ids,
    )
    temporal_verify_support, temporal_verify_target = _observation_subset(
        temporal_support_observations,
        temporal_target_observations,
        temporal_verify_ids,
    )
    if len(fit_ids) == fit_budget:
        fit_result = solve_same_time_multiview_pnp(
            support_cameras=support_cameras,
            support_observations=fit_support,
            target_observations=fit_target,
            target_template=target_template,
            config=config,
        )
    else:
        fit_result = {
            "success": False,
            "reason": "insufficient_equal_budget_temporal_fit_points",
        }
    fit_verification = (
        verify_multiseed_inlier_support(
            inlier_point_ids=np.asarray(fit_result["inlier_point_ids"]),
            point_ranges=point_ranges,
        )
        if bool(fit_result.get("success"))
        else {"verified": False, "reason": str(fit_result.get("reason"))}
    )
    candidate = (
        fit_result.get("camera")
        if bool(fit_result.get("success"))
        and bool(fit_verification.get("verified"))
        else None
    )

    validation_inputs = [
        (
            "static_map_same_frame_pnp_holdout",
            base_validation_support_observations,
            base_validation_target_observations,
        ),
        (
            "temporal_track_holdout",
            temporal_verify_support,
            temporal_verify_target,
        ),
    ]
    validation_rows: list[dict[str, Any]] = []
    candidate_scores: list[Mapping[str, Any]] = []
    threshold_px = float(config.pnp_reprojection_error_px)
    for source, verify_support, verify_target in validation_inputs:
        if isinstance(candidate, CameraParams):
            scores = score_same_time_multiview_poses(
                support_cameras=support_cameras,
                support_observations=verify_support,
                target_observations=verify_target,
                target_cameras={"temporal": candidate},
                config=config,
            )
            candidate_score = scores["temporal"]
        else:
            candidate_score = {
                "success": False,
                "reason": "candidate_fit_failed",
            }
        absolute_check = _absolute_geometry_validation(
            score=candidate_score,
            config=config,
        )
        absolute_geometry_pass = bool(absolute_check["valid"])
        candidate_scores.append(candidate_score)
        validation_rows.append(
            {
                "validation_source": source,
                "validation_point_count": int(len(verify_target)),
                "absolute_geometry_threshold_px": threshold_px,
                "absolute_geometry_pass": absolute_geometry_pass,
                "absolute_geometry_check": absolute_check,
                "validation_prefers_track": absolute_geometry_pass,
                "base_score": {
                    "success": False,
                    "reason": "base_pose_missing",
                },
                "temporal_score": _score_diagnostics(candidate_score),
            }
        )

    combined_temporal = _combined_crossfit_score(candidate_scores)
    both_absolute_checks_pass = bool(
        len(validation_rows) == 2
        and all(row["absolute_geometry_pass"] for row in validation_rows)
    )
    valid = bool(
        isinstance(candidate, CameraParams)
        and fit_verification.get("verified")
        and combined_temporal.get("success")
        and both_absolute_checks_pass
    )
    point_id_groups = [fit_ids, base_verify_ids, temporal_verify_ids]
    return {
        "valid": valid,
        "reason": "ok" if valid else "missing_base_fit_or_absolute_score_failed",
        "point_count": int(len(temporal_ids)),
        "fold_point_counts": [int(len(ids)) for ids in point_id_groups],
        "fold_point_ids_sha256": [
            hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
            for ids in point_id_groups
        ],
        "fold_rows": validation_rows,
        "both_directions_prefer_track": both_absolute_checks_pass,
        "static_score": {"success": False, "reason": "base_pose_missing"},
        "track_score": combined_temporal,
        "commit_candidate_protocol": (
            "missing-base-dual-source-absolute-validation-v1"
        ),
        "commit_candidate_fit_point_count": int(len(fit_ids)),
        "commit_candidate_multiseed_verified": bool(
            fit_verification.get("verified")
        ),
        "commit_candidate_verification_reason": str(
            fit_verification.get("reason")
        ),
        "commit_candidate_solver": _score_diagnostics(fit_result),
        "commit_camera": candidate if valid else None,
        "_candidate_cameras": [candidate, candidate],
    }


def _attested_multiseed_missing_base_result(
    *,
    prediction_row: Mapping[str, Any],
    temporal_camera: CameraParams | None,
) -> dict[str, Any]:
    """Validate the exact upstream pooled-track pose and its multiseed proof."""

    solver = prediction_row.get("solver") or {}
    verification = prediction_row.get("verification") or {}
    if not isinstance(solver, Mapping) or not isinstance(verification, Mapping):
        raise ValueError("temporal solver or verification diagnostics are malformed")
    solver_success = bool(solver.get("success", False))
    candidate_success = bool(solver.get("candidate_success", solver_success))
    verified = bool(verification.get("verified", False))
    if solver_success != bool(candidate_success and verified):
        raise ValueError("temporal solver and multiseed verification disagree")
    if solver_success != isinstance(temporal_camera, CameraParams):
        raise ValueError("temporal solver/camera availability mismatch")
    selected_count = int(
        (prediction_row.get("observation") or {}).get("selected_count", 0)
    )
    valid = bool(solver_success and verified and temporal_camera is not None)
    return {
        "valid": valid,
        "reason": "ok" if valid else "attested_multiseed_candidate_unavailable",
        "point_count": selected_count,
        "fold_point_counts": [],
        "fold_point_ids_sha256": [],
        "fold_rows": [
            {
                "validation_source": "temporal_multiseed_inlier_support",
                "validation_point_count": selected_count,
                "validation_prefers_track": valid,
                "verification": dict(verification),
            }
        ],
        "both_directions_prefer_track": valid,
        "static_score": {"success": False, "reason": "base_pose_missing"},
        "track_score": dict(solver),
        "commit_candidate_protocol": "attested-pooled-multiseed-track-pnp-v1",
        "commit_candidate_fit_point_count": selected_count,
        "commit_candidate_multiseed_verified": verified,
        "commit_candidate_verification_reason": str(
            verification.get("reason", "missing_verification")
        ),
        "commit_candidate_solver": _score_diagnostics(solver),
        "commit_camera": temporal_camera if valid else None,
        "_candidate_cameras": [temporal_camera, temporal_camera],
    }


def _absolute_geometry_validation(
    *,
    score: Mapping[str, Any],
    config: SameTimePnPConfig,
) -> dict[str, Any]:
    """Apply the existing PnP count and raw reprojection gates to a fixed pose."""

    minimum_inliers = max(4, int(config.min_pnp_inliers))
    threshold = float(config.pnp_reprojection_error_px)
    inlier_count = int(score.get("inlier_count", 0))
    raw_ratio = score.get("inlier_ratio")
    raw_median = score.get("reprojection_median_px")
    ratio = float(raw_ratio) if raw_ratio is not None else float("nan")
    median = float(raw_median) if raw_median is not None else float("nan")
    valid = bool(
        score.get("success")
        and inlier_count >= minimum_inliers
        and np.isfinite(median)
        and median <= threshold
    )
    return {
        "valid": valid,
        "reason": "ok" if valid else "geometry_not_dominant",
        "minimum_inlier_count": int(minimum_inliers),
        "reprojection_threshold_px": threshold,
        "inlier_count": inlier_count,
        "inlier_ratio": ratio if np.isfinite(ratio) else None,
        "reprojection_median_px": median if np.isfinite(median) else None,
    }


def balanced_refit_point_ids(
    *,
    point_ids: list[int] | tuple[int, ...],
    point_ranges: list[Mapping[str, Any]],
    budget: int,
) -> list[int]:
    """Select a deterministic, seed-balanced subset without losing quality order."""

    limit = int(budget)
    if limit <= 0:
        raise ValueError("final refit point budget must be positive")
    ranges = [tuple(int(value) for value in row["point_range"]) for row in point_ranges]
    if not ranges or any(start < 0 or stop < start for start, stop in ranges):
        raise ValueError("invalid temporal seed point range")
    buckets: list[list[int]] = [[] for _range in ranges]
    seen: set[int] = set()
    for raw_point_id in point_ids:
        point_id = int(raw_point_id)
        if point_id in seen:
            raise ValueError("final refit point IDs must be unique")
        seen.add(point_id)
        memberships = [
            index
            for index, (start, stop) in enumerate(ranges)
            if start <= point_id < stop
        ]
        if len(memberships) != 1:
            raise ValueError("each final refit point must belong to exactly one seed bank")
        buckets[memberships[0]].append(point_id)
    selected: list[int] = []
    offsets = [0 for _bucket in buckets]
    while len(selected) < min(limit, len(seen)):
        added = False
        for index, bucket in enumerate(buckets):
            if offsets[index] >= len(bucket) or len(selected) >= limit:
                continue
            selected.append(bucket[offsets[index]])
            offsets[index] += 1
            added = True
        if not added:
            break
    return selected


def _combined_crossfit_score(scores: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(scores) != 2 or not all(bool(score.get("success")) for score in scores):
        return {"success": False, "reason": "crossfit_candidate_score_failed"}
    values = np.concatenate(
        [
            np.asarray(score["truncated_reprojection_values_px"], dtype=np.float64)
            for score in scores
        ]
    )
    if len(values) == 0 or not np.isfinite(values).all():
        return {"success": False, "reason": "crossfit_residuals_invalid"}
    return {
        "success": True,
        "reason": "ok",
        "truncated_reprojection_mean_px": float(np.mean(values)),
        "truncated_reprojection_values_px": values.tolist(),
    }


def _crossfit_geometry_scores(
    *,
    support_cameras: Mapping[str, CameraParams],
    support_observations: Mapping[int, Mapping[str, np.ndarray]],
    target_observations: Mapping[int, np.ndarray],
    static_camera: CameraParams,
    target_template: CameraParams,
    point_ranges: list[Mapping[str, Any]],
    config: SameTimePnPConfig,
    fold_point_ids: tuple[list[int], list[int]] | None = None,
    final_refit_point_budget: int | None = None,
    commit_candidate_strategy: str = "refit",
) -> dict[str, Any]:
    if commit_candidate_strategy not in {
        "refit",
        "best_heldout_fold",
        "fixed_fold0",
    }:
        raise ValueError(
            f"unsupported crossfit commit candidate strategy: {commit_candidate_strategy}"
        )
    if (
        commit_candidate_strategy in {"best_heldout_fold", "fixed_fold0"}
        and final_refit_point_budget is not None
    ):
        raise ValueError(
            f"{commit_candidate_strategy} does not perform a final refit"
        )
    point_ids = list(target_observations)
    if fold_point_ids is None:
        first_ids, second_ids = crossfit_point_id_folds(
            point_ids=point_ids,
            point_ranges=point_ranges,
        )
    else:
        target_point_ids = set(int(point_id) for point_id in point_ids)
        first_ids = [
            int(point_id)
            for point_id in fold_point_ids[0]
            if int(point_id) in target_point_ids
        ]
        second_ids = [
            int(point_id)
            for point_id in fold_point_ids[1]
            if int(point_id) in target_point_ids
        ]
        if set(first_ids) & set(second_ids):
            raise ValueError("explicit crossfit folds must be disjoint")
        if set(first_ids) | set(second_ids) != target_point_ids:
            raise ValueError("explicit crossfit folds must cover target observations")
    fold_ids = [first_ids, second_ids]
    fold_rows: list[dict[str, Any]] = []
    static_scores: list[Mapping[str, Any]] = []
    track_scores: list[Mapping[str, Any]] = []
    candidate_cameras: list[CameraParams | None] = []
    fit_results: list[Mapping[str, Any]] = []
    fit_verifications: list[Mapping[str, Any]] = []
    for fit_index in range(2):
        validation_index = 1 - fit_index
        fit_support, fit_target = _observation_subset(
            support_observations,
            target_observations,
            fold_ids[fit_index],
        )
        validation_support, validation_target = _observation_subset(
            support_observations,
            target_observations,
            fold_ids[validation_index],
        )
        fit_result = solve_same_time_multiview_pnp(
            support_cameras=support_cameras,
            support_observations=fit_support,
            target_observations=fit_target,
            target_template=target_template,
            config=config,
        )
        verification = (
            verify_multiseed_inlier_support(
                inlier_point_ids=np.asarray(fit_result["inlier_point_ids"]),
                point_ranges=point_ranges,
            )
            if bool(fit_result.get("success"))
            else {"verified": False, "reason": str(fit_result.get("reason"))}
        )
        if bool(fit_result.get("success")) and bool(verification.get("verified")):
            candidate_scores = score_same_time_multiview_poses(
                support_cameras=support_cameras,
                support_observations=validation_support,
                target_observations=validation_target,
                target_cameras={
                    "static": static_camera,
                    "track": fit_result["camera"],
                },
                config=config,
            )
        else:
            candidate_scores = {
                "static": {"success": False, "reason": "crossfit_fit_failed"},
                "track": {"success": False, "reason": "crossfit_fit_failed"},
            }
        static_scores.append(candidate_scores["static"])
        track_scores.append(candidate_scores["track"])
        candidate_cameras.append(
            fit_result["camera"] if bool(fit_result.get("success")) else None
        )
        fit_results.append(fit_result)
        fit_verifications.append(verification)
        static_value = candidate_scores["static"].get(
            "truncated_reprojection_mean_px"
        )
        track_value = candidate_scores["track"].get(
            "truncated_reprojection_mean_px"
        )
        prefers_track = bool(
            static_value is not None
            and track_value is not None
            and float(track_value) < float(static_value)
        )
        fold_rows.append(
            {
                "fit_fold": int(fit_index),
                "fit_point_count": int(len(fold_ids[fit_index])),
                "validation_point_count": int(len(fold_ids[validation_index])),
                "solver_success": bool(fit_result.get("success")),
                "solver_reason": str(fit_result.get("reason")),
                "multiseed_verified": bool(verification.get("verified")),
                "verification_reason": str(verification.get("reason")),
                "validation_prefers_track": prefers_track,
                "static_score": _score_diagnostics(candidate_scores["static"]),
                "track_score": _score_diagnostics(candidate_scores["track"]),
            }
        )

    static_score = _combined_crossfit_score(static_scores)
    track_score = _combined_crossfit_score(track_scores)
    selected_fit_fold: int | None = None
    if commit_candidate_strategy == "best_heldout_fold":
        heldout_gains = [
            float(row["static_score"]["truncated_reprojection_mean_px"])
            - float(row["track_score"]["truncated_reprojection_mean_px"])
            if row["static_score"].get("success")
            and row["track_score"].get("success")
            else float("-inf")
            for row in fold_rows
        ]
        selected_fit_fold = int(np.argmax(np.asarray(heldout_gains)))
        commit_fit_result = fit_results[selected_fit_fold]
        commit_verification = fit_verifications[selected_fit_fold]
        commit_fit_point_ids = fold_ids[selected_fit_fold]
        commit_candidate_protocol = "crossfit-best-heldout-fold-v1"
    elif commit_candidate_strategy == "fixed_fold0":
        selected_fit_fold = 0
        commit_fit_result = fit_results[selected_fit_fold]
        commit_verification = fit_verifications[selected_fit_fold]
        commit_fit_point_ids = fold_ids[selected_fit_fold]
        commit_candidate_protocol = "crossfit-fixed-fold0-oof-v1"
    else:
        commit_fit_point_ids = (
            list(point_ids)
            if final_refit_point_budget is None
            else balanced_refit_point_ids(
                point_ids=point_ids,
                point_ranges=point_ranges,
                budget=int(final_refit_point_budget),
            )
        )
        refit_support, refit_target = _observation_subset(
            support_observations,
            target_observations,
            commit_fit_point_ids,
        )
        commit_fit_result = solve_same_time_multiview_pnp(
            support_cameras=support_cameras,
            support_observations=refit_support,
            target_observations=refit_target,
            target_template=target_template,
            config=config,
        )
        commit_verification = (
            verify_multiseed_inlier_support(
                inlier_point_ids=np.asarray(commit_fit_result["inlier_point_ids"]),
                point_ranges=point_ranges,
            )
            if bool(commit_fit_result.get("success"))
            else {
                "verified": False,
                "reason": str(commit_fit_result.get("reason")),
            }
        )
        commit_candidate_protocol = (
            "crossfit-select-then-seed-balanced-bounded-refit-v2"
            if final_refit_point_budget is not None
            else "crossfit-select-then-full-refit-v1"
        )
    valid = bool(
        static_score.get("success")
        and track_score.get("success")
        and all(camera is not None for camera in candidate_cameras)
        and commit_fit_result.get("success")
        and commit_verification.get("verified")
    )
    fold_hashes = [
        hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
        for ids in fold_ids
    ]
    return {
        "valid": valid,
        "reason": "ok" if valid else "crossfit_fit_or_score_failed",
        "point_count": int(len(point_ids)),
        "fold_point_counts": [int(len(first_ids)), int(len(second_ids))],
        "fold_point_ids_sha256": fold_hashes,
        "fold_rows": fold_rows,
        "both_directions_prefer_track": bool(
            valid and all(row["validation_prefers_track"] for row in fold_rows)
        ),
        "static_score": static_score,
        "track_score": track_score,
        "commit_candidate_protocol": commit_candidate_protocol,
        "selected_fit_fold": selected_fit_fold,
        "commit_candidate_fit_point_count": int(len(commit_fit_point_ids)),
        "commit_candidate_multiseed_verified": bool(
            commit_verification.get("verified")
        ),
        "commit_candidate_verification_reason": str(
            commit_verification.get("reason")
        ),
        "commit_candidate_solver": _score_diagnostics(commit_fit_result),
        "commit_camera": commit_fit_result.get("camera") if valid else None,
        "_candidate_cameras": candidate_cameras,
    }


def select_band_commit_camera(
    *,
    node_crossfit: Mapping[str, Any],
    band_crossfit: Mapping[str, Any],
    candidate_protocol: str = "legacy_crossfit_refit",
) -> CameraParams | None:
    """Return the camera that was actually verified by the chosen protocol."""

    if not bool(node_crossfit.get("valid")) or not bool(
        band_crossfit.get("valid")
    ):
        return None
    source = (
        band_crossfit
        if candidate_protocol == "crossfit_fixed_fold"
        else node_crossfit
    )
    camera = source.get("commit_camera")
    return camera if isinstance(camera, CameraParams) else None


def export_geometry_verified_hybrid(
    *,
    runtime_scene: Path,
    base_pose_root: Path,
    base_pose_method: str,
    base_prediction_attestation: Path | None = None,
    track_output_root: Path,
    output_root: Path,
    commit_policy: str = "mean",
    commit_scope: str = "node",
    band_graph_policy: str = "atomic_supported",
    candidate_protocol: str = "exact_holdout",
    final_refit_point_budget: int | None = None,
    missing_base_repair_protocol: str = "none",
    successful_base_policy: str = "verify_commit",
) -> dict[str, Any]:
    if base_pose_method not in {"static_pnp", "static_map_same_frame_pnp"}:
        raise ValueError(f"unsupported base pose method: {base_pose_method}")
    if commit_scope not in {"node", "band"}:
        raise ValueError(f"unsupported commit scope: {commit_scope}")
    if band_graph_policy not in {"atomic_supported", "verified_edges"}:
        raise ValueError(f"unsupported band graph policy: {band_graph_policy}")
    if commit_scope != "band" and band_graph_policy != "atomic_supported":
        raise ValueError("verified-edge graph policy requires band commits")
    if candidate_protocol not in {
        "exact_holdout",
        "dual_source_holdout",
        "legacy_crossfit_refit",
        "crossfit_verified_fold",
        "crossfit_fixed_fold",
    }:
        raise ValueError(f"unsupported candidate protocol: {candidate_protocol}")
    no_refit_protocols = {
        "exact_holdout",
        "dual_source_holdout",
        "crossfit_verified_fold",
        "crossfit_fixed_fold",
    }
    node_only_protocols = {
        "exact_holdout",
        "dual_source_holdout",
        "crossfit_verified_fold",
    }
    if candidate_protocol in node_only_protocols and commit_scope != "node":
        raise ValueError(f"{candidate_protocol} currently supports node commits only")
    if (
        candidate_protocol in no_refit_protocols
        and final_refit_point_budget is not None
    ):
        raise ValueError(f"{candidate_protocol} never refits after verification")
    if candidate_protocol == "dual_source_holdout":
        if base_pose_method != "static_map_same_frame_pnp":
            raise ValueError(
                "dual_source_holdout requires static_map_same_frame_pnp as the base"
            )
    allowed_missing_base_protocols = {
        "none",
        "dual_source_absolute",
        "attested_multiseed",
    }
    if missing_base_repair_protocol not in allowed_missing_base_protocols:
        raise ValueError(
            "unsupported missing-base repair protocol: "
            f"{missing_base_repair_protocol}"
        )
    if successful_base_policy not in {"verify_commit", "retain"}:
        raise ValueError(
            f"unsupported successful-base policy: {successful_base_policy}"
        )
    if successful_base_policy == "retain" and commit_scope != "node":
        raise ValueError("successful-base retain policy requires node commits")
    repair_missing_base = missing_base_repair_protocol != "none"
    if repair_missing_base and (
        candidate_protocol != "dual_source_holdout" or commit_scope != "node"
    ):
        raise ValueError(
            "missing-base repair requires node-scoped dual_source_holdout"
        )
    if final_refit_point_budget is not None and int(final_refit_point_budget) <= 0:
        raise ValueError("final_refit_point_budget must be positive when provided")
    runtime = Path(runtime_scene).resolve()
    base_root = Path(base_pose_root).resolve()
    track_output = Path(track_output_root).resolve()
    output = Path(output_root).resolve()
    runtime_manifest = json.loads(
        (runtime / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    runtime_digest = str(
        json.loads((runtime / "checksums.sha256.json").read_text(encoding="utf-8"))[
            "tree_digest"
        ]
    )
    runtime_views = [str(view) for view in runtime_manifest["runtime_views"]]
    static_views = [str(view) for view in runtime_manifest["observed_short_views"]]
    legal_targets = {str(view) for view in runtime_manifest["long_views"]}

    prediction_path = track_output / "predictions_pre_gt.json"
    attestation = json.loads(
        (track_output / "prediction_attestation.json").read_text(encoding="utf-8")
    )
    prediction_sha256 = _sha256(prediction_path)
    if prediction_sha256 != str(attestation["sha256"]):
        raise ValueError("track prediction attestation hash does not match")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if bool(predictions.get("gt_used", True)) or bool(predictions.get("nvs_used", True)):
        raise ValueError("geometry verification requires pre-GT, pre-NVS predictions")
    if str(predictions["runtime_tree_digest"]) != runtime_digest:
        raise ValueError("track predictions were generated from a different runtime tree")
    if not bool(predictions.get("verify_commit_used", False)):
        raise ValueError("geometry verification requires verified track candidates")
    if str(predictions.get("track_source_mode")) != "shared_temporal_bundle":
        raise ValueError("geometry verification requires a shared temporal track bundle")

    track_roots = [Path(path).resolve() for path in predictions.get("track_roots", [])]
    if len(track_roots) != 1:
        raise ValueError("geometry verification requires exactly one temporal track root")
    track_root = track_roots[0]
    track_hashes = _verify_track_files(track_root=track_root, predictions=predictions)
    fit_point_budget = int(predictions.get("fit_point_budget", 0))
    if fit_point_budget <= 0:
        raise ValueError("track prediction manifest is missing a positive fit point budget")
    candidate_point_budget = int(
        (3 if candidate_protocol == "exact_holdout" else 2) * fit_point_budget
    )

    track_meta = json.loads((track_root / "meta.json").read_text(encoding="utf-8"))
    track_views = (track_root / "views.txt").read_text(encoding="utf-8").split()
    if set(track_views) - set(runtime_views):
        raise ValueError("track bundle contains a non-runtime view")
    if set(static_views) - set(track_views):
        raise ValueError("track bundle is missing a public static view")
    tracks = np.load(track_root / "tracks.npy", mmap_mode="r")
    visibility = np.load(track_root / "visibility.npy", mmap_mode="r")
    quality_path = track_root / "track_quality.npy"
    track_quality = np.load(quality_path, mmap_mode="r") if quality_path.is_file() else None
    point_ranges = seed_point_ranges(meta=track_meta, total_points=int(tracks.shape[2]))
    balanced_ranges = [tuple(row["point_range"]) for row in point_ranges]
    track_start = int(track_meta["start_frame"])
    track_stop = track_start + int(track_meta["num_frames"])

    frames = [int(frame) for frame in predictions["frames"]]
    target_views = [str(view) for view in predictions["target_views"]]
    if set(target_views) - legal_targets:
        raise ValueError("track prediction target is absent from the runtime scene")
    if len(target_views) != len(set(target_views)):
        raise ValueError("track prediction target views must be unique")
    export_views = static_views + target_views
    if any(frame < track_start or frame >= track_stop for frame in frames):
        raise ValueError("prediction frame lies outside the temporal track bundle")
    expected_track_views = static_views + target_views
    if track_views != expected_track_views:
        raise ValueError(
            "track bundle must contain exactly public static views followed by declared targets"
        )
    if [str(view) for view in track_meta.get("views", [])] != expected_track_views:
        raise ValueError("track metadata view list does not match the declared train-only split")
    validate_track_seed_builder(
        seed_builder=str(track_meta.get("seed_builder")),
        commit_scope=commit_scope,
    )
    expected_data_root = (runtime / "images").resolve()
    if Path(str(track_meta.get("data_root", ""))).resolve() != expected_data_root:
        raise ValueError("track bundle was generated from a different image root")
    seed_diagnostics = track_meta.get("seed_builder_diag_by_seed")
    if not isinstance(seed_diagnostics, Mapping) or not seed_diagnostics:
        raise ValueError("track metadata is missing per-seed closure diagnostics")
    for seed_key, diagnostics in seed_diagnostics.items():
        if [str(view) for view in diagnostics.get("target_views", [])] != target_views:
            raise ValueError(
                f"seed {seed_key} target set does not match the declared train-only split"
            )
    rows_by_node: dict[tuple[int, str], dict[str, Any]] = {}
    for row in predictions["rows"]:
        key = (int(row["frame"]), str(row["view"]))
        if key in rows_by_node:
            raise ValueError(f"duplicate track prediction row: {key}")
        rows_by_node[key] = row
    expected_nodes = {(frame, view) for frame in frames for view in target_views}
    if set(rows_by_node) != expected_nodes:
        raise ValueError("track prediction rows do not cover the declared frame/view grid")

    base_lineage = _verify_base_pose_lineage(
        base_pose_root=base_root,
        base_pose_method=base_pose_method,
        base_prediction_attestation=base_prediction_attestation,
        runtime_digest=runtime_digest,
        runtime_scene=runtime,
        frames=frames,
        static_views=static_views,
        target_views=target_views,
        require_track_bundles=(candidate_protocol == "dual_source_holdout"),
    )

    support_cameras = CameraLoader(runtime / "cameras/static").load_cameras(static_views)
    target_templates = (
        CameraIntrinsicsLoader(runtime / "cameras").load_camera_templates(
            target_views
        )
        if repair_missing_base
        else {}
    )
    if missing_base_repair_protocol == "attested_multiseed":
        if str(predictions.get("temporal_seed_policy")) != "verified_all":
            raise ValueError(
                "attested missing-base repair requires verified_all temporal seeds"
            )
        if str(predictions.get("verification_protocol")) != (
            "independent-temporal-seed-bank-inlier-support-v1"
        ):
            raise ValueError(
                "attested missing-base repair requires the frozen multiseed verifier"
            )
    temporal_pose_root = (
        _verify_prediction_pose_lineage(
            predictions=predictions,
            frames=frames,
            label="temporal",
        )
        if missing_base_repair_protocol == "attested_multiseed"
        else None
    )
    score_config = _config_from_prediction(predictions)
    if candidate_protocol == "dual_source_holdout":
        if int(base_lineage["fit_point_budget"]) != fit_point_budget:
            raise ValueError("base and temporal candidate fit budgets differ")
        if _config_from_prediction({"config": base_lineage["config"]}) != score_config:
            raise ValueError("base and temporal PnP configurations differ")
    base_rows_by_node = {
        (int(row["frame"]), str(row["view"])): row
        for row in base_lineage["prediction_rows"]
    }
    if candidate_protocol == "dual_source_holdout" and set(base_rows_by_node) != expected_nodes:
        raise ValueError("base matcher prediction rows do not cover the node grid")
    pose_output = output / "pose_geometry_verified_hybrid"
    pose_output.mkdir(parents=True, exist_ok=False)
    source_rows: list[dict[str, Any]] = []
    base_camera_hashes = dict(base_lineage["camera_hashes"])
    output_camera_hashes: dict[str, dict[str, str]] = {}
    available_output_views_by_frame: dict[str, list[str]] = {}
    band_rows_by_frame: dict[str, dict[str, Any]] = {}
    track_commit_count = 0
    missing_base_track_commit_count = 0
    missing_base_count = 0
    unregistered_count = 0
    for frame in frames:
        frame_name = f"f{frame:06d}"
        frame_local = frame - track_start
        base_camera_root = base_root / frame_name / "cameras_pred"
        # Export training cameras only; the evaluator loads held-out cameras separately.
        base_cameras, missing_base_views = _load_base_frame_cameras(
            camera_root=base_camera_root,
            static_views=static_views,
            target_views=target_views,
            allow_missing_targets=repair_missing_base,
        )
        missing_base_view_set = set(missing_base_views)
        missing_base_count += len(missing_base_views)
        temporal_camera_loader: CameraLoader | None = None
        temporal_static_cameras: dict[str, CameraParams] = {}
        if missing_base_repair_protocol == "attested_multiseed":
            assert temporal_pose_root is not None
            temporal_camera_loader = CameraLoader(
                temporal_pose_root / frame_name / "cameras_pred"
            )
            temporal_static_cameras = temporal_camera_loader.load_cameras(
                static_views
            )
        if candidate_protocol == "dual_source_holdout":
            for view in target_views:
                base_solver_success = prediction_row_base_pose_available(
                    base_rows_by_node[(frame, view)]
                )
                if base_solver_success != (view in base_cameras):
                    raise ValueError(
                        "base solver/camera availability mismatch: "
                        f"frame={frame} view={view}"
                    )
        for static_view in static_views:
            expected = support_cameras[static_view]
            actual = base_cameras[static_view]
            if not (
                np.allclose(actual.K, expected.K, rtol=0.0, atol=1e-9)
                and np.allclose(actual.R, expected.R, rtol=0.0, atol=1e-9)
                and np.allclose(actual.T, expected.T, rtol=0.0, atol=1e-9)
                and np.allclose(actual.D, expected.D, rtol=0.0, atol=1e-9)
            ):
                raise ValueError(
                    f"base pose root is not in the runtime static-camera gauge: "
                    f"frame={frame} view={static_view}"
                )
            if temporal_camera_loader is not None:
                temporal_static = temporal_static_cameras[static_view]
                if not (
                    np.allclose(
                        temporal_static.K, expected.K, rtol=0.0, atol=1e-9
                    )
                    and np.allclose(
                        temporal_static.R, expected.R, rtol=0.0, atol=1e-9
                    )
                    and np.allclose(
                        temporal_static.T, expected.T, rtol=0.0, atol=1e-9
                    )
                    and np.allclose(
                        temporal_static.D, expected.D, rtol=0.0, atol=1e-9
                    )
                ):
                    raise ValueError(
                        "temporal pose root is not in the runtime static-camera "
                        f"gauge: frame={frame} view={static_view}"
                    )

        base_validation_by_view: dict[
            str, tuple[dict[int, Mapping[str, np.ndarray]], dict[int, np.ndarray]]
        ] = {}
        base_validation_diagnostics_by_view: dict[str, dict[str, Any]] = {}
        base_validation_source_by_view: dict[str, str] = {}
        if candidate_protocol == "dual_source_holdout":
            base_track_root = Path(
                base_lineage["track_root_by_frame"][frame]
            )
            base_track_views = (
                base_track_root / "views.txt"
            ).read_text(encoding="utf-8").split()
            base_tracks = np.load(base_track_root / "tracks.npy", mmap_mode="r")
            base_visibility = np.load(
                base_track_root / "visibility.npy", mmap_mode="r"
            )
            base_quality_path = base_track_root / "track_quality.npy"
            base_quality = (
                np.load(base_quality_path, mmap_mode="r")
                if base_quality_path.is_file()
                else None
            )
            for view in target_views:
                base_support_all, base_target_all, base_diag = build_node_observations(
                    tracks=base_tracks,
                    visibility=base_visibility,
                    track_quality=base_quality,
                    view_ids=base_track_views,
                    static_views=static_views,
                    target_view=view,
                    frame_local=0,
                    min_support_views=int(score_config.min_support_views),
                    max_points=int(2 * fit_point_budget),
                )
                base_point_ids = list(base_target_all)
                base_solver_success = prediction_row_base_pose_available(
                    base_rows_by_node[(frame, view)]
                )
                base_validation = select_prediction_row_base_validation_point_ids(
                    point_ids=base_point_ids,
                    prediction_row=base_rows_by_node[(frame, view)],
                    fit_point_budget=fit_point_budget,
                )
                selected_source = str(
                    base_rows_by_node[(frame, view)].get("selected_source", "")
                )
                if selected_source == "temporal_interpolation":
                    upstream_count = 0
                elif selected_source in {"primary", "rescue", "tertiary"}:
                    upstream_count = int(
                        (
                            base_rows_by_node[(frame, view)].get(
                                f"{selected_source}_solver"
                            )
                            or {}
                        ).get("candidate_target_observation_count", -1)
                    )
                else:
                    upstream_count = int(
                        base_rows_by_node[(frame, view)]
                        .get("observation", {})
                        .get("selected_count", -1)
                    )
                base_verify_ids = list(base_validation["point_ids"])
                base_diag["validation_selection_reason"] = str(
                    base_validation["reason"]
                )
                base_diag["validation_reuses_base_fit_points"] = bool(
                    base_validation["fit_reuse"]
                )
                base_validation_source = str(base_validation["source"])
                if (
                    repair_missing_base
                    and not base_solver_success
                    and view not in base_cameras
                ):
                    # These matcher observations are independent of the
                    # temporal candidate, so the failed base fit does not
                    # consume them for missing-base absolute validation.
                    base_verify_ids = base_point_ids[:fit_point_budget]
                    base_diag["validation_disabled_reason"] = None
                    base_diag["validation_reuse_reason"] = (
                        "base_solver_failed_independent_matcher_absolute_check"
                    )
                    base_validation_source = (
                        "static_map_same_frame_pnp_failed_fit_absolute_check"
                    )
                base_validation_by_view[view] = _observation_subset(
                    base_support_all,
                    base_target_all,
                    base_verify_ids,
                )
                base_diag.update(
                    {
                        "attested_fit_point_count": upstream_count,
                        "reconstructed_fit_prefix_count": int(
                            min(len(base_point_ids), fit_point_budget)
                        ),
                        "validation_point_count": int(len(base_verify_ids)),
                    }
                )
                base_validation_diagnostics_by_view[view] = base_diag
                base_validation_source_by_view[view] = base_validation_source

        observations_by_view: dict[
            str, tuple[dict[int, Mapping[str, np.ndarray]], dict[int, np.ndarray]]
        ] = {}
        temporal_observation_diagnostics_by_view: dict[str, dict[str, Any]] = {}
        excluded_seed_ranges = (
            [
                tuple(row["point_range"])
                for row in point_ranges
                if int(row["seed_frame"]) == frame
            ]
            if candidate_protocol == "dual_source_holdout"
            else []
        )
        for view in target_views:
            crossfit_support, crossfit_target, crossfit_observation_diag = build_node_observations(
                tracks=tracks,
                visibility=visibility,
                track_quality=track_quality,
                view_ids=track_views,
                static_views=static_views,
                target_view=view,
                frame_local=frame_local,
                min_support_views=int(score_config.min_support_views),
                max_points=candidate_point_budget,
                excluded_point_ranges=excluded_seed_ranges,
                balanced_point_ranges=balanced_ranges,
            )
            observations_by_view[view] = (crossfit_support, crossfit_target)
            crossfit_observation_diag["excluded_current_seed_ranges"] = [
                list(point_range) for point_range in excluded_seed_ranges
            ]
            temporal_observation_diagnostics_by_view[view] = (
                crossfit_observation_diag
            )

        global_fold_ids: tuple[list[int], list[int]] | None = None
        if commit_scope == "band":
            union_point_ids = sorted(
                {
                    int(point_id)
                    for _support, target in observations_by_view.values()
                    for point_id in target
                }
            )
            visibility_strata = {
                point_id: tuple(
                    view
                    for view in target_views
                    if point_id in observations_by_view[view][1]
                )
                for point_id in union_point_ids
            }
            global_fold_ids = stratified_crossfit_point_id_folds(
                point_ids=union_point_ids,
                point_ranges=point_ranges,
                strata_by_point_id=visibility_strata,
            )

        node_results: dict[str, dict[str, Any]] = {}
        for view in target_views:
            prediction_row = rows_by_node[(frame, view)]
            if (
                repair_missing_base
                and view in missing_base_view_set
            ):
                if missing_base_repair_protocol == "attested_multiseed":
                    assert temporal_camera_loader is not None
                    try:
                        temporal_camera = temporal_camera_loader.load_camera(view)
                    except ValueError as error:
                        if "not found in extrinsics file" not in str(error):
                            raise
                        temporal_camera = None
                    if temporal_camera is not None:
                        template = target_templates[view]
                        if not (
                            np.allclose(
                                temporal_camera.K,
                                template.K,
                                rtol=0.0,
                                atol=1e-9,
                            )
                            and np.allclose(
                                temporal_camera.D,
                                template.D,
                                rtol=0.0,
                                atol=1e-9,
                            )
                            and temporal_camera.H == template.H
                            and temporal_camera.W == template.W
                        ):
                            raise ValueError(
                                "temporal target intrinsics differ from runtime: "
                                f"frame={frame} view={view}"
                            )
                    missing_geometry = _attested_multiseed_missing_base_result(
                        prediction_row=prediction_row,
                        temporal_camera=temporal_camera,
                    )
                else:
                    crossfit_support, crossfit_target = observations_by_view[view]
                    base_verify_support, base_verify_target = (
                        base_validation_by_view[view]
                    )
                    missing_geometry = _missing_base_dual_source_geometry_scores(
                        support_cameras=support_cameras,
                        temporal_support_observations=crossfit_support,
                        temporal_target_observations=crossfit_target,
                        base_validation_support_observations=base_verify_support,
                        base_validation_target_observations=base_verify_target,
                        target_template=target_templates[view],
                        point_ranges=point_ranges,
                        fit_point_budget=fit_point_budget,
                        config=score_config,
                    )
                commit_missing = bool(
                    missing_geometry["valid"]
                    and missing_geometry["both_directions_prefer_track"]
                    and isinstance(
                        missing_geometry.get("commit_camera"), CameraParams
                    )
                )
                if commit_missing:
                    base_cameras[view] = missing_geometry["commit_camera"]
                    track_commit_count += 1
                    missing_base_track_commit_count += 1
                    action = "commit_temporal"
                    reason = (
                        "base_missing_attested_multiseed_track_verified"
                        if missing_base_repair_protocol == "attested_multiseed"
                        else "base_missing_dual_source_absolute_geometry_verified"
                    )
                else:
                    unregistered_count += 1
                    action = "unregistered"
                    reason = (
                        "base_and_temporal_unresolved"
                        if missing_base_repair_protocol == "attested_multiseed"
                        else "base_missing_absolute_geometry_inconclusive"
                    )
                candidate_solver = missing_geometry["commit_candidate_solver"]
                node_results[view] = {
                    "availability_action": action,
                    "availability_reason": reason,
                    "missing_base_geometry": missing_geometry,
                    "prediction_row": prediction_row,
                    "solver_success": bool(
                        candidate_solver.get("success", False)
                    ),
                    "verified": bool(
                        missing_geometry[
                            "commit_candidate_multiseed_verified"
                        ]
                    ),
                    "temporal_camera_available": bool(
                        isinstance(
                            missing_geometry.get("commit_camera"), CameraParams
                        )
                    ),
                }
                continue
            crossfit_support, crossfit_target = observations_by_view[view]
            if candidate_protocol == "dual_source_holdout":
                base_verify_support, base_verify_target = (
                    base_validation_by_view[view]
                )
                node_crossfit = _dual_source_holdout_geometry_scores(
                    support_cameras=support_cameras,
                    temporal_support_observations=crossfit_support,
                    temporal_target_observations=crossfit_target,
                    base_validation_support_observations=base_verify_support,
                    base_validation_target_observations=base_verify_target,
                    base_validation_source=base_validation_source_by_view[view],
                    base_camera=base_cameras[view],
                    target_template=base_cameras[view],
                    point_ranges=point_ranges,
                    fit_point_budget=fit_point_budget,
                    config=score_config,
                )
            else:
                geometry_score_fn = (
                    _exact_holdout_geometry_scores
                    if candidate_protocol == "exact_holdout"
                    else _crossfit_geometry_scores
                )
                geometry_kwargs = {
                    "support_cameras": support_cameras,
                    "support_observations": crossfit_support,
                    "target_observations": crossfit_target,
                    "static_camera": base_cameras[view],
                    "target_template": base_cameras[view],
                    "point_ranges": point_ranges,
                    "config": score_config,
                }
                if candidate_protocol == "legacy_crossfit_refit":
                    geometry_kwargs["final_refit_point_budget"] = (
                        final_refit_point_budget
                    )
                elif candidate_protocol == "crossfit_verified_fold":
                    geometry_kwargs["commit_candidate_strategy"] = (
                        "best_heldout_fold"
                    )
                elif candidate_protocol == "crossfit_fixed_fold":
                    geometry_kwargs["commit_candidate_strategy"] = "fixed_fold0"
                node_crossfit = geometry_score_fn(**geometry_kwargs)
            band_crossfit = (
                _crossfit_geometry_scores(
                    support_cameras=support_cameras,
                    support_observations=crossfit_support,
                    target_observations=crossfit_target,
                    static_camera=base_cameras[view],
                    target_template=base_cameras[view],
                    point_ranges=point_ranges,
                    config=score_config,
                    fold_point_ids=global_fold_ids,
                    final_refit_point_budget=final_refit_point_budget,
                    commit_candidate_strategy=(
                        "fixed_fold0"
                        if candidate_protocol == "crossfit_fixed_fold"
                        else "refit"
                    ),
                )
                if global_fold_ids is not None
                else node_crossfit
            )
            static_score = node_crossfit["static_score"]
            track_score = node_crossfit["track_score"]
            if candidate_protocol in {
                "exact_holdout",
                "dual_source_holdout",
                "crossfit_verified_fold",
                "crossfit_fixed_fold",
            }:
                solver_success = bool(
                    node_crossfit["commit_candidate_solver"].get("success", False)
                )
                verified = bool(
                    node_crossfit["commit_candidate_multiseed_verified"]
                )
            else:
                solver_success = bool(
                    prediction_row.get("solver", {}).get("success", False)
                )
                verified = bool(
                    (prediction_row.get("verification") or {}).get("verified", False)
                )
            static_value = static_score.get("truncated_reprojection_mean_px")
            track_value = track_score.get("truncated_reprojection_mean_px")
            commit, decision_reason = decide_track_commit(
                solver_success=solver_success,
                multiseed_verified=verified,
                static_score=static_score,
                track_score=track_score,
                commit_policy=commit_policy,
            )
            if not bool(node_crossfit["valid"]):
                commit = False
                decision_reason = "crossfit_inconclusive_keep_base"
            if commit and not bool(node_crossfit["both_directions_prefer_track"]):
                commit = False
                decision_reason = "crossfit_directions_disagree_keep_base"
            if candidate_protocol == "dual_source_holdout":
                if commit:
                    decision_reason = (
                        "matcher_and_temporal_holdouts_prefer_temporal"
                    )
                elif not bool(node_crossfit["valid"]):
                    decision_reason = "dual_source_holdout_inconclusive_keep_base"
                else:
                    decision_reason = "dual_source_holdouts_disagree_keep_base"
            if successful_base_policy == "retain":
                commit = False
                decision_reason = "successful_base_policy_retain"
            band_static_score = band_crossfit["static_score"]
            band_track_score = band_crossfit["track_score"]
            band_commit, band_decision_reason = decide_track_commit(
                solver_success=solver_success,
                multiseed_verified=verified,
                static_score=band_static_score,
                track_score=band_track_score,
                commit_policy=commit_policy,
            )
            if not bool(band_crossfit["valid"]):
                band_commit = False
                band_decision_reason = "band_crossfit_inconclusive_keep_base"
            if band_commit and not bool(
                band_crossfit["both_directions_prefer_track"]
            ):
                band_commit = False
                band_decision_reason = "band_crossfit_directions_disagree_keep_base"
            if successful_base_policy == "retain":
                band_commit = False
                band_decision_reason = "successful_base_policy_retain"
            confidence = paired_reprojection_confidence(static_score, track_score)
            node_results[view] = {
                "prediction_row": prediction_row,
                "node_crossfit": node_crossfit,
                "band_crossfit": band_crossfit,
                "static_score": static_score,
                "track_score": track_score,
                "static_value": static_value,
                "track_value": track_value,
                "solver_success": solver_success,
                "verified": verified,
                "node_commit": bool(commit),
                "node_decision_reason": decision_reason,
                "band_node_commit": bool(band_commit),
                "band_node_decision_reason": band_decision_reason,
                "confidence": confidence,
            }

        pair_scores: dict[tuple[str, str], dict[str, Any]] = {}
        band_decisions: dict[str, Any] | None = None
        if commit_scope == "band":
            assert global_fold_ids is not None
            for left, right in combinations(target_views, 2):
                pair_scores[(left, right)] = score_target_pair_crossfit(
                    left_observations=observations_by_view[left][1],
                    right_observations=observations_by_view[right][1],
                    static_left_camera=base_cameras[left],
                    static_right_camera=base_cameras[right],
                    candidate_left_cameras=node_results[left]["band_crossfit"][
                        "_candidate_cameras"
                    ],
                    candidate_right_cameras=node_results[right]["band_crossfit"][
                        "_candidate_cameras"
                    ],
                    fold_point_ids=global_fold_ids,
                    min_shared_points=8,
                    truncation_px=float(score_config.pnp_reprojection_error_px),
                )
            band_decision_fn = (
                decide_target_edge_supported_commits
                if band_graph_policy == "verified_edges"
                else decide_target_band_commits
            )
            band_decisions = band_decision_fn(
                target_views=target_views,
                node_commits={
                    view: bool(node_results[view]["band_node_commit"])
                    for view in target_views
                },
                pair_scores=pair_scores,
            )
            band_rows_by_frame[frame_name] = {
                "band_graph_policy": str(band_graph_policy),
                "fold_protocol": "seed-by-target-visibility-stratified-two-fold-v1",
                "fold_point_counts": [len(fold) for fold in global_fold_ids],
                "fold_point_ids_sha256": [
                    hashlib.sha256(np.asarray(fold, dtype=np.int64).tobytes()).hexdigest()
                    for fold in global_fold_ids
                ],
                "pair_scores": {
                    f"{left}:{right}": score
                    for (left, right), score in sorted(pair_scores.items())
                },
                **band_decisions,
            }

        for view in target_views:
            result = node_results[view]
            if "availability_action" in result:
                action = str(result["availability_action"])
                prediction_row = result["prediction_row"]
                missing_geometry = result.get("missing_base_geometry")
                geometry = (
                    missing_geometry
                    if isinstance(missing_geometry, Mapping)
                    else {}
                )
                source_rows.append(
                    {
                        "frame": int(frame),
                        "view": view,
                        "source": (
                            "temporal_track_repair"
                            if action == "commit_temporal"
                            else "unresolved"
                        ),
                        "decision_reason": str(result["availability_reason"]),
                        "missing_base_repair": True,
                        "base_pose_available": False,
                        "base_solver_success": False,
                        "node_commit": action == "commit_temporal",
                        "band_node_commit": False,
                        "band_node_decision_reason": "not_applicable",
                        "band_commit": action == "commit_temporal",
                        "band_component_views": [view],
                        "solver_success": bool(result["solver_success"]),
                        "multiseed_verified": bool(result["verified"]),
                        "temporal_camera_available": bool(
                            result["temporal_camera_available"]
                        ),
                        "fit_point_count": int(
                            prediction_row.get("observation", {}).get(
                                "selected_count", 0
                            )
                        ),
                        "upstream_prediction_fit_count": int(
                            prediction_row.get("observation", {}).get(
                                "selected_count", 0
                            )
                        ),
                        "base_validation_observation": (
                            base_validation_diagnostics_by_view.get(view)
                        ),
                        "temporal_observation": (
                            temporal_observation_diagnostics_by_view.get(view)
                        ),
                        "crossfit_valid": bool(geometry.get("valid", False)),
                        "crossfit_reason": str(
                            geometry.get(
                                "reason", "missing_base_candidate_unavailable"
                            )
                        ),
                        "crossfit_point_count": int(
                            geometry.get("point_count", 0)
                        ),
                        "crossfit_fold_point_counts": geometry.get(
                            "fold_point_counts", []
                        ),
                        "crossfit_fold_point_ids_sha256": geometry.get(
                            "fold_point_ids_sha256", []
                        ),
                        "crossfit_fold_rows": geometry.get("fold_rows", []),
                        "both_directions_prefer_track": bool(
                            geometry.get(
                                "both_directions_prefer_track", False
                            )
                        ),
                        "validation_prefers_temporal": bool(
                            geometry.get(
                                "both_directions_prefer_track", False
                            )
                        ),
                        "commit_candidate_protocol": geometry.get(
                            "commit_candidate_protocol"
                        ),
                        "commit_candidate_fit_point_count": int(
                            geometry.get(
                                "commit_candidate_fit_point_count", 0
                            )
                        ),
                        "commit_candidate_multiseed_verified": bool(
                            geometry.get(
                                "commit_candidate_multiseed_verified", False
                            )
                        ),
                        "commit_candidate_verification_reason": str(
                            geometry.get(
                                "commit_candidate_verification_reason",
                                "candidate_unavailable",
                            )
                        ),
                        "commit_candidate_solver": geometry.get(
                            "commit_candidate_solver", {}
                        ),
                        "base_score": _score_diagnostics(
                            geometry.get("static_score", {})
                        ),
                        "temporal_track_score": _score_diagnostics(
                            geometry.get("track_score", {})
                        ),
                        "paired_reprojection_confidence": {
                            "valid": False,
                            "reason": "base_pose_unavailable",
                        },
                    }
                )
                continue
            crossfit = result["node_crossfit"]
            band_crossfit = result["band_crossfit"]
            node_commit = bool(result["node_commit"])
            commit = (
                bool(band_decisions["commit_by_view"][view])
                if band_decisions is not None
                else node_commit
            )
            component = (
                next(
                    component
                    for component in band_decisions["components"]
                    if view in component
                )
                if band_decisions is not None
                else [view]
            )
            decision_reason = str(result["node_decision_reason"])
            if commit_scope == "band" and band_graph_policy == "verified_edges":
                decision_reason = (
                    "node_and_cross_target_oof_geometry_prefer_track"
                    if commit
                    else "cross_target_edge_not_verified_keep_base"
                )
            elif commit_scope == "band" and len(component) > 1:
                decision_reason = (
                    "band_base_and_cross_target_geometry_prefer_track"
                    if commit
                    else "band_geometry_not_dominant_keep_base"
                )
            if commit:
                commit_camera = select_band_commit_camera(
                    node_crossfit=crossfit,
                    band_crossfit=band_crossfit,
                    candidate_protocol=candidate_protocol,
                )
                if commit_camera is None:
                    raise RuntimeError(
                        f"committed band candidate is unavailable: frame={frame} view={view}"
                    )
                base_cameras[view] = commit_camera
                track_commit_count += 1
            prediction_row = result["prediction_row"]
            source_rows.append(
                {
                    "frame": int(frame),
                    "view": view,
                    "source": (
                        "temporal_track_repair" if commit else base_pose_method
                    ),
                    "decision_reason": decision_reason,
                    "missing_base_repair": False,
                    "base_pose_available": True,
                    "base_solver_success": True,
                    "node_commit": node_commit,
                    "band_node_commit": bool(result["band_node_commit"]),
                    "band_node_decision_reason": str(
                        result["band_node_decision_reason"]
                    ),
                    "band_commit": bool(commit),
                    "band_component_views": component,
                    "solver_success": bool(result["solver_success"]),
                    "multiseed_verified": bool(result["verified"]),
                    "fit_point_count": int(
                        prediction_row.get("observation", {}).get(
                            "selected_count", fit_point_budget
                        )
                    ),
                    "upstream_prediction_fit_count": int(
                        prediction_row.get("observation", {}).get(
                            "selected_count", fit_point_budget
                        )
                    ),
                    "base_validation_observation": (
                        base_validation_diagnostics_by_view.get(view)
                    ),
                    "temporal_observation": (
                        temporal_observation_diagnostics_by_view.get(view)
                    ),
                    "crossfit_valid": bool(crossfit["valid"]),
                    "crossfit_reason": str(crossfit["reason"]),
                    "crossfit_point_count": int(crossfit["point_count"]),
                    "crossfit_fold_point_counts": crossfit["fold_point_counts"],
                    "crossfit_fold_point_ids_sha256": crossfit[
                        "fold_point_ids_sha256"
                    ],
                    "crossfit_fold_rows": crossfit["fold_rows"],
                    "both_directions_prefer_track": bool(
                        crossfit["both_directions_prefer_track"]
                    ),
                    "validation_prefers_temporal": bool(
                        crossfit["both_directions_prefer_track"]
                    ),
                    "commit_candidate_protocol": crossfit[
                        "commit_candidate_protocol"
                    ],
                    "selected_fit_fold": crossfit.get("selected_fit_fold"),
                    "commit_candidate_fit_point_count": int(
                        crossfit["commit_candidate_fit_point_count"]
                    ),
                    "commit_candidate_multiseed_verified": bool(
                        crossfit["commit_candidate_multiseed_verified"]
                    ),
                    "commit_candidate_verification_reason": str(
                        crossfit["commit_candidate_verification_reason"]
                    ),
                    "commit_candidate_solver": crossfit[
                        "commit_candidate_solver"
                    ],
                    "band_crossfit_valid": bool(band_crossfit["valid"]),
                    "band_crossfit_reason": str(band_crossfit["reason"]),
                    "band_crossfit_point_count": int(band_crossfit["point_count"]),
                    "band_crossfit_fold_point_counts": band_crossfit[
                        "fold_point_counts"
                    ],
                    "band_crossfit_fold_point_ids_sha256": band_crossfit[
                        "fold_point_ids_sha256"
                    ],
                    "band_crossfit_fold_rows": band_crossfit["fold_rows"],
                    "band_both_directions_prefer_track": bool(
                        band_crossfit["both_directions_prefer_track"]
                    ),
                    "base_score_px": (
                        float(result["static_value"])
                        if result["static_value"] is not None
                        else None
                    ),
                    "temporal_track_score_px": (
                        float(result["track_value"])
                        if result["track_value"] is not None
                        else None
                    ),
                    "base_score": _score_diagnostics(result["static_score"]),
                    "temporal_track_score": _score_diagnostics(
                        result["track_score"]
                    ),
                    "paired_reprojection_confidence": result["confidence"],
                }
            )
        frame_output = pose_output / frame_name / "cameras_pred"
        save_cameras(frame_output, base_cameras)
        output_camera_hashes[frame_name] = _camera_file_hashes(frame_output)
        available_output_views_by_frame[frame_name] = [
            view for view in export_views if view in base_cameras
        ]

    manifest = {
        "protocol": (
            "hidden-parent-fixed-fold-edge-verified-hybrid-v1"
            if (
                candidate_protocol == "crossfit_fixed_fold"
                and commit_scope == "band"
                and band_graph_policy == "verified_edges"
            )
            else "hidden-parent-fixed-fold-atomic-supported-hybrid-v1"
            if (
                candidate_protocol == "crossfit_fixed_fold"
                and commit_scope == "band"
                and band_graph_policy == "atomic_supported"
            )
            else "track-driven-public-static-pnp-missing-closure-v1"
            if (
                missing_base_repair_protocol == "attested_multiseed"
                and successful_base_policy == "retain"
            )
            else
            "hidden-parent-attested-multiseed-missing-base-closure-v5"
            if missing_base_repair_protocol == "attested_multiseed"
            else
            "hidden-parent-dual-source-absolute-missing-base-closure-v4"
            if (
                candidate_protocol == "dual_source_holdout"
                and missing_base_repair_protocol == "dual_source_absolute"
            )
            else "hidden-parent-dual-source-capped-fit-temporal-closure-v2"
            if candidate_protocol == "dual_source_holdout"
            else "hidden-parent-exact-holdout-temporal-closure-v2"
            if candidate_protocol == "exact_holdout"
            else "hidden-parent-crossfit-verified-fold-geometry-hybrid-v1"
            if candidate_protocol == "crossfit_verified_fold"
            else
            "hidden-parent-stratified-band-crossfit-bounded-refit-hybrid-v6"
            if commit_scope == "band" and final_refit_point_budget is not None
            else "hidden-parent-crossfit-bounded-refit-geometry-hybrid-v5"
            if final_refit_point_budget is not None
            else "hidden-parent-stratified-band-crossfit-full-refit-hybrid-v5"
            if commit_scope == "band"
            else "hidden-parent-crossfit-full-refit-geometry-hybrid-v4"
        ),
        "score_protocol": (
            "fixed-fold0-fit-fold1-verify-bidirectional-node-and-edge-v1"
            if (
                candidate_protocol == "crossfit_fixed_fold"
                and commit_scope == "band"
                and band_graph_policy == "verified_edges"
            )
            else "fixed-fold0-fit-fold1-verify-bidirectional-node-atomic-band-v1"
            if (
                candidate_protocol == "crossfit_fixed_fold"
                and commit_scope == "band"
                and band_graph_policy == "atomic_supported"
            )
            else "missing-base-attested-multiseed-inlier-closure-v1"
            if missing_base_repair_protocol == "attested_multiseed"
            else
            "equal-fit-independent-matcher-and-temporal-absolute-holdouts-v4"
            if (
                candidate_protocol == "dual_source_holdout"
                and missing_base_repair_protocol == "dual_source_absolute"
            )
            else
            "capped-fit-base-holdout-or-conservative-fit-reuse-plus-temporal-unseen-v2"
            if candidate_protocol == "dual_source_holdout"
            else "seed-balanced-quality-prefix-2to1-fit-unseen-verify-v4"
            if candidate_protocol == "exact_holdout"
            else "two-fold-fit150-heldout150-verified-candidate-selection-v1"
            if candidate_protocol == "crossfit_verified_fold"
            else
            "stratified-two-fold-static-and-cross-target-track-validation-v2"
            if commit_scope == "band"
            else "two-fold-exact-candidate-common-track-validation-v1"
        ),
        "commit_policy": str(commit_policy),
        "commit_scope": str(commit_scope),
        "band_graph_policy": str(band_graph_policy),
        "candidate_protocol": str(candidate_protocol),
        "missing_base_repair_protocol": str(missing_base_repair_protocol),
        "successful_base_policy": str(successful_base_policy),
        "repair_missing_base_with_verified_track": bool(
            repair_missing_base
        ),
        "missing_base_temporal_fit_policy": (
            {
                "fit_point_budget_role": "maximum",
                "candidate_source": "attested_upstream_temporal_pose",
                "verification_source": (
                    "existing_multiseed_inlier_support"
                ),
                "extra_numeric_hyperparameters": 0,
            }
            if missing_base_repair_protocol == "attested_multiseed"
            else
            {
                "fit_point_budget_role": "exact",
                "absolute_reprojection_threshold_source": (
                    "existing_pnp_reprojection_error_px"
                ),
                "extra_numeric_hyperparameters": 0,
            }
            if missing_base_repair_protocol == "dual_source_absolute"
            else None
        ),
        "runtime_scene": str(runtime),
        "runtime_tree_digest": runtime_digest,
        "pose_root": str(pose_output),
        "base_pose_root": str(base_root),
        "base_pose_method": str(base_pose_method),
        "base_prediction_attestation": base_lineage["attestation_path"],
        "base_prediction_path": base_lineage["prediction_path"],
        "base_prediction_sha256": base_lineage["prediction_sha256"],
        "base_lineage_protocol": base_lineage["protocol"],
        "base_matcher_invocations": base_lineage["matcher_invocations"],
        "temporal_seed_matcher_invocations": int(len(point_ranges)),
        "total_matcher_invocations": (
            None
            if base_lineage["matcher_invocations"] is None
            else int(base_lineage["matcher_invocations"] + len(point_ranges))
        ),
        "track_output_root": str(track_output),
        "track_root": str(track_root),
        "track_prediction_sha256": prediction_sha256,
        "track_file_hashes": track_hashes,
        "fit_point_budget": fit_point_budget,
        "candidate_point_budget": candidate_point_budget,
        "candidate_fit_budget_target": (
            int(2 * fit_point_budget)
            if candidate_protocol == "exact_holdout"
            else int(fit_point_budget)
        ),
        "candidate_fit_budget_role": (
            "maximum"
            if candidate_protocol == "dual_source_holdout"
            else "protocol_specific"
        ),
        "candidate_verify_budget_target": int(fit_point_budget),
        "base_validation_budget_target": (
            int(fit_point_budget)
            if candidate_protocol == "dual_source_holdout"
            else None
        ),
        "temporal_validation_budget_target": (
            int(fit_point_budget)
            if candidate_protocol == "dual_source_holdout"
            else None
        ),
        "base_validation_fit_reuse_is_conservative": bool(
            candidate_protocol == "dual_source_holdout"
        ),
        "current_seed_leave_one_out": bool(
            candidate_protocol == "dual_source_holdout"
        ),
        "crossfit_point_budget": candidate_point_budget,
        "final_refit_point_budget": (
            None
            if candidate_protocol in no_refit_protocols
            else int(final_refit_point_budget)
            if final_refit_point_budget is not None
            else int(2 * fit_point_budget)
        ),
        "frames": frames,
        "target_views": target_views,
        "output_views": export_views,
        "track_commit_count": int(track_commit_count),
        "missing_base_count": int(missing_base_count),
        "missing_base_track_commit_count": int(
            missing_base_track_commit_count
        ),
        "unresolved_count": int(unregistered_count),
        "unregistered_count": int(unregistered_count),
        "unresolved_nodes": [
            {"frame": int(row["frame"]), "view": str(row["view"])}
            for row in source_rows
            if row["source"] == "unresolved"
        ],
        "base_fallback_count": int(
            sum(row["source"] == base_pose_method for row in source_rows)
        ),
        "temporal_commit_count": int(track_commit_count),
        "base_retain_count": int(
            sum(row["source"] == base_pose_method for row in source_rows)
        ),
        "source_rows": source_rows,
        "available_output_views_by_frame": available_output_views_by_frame,
        "band_rows_by_frame": band_rows_by_frame,
        "input_camera_hashes": {
            "base": base_camera_hashes,
        },
        "output_camera_hashes": output_camera_hashes,
        "gt_used": False,
        "nvs_used": False,
    }
    _write_json(output / "hybrid_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Commit a track repair only when it wins on common held-out geometry."
    )
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--base-pose-root", type=Path, required=True)
    parser.add_argument(
        "--base-pose-method",
        choices=["static_pnp", "static_map_same_frame_pnp"],
        required=True,
        help="Method that produced the pose retained when temporal repair is rejected.",
    )
    parser.add_argument(
        "--base-prediction-attestation",
        type=Path,
        help="Required pre-GT pose attestation for static_map_same_frame_pnp.",
    )
    parser.add_argument("--track-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--commit-policy",
        choices=["mean", "confidence95"],
        default="mean",
    )
    parser.add_argument(
        "--commit-scope",
        choices=["node", "band"],
        default="node",
    )
    parser.add_argument(
        "--band-graph-policy",
        choices=["atomic_supported", "verified_edges"],
        default="atomic_supported",
    )
    parser.add_argument(
        "--candidate-protocol",
        choices=[
            "exact_holdout",
            "dual_source_holdout",
            "legacy_crossfit_refit",
            "crossfit_verified_fold",
            "crossfit_fixed_fold",
        ],
        default="exact_holdout",
    )
    parser.add_argument(
        "--final-refit-point-budget",
        type=int,
        default=0,
        help="Positive value caps the committed candidate refit; 0 preserves legacy full refit.",
    )
    parser.add_argument(
        "--missing-base-repair-protocol",
        choices=["none", "dual_source_absolute", "attested_multiseed"],
        default="none",
        help=(
            "Protocol used only when same-time Static+PnP has no target pose."
        ),
    )
    parser.add_argument(
        "--successful-base-policy",
        choices=["verify_commit", "retain"],
        default="verify_commit",
        help=(
            "Whether successful Static+PnP nodes remain eligible for temporal "
            "replacement or are retained byte-for-byte."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        if not bool(args.overwrite):
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    summary = export_geometry_verified_hybrid(
        runtime_scene=args.runtime_scene,
        base_pose_root=args.base_pose_root,
        base_pose_method=str(args.base_pose_method),
        base_prediction_attestation=args.base_prediction_attestation,
        track_output_root=args.track_output_root,
        output_root=args.output_root,
        commit_policy=str(args.commit_policy),
        commit_scope=str(args.commit_scope),
        band_graph_policy=str(args.band_graph_policy),
        candidate_protocol=str(args.candidate_protocol),
        final_refit_point_budget=(
            int(args.final_refit_point_budget)
            if int(args.final_refit_point_budget) > 0
            else None
        ),
        missing_base_repair_protocol=str(args.missing_base_repair_protocol),
        successful_base_policy=str(args.successful_base_policy),
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "protocol",
                    "score_protocol",
                    "commit_policy",
                    "commit_scope",
                    "band_graph_policy",
                    "candidate_protocol",
                    "base_pose_method",
                    "final_refit_point_budget",
                    "missing_base_repair_protocol",
                    "successful_base_policy",
                    "track_commit_count",
                    "missing_base_track_commit_count",
                    "unregistered_count",
                    "base_fallback_count",
                    "gt_used",
                    "nvs_used",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
