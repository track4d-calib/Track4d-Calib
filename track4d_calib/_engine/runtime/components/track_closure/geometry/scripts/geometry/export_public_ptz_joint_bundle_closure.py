#!/usr/bin/env python3
"""Export a cross-fit, track-driven shared-center PTZ repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader, CameraParams, save_cameras
from track4d_calib import track_joint_ptz_bundle as joint_solver_module
from track4d_calib.track_crossview_verify import (
    CrossViewTrackVerifierConfig,
    stable_point_id_validation_groups,
)
from track4d_calib.track_joint_ptz_bundle import (
    crossfit_joint_ptz_shared_center,
    decide_joint_ptz_band_model_commit,
)
from track4d_calib.track_temporal_static_filter import (
    evaluate_fixed_static_landmark_identity,
    select_temporally_static_track_ids,
)
from scripts.geometry.evaluate_track_pose_after_stage_a_seal import (
    _verify_stage_a_seal,
)
from scripts.geometry.export_public_ptz_track_center_closure import (
    COMMIT_POLICY,
    _build_node_point_bank,
    _collect_successful_nodes,
    _load_attested_prediction,
    _load_independent_track_bundles,
    _load_successful_cameras,
    _load_track_bundle,
    _load_track_bundle_payload,
    _verify_runtime_tree,
)
from scripts.geometry.run_track_closure import (
    camera_root_attestation,
    seed_point_ranges,
    track_file_attestation,
)


METHOD = "public_ptz_joint_track_bundle_synchronous_geometry_crossfit_v9"
INDEPENDENT_CONTROL_METHOD = (
    "public_ptz_joint_independent_match_synchronous_geometry_crossfit_control_v1"
)
MISSING_BASE_METHOD = (
    "public_ptz_joint_track_bundle_synchronous_geometry_crossfit_"
    "missing_se3_fallback_v2"
)
MIXED_BASE_METHOD = (
    "public_ptz_joint_track_bundle_synchronous_geometry_crossfit_"
    "complete_view_repair_v1"
)
VERIFIED_VIEW_METHOD = (
    "public_ptz_joint_track_bundle_synchronous_geometry_crossfit_"
    "verified_view_repair_v1"
)
TEMPORAL_BOOTSTRAP_METHOD = (
    "public_ptz_joint_track_bundle_synchronous_geometry_crossfit_"
    "zero_static_anchor_temporal_bootstrap_v1"
)
TEMPORAL_BOOTSTRAP_PROTOCOL = (
    "verified_temporal_track_pnp_zero_static_anchor_bootstrap_v1"
)
BAND_COMMIT_POLICY = (
    "heldout robust cost decreases and fold projection consistency is "
    "within the focus threshold"
)
TRACK_GEOMETRY_POLICY = (
    "per-frame synchronous multiview tracks passing public-camera geometry"
)
LANDMARK_GEOMETRY_POLICY = (
    "one independently triangulated XYZ per track and frame"
)
STATIC_LANDMARK_METHOD_SUFFIX = "public_static_temporal_landmarks_v1"
FULL_PARENT_OBSERVATION_METHOD_SUFFIX = (
    "attested_parent_complete_temporal_observations_v1"
)


def load_attested_parent_temporal_bundle(
    *,
    landmark_bundle: Mapping[str, Any],
    runtime_views: Sequence[str],
    static_views: Sequence[str],
    target_views: Sequence[str],
    frames: Sequence[int],
    observation_track_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact full-track parent attested by a refined local bundle."""

    landmark_root = Path(landmark_bundle["root"]).resolve()
    landmark_meta = dict(landmark_bundle["meta"])
    parent_value = landmark_meta.get("parent_track_root")
    expected_hashes = landmark_meta.get("parent_track_files_sha256")
    if not parent_value or not isinstance(expected_hashes, Mapping):
        raise ValueError("landmark bundle does not attest a parent track payload")
    parent_root = Path(str(parent_value)).resolve()
    parent_hashes = track_file_attestation(parent_root)
    normalized_expected_hashes = {
        str(key): str(value) for key, value in expected_hashes.items()
    }
    if normalized_expected_hashes != parent_hashes:
        raise ValueError("parent track payload differs from the landmark attestation")

    observation_root = (
        parent_root
        if observation_track_root is None
        else Path(observation_track_root).resolve()
    )
    observation_derivation: dict[str, Any]
    if observation_root == parent_root:
        observation_derivation = {"policy": "exact_attested_parent"}
    else:
        observation_meta_path = observation_root / "meta.json"
        if not observation_meta_path.is_file():
            raise ValueError("complete observation bundle has no metadata")
        observation_meta = json.loads(
            observation_meta_path.read_text(encoding="utf-8")
        )
        if Path(str(observation_meta.get("parent_track_root", ""))).resolve() != parent_root:
            raise ValueError("complete observation bundle has a different parent")
        if {
            str(key): str(value)
            for key, value in dict(
                observation_meta.get("parent_track_files_sha256") or {}
            ).items()
        } != parent_hashes:
            raise ValueError("complete observation bundle parent hashes differ")
        refinement = dict(observation_meta.get("track_refinement") or {})
        landmark_refinement = dict(landmark_meta.get("track_refinement") or {})
        if str(refinement.get("window_mode")) != "full":
            raise ValueError("derived complete observations require full-window refinement")
        if float(refinement.get("forward_backward_threshold_px", float("nan"))) != float(
            landmark_refinement.get("forward_backward_threshold_px", float("nan"))
        ):
            raise ValueError("complete and landmark LK thresholds differ")
        observation_derivation = {
            "policy": "full_window_lk_from_exact_attested_parent",
            "track_refinement": refinement,
        }

    parent_bundle = _load_track_bundle_payload(
        root=observation_root,
        runtime_views=runtime_views,
        static_views=static_views,
        target_views=target_views,
        frames=frames,
    )
    if parent_bundle["views"] != landmark_bundle["views"]:
        raise ValueError("parent and landmark bundles have different view ordering")
    if tuple(parent_bundle["tracks"].shape) != tuple(landmark_bundle["tracks"].shape):
        raise ValueError("parent and landmark track arrays have different shapes")
    if tuple(parent_bundle["visibility"].shape) != tuple(
        landmark_bundle["visibility"].shape
    ):
        raise ValueError("parent and landmark visibility arrays have different shapes")
    layout_fields = (
        "start_frame",
        "num_frames",
        "num_points",
        "seed_frames",
        "num_points_by_seed",
    )
    for field in layout_fields:
        if parent_bundle["meta"].get(field) != landmark_meta.get(field):
            raise ValueError(f"parent and landmark track layout differ at {field}")

    parent_points = observation_root / "points_t0.npy"
    landmark_points = landmark_root / "points_t0.npy"
    if not parent_points.is_file() or not landmark_points.is_file():
        raise ValueError("both track bundles must contain points_t0.npy")
    parent_points_hash = _sha256(parent_points)
    landmark_points_hash = _sha256(landmark_points)
    if parent_points_hash != landmark_points_hash:
        raise ValueError("parent and landmark point axes are not byte-identical")
    audit = {
        "protocol": "refined_bundle_attested_complete_parent_v1",
        "landmark_track_root": str(landmark_root),
        "observation_track_root": str(observation_root),
        "source_parent_track_root": str(parent_root),
        "source_parent_file_sha256": parent_hashes,
        "observation_file_sha256": track_file_attestation(observation_root),
        "observation_derivation": observation_derivation,
        "points_t0_sha256": parent_points_hash,
        "views_order_identical": True,
        "array_shapes_identical": True,
        "point_axis_identical": True,
        "layout_fields_identical": list(layout_fields),
        "gt_fields_used": [],
        "nvs_fields_used": [],
    }
    return parent_bundle, audit


def temporal_static_landmark_geometry(
    filter_result: Mapping[str, Any],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Validate and index the fixed public-static landmarks from the filter."""

    point_ids = np.asarray(
        filter_result.get("selected_point_ids", []), dtype=np.int64
    ).reshape(-1)
    xyz = np.asarray(
        filter_result.get("selected_landmark_xyz", []), dtype=np.float64
    ).reshape(-1, 3)
    if len(point_ids) != len(xyz):
        raise ValueError("temporal-static point IDs and XYZ have different lengths")
    if len(point_ids) != len(np.unique(point_ids)):
        raise ValueError("temporal-static point IDs must be unique")
    if np.any(point_ids < 0) or not np.all(np.isfinite(xyz)):
        raise ValueError("temporal-static landmarks must be finite and non-negative")
    return point_ids, {
        int(point_id): point.copy()
        for point_id, point in zip(point_ids, xyz, strict=True)
    }


def summarize_temporal_observation_coverage(
    *,
    observations_by_view: Mapping[str, Sequence[Mapping[str, Any]]],
    frames: Sequence[int],
    target_views: Sequence[str],
    num_folds: int,
    min_inliers: int,
    min_successful_frames: int,
) -> dict[str, Any]:
    """Summarize the exact point-ID split before running pose optimization."""

    folds = int(num_folds)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    rows = []
    by_view: dict[str, dict[str, Any]] = {}
    for view in [str(value) for value in target_views]:
        observations = list(observations_by_view[view])
        if [int(row["frame"]) for row in observations] != [
            int(frame) for frame in frames
        ]:
            raise ValueError("coverage observations do not match the frame protocol")
        successful_frames = 0
        for observation in observations:
            point_ids = np.asarray(
                observation["point_ids"], dtype=np.int64
            ).reshape(-1)
            groups = stable_point_id_validation_groups(point_ids)
            fold_rows = []
            for validation_fold in range(folds):
                validation_count = int(
                    np.count_nonzero(np.mod(groups, folds) == validation_fold)
                )
                fold_rows.append(
                    {
                        "validation_fold": validation_fold,
                        "fit_count": int(len(point_ids) - validation_count),
                        "verify_count": validation_count,
                    }
                )
            usable = len(point_ids) >= int(min_inliers)
            successful_frames += int(usable)
            rows.append(
                {
                    "frame": int(observation["frame"]),
                    "view": view,
                    "effective_landmark_count": int(len(point_ids)),
                    "meets_min_inliers": bool(usable),
                    "folds": fold_rows,
                }
            )
        by_view[view] = {
            "successful_frame_count": int(successful_frames),
            "required_successful_frame_count": int(min_successful_frames),
            "passes": bool(successful_frames >= int(min_successful_frames)),
            "minimum_effective_landmarks": int(
                min(
                    row["effective_landmark_count"]
                    for row in rows
                    if row["view"] == view
                )
            ),
        }
    return {
        "rows": rows,
        "by_view": by_view,
        "all_views_pass": all(row["passes"] for row in by_view.values()),
        "num_folds": folds,
        "min_inliers": int(min_inliers),
        "min_successful_frames": int(min_successful_frames),
        "gt_fields_used": [],
        "nvs_fields_used": [],
    }


def effective_method(
    missing_base_policy: str,
    *,
    observation_source: str = "temporal_tracks",
) -> str:
    source = str(observation_source)
    if source == "independent_same_frame":
        if str(missing_base_policy) == "temporal_bootstrap":
            raise ValueError(
                "temporal_bootstrap requires temporal_tracks observations"
            )
        if str(missing_base_policy) != "error":
            return f"{INDEPENDENT_CONTROL_METHOD}_{missing_base_policy}"
        return INDEPENDENT_CONTROL_METHOD
    if source != "temporal_tracks":
        raise ValueError(f"unsupported observation source: {source}")
    policy = str(missing_base_policy)
    if policy == "error":
        return METHOD
    if policy == "interpolate":
        return MISSING_BASE_METHOD
    if policy == "interpolate_complete_views":
        return MIXED_BASE_METHOD
    if policy == "interpolate_verified_views":
        return VERIFIED_VIEW_METHOD
    if policy == "temporal_bootstrap":
        return TEMPORAL_BOOTSTRAP_METHOD
    raise ValueError(f"unsupported missing-base policy: {policy}")


def namespace_independent_observation_ids(
    observation: Mapping[str, Any],
    *,
    frame_ordinal: int,
) -> dict[str, Any]:
    """Prevent unrelated per-frame matcher IDs from becoming temporal IDs."""

    if int(frame_ordinal) < 0:
        raise ValueError("frame_ordinal must be non-negative")
    point_ids = np.asarray(observation["point_ids"], dtype=np.int64)
    if np.any(point_ids < 0) or np.any(point_ids >= (1 << 32)):
        raise ValueError("independent point IDs must fit in unsigned 32 bits")
    output = dict(observation)
    output["point_ids"] = point_ids + (int(frame_ordinal) << 32)
    return output


def observation_bundle_for_frame(
    *,
    observation_source: str,
    frame: int,
    temporal_bundle: Mapping[str, Any] | None,
    independent_bundles: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    source = str(observation_source)
    if source == "temporal_tracks":
        if temporal_bundle is None:
            raise ValueError("missing temporal track bundle")
        return temporal_bundle
    if source == "independent_same_frame":
        try:
            return independent_bundles[int(frame)]
        except KeyError as exc:
            raise KeyError(
                f"missing independent same-frame bundle for frame {int(frame)}"
            ) from exc
    raise ValueError(f"unsupported observation source: {source}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_file(module: Any) -> str:
    path = getattr(module, "__file__", None)
    if path is None:
        raise ValueError(f"module has no source file: {module!r}")
    return str(Path(path).resolve())


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if str(key) != "cameras"
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sealed_selector_protocol(seal_path: Path) -> tuple[str, str]:
    seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
    selector_subdir = str(seal.get("selector_subdir", ""))
    if (
        not selector_subdir
        or Path(selector_subdir).name != selector_subdir
        or selector_subdir in {".", ".."}
    ):
        raise ValueError("Stage-A seal has an invalid selector_subdir")
    commit_policy = str(seal.get("commit_policy", ""))
    if commit_policy != COMMIT_POLICY:
        raise ValueError(
            "Stage-A seal must use the confidence95 commit policy: "
            f"got {commit_policy!r}"
        )
    return selector_subdir, commit_policy


def extract_joint_observation(
    *,
    frame: int,
    view: str,
    point_bank: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    views = [str(value) for value in bundle["views"]]
    focus = str(view)
    if focus not in views:
        raise KeyError(f"target view {focus!r} is absent from track bundle")
    local = int(frame) - int(bundle["meta"]["start_frame"])
    focus_index = views.index(focus)
    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    xyz = np.asarray(point_bank.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    if len(point_ids) != len(xyz):
        raise ValueError("point bank point_ids and xyz have different lengths")
    tracks = np.asarray(bundle["tracks"][focus_index, local], dtype=np.float64)
    visibility = np.asarray(
        bundle["visibility"][focus_index, local], dtype=bool
    ).reshape(-1)
    in_range = (point_ids >= 0) & (point_ids < len(tracks))
    safe_ids = point_ids[in_range]
    safe_xyz = xyz[in_range]
    image_points = tracks[safe_ids]
    valid = (
        visibility[safe_ids]
        & np.isfinite(safe_xyz).all(axis=1)
        & np.isfinite(image_points).all(axis=1)
    )
    return {
        "frame": int(frame),
        "point_ids": safe_ids[valid],
        "xyz": safe_xyz[valid],
        "image_points": image_points[valid],
    }


def _camera_equal(first: CameraParams, second: CameraParams) -> bool:
    return (
        first.view_id == second.view_id
        and first.H == second.H
        and first.W == second.W
        and np.array_equal(first.K, second.K)
        and np.array_equal(first.R, second.R)
        and np.array_equal(first.T, second.T)
        and np.array_equal(first.D, second.D)
    )


def complete_base_camera_grid(
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
    base_cameras: Mapping[tuple[int, str], CameraParams],
) -> tuple[dict[tuple[int, str], CameraParams], dict[str, Any]]:
    """Fill failed Static+PnP nodes without changing successful cameras."""

    ordered_frames = [int(frame) for frame in frames]
    ordered_views = [str(view) for view in target_views]
    if ordered_frames != sorted(set(ordered_frames)):
        raise ValueError("frames must be sorted and unique")
    if len(ordered_views) != len(set(ordered_views)):
        raise ValueError("target views must be unique")

    completed = dict(base_cameras)
    missing_rows: list[dict[str, Any]] = []
    for view in ordered_views:
        anchors = [
            frame for frame in ordered_frames if (frame, view) in base_cameras
        ]
        if not anchors:
            raise ValueError(f"no successful Static+PnP camera for view {view}")
        template = base_cameras[(anchors[0], view)]
        for frame in anchors[1:]:
            camera = base_cameras[(frame, view)]
            if (
                camera.view_id != template.view_id
                or camera.H != template.H
                or camera.W != template.W
                or not np.array_equal(camera.K, template.K)
                or not np.array_equal(camera.D, template.D)
            ):
                raise ValueError(
                    f"cannot interpolate time-varying intrinsics for view {view}"
                )

        for frame in ordered_frames:
            node = (frame, view)
            if node in completed:
                continue
            left = max((value for value in anchors if value < frame), default=None)
            right = min((value for value in anchors if value > frame), default=None)
            if left is not None and right is not None:
                alpha = (frame - left) / (right - left)
                left_camera = base_cameras[(left, view)]
                right_camera = base_cameras[(right, view)]
                center = (
                    (1.0 - alpha) * np.asarray(left_camera.camera_center)
                    + alpha * np.asarray(right_camera.camera_center)
                )
                slerp = Slerp(
                    np.array([float(left), float(right)], dtype=np.float64),
                    Rotation.from_matrix(
                        np.stack([left_camera.R.T, right_camera.R.T], axis=0)
                    ),
                )
                rotation = slerp([float(frame)]).as_matrix()[0].T
                policy = "linear_center_slerp_rotation"
                source = left_camera
            else:
                boundary = left if left is not None else right
                if boundary is None:  # pragma: no cover - guarded by anchors above.
                    raise RuntimeError("missing interpolation boundary")
                source = base_cameras[(boundary, view)]
                center = np.asarray(source.camera_center, dtype=np.float64).copy()
                rotation = np.asarray(source.R, dtype=np.float64).copy()
                alpha = None
                policy = "nearest_boundary_hold"
            completed[node] = CameraParams(
                view_id=str(source.view_id),
                K=np.asarray(source.K, dtype=np.float64).copy(),
                R=np.asarray(rotation, dtype=np.float64),
                T=-np.asarray(rotation, dtype=np.float64) @ np.asarray(center),
                D=np.asarray(source.D, dtype=np.float64).copy(),
                H=int(source.H),
                W=int(source.W),
            )
            missing_rows.append(
                {
                    "frame": frame,
                    "view": view,
                    "left_frame": left,
                    "right_frame": right,
                    "alpha": alpha,
                    "policy": policy,
                }
            )

    expected = {(frame, view) for frame in ordered_frames for view in ordered_views}
    if not expected.issubset(completed):
        raise RuntimeError("completed camera grid is incomplete")
    return completed, {
        "protocol": "static_pnp_missing_se3_interpolation_v1",
        "successful_node_count": len(expected) - len(missing_rows),
        "missing_node_count": len(missing_rows),
        "missing_nodes": missing_rows,
        "successful_nodes_preserved_exactly": all(
            completed[node] is base_cameras[node]
            for node in expected & set(base_cameras)
        ),
        "new_numeric_hyperparameters": 0,
    }


def resolve_base_camera_grid(
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
    base_cameras: Mapping[tuple[int, str], CameraParams],
    missing_base_policy: str,
) -> tuple[dict[tuple[int, str], CameraParams], dict[str, Any]]:
    expected = {
        (int(frame), str(view)) for frame in frames for view in target_views
    }
    actual = set(base_cameras)
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra:
        raise ValueError(f"Static+PnP camera grid has extra nodes: {extra}")
    policy = str(missing_base_policy)
    if policy not in {
        "error",
        "interpolate",
        "interpolate_complete_views",
        "interpolate_verified_views",
    }:
        raise ValueError(f"unsupported missing-base policy: {policy}")
    if missing and policy == "error":
        raise ValueError(
            "joint repair requires a complete strong Static+PnP baseline; "
            f"missing={missing}, extra={extra}"
        )
    if policy in {
        "interpolate",
        "interpolate_complete_views",
        "interpolate_verified_views",
    }:
        completed, interpolation = complete_base_camera_grid(
            frames=frames,
            target_views=target_views,
            base_cameras=base_cameras,
        )
    else:
        completed = dict(base_cameras)
        interpolation = {
            "protocol": "disabled_complete_static_pnp",
            "successful_node_count": len(expected),
            "missing_node_count": 0,
            "missing_nodes": [],
            "successful_nodes_preserved_exactly": True,
            "new_numeric_hyperparameters": 0,
        }
    return completed, {
        "missing_base_policy": policy,
        "raw_static_pnp_success_count": len(actual),
        "raw_static_pnp_missing_count": len(missing),
        "raw_static_pnp_missing_nodes": [
            {"frame": frame, "view": view} for frame, view in missing
        ],
        "interpolation": interpolation,
    }


def partition_target_views_by_base_support(
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
    base_cameras: Mapping[tuple[int, str], CameraParams],
) -> tuple[list[str], list[str]]:
    """Separate views with an absolute PnP anchor from unavailable views."""

    frame_values = [int(frame) for frame in frames]
    supported: list[str] = []
    unavailable: list[str] = []
    for view in [str(value) for value in target_views]:
        destination = (
            supported
            if any((frame, view) in base_cameras for frame in frame_values)
            else unavailable
        )
        destination.append(view)
    return supported, unavailable


def prepare_target_camera_grid(
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
    base_cameras: Mapping[tuple[int, str], CameraParams],
    verified_temporal_cameras: Mapping[tuple[int, str], CameraParams],
    missing_base_policy: str,
) -> tuple[
    dict[tuple[int, str], CameraParams],
    list[str],
    list[str],
    dict[str, Any],
]:
    """Prepare Ours' grid, optionally bootstrapping zero-anchor views."""

    ordered_frames = [int(frame) for frame in frames]
    ordered_views = [str(view) for view in target_views]
    expected = {
        (frame, view) for frame in ordered_frames for view in ordered_views
    }
    base_nodes = set(base_cameras)
    temporal_nodes = set(verified_temporal_cameras)
    if base_nodes - expected:
        raise ValueError(
            f"Static+PnP camera grid has extra nodes: {sorted(base_nodes - expected)}"
        )
    if temporal_nodes - expected:
        raise ValueError(
            "verified temporal camera grid has extra nodes: "
            f"{sorted(temporal_nodes - expected)}"
        )

    policy = str(missing_base_policy)
    valid_policies = {
        "error",
        "interpolate",
        "interpolate_complete_views",
        "interpolate_verified_views",
        "temporal_bootstrap",
    }
    if policy not in valid_policies:
        raise ValueError(f"unsupported missing-base policy: {policy}")
    bootstrap_enabled = policy == "temporal_bootstrap"

    selected_anchors = dict(base_cameras)
    supported: list[str] = []
    unavailable: list[str] = []
    bootstrapped_views: list[str] = []
    bootstrapped_nodes: list[tuple[int, str]] = []
    view_rows: list[dict[str, Any]] = []
    for view in ordered_views:
        static_nodes = sorted(
            node for node in base_nodes if node[1] == view
        )
        available_temporal_nodes = sorted(
            node for node in temporal_nodes if node[1] == view
        )
        if static_nodes:
            selected_source = "static_pnp"
            selected_count = len(static_nodes)
            supportable = True
        elif bootstrap_enabled and available_temporal_nodes:
            selected_source = "verified_temporal_track_pnp"
            selected_count = len(available_temporal_nodes)
            supportable = True
            bootstrapped_views.append(view)
            bootstrapped_nodes.extend(available_temporal_nodes)
            for node in available_temporal_nodes:
                selected_anchors[node] = verified_temporal_cameras[node]
        else:
            selected_source = "unavailable"
            selected_count = 0
            supportable = False
        (supported if supportable else unavailable).append(view)
        view_rows.append(
            {
                "view": view,
                "static_pnp_anchor_count": len(static_nodes),
                "verified_temporal_anchor_count": len(available_temporal_nodes),
                "selected_anchor_source": selected_source,
                "selected_anchor_count": selected_count,
                "supportable": supportable,
            }
        )

    selected_anchors = {
        node: camera
        for node, camera in selected_anchors.items()
        if node[1] in supported
    }
    completion_policy = "interpolate" if bootstrap_enabled else policy
    if supported:
        completed, completion = resolve_base_camera_grid(
            frames=ordered_frames,
            target_views=supported,
            base_cameras=selected_anchors,
            missing_base_policy=completion_policy,
        )
    else:
        completed = {}
        completion = {
            "missing_base_policy": completion_policy,
            "raw_static_pnp_success_count": 0,
            "raw_static_pnp_missing_count": 0,
            "raw_static_pnp_missing_nodes": [],
            "interpolation": {
                "protocol": "not_run_no_supported_target_views",
                "successful_node_count": 0,
                "missing_node_count": 0,
                "missing_nodes": [],
                "successful_nodes_preserved_exactly": True,
                "new_numeric_hyperparameters": 0,
            },
        }

    supported_expected = {
        (frame, view) for frame in ordered_frames for view in supported
    }
    raw_static_nodes = base_nodes & supported_expected
    raw_static_missing = sorted(supported_expected - raw_static_nodes)
    completion.update(
        {
            "missing_base_policy": policy,
            "completion_policy": completion_policy,
            "raw_static_pnp_success_count": len(raw_static_nodes),
            "raw_static_pnp_missing_count": len(raw_static_missing),
            "raw_static_pnp_missing_nodes": [
                {"frame": frame, "view": view}
                for frame, view in raw_static_missing
            ],
            "initial_anchor_success_count": len(selected_anchors),
            "initial_anchor_missing_count": len(
                supported_expected - set(selected_anchors)
            ),
            "temporal_bootstrap": {
                "protocol": TEMPORAL_BOOTSTRAP_PROTOCOL,
                "enabled": bootstrap_enabled,
                "used": bool(bootstrapped_views),
                "bootstrapped_views": bootstrapped_views,
                "bootstrapped_nodes": [
                    {"frame": frame, "view": view}
                    for frame, view in bootstrapped_nodes
                ],
                "view_rows": view_rows,
                "static_pnp_anchors_preserved_exactly": all(
                    selected_anchors.get(node) is camera
                    for node, camera in base_cameras.items()
                ),
                "GT_used": False,
                "NVS_used": False,
            },
        }
    )
    return completed, supported, unavailable, completion


def should_use_initial_camera_bootstrap(
    *,
    view: str,
    base_completion: Mapping[str, Any],
) -> bool:
    """Enable optimizer fallback only for a temporally bootstrapped view."""

    audit = base_completion.get("temporal_bootstrap", {})
    if not isinstance(audit, Mapping) or not bool(audit.get("enabled", False)):
        return False
    return str(view) in {
        str(value) for value in audit.get("bootstrapped_views", [])
    }


def should_use_joint_refined_rotation_fallback(
    *,
    view: str,
    base_completion: Mapping[str, Any],
) -> bool:
    """Retain joint-refined rotations only for temporal-bootstrap views."""

    return should_use_initial_camera_bootstrap(
        view=view,
        base_completion=base_completion,
    )


def select_committed_nodes(
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
    decisions: Mapping[str, Mapping[str, Any]],
    raw_base_nodes: set[tuple[int, str]],
    missing_base_policy: str,
) -> set[tuple[int, str]]:
    """Choose write-back nodes under the explicitly selected commit scope."""

    expected = {
        (int(frame), str(view)) for frame in frames for view in target_views
    }
    policy = str(missing_base_policy)
    if policy in {"error", "interpolate_verified_views"}:
        eligible = expected
    elif policy in {"interpolate", "temporal_bootstrap"}:
        eligible = expected - set(raw_base_nodes)
    elif policy == "interpolate_complete_views":
        complete_views = {
            view
            for view in [str(value) for value in target_views]
            if all((int(frame), view) in raw_base_nodes for frame in frames)
        }
        eligible = {
            node
            for node in expected
            if node[1] in complete_views or node not in raw_base_nodes
        }
    else:
        raise ValueError(f"unsupported missing-base policy: {policy}")
    return {
        node
        for node in eligible
        if bool(decisions.get(node[1], {}).get("commit", False))
    }


def select_final_verified_camera(
    *,
    node: tuple[int, str],
    base_camera: CameraParams,
    candidate_camera: CameraParams | None,
    committed_nodes: set[tuple[int, str]],
    temporal_bootstrap_views: set[str],
) -> tuple[CameraParams | None, str]:
    """Materialize only verification-accepted temporal-bootstrap cameras."""

    key = (int(node[0]), str(node[1]))
    if key in committed_nodes:
        if candidate_camera is None:
            raise ValueError(f"committed node has no candidate camera: {key}")
        return candidate_camera, "verified_candidate"
    if key[1] in {str(view) for view in temporal_bootstrap_views}:
        return None, "verification_rejected_temporal_bootstrap_omitted"
    return base_camera, "base_passthrough"


def initial_camera_grid_matches_raw_static_pnp(
    *,
    base_cameras: Mapping[tuple[int, str], CameraParams],
    raw_base_cameras: Mapping[tuple[int, str], CameraParams],
) -> bool:
    """Check that repair starts from the exact complete Static+PnP grid."""

    return set(base_cameras) == set(raw_base_cameras) and all(
        _camera_equal(base_cameras[node], raw_base_cameras[node])
        for node in raw_base_cameras
    )


def build_pose_root_manifest_fields(
    *,
    pure_static_pnp_input_pose_root: Path,
    ours_initialization_pose_root: Path,
    verified_temporal_pose_root: Path | None,
    temporal_bootstrap: Mapping[str, Any],
    missing_base_policy: str,
) -> dict[str, Any]:
    """Label raw input and Ours initialization roots without ambiguity."""

    pure_root = str(pure_static_pnp_input_pose_root)
    initialization_root = str(ours_initialization_pose_root)
    temporal_root = (
        str(verified_temporal_pose_root)
        if verified_temporal_pose_root is not None
        else None
    )
    bootstrap_used = bool(temporal_bootstrap.get("used", False))
    return {
        "pure_static_pnp_input_pose_root": pure_root,
        "ours_initialization_pose_root": initialization_root,
        # Retained compatibility aliases; their canonical roles are explicit.
        "base_pose_root": pure_root,
        "pure_static_pnp_baseline_pose_root": pure_root,
        "baseline_pose_root": initialization_root,
        "legacy_pose_root_aliases": {
            "base_pose_root": "pure_static_pnp_input_pose_root",
            "baseline_pose_root": "ours_initialization_pose_root",
        },
        "baseline_pose_semantics": (
            "ours_initialization_with_verified_temporal_bootstrap_not_"
            "pure_static_pnp_baseline"
            if bootstrap_used
            else "completed_static_pnp_initialization"
        ),
        "camera_initialization_provenance": {
            **temporal_bootstrap,
            "missing_base_policy": str(missing_base_policy),
            "pure_static_pnp_input_pose_root": pure_root,
            "ours_initialization_pose_root": initialization_root,
            "verified_temporal_pose_root": temporal_root,
            # Compatibility alias for manifests written before explicit roles.
            "pure_static_pnp_pose_root": pure_root,
        },
    }


def build_final_target_availability_audit(
    *,
    target_views: Sequence[str],
    initially_unavailable_target_views: Sequence[str],
    verification_rejected_bootstrap_views: Sequence[str],
) -> dict[str, Any]:
    """Summarize availability after temporal-bootstrap verification."""

    ordered_views = [str(view) for view in target_views]
    unavailable_set = {str(view) for view in initially_unavailable_target_views}
    rejected_set = {str(view) for view in verification_rejected_bootstrap_views}
    unknown = (unavailable_set | rejected_set) - set(ordered_views)
    if unknown:
        raise ValueError(
            "final availability audit contains unknown target views: "
            f"{sorted(unknown)}"
        )
    final_unavailable = [
        view
        for view in ordered_views
        if view in unavailable_set or view in rejected_set
    ]
    final_available = [
        view for view in ordered_views if view not in final_unavailable
    ]
    return {
        "initially_unavailable_target_views": [
            view for view in ordered_views if view in unavailable_set
        ],
        "verification_rejected_bootstrap_views": [
            view for view in ordered_views if view in rejected_set
        ],
        "final_available_target_views": final_available,
        "final_unavailable_target_views": final_unavailable,
        "final_pose_complete": not bool(final_unavailable),
        "zero_anchor_views_omitted_from_both_methods": bool(final_unavailable),
        "zero_anchor_views_omitted_from_both_methods_views": final_unavailable,
        "zero_anchor_views_omitted_from_both_methods_semantics": (
            "omitted_from_pure_static_pnp_and_final_verified_ours"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = Path(args.runtime_scene).resolve()
    stage = Path(args.stage_a_root).resolve()
    seal = Path(args.stage_a_seal).resolve()
    output = Path(args.output_root).resolve()
    frames = [int(frame) for frame in args.frames]
    target_views = [str(view) for view in args.target_views]
    observation_source = str(args.observation_source)
    method = effective_method(
        str(args.missing_base_policy),
        observation_source=observation_source,
    )
    use_temporal_static_landmarks = bool(args.temporal_static_landmarks)
    use_parent_observations = bool(
        args.use_attested_parent_temporal_observations
    )
    use_static_xyz_identity_gate = bool(args.static_xyz_identity_gate)
    if args.complete_temporal_observation_track_root and not use_parent_observations:
        raise ValueError(
            "complete temporal observation root requires attested-parent mode"
        )
    if use_temporal_static_landmarks:
        if observation_source != "temporal_tracks":
            raise ValueError(
                "temporal-static landmarks require temporal_tracks observations"
            )
        method = f"{method}_{STATIC_LANDMARK_METHOD_SUFFIX}"
    if use_parent_observations:
        if not use_temporal_static_landmarks or observation_source != "temporal_tracks":
            raise ValueError(
                "attested parent observations require temporal-static landmarks"
            )
        method = f"{method}_{FULL_PARENT_OBSERVATION_METHOD_SUFFIX}"
    if use_static_xyz_identity_gate and not use_temporal_static_landmarks:
        raise ValueError("static-XYZ identity gating requires temporal-static landmarks")
    if frames != sorted(set(frames)):
        raise ValueError("frames must be sorted and unique")
    if len(target_views) != len(set(target_views)):
        raise ValueError("target views must be unique")
    if output.is_relative_to(runtime) or output.is_relative_to(stage):
        raise ValueError("output must be outside sealed input trees")

    selector_subdir, commit_policy = _sealed_selector_protocol(seal)
    if commit_policy != COMMIT_POLICY:
        raise ValueError("unexpected Stage-A commit policy")
    seal_record = _verify_stage_a_seal(
        runtime_scene=runtime,
        stage_a_root=stage,
        stage_a_seal=seal,
        selector_subdir=selector_subdir,
        expected_commit_policy=commit_policy,
        frames=frames,
        target_views=target_views,
    )
    runtime_manifest, runtime_checksums = _verify_runtime_tree(runtime)
    static_views = [str(view) for view in runtime_manifest["observed_short_views"]]
    if set(target_views) - {str(view) for view in runtime_manifest["long_views"]}:
        raise ValueError("target view is absent from runtime long views")

    base_dir = stage / "base_same_frame_pnp"
    temporal_dir = stage / "temporal_pose_fit150"
    pure_static_pnp_input_pose_root = (
        base_dir / "pose_track_same_time_pnp"
    )
    temporal_pose_root = temporal_dir / "pose_track_same_time_pnp"
    base_prediction = _load_attested_prediction(
        attestation_path=base_dir / "prediction_attestation.json",
        expected_prediction_path=base_dir / "predictions_pre_gt.json",
        expected_pose_root=pure_static_pnp_input_pose_root,
        runtime=runtime,
        runtime_digest=str(runtime_checksums["tree_digest"]),
        frames=frames,
        target_views=target_views,
        expected_track_mode="independent_per_frame_bundle",
    )
    temporal_prediction: Mapping[str, Any] | None = None
    temporal_bundle: Mapping[str, Any] | None = None
    temporal_landmark_bundle: Mapping[str, Any] | None = None
    parent_observation_audit: dict[str, Any] | None = None
    independent_bundles: dict[int, dict[str, Any]] = {}
    if observation_source == "temporal_tracks":
        temporal_prediction = _load_attested_prediction(
            attestation_path=temporal_dir / "prediction_attestation.json",
            expected_prediction_path=temporal_dir / "predictions_pre_gt.json",
            expected_pose_root=temporal_pose_root,
            runtime=runtime,
            runtime_digest=str(runtime_checksums["tree_digest"]),
            frames=frames,
            target_views=target_views,
            expected_track_mode="shared_temporal_bundle",
        )
        track_root = stage / "temporal_targetstar_stride5_lkfb"
        temporal_bundle = _load_track_bundle(
            track_root=track_root,
            prediction=temporal_prediction,
            runtime_views=runtime_manifest["runtime_views"],
            static_views=static_views,
            target_views=target_views,
            frames=frames,
        )
        temporal_landmark_bundle = temporal_bundle
        if use_parent_observations:
            temporal_bundle, parent_observation_audit = (
                load_attested_parent_temporal_bundle(
                    landmark_bundle=temporal_landmark_bundle,
                    runtime_views=runtime_manifest["runtime_views"],
                    static_views=static_views,
                    target_views=target_views,
                    frames=frames,
                    observation_track_root=(
                        Path(args.complete_temporal_observation_track_root)
                        if args.complete_temporal_observation_track_root
                        else None
                    ),
                )
            )
            observation_roots = [
                Path(temporal_bundle["root"]).resolve(),
                Path(temporal_landmark_bundle["root"]).resolve(),
            ]
        else:
            observation_roots = [Path(track_root).resolve()]
    else:
        independent_bundles = _load_independent_track_bundles(
            prediction=base_prediction,
            runtime_views=runtime_manifest["runtime_views"],
            static_views=static_views,
            target_views=target_views,
            frames=frames,
        )
        observation_roots = [
            Path(independent_bundles[frame]["root"]).resolve()
            for frame in frames
        ]

    base_success = _collect_successful_nodes(base_prediction["rows"])
    raw_base_cameras = _load_successful_cameras(
        pose_root=pure_static_pnp_input_pose_root,
        successful_nodes=base_success,
        frames=frames,
        target_views=target_views,
    )
    verified_temporal_cameras = (
        _load_successful_cameras(
            pose_root=temporal_pose_root,
            successful_nodes=_collect_successful_nodes(
                temporal_prediction["rows"]
            ),
            frames=frames,
            target_views=target_views,
        )
        if (
            temporal_prediction is not None
            and str(args.missing_base_policy) == "temporal_bootstrap"
        )
        else {}
    )
    (
        base_cameras,
        repair_target_views,
        unavailable_target_views,
        base_completion,
    ) = prepare_target_camera_grid(
        frames=frames,
        target_views=target_views,
        base_cameras=raw_base_cameras,
        verified_temporal_cameras=verified_temporal_cameras,
        missing_base_policy=str(args.missing_base_policy),
    )
    if not repair_target_views:
        raise ValueError(
            "no target view has a successful Static+PnP or opted-in verified "
            "temporal anchor"
        )
    temporal_bootstrap = base_completion["temporal_bootstrap"]
    support_cameras = CameraLoader(runtime / "cameras/static").load_cameras(
        static_views
    )
    config = CrossViewTrackVerifierConfig(
        min_context_views=2,
        context_inlier_px=float(args.support_reprojection_px),
        focus_inlier_px=float(args.focus_reprojection_px),
        max_pair_hypotheses=24,
    )
    temporal_static_filter: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled" if not use_temporal_static_landmarks else "not_run",
    }
    static_point_ids: np.ndarray | None = None
    static_xyz_by_point_id: dict[int, np.ndarray] | None = None
    if use_temporal_static_landmarks:
        if temporal_landmark_bundle is None:
            raise RuntimeError("temporal track bundle is unavailable")
        point_ids_by_seed = {
            int(row["seed_frame"]): np.arange(
                int(row["point_range"][0]),
                int(row["point_range"][1]),
                dtype=np.int64,
            )
            for row in seed_point_ranges(
                meta=temporal_landmark_bundle["meta"],
                total_points=int(temporal_landmark_bundle["tracks"].shape[2]),
            )
        }
        temporal_static_filter = select_temporally_static_track_ids(
            tracks=np.asarray(temporal_landmark_bundle["tracks"]),
            visibility=np.asarray(temporal_landmark_bundle["visibility"]),
            view_ids=temporal_landmark_bundle["views"],
            cameras_by_frame={int(frame): support_cameras for frame in frames},
            context_views=static_views,
            point_ids_by_seed=point_ids_by_seed,
            track_start_frame=int(
                temporal_landmark_bundle["meta"]["start_frame"]
            ),
            sample_frames=frames,
            config=config,
            reprojection_gate_px=float(args.support_reprojection_px),
        )
        temporal_static_filter["enabled"] = True
        static_point_ids, static_xyz_by_point_id = temporal_static_landmark_geometry(
            temporal_static_filter
        )
        if len(static_point_ids) < int(args.min_inliers):
            raise ValueError(
                "temporal-static filter retained fewer landmarks than min-inliers"
            )
    identity_by_frame: dict[int, dict[str, Any]] = {}
    identity_audit_rows: list[dict[str, Any]] = []
    if use_static_xyz_identity_gate:
        if temporal_bundle is None or static_point_ids is None:
            raise RuntimeError("static-XYZ identity gate has no temporal landmarks")
        static_xyz = np.asarray(
            [static_xyz_by_point_id[int(point_id)] for point_id in static_point_ids],
            dtype=np.float64,
        )
        for frame in frames:
            local = int(frame) - int(temporal_bundle["meta"]["start_frame"])
            identity = evaluate_fixed_static_landmark_identity(
                tracks_frame=np.asarray(temporal_bundle["tracks"][:, local]),
                visibility_frame=np.asarray(
                    temporal_bundle["visibility"][:, local]
                ),
                view_ids=temporal_bundle["views"],
                cameras_by_view=support_cameras,
                context_views=static_views,
                point_ids=static_point_ids,
                landmark_xyz=static_xyz,
                reprojection_gate_px=float(args.support_reprojection_px),
                min_context_views=int(config.min_context_views),
            )
            identity_by_frame[int(frame)] = identity
            identity_audit_rows.append(
                {
                    key: value
                    for key, value in identity.items()
                    if key not in {"accepted_point_ids", "point_rows"}
                }
                | {"frame": int(frame)}
            )
    point_bank_rows: list[dict[str, Any]] = []
    observations_by_view: dict[str, list[dict[str, Any]]] = {
        view: [] for view in repair_target_views
    }
    for frame_ordinal, frame in enumerate(frames):
        bundle = observation_bundle_for_frame(
            observation_source=observation_source,
            frame=frame,
            temporal_bundle=temporal_bundle,
            independent_bundles=independent_bundles,
        )
        point_ranges = seed_point_ranges(
            meta=bundle["meta"],
            total_points=int(bundle["tracks"].shape[2]),
        )
        for view in repair_target_views:
            allowed_ids = static_point_ids
            if use_static_xyz_identity_gate:
                allowed_ids = np.asarray(
                    identity_by_frame[int(frame)]["accepted_point_ids"],
                    dtype=np.int64,
                )
            bank, diagnostics = _build_node_point_bank(
                frame=frame,
                view=view,
                bundle=bundle,
                static_views=static_views,
                support_cameras=support_cameras,
                point_ranges=point_ranges,
                fit_budget=int(args.fit_budget),
                config=config,
                allowed_point_ids=allowed_ids,
                fixed_xyz_by_point_id=static_xyz_by_point_id,
            )
            observation = extract_joint_observation(
                frame=frame,
                view=view,
                point_bank=bank,
                bundle=bundle,
            )
            if observation_source == "independent_same_frame":
                observation = namespace_independent_observation_ids(
                    observation,
                    frame_ordinal=frame_ordinal,
                )
            observations_by_view[view].append(observation)
            focus_index = bundle["views"].index(str(view))
            local = int(frame) - int(bundle["meta"]["start_frame"])
            target_visibility = np.asarray(
                bundle["visibility"][focus_index, local], dtype=bool
            )
            target_visible_landmark_count = (
                int(np.count_nonzero(target_visibility[static_point_ids]))
                if static_point_ids is not None
                else int(np.count_nonzero(target_visibility))
            )
            point_bank_rows.append(
                {
                    "frame": frame,
                    "view": view,
                    "target_visible_landmark_count": target_visible_landmark_count,
                    "static_xyz_identity_accepted_count": int(len(allowed_ids)),
                    **diagnostics,
                    "joint_observation_count": int(len(observation["point_ids"])),
                }
            )

    minimum_frames = max(3, len(frames) // 2 + 1)
    coverage_audit = summarize_temporal_observation_coverage(
        observations_by_view=observations_by_view,
        frames=frames,
        target_views=repair_target_views,
        num_folds=int(args.num_folds),
        min_inliers=int(args.min_inliers),
        min_successful_frames=minimum_frames,
    )
    coverage_audit.update(
        {
            "protocol": method,
            "observation_roots": [str(root) for root in observation_roots],
            "parent_observation_attestation": parent_observation_audit,
            "static_xyz_identity_gate": {
                "enabled": use_static_xyz_identity_gate,
                "frames": identity_audit_rows,
            },
            "point_bank_rows": point_bank_rows,
        }
    )
    if bool(args.coverage_audit_only):
        if output.exists():
            if not bool(args.overwrite):
                raise FileExistsError(output)
            shutil.rmtree(output)
        output.mkdir(parents=True)
        _write_json(output / "COVERAGE_AUDIT.json", coverage_audit)
        return {
            "coverage_audit_only": True,
            "coverage_audit": coverage_audit,
            "output_root": str(output),
            "pose_root": None,
            "committed_views": [],
        }
    if use_static_xyz_identity_gate and not bool(coverage_audit["all_views_pass"]):
        raise ValueError(
            "static-XYZ-gated temporal observations failed the frozen coverage gate"
        )

    results: dict[str, dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    candidate_cameras: dict[tuple[int, str], CameraParams] = {}
    for view in repair_target_views:
        initial = {frame: base_cameras[(frame, view)] for frame in frames}
        result = crossfit_joint_ptz_shared_center(
            observations=observations_by_view[view],
            initial_cameras=initial,
            reprojection_error_px=float(args.focus_reprojection_px),
            min_inliers_per_frame=int(args.min_inliers),
            min_successful_frames=minimum_frames,
            num_folds=int(args.num_folds),
            rng_seed=0,
            allow_initial_camera_bootstrap=(
                should_use_initial_camera_bootstrap(
                    view=view,
                    base_completion=base_completion,
                )
            ),
            allow_joint_refined_rotation_fallback=(
                should_use_joint_refined_rotation_fallback(
                    view=view,
                    base_completion=base_completion,
                )
            ),
        )
        decision = decide_joint_ptz_band_model_commit(
            result,
            focus_inlier_px=float(args.focus_reprojection_px),
            frame_count=len(frames),
        )
        results[view] = result
        decisions[view] = decision
        if bool(result.get("success", False)):
            for frame in frames:
                candidate_cameras[(frame, view)] = result["cameras"][frame]

    if output.exists():
        if not bool(args.overwrite):
            raise FileExistsError(output)
        shutil.rmtree(output)
    temporal_bootstrap_used = bool(temporal_bootstrap["used"])
    ours_initialization_pose_root = output / (
        "pose_ours_temporal_bootstrap_initialization"
        if temporal_bootstrap_used
        else "pose_static_pnp_interpolated"
    )
    candidate_root = output / "pose_joint_track_bundle_candidates"
    pose_root = output / "pose_joint_track_bundle_verified"
    committed_nodes = select_committed_nodes(
        frames=frames,
        target_views=repair_target_views,
        decisions=decisions,
        raw_base_nodes=set(raw_base_cameras),
        missing_base_policy=str(args.missing_base_policy),
    )
    committed_views = {view for _, view in committed_nodes}
    temporal_bootstrap_views = {
        str(view) for view in temporal_bootstrap["bootstrapped_views"]
    }
    committed_bootstrap_views = [
        view
        for view in target_views
        if view in temporal_bootstrap_views
        and all((frame, view) in committed_nodes for frame in frames)
    ]
    committed_bootstrap_view_set = set(committed_bootstrap_views)
    final_export_committed_nodes = {
        node
        for node in committed_nodes
        if node[1] not in temporal_bootstrap_views
        or node[1] in committed_bootstrap_view_set
    }
    verification_rejected_bootstrap_views = [
        view
        for view in target_views
        if view in temporal_bootstrap_views
        and view not in committed_bootstrap_view_set
    ]
    final_availability = build_final_target_availability_audit(
        target_views=target_views,
        initially_unavailable_target_views=unavailable_target_views,
        verification_rejected_bootstrap_views=(
            verification_rejected_bootstrap_views
        ),
    )
    final_unavailable_target_views = final_availability[
        "final_unavailable_target_views"
    ]
    final_available_target_views = final_availability[
        "final_available_target_views"
    ]
    passthrough_checks: list[bool] = []
    final_pose_selection_rows: list[dict[str, Any]] = []
    for frame in frames:
        baseline_frame = dict(support_cameras)
        candidate_frame = dict(support_cameras)
        selected_frame = dict(support_cameras)
        for view in repair_target_views:
            node = (frame, view)
            baseline_frame[view] = base_cameras[node]
            candidate_frame[view] = candidate_cameras.get(node, base_cameras[node])
            selected_camera, selection_source = select_final_verified_camera(
                node=node,
                base_camera=base_cameras[node],
                candidate_camera=candidate_cameras.get(node),
                committed_nodes=final_export_committed_nodes,
                temporal_bootstrap_views=temporal_bootstrap_views,
            )
            if selected_camera is None:
                selected_frame.pop(view, None)
            else:
                selected_frame[view] = selected_camera
            final_pose_selection_rows.append(
                {
                    "frame": frame,
                    "view": view,
                    "source": selection_source,
                }
            )
            if selection_source == "base_passthrough":
                passthrough_checks.append(
                    _camera_equal(selected_frame[view], base_cameras[node])
                )
        save_cameras(
            ours_initialization_pose_root
            / f"f{frame:06d}"
            / "cameras_pred",
            baseline_frame,
        )
        save_cameras(
            candidate_root / f"f{frame:06d}" / "cameras_pred", candidate_frame
        )
        save_cameras(pose_root / f"f{frame:06d}" / "cameras_pred", selected_frame)
    if not all(passthrough_checks):
        raise RuntimeError("non-committed Static+PnP camera changed")

    initial_grid_equal = initial_camera_grid_matches_raw_static_pnp(
        base_cameras=base_cameras,
        raw_base_cameras=raw_base_cameras,
    )
    complete_views = {
        view
        for view in repair_target_views
        if all((frame, view) in raw_base_cameras for frame in frames)
    }
    partial_views = set(repair_target_views) - complete_views
    partial_successes_preserved = all(
        node not in committed_nodes
        and _camera_equal(base_cameras[node], raw_base_cameras[node])
        for node in raw_base_cameras
        if node[1] in partial_views
    )
    pose_root_manifest_fields = build_pose_root_manifest_fields(
        pure_static_pnp_input_pose_root=pure_static_pnp_input_pose_root,
        ours_initialization_pose_root=ours_initialization_pose_root,
        verified_temporal_pose_root=(
            temporal_pose_root
            if bool(temporal_bootstrap["enabled"])
            else None
        ),
        temporal_bootstrap=temporal_bootstrap,
        missing_base_policy=str(args.missing_base_policy),
    )
    initialization_output_camera_hashes = {
        f"f{frame:06d}": camera_root_attestation(
            ours_initialization_pose_root
            / f"f{frame:06d}"
            / "cameras_pred"
        )
        for frame in frames
    }
    manifest = {
        "protocol": method,
        "method": method,
        "complete_base_source_method": (
            TEMPORAL_BOOTSTRAP_METHOD if temporal_bootstrap_used else METHOD
        ),
        "runtime_scene": str(runtime),
        "runtime_tree_digest": runtime_checksums["tree_digest"],
        "stage_a_root": str(stage),
        "stage_a_seal": seal_record,
        "frames": frames,
        "static_views": static_views,
        "target_views": target_views,
        "repair_target_views": repair_target_views,
        "initially_unavailable_target_views": final_availability[
            "initially_unavailable_target_views"
        ],
        # Compatibility name: this is pre-verification availability only.
        "unavailable_target_views": unavailable_target_views,
        "verification_committed_bootstrap_views": committed_bootstrap_views,
        "verification_rejected_bootstrap_views": final_availability[
            "verification_rejected_bootstrap_views"
        ],
        "final_available_target_views": final_available_target_views,
        "final_unavailable_target_views": final_unavailable_target_views,
        "final_pose_complete": final_availability["final_pose_complete"],
        "final_pose_completeness": {
            "policy": (
                "temporal_bootstrap_views_require_full_heldout_verified_commit"
            ),
            "available_target_views": final_available_target_views,
            "unavailable_target_views": final_unavailable_target_views,
            "all_requested_target_views_available": final_availability[
                "final_pose_complete"
            ],
            "initially_unavailable_target_views": final_availability[
                "initially_unavailable_target_views"
            ],
            "verification_rejected_bootstrap_views": final_availability[
                "verification_rejected_bootstrap_views"
            ],
            "initialization_and_candidate_roots_may_retain_rejected_views": True,
        },
        **pose_root_manifest_fields,
        "base_completion": base_completion,
        "ours_initialization_output_camera_hashes": (
            initialization_output_camera_hashes
        ),
        "baseline_output_camera_hashes": initialization_output_camera_hashes,
        "legacy_output_hash_aliases": {
            "baseline_output_camera_hashes": (
                "ours_initialization_output_camera_hashes"
            )
        },
        "observation_source": observation_source,
        "observation_roots": [str(root) for root in observation_roots],
        "observation_files_sha256": {
            str(root): track_file_attestation(root) for root in observation_roots
        },
        "track_root": (
            str(observation_roots[0])
            if observation_source == "temporal_tracks"
            else None
        ),
        "track_files_sha256": (
            track_file_attestation(observation_roots[0])
            if observation_source == "temporal_tracks"
            else {
                str(root): track_file_attestation(root)
                for root in observation_roots
            }
        ),
        "parent_observation_attestation": parent_observation_audit,
        "coverage_audit": coverage_audit,
        "config": {
            "fit_point_budget": int(args.fit_budget),
            "support_reprojection_px": float(args.support_reprojection_px),
            "focus_reprojection_px": float(args.focus_reprojection_px),
            "min_inliers_per_frame": int(args.min_inliers),
            "min_successful_frames": int(minimum_frames),
            "num_crossfit_folds": int(args.num_folds),
            "temporal_static_landmarks": bool(use_temporal_static_landmarks),
            "use_attested_parent_temporal_observations": bool(
                use_parent_observations
            ),
            "static_xyz_identity_gate": bool(use_static_xyz_identity_gate),
            "new_scene_specific_numeric_hyperparameters": 0,
        },
        "temporal_static_filter": temporal_static_filter,
        "candidate_generation": {
            "model": "one shared center plus one rotation per frame",
            "initial_camera_bootstrap": (
                TEMPORAL_BOOTSTRAP_PROTOCOL
                if temporal_bootstrap_used
                else "static_pnp_only"
            ),
            "fold_optimization_initialization": {
                "policy": (
                    "pnp_then_supplied_initial_camera_fallback_for_"
                    "temporally_bootstrapped_views"
                ),
                "enabled_views": list(
                    temporal_bootstrap["bootstrapped_views"]
                ),
                "heldout_track_verification_unchanged": True,
                "new_numeric_hyperparameters": 0,
            },
            "fold_final_rotation_selection": {
                "policy": (
                    "independently_supported_rotation_then_joint_refined_"
                    "rotation_fallback"
                ),
                "enabled_views": list(
                    temporal_bootstrap["bootstrapped_views"]
                ),
                "independently_supported_rotations_take_precedence": True,
                "minimum_support_uses_existing_successful_frame_gate": True,
                "heldout_track_verification_unchanged": True,
                "new_numeric_hyperparameters": 0,
            },
            "bootstrap": (
                f"{observation_source} PnP per frame then geometric median center"
            ),
            "refinement": "equal-frame robust joint track reprojection",
            "sparse_frame_fallback": (
                "retain Static+PnP rotation but apply the verified shared center"
            ),
            "fit_verify_split": (
                "stable_track_id_splitmix64_modulo"
                if observation_source == "temporal_tracks"
                else "frame_namespaced_match_id_splitmix64_modulo"
            ),
            "track_geometry": TRACK_GEOMETRY_POLICY,
            "landmark_geometry": (
                "one public-static temporally pooled XYZ per persistent track"
                if use_temporal_static_landmarks
                else LANDMARK_GEOMETRY_POLICY
            ),
            "dynamic_track_policy": (
                "public-static multiframe reprojection filtering; persistent "
                "track identity shares one temporally pooled XYZ"
                if use_temporal_static_landmarks
                else "temporal identity retained for cross-fit partitioning; "
                "XYZ is independently triangulated per frame"
                if observation_source == "temporal_tracks"
                else "same-frame matcher observations only; no temporal identity"
            ),
        },
        "commit_policy": BAND_COMMIT_POLICY,
        "point_bank_rows": point_bank_rows,
        "synchronous_geometry": {
            "point_bank_scope": (
                "cross_frame_public_static_landmarks"
                if use_temporal_static_landmarks
                else "independent_per_frame"
            ),
            "track_id_scope": (
                "shared only for cross-fit partitioning"
                if observation_source == "temporal_tracks"
                else "frame-namespaced independent matcher IDs"
            ),
            "cross_time_xyz_shared": bool(use_temporal_static_landmarks),
            "numeric_hyperparameters_added": 0,
        },
        "view_results": results,
        "view_decisions": decisions,
        "committed_views": sorted(committed_views),
        "committed_nodes": [
            {"frame": frame, "view": view}
            for frame, view in sorted(committed_nodes)
        ],
        "final_exported_committed_nodes": [
            {"frame": frame, "view": view}
            for frame, view in sorted(final_export_committed_nodes)
        ],
        "candidate_pose_root": str(candidate_root),
        "candidate_output_camera_hashes": {
            f"f{frame:06d}": camera_root_attestation(
                candidate_root / f"f{frame:06d}" / "cameras_pred"
            )
            for frame in frames
        },
        "pose_root": str(pose_root),
        "final_pose_selection_rows": final_pose_selection_rows,
        "output_camera_hashes": {
            f"f{frame:06d}": camera_root_attestation(
                pose_root / f"f{frame:06d}" / "cameras_pred"
            )
            for frame in frames
        },
        "fairness": {
            "GT_used": False,
            "NVS_used": False,
            "heldout_test_views_used": False,
            "complete_strong_same_time_static_pnp_required": bool(
                base_completion["raw_static_pnp_missing_count"] == 0
            ),
            "raw_static_pnp_success_and_failure_retained": True,
            "zero_anchor_views_omitted_from_both_methods": final_availability[
                "zero_anchor_views_omitted_from_both_methods"
            ],
            "zero_anchor_views_omitted_from_both_methods_views": (
                final_availability[
                    "zero_anchor_views_omitted_from_both_methods_views"
                ]
            ),
            "zero_anchor_views_omitted_from_both_methods_semantics": (
                final_availability[
                    "zero_anchor_views_omitted_from_both_methods_semantics"
                ]
            ),
            "pure_static_pnp_baseline_modified": False,
            "pure_static_pnp_zero_anchor_policy": "unavailable",
            "ours_verified_temporal_bootstrap_used": temporal_bootstrap_used,
            "ours_verified_temporal_bootstrap_views": list(
                temporal_bootstrap["bootstrapped_views"]
            ),
            "initial_camera_optimizer_fallback_is_ours_only": bool(
                temporal_bootstrap_used
            ),
            "initial_camera_optimizer_fallback_changes_heldout_scoring": False,
            "joint_refined_rotation_fallback_is_ours_only": bool(
                temporal_bootstrap_used
            ),
            "joint_refined_rotation_fallback_changes_heldout_scoring": False,
            "rejected_temporal_bootstrap_omitted_from_final_pose": True,
            "final_verified_pose_complete": final_availability[
                "final_pose_complete"
            ],
            "ours_initialization_missing_pose_policy": str(
                args.missing_base_policy
            ),
            # Compatibility alias; this policy governs Ours' initialization.
            "baseline_missing_pose_policy": str(args.missing_base_policy),
            "raw_static_pnp_grid_complete_and_used_without_interpolation": (
                initial_grid_equal
            ),
            "successful_raw_static_pnp_nodes_preserved_in_completed_grid": all(
                _camera_equal(base_cameras[node], raw_base_cameras[node])
                for node in raw_base_cameras
            ),
            "baseline_and_ours_start_from_same_completed_camera_grid": bool(
                not temporal_bootstrap_used
            ),
            "commit_scope": {
                "error": "verified_complete_view",
                "interpolate": "raw_static_pnp_missing_nodes_only",
                "interpolate_complete_views": (
                    "verified_complete_views_and_raw_missing_nodes_only_"
                    "for_partial_views"
                ),
                "interpolate_verified_views": (
                    "all_nodes_of_each_heldout_verified_view"
                ),
                "temporal_bootstrap": (
                    "raw_static_pnp_missing_nodes_and_zero_anchor_views_"
                    "initialized_by_verified_temporal_track_pnp"
                ),
            }[str(args.missing_base_policy)],
            "successful_static_pnp_cameras_preserved_exactly": bool(
                base_completion["interpolation"][
                    "successful_nodes_preserved_exactly"
                ]
            ),
            "successful_static_pnp_cameras_preserved_in_ours_exactly": all(
                _camera_equal(base_cameras[node], raw_base_cameras[node])
                and node not in committed_nodes
                for node in raw_base_cameras
            ),
            "partial_view_successful_pnp_nodes_preserved_in_ours_exactly": (
                partial_successes_preserved
            ),
            "complete_view_commit_requires_heldout_track_improvement": True,
            "runtime_and_images_identical_to_static_pnp": True,
            "static_camera_map_identical_to_static_pnp": True,
            "temporal_tracks_are_only_additional_method_input": bool(
                observation_source == "temporal_tracks"
            ),
            "synchronous_triangulation_uses_only_public_static_cameras": True,
            "cross_time_static_landmark_assumption": bool(
                use_temporal_static_landmarks
            ),
            "temporal_static_filter_uses_only_public_static_tracks": bool(
                use_temporal_static_landmarks
            ),
            "noncommitted_base_cameras_preserved_exactly": bool(
                all(passthrough_checks)
            ),
        },
        "code": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "joint_solver": _module_file(joint_solver_module),
            "joint_solver_sha256": _sha256(_module_file(joint_solver_module)),
        },
    }
    manifest_path = output / "closure_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        output / "prediction_attestation.json",
        {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--stage-a-seal", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-views", nargs="+", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--fit-budget", type=int, default=150)
    parser.add_argument("--support-reprojection-px", type=float, default=2.5)
    parser.add_argument("--focus-reprojection-px", type=float, default=4.0)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument(
        "--observation-source",
        choices=["temporal_tracks", "independent_same_frame"],
        default="temporal_tracks",
    )
    parser.add_argument(
        "--temporal-static-landmarks",
        action="store_true",
        help=(
            "Use public-static multiframe track filtering and one fixed XYZ "
            "per persistent temporal track."
        ),
    )
    parser.add_argument(
        "--use-attested-parent-temporal-observations",
        action="store_true",
        help=(
            "Use the complete parent observations cryptographically attested "
            "by the refined landmark bundle."
        ),
    )
    parser.add_argument(
        "--complete-temporal-observation-track-root",
        type=Path,
        default=None,
        help=(
            "Optional full-window LK child of the exact attested parent; only "
            "valid with --use-attested-parent-temporal-observations."
        ),
    )
    parser.add_argument(
        "--static-xyz-identity-gate",
        action="store_true",
        help=(
            "Require each temporal landmark observation to retain support from "
            "the existing fixed-XYZ public-static reprojection gate."
        ),
    )
    parser.add_argument(
        "--coverage-audit-only",
        action="store_true",
        help="Write pre-optimization track coverage and stop before pose solving.",
    )
    parser.add_argument(
        "--missing-base-policy",
        choices=[
            "error",
            "interpolate",
            "interpolate_complete_views",
            "interpolate_verified_views",
            "temporal_bootstrap",
        ],
        default="error",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    payload = run(parse_args())
    print(
        json.dumps(
            {
                "pose_root": payload["pose_root"],
                "committed_views": payload["committed_views"],
            },
            indent=2,
            sort_keys=True,
        )
    )
