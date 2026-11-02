#!/usr/bin/env python3
"""Build and verify five-frame same-time multi-PTZ rotation closures."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import replace
import hashlib
from itertools import combinations
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader, CameraParams, save_cameras
from track4d_calib.direct_multi_ptz_rotation_ba import (
    DirectRotationBAConfig,
    solve_direct_rotation_component,
)
from track4d_calib.same_time_multi_ptz_closure import (
    candidate_fit_residuals,
    materialize_candidate_pose_path,
    materialize_fixed_center_rotation_path,
    pooled_noninferiority_guard,
    pooled_pareto_score,
    split_frame_local_track_ids,
    target_components_from_window_visibility,
)
from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    triangulate_same_time_multiview_points,
)
from track4d_calib.track_joint_ptz_bundle import (
    _camera_reprojection_errors,
    solve_fixed_center_ptz_rotations,
    solve_joint_ptz_shared_center,
    split_joint_ptz_track_groups,
)
from scripts.geometry.export_same_time_dynamic_track_closure import (
    build_time_local_observation,
    load_static_pnp_confidence,
    point_ids_in_frame_window,
    seed_frame_by_point,
    weighted_geometric_median,
)
from scripts.geometry.export_static_pnp_inlier_seeded_track_closure import (
    _load_bundle,
)
from scripts.geometry.run_cross_dynamic_band_ba_closure import (
    sampson_distances_px,
)
from scripts.geometry.run_track_closure import (
    build_node_observations,
)


PROTOCOL = "MULTIVIEW_TRACK-same-time-multi-PTZ-window-closure-v1"
DIRECT_PROTOCOL = "ROTATION_BA-direct-multi-PTZ-rotation-residual-BA-v1"
DIRECT_CENTER_PROTOCOL = "POSE_BA-direct-multi-PTZ-pose-residual-BA-v1"
DIRECT_CENTER_INIT_PROTOCOL = "POSE_INIT-direct-multi-PTZ-pose-BA-joint-init-v1"
DIRECT_GUARD_PROTOCOL = "STATIC_GUARD-direct-multi-PTZ-independent-static-guard-v1"
DIRECT_STATIC_CACHE_PROTOCOL = (
    "expanded-direct-target-static-appended-observation-cache-v1"
)
DIRECT_ROMA_STATIC_PROTOCOL = (
    "CENTER_CLOSURE-direct-BA-plus-fixed-static-RoMa-anchor-v1"
)
DIRECT_ROMA_PROTECTED_PROTOCOL = (
    "PTZ_BUNDLE-direct-BA-RoMa-anchor-high-confidence-protection-v1"
)
ROMA_STATIC_CACHE_PROTOCOL = (
    "roma-fast-bidirectional-component-anchor-merged-v1"
)
DIRECT_ARMS = frozenset({"D", "E"})
TEMPORAL_OBSERVATION_ARMS = frozenset({"C", "D", "E"})
STATIC_VIEWS = [
    "00", "02", "03", "05", "06", "08", "09", "11",
    "12", "14", "15", "17", "18", "20", "21", "23",
    "24", "26", "27", "29", "30", "32", "33", "35",
    "36", "38", "39", "41", "42", "44", "45", "47",
]
TARGET_VIEWS = ["81", "82", "83", "84", "86", "87", "88", "89", "91", "92", "93", "94"]
if os.environ.get("TRACK4D_CALIB_RIG_CONTRACT"):
    from relay_contract import STATIC_VIEWS as _statics, TARGET_VIEWS as _dynamic
    STATIC_VIEWS = list(_statics)
    TARGET_VIEWS = list(_dynamic)
SPATIAL_SOURCE_OFFSET = 1_000_000_000
_SPATIAL_OBSERVATION_CONTEXT: dict[str, Any] | None = None
_TEMPORAL_OBSERVATION_CONTEXT: dict[str, Any] | None = None
_DIRECT_COMPONENT_CONTEXT: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, CameraParams):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if not isinstance(item, CameraParams)
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _camera_equal(left: CameraParams, right: CameraParams) -> bool:
    return bool(
        left.view_id == right.view_id
        and left.H == right.H
        and left.W == right.W
        and np.array_equal(left.K, right.K)
        and np.array_equal(left.R, right.R)
        and np.array_equal(left.T, right.T)
        and np.array_equal(left.D, right.D)
    )


def _load_pose_grid(root: Path, frames: Sequence[int]) -> dict[int, dict[str, CameraParams]]:
    return {
        int(frame): CameraLoader(
            Path(root) / f"f{int(frame):06d}" / "cameras_pred"
        ).load_cameras(TARGET_VIEWS)
        for frame in frames
    }


def _load_spatial_bundles(root: Path, frames: Sequence[int]) -> dict[int, dict[str, Any]]:
    bundles: dict[int, dict[str, Any]] = {}
    expected = STATIC_VIEWS + TARGET_VIEWS
    for frame in frames:
        frame_root = Path(root) / f"f{int(frame):06d}"
        bundle = _load_bundle(frame_root)
        if list(bundle["views"]) != expected:
            raise RuntimeError(f"frame {frame}: spatial track view roster mismatch")
        builder = str(bundle["meta"].get("seed_builder"))
        if (
            str(bundle["meta"].get("mask_mode")) != "all"
            or int(bundle["meta"].get("min_visible_views", -1)) != 3
            or int(bundle["meta"].get("seed_bridge_min_points_per_view", -1)) != 150
        ):
            raise RuntimeError(f"frame {frame}: spatial track protocol changed")
        if builder == "spsg_target_verified_closure":
            builder_rows = bundle["meta"].get("seed_builder_diag_by_seed", {})
            if not builder_rows or any(
                float(row.get("f_ransac_px", float("nan"))) != 2.0
                or str(row.get("feature_conf_name")) != "superpoint_aachen"
                for row in builder_rows.values()
            ):
                raise RuntimeError(f"frame {frame}: SPSG matching protocol changed")
        elif builder == "anchor_lk_propagated_spatial_closure":
            if (
                str(bundle["meta"].get("protocol"))
                != "global9-bidirectional-allview-anchor-lk-spatial-closure-v1"
                or str(bundle["meta"].get("status"))
                != "sealed_public_track_propagation_before_pose_GT_or_NVS"
                or float(bundle["meta"].get("fb_threshold_px", float("nan")))
                != 1.5
                or int(bundle["meta"].get("min_static_support", -1)) != 2
                or bool(bundle["meta"].get("GT_used", True))
                or bool(bundle["meta"].get("NVS_used", True))
                or bool(bundle["meta"].get("test_RGB_used", True))
            ):
                raise RuntimeError(
                    f"frame {frame}: propagated spatial track protocol changed"
                )
            anchors = list(bundle["meta"].get("anchor_frames_global", []))
            coverage = dict(bundle["meta"].get("coverage", {}))
            per_target = dict(
                coverage.get("per_target_triangulatable_points", {})
            )
            if (
                len(anchors) not in (1, 2)
                or sorted(per_target) != sorted(TARGET_VIEWS)
                or int(coverage.get("minimum", -1)) < 12
            ):
                raise RuntimeError(
                    f"frame {frame}: propagated spatial coverage is insufficient"
                )
        else:
            raise RuntimeError(
                f"frame {frame}: unsupported spatial track provenance {builder!r}"
            )
        if int(bundle["meta"].get("start_frame", -1)) != int(frame):
            raise RuntimeError(f"frame {frame}: spatial track frame identity mismatch")
        tracks = np.asarray(bundle["tracks"])
        if tracks.shape[1] != 1:
            raise RuntimeError("spatial bundle must contain exactly one frame")
        bundles[int(frame)] = bundle
    return bundles


def _track_file_hashes(root: Path, frames: Sequence[int]) -> dict[str, str]:
    names = ("tracks.npy", "visibility.npy", "points_t0.npy", "views.txt", "meta.json")
    hashes: dict[str, str] = {}
    for frame in frames:
        for name in names:
            path = Path(root) / f"f{int(frame):06d}" / name
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[str(path.relative_to(root))] = _sha256(path)
    return hashes


def _load_temporal_bundle_with_segment_adapter(
    root: Path, frames: Sequence[int]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load a temporal bundle, optionally joining contiguous shards in RAM."""

    source = Path(root).resolve()
    adapter = source / "SEGMENTED_TEMPORAL_BUNDLE.json"
    names = ("tracks.npy", "visibility.npy", "points_t0.npy", "views.txt", "meta.json")
    if not adapter.is_file():
        bundle = _load_bundle(source)
        return bundle, {name: _sha256(source / name) for name in names}

    payload = json.loads(adapter.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_single120_segment_adapter":
        raise ValueError("temporal segment adapter is not frozen")
    expected_frames = list(map(int, frames))
    rows = list(payload.get("segments", []))
    if not rows:
        raise ValueError("temporal segment adapter is empty")
    bundles: list[tuple[int, int, Path, dict[str, Any]]] = []
    covered: list[int] = []
    hashes = {"SEGMENTED_TEMPORAL_BUNDLE.json": _sha256(adapter)}
    for index, row in enumerate(rows):
        start = int(row["global_start"])
        count = int(row["frame_count"])
        segment_root = Path(row["root"]).resolve()
        bundle = _load_bundle(segment_root)
        if int(bundle["meta"].get("start_frame", -1)) != 0:
            raise ValueError("temporal source shard must use local frame zero")
        if int(bundle["meta"].get("num_frames", -1)) != count:
            raise ValueError("temporal source shard frame count mismatch")
        covered.extend(range(start, start + count))
        bundles.append((start, count, segment_root, bundle))
        for name in names:
            hashes[f"segment{index}/{name}"] = _sha256(segment_root / name)
    if covered != expected_frames:
        raise ValueError("temporal segment adapter does not exactly cover frames")

    views = list(bundles[0][3]["views"])
    total_points = sum(int(np.asarray(bundle["tracks"]).shape[2]) for *_x, bundle in bundles)
    view_count = len(views)
    frame_count = len(expected_frames)
    tracks = np.zeros((view_count, frame_count, total_points, 2), dtype=np.float32)
    visibility = np.zeros((view_count, frame_count, total_points), dtype=bool)
    quality = np.zeros((view_count, frame_count, total_points), dtype=np.float32)
    points_t0 = np.zeros((view_count, total_points, 2), dtype=np.float32)
    seed_frames: list[int] = []
    seed_counts: list[int] = []
    point_offset = 0
    for start, count, segment_root, bundle in bundles:
        if list(bundle["views"]) != views:
            raise ValueError("temporal source shard view roster mismatch")
        source_tracks = np.asarray(bundle["tracks"], dtype=np.float32)
        source_visibility = np.asarray(bundle["visibility"], dtype=bool)
        source_quality = np.asarray(
            bundle.get("track_quality", source_visibility.astype(np.float32)),
            dtype=np.float32,
        )


        source_points = np.asarray(
            np.load(segment_root / "points_t0.npy", allow_pickle=False),
            dtype=np.float32,
        )
        points = int(source_tracks.shape[2])
        point_slice = slice(point_offset, point_offset + points)
        frame_slice = slice(start, start + count)
        tracks[:, frame_slice, point_slice] = source_tracks
        visibility[:, frame_slice, point_slice] = source_visibility
        quality[:, frame_slice, point_slice] = source_quality
        points_t0[:, point_slice] = source_points
        local_seeds = [int(value) for value in bundle["meta"].get("seed_frames", [])]
        local_counts = [int(value) for value in bundle["meta"].get("num_points_by_seed", [])]
        if sum(local_counts) != points:
            raise ValueError("temporal source shard seed blocks do not cover points")
        seed_frames.extend(start + value for value in local_seeds)
        seed_counts.extend(local_counts)
        point_offset += points
    meta = dict(bundles[0][3]["meta"])
    meta.update(
        {
            "start_frame": 0,
            "seed_frame": seed_frames[0],
            "seed_frames": seed_frames,
            "num_frames": frame_count,
            "num_points": total_points,
            "num_points_by_seed": seed_counts,
            "num_seeds": len(seed_frames),
            "single120_segment_adapter": str(adapter),
        }
    )
    return {
        "tracks": tracks,
        "visibility": visibility,
        "track_quality": quality,
        "points_t0": points_t0,
        "views": views,
        "meta": meta,
    }, hashes


def _build_spatial_observation(
    *,
    bundle: Mapping[str, Any],
    support_cameras: Mapping[str, CameraParams],
    target_view: str,
    frame: int,
    point_ids: np.ndarray,
    config: SameTimePnPConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support, target, raw = build_node_observations(
        tracks=bundle["tracks"],
        visibility=bundle["visibility"],
        track_quality=bundle.get("track_quality"),
        view_ids=bundle["views"],
        static_views=STATIC_VIEWS,
        target_view=str(target_view),
        frame_local=0,
        min_support_views=int(config.min_support_views),
        max_points=150,
        allowed_point_ids=np.asarray(point_ids, dtype=np.int64),
    )
    triangulated = triangulate_same_time_multiview_points(
        support_cameras=support_cameras,
        support_observations=support,
        target_observations=target,
        config=config,
    )
    physical = np.asarray(triangulated["point_ids"], dtype=np.int64)
    namespaced = SPATIAL_SOURCE_OFFSET + (int(frame) + 1) * 1_000_000 + physical
    triangulation_by_id = {
        int(row["point_id"]): row
        for row in triangulated.get("triangulation_rows", [])
    }
    observation = {
        "frame": int(frame),
        "point_ids": namespaced,
        "validation_group_ids": namespaced.copy(),
        "xyz": np.asarray(triangulated["object_points"], dtype=np.float64),
        "image_points": np.asarray(triangulated["image_points"], dtype=np.float64),
        "support_inlier_count": np.asarray(
            [
                triangulation_by_id[int(point_id)]["support_inlier_count"]
                for point_id in physical
            ],
            dtype=np.int64,
        ),
        "support_reprojection_median_px": np.asarray(
            [
                triangulation_by_id[int(point_id)][
                    "support_reprojection_median_px"
                ]
                for point_id in physical
            ],
            dtype=np.float64,
        ),
        "max_triangulation_angle_deg": np.asarray(
            [
                triangulation_by_id[int(point_id)][
                    "max_triangulation_angle_deg"
                ]
                for point_id in physical
            ],
            dtype=np.float64,
        ),
    }
    rows = list(triangulated.get("triangulation_rows", []))
    return observation, {
        **raw,
        "triangulated_count": int(triangulated["triangulated_count"]),
        "support_inlier_count": int(triangulated["support_observation_inlier_count"]),
        "positive_depth": int(len(rows)),
        "triangulation_angle_median_deg": (
            float(np.median([row["max_triangulation_angle_deg"] for row in rows]))
            if rows
            else None
        ),
    }


def _build_spatial_job_from_context(
    job: tuple[str, int, str],
) -> tuple[str, int, str, dict[str, Any], dict[str, Any]]:
    if _SPATIAL_OBSERVATION_CONTEXT is None:
        raise RuntimeError("spatial observation worker context is unset")
    view, frame, role = job
    bundle = _SPATIAL_OBSERVATION_CONTEXT["bundles"][int(frame)]
    point_ids_by_frame = _SPATIAL_OBSERVATION_CONTEXT.get("point_ids_by_frame")
    point_ids = (
        np.arange(np.asarray(bundle["tracks"]).shape[2], dtype=np.int64)
        if point_ids_by_frame is None
        else np.asarray(point_ids_by_frame[int(frame)], dtype=np.int64)
    )
    split = split_frame_local_track_ids(frame=int(frame), point_ids=point_ids)
    observation, diagnostic = _build_spatial_observation(
        bundle=bundle,
        support_cameras=_SPATIAL_OBSERVATION_CONTEXT["support_cameras"],
        target_view=str(view),
        frame=int(frame),
        point_ids=split[str(role)],
        config=_SPATIAL_OBSERVATION_CONTEXT["config"],
    )
    return str(view), int(frame), str(role), observation, diagnostic


def _build_temporal_view_from_context(
    view: str,
) -> tuple[str, dict[str, Any]]:
    if _TEMPORAL_OBSERVATION_CONTEXT is None:
        raise RuntimeError("temporal observation worker context is unset")
    context = _TEMPORAL_OBSERVATION_CONTEXT
    rows = []
    for frame in context["frames"]:
        allowed = point_ids_in_frame_window(
            context["seed_axis"], frame=int(frame), radius=2
        )
        observation, _ = build_time_local_observation(
            bundle=context["bundle"],
            support_cameras=context["support_cameras"],
            static_views=STATIC_VIEWS,
            target_view=str(view),
            frame=int(frame),
            allowed_point_ids=allowed,
            config=context["config"],
            max_points=150,
        )
        rows.append(observation)
    split = split_joint_ptz_track_groups(
        observations=rows,
        initial_cameras={
            int(frame): context["base_grid"][int(frame)][str(view)]
            for frame in context["frames"]
        },
    )
    if not bool(split["group_disjoint"]):
        raise RuntimeError("temporal fit/confirm/final IDs overlap")
    return str(view), split


def _observation_executor(kind: str, workers: int) -> concurrent.futures.Executor:
    if kind == "thread":
        return concurrent.futures.ThreadPoolExecutor(max_workers=int(workers))
    if kind == "process":
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=multiprocessing.get_context("fork"),
        )
    raise ValueError(f"unsupported observation executor: {kind}")


def _merge_observations(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    if int(first["frame"]) != int(second["frame"]):
        raise ValueError("cannot merge observations from different frames")
    return {
        "frame": int(first["frame"]),
        "point_ids": np.concatenate([first["point_ids"], second["point_ids"]]),
        "validation_group_ids": np.concatenate(
            [first["validation_group_ids"], second["validation_group_ids"]]
        ),
        "xyz": np.concatenate([first["xyz"], second["xyz"]], axis=0),
        "image_points": np.concatenate(
            [first["image_points"], second["image_points"]], axis=0
        ),
    }


def _direct_fit_observation(
    *,
    spatial: Mapping[str, Any],
    temporal: Mapping[str, Any],
    use_temporal: bool,
) -> Mapping[str, Any]:
    return _merge_observations(temporal, spatial) if use_temporal else spatial


def _validated_direct_static_observations(
    payload: Mapping[str, Any],
    *,
    required_keys: set[tuple[str, int, str]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    if str(payload.get("protocol")) != DIRECT_STATIC_CACHE_PROTOCOL:
        raise RuntimeError("external spatial cache protocol is not RELAY_CLOSURE appended-static")
    observations = payload.get("spatial_observations")
    if not isinstance(observations, Mapping):
        raise RuntimeError("external spatial cache lacks spatial observations")
    missing = required_keys - set(observations)
    if missing:
        raise RuntimeError(
            f"external spatial cache lacks {len(missing)} required observations"
        )
    prefix_rows = payload.get("prefix_audit")
    if not isinstance(prefix_rows, list) or len(prefix_rows) != 30:
        raise RuntimeError("external spatial cache lacks the 30-frame prefix audit")
    required_attestations = (
        "tracks_prefix_equal_nan",
        "visibility_prefix_equal",
        "points_t0_prefix_equal_nan",
        "appended_exactly_one_target",
    )
    if any(
        not all(bool(row.get(key)) for key in required_attestations)
        or int(row.get("appended_count", 0)) <= 0
        or int(row.get("appended_min_static_support", 0)) < 2
        for row in prefix_rows
    ):
        raise RuntimeError("external spatial cache failed its independent-prefix audit")
    return {key: dict(observations[key]) for key in required_keys}


def _validated_roma_static_epipolar_observations(
    payload: Mapping[str, Any],
    *,
    frames: Sequence[int],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    if str(payload.get("protocol")) != ROMA_STATIC_CACHE_PROTOCOL:
        raise RuntimeError("RoMa static cache protocol mismatch")
    if str(payload.get("status")) != "complete_public_match_cache":
        raise RuntimeError("RoMa static cache is incomplete")
    attestation = payload.get("attestation", {})
    if any(
        bool(attestation.get(key))
        for key in (
            "GT_used",
            "NVS_used",
            "test_RGB_or_cameras_used",
            "reference_pose_used",
        )
    ):
        raise RuntimeError("RoMa static cache method-access attestation failed")
    if [int(frame) for frame in payload.get("frames", [])] != [
        int(frame) for frame in frames
    ]:
        raise RuntimeError("RoMa static cache frame roster mismatch")
    observations = payload.get("observations")
    if not isinstance(observations, Mapping):
        raise RuntimeError("RoMa static cache lacks observations")
    validated: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw_key, raw_row in observations.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 3:
            raise RuntimeError("RoMa static observation key is malformed")
        view, frame, role = str(raw_key[0]), int(raw_key[1]), str(raw_key[2])
        if view not in TARGET_VIEWS or frame not in frames:
            raise RuntimeError("RoMa static observation is outside the frozen roster")
        if role not in {"fit", "confirm", "final"}:
            raise RuntimeError("RoMa static observation role is invalid")
        row = dict(raw_row)
        static_view = str(row.get("static_view"))
        if static_view not in STATIC_VIEWS:
            raise RuntimeError("RoMa static observation references a non-static view")
        point_ids = np.asarray(row.get("point_ids"), dtype=np.int64).reshape(-1)
        left = np.asarray(row.get("static_points"), dtype=np.float64).reshape(-1, 2)
        right = np.asarray(row.get("target_points"), dtype=np.float64).reshape(-1, 2)
        if not (len(point_ids) == len(left) == len(right)):
            raise RuntimeError("RoMa static observation arrays have unequal length")
        if len(np.unique(point_ids)) != len(point_ids):
            raise RuntimeError("RoMa static observation point IDs are not unique")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise RuntimeError("RoMa static observation contains non-finite points")
        validated[(view, frame, role)] = {
            "static_view": static_view,
            "point_ids": point_ids,
            "static_points": left,
            "target_points": right,
        }
    for view, frame, _role in validated:
        role_sets = [
            set(validated.get((view, frame, role), {}).get("point_ids", []))
            for role in ("fit", "confirm", "final")
        ]
        if any(
            role_sets[first] & role_sets[second]
            for first in range(3)
            for second in range(first + 1, 3)
        ):
            raise RuntimeError("RoMa static fit/confirm/final IDs overlap")
    return validated


def _errors_for_observation(camera: CameraParams, observation: Mapping[str, Any]) -> np.ndarray:
    return _camera_reprojection_errors(
        camera=camera,
        xyz=np.asarray(observation["xyz"], dtype=np.float64),
        image_points=np.asarray(observation["image_points"], dtype=np.float64),
    )


def _spatial_errors(
    *,
    bundles: Mapping[int, Mapping[str, Any]],
    observations: Mapping[tuple[str, int, str], Mapping[str, Any]],
    component: Sequence[str],
    frames: Sequence[int],
    role: str,
    base_grid: Mapping[int, Mapping[str, CameraParams]],
    candidate_grid: Mapping[int, Mapping[str, CameraParams]],
) -> dict[str, Any]:
    targets = [str(view) for view in component]
    static_base: dict[str, np.ndarray] = {}
    static_candidate: dict[str, np.ndarray] = {}
    pair_base: dict[str, np.ndarray] = {}
    pair_candidate: dict[str, np.ndarray] = {}
    supported_edges: set[tuple[str, str]] = set()
    pair_count_by_edge: dict[str, int] = {}
    for frame in frames:
        bundle = bundles[int(frame)]
        tracks = np.asarray(bundle["tracks"])[:, 0]
        visibility = np.asarray(bundle["visibility"], dtype=bool)[:, 0]
        views = list(bundle["views"])
        point_count = tracks.shape[1]
        role_ids = split_frame_local_track_ids(
            frame=int(frame), point_ids=np.arange(point_count, dtype=np.int64)
        )[str(role)]
        for view in targets:
            observation = observations[(view, int(frame), str(role))]
            key = f"{frame}:{view}"
            static_base[key] = _errors_for_observation(
                base_grid[int(frame)][view], observation
            )
            static_candidate[key] = _errors_for_observation(
                candidate_grid[int(frame)][view], observation
            )
        for left, right in combinations(targets, 2):
            valid = (
                visibility[views.index(left), role_ids]
                & visibility[views.index(right), role_ids]
                & np.isfinite(tracks[views.index(left), role_ids]).all(axis=1)
                & np.isfinite(tracks[views.index(right), role_ids]).all(axis=1)
            )
            selected = role_ids[valid]
            if not len(selected):
                continue
            key = f"{frame}:{left}:{right}"
            left_points = tracks[views.index(left), selected]
            right_points = tracks[views.index(right), selected]
            pair_base[key] = sampson_distances_px(
                target_camera=base_grid[int(frame)][left],
                support_camera=base_grid[int(frame)][right],
                target_points=left_points,
                support_points=right_points,
            )
            pair_candidate[key] = sampson_distances_px(
                target_camera=candidate_grid[int(frame)][left],
                support_camera=candidate_grid[int(frame)][right],
                target_points=left_points,
                support_points=right_points,
            )
            edge = (left, right)
            pair_count_by_edge[f"{left}:{right}"] = (
                pair_count_by_edge.get(f"{left}:{right}", 0) + len(selected)
            )
            if pair_count_by_edge[f"{left}:{right}"] >= 8:
                supported_edges.add(edge)

    adjacency = {view: set() for view in targets}
    for left, right in supported_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    reached = set()
    queue = [targets[0]] if targets else []
    while queue:
        current = queue.pop()
        if current in reached:
            continue
        reached.add(current)
        queue.extend(sorted(adjacency[current] - reached))
    graph_connected = bool(targets and reached == set(targets))
    static_score = pooled_pareto_score(
        base_errors_by_node=static_base,
        candidate_errors_by_node=static_candidate,
        threshold_px=4.0,
    )
    pair_score = pooled_pareto_score(
        base_errors_by_node=pair_base,
        candidate_errors_by_node=pair_candidate,
        threshold_px=4.0,
    )
    static_guard = pooled_noninferiority_guard(
        base_errors_by_node=static_base,
        candidate_errors_by_node=static_candidate,
        threshold_px=4.0,
    )
    commit = bool(
        graph_connected and static_score.get("commit") and pair_score.get("commit")
    )
    return {
        "commit": commit,
        "reason": (
            "heldout_static_and_PTZ_pair_geometry_improved"
            if commit
            else "heldout_spatial_geometry_failed"
        ),
        "role": str(role),
        "static": static_score,
        "static_noninferiority_guard": static_guard,
        "PTZ_pair": pair_score,
        "supported_target_graph_connected": graph_connected,
        "supported_target_edges": [list(edge) for edge in sorted(supported_edges)],
        "pair_count_by_edge": pair_count_by_edge,
    }


def _temporal_score(
    *,
    observations: Mapping[tuple[str, int, str], Mapping[str, Any]],
    component: Sequence[str],
    frames: Sequence[int],
    role: str,
    base_grid: Mapping[int, Mapping[str, CameraParams]],
    candidate_grid: Mapping[int, Mapping[str, CameraParams]],
) -> dict[str, Any]:
    base: dict[str, np.ndarray] = {}
    candidate: dict[str, np.ndarray] = {}
    for frame in frames:
        for view in component:
            observation = observations[(str(view), int(frame), str(role))]
            if not len(np.asarray(observation["point_ids"])):
                continue
            key = f"{frame}:{view}"
            base[key] = _errors_for_observation(base_grid[int(frame)][str(view)], observation)
            candidate[key] = _errors_for_observation(
                candidate_grid[int(frame)][str(view)], observation
            )
    strict = pooled_pareto_score(
        base_errors_by_node=base,
        candidate_errors_by_node=candidate,
        threshold_px=4.0,
    )
    return {
        **strict,
        "noninferiority_guard": pooled_noninferiority_guard(
            base_errors_by_node=base,
            candidate_errors_by_node=candidate,
            threshold_px=4.0,
        ),
    }


def _fit_multi_ptz_candidate(
    *,
    component: Sequence[str],
    frames: Sequence[int],
    base_grid: Mapping[int, Mapping[str, CameraParams]],
    fixed_centers: Mapping[str, np.ndarray],
    fit_observations: Mapping[tuple[str, int, str], Mapping[str, Any]],
    temporal_observations: Mapping[tuple[str, int, str], Mapping[str, Any]],
    bundles: Mapping[int, Mapping[str, Any]],
    fit_residual_mode: str,
    center_path_mode: str,
) -> tuple[dict[int, dict[str, CameraParams]], dict[str, Any]]:
    endpoints: dict[tuple[int, str], np.ndarray] = {}
    solver_rows: dict[str, Any] = {}
    for view in component:
        initial = {int(frame): base_grid[int(frame)][str(view)] for frame in frames}
        rows = [
            _merge_observations(
                temporal_observations[(str(view), int(frame), "fit")],
                fit_observations[(str(view), int(frame), "fit")],
            )
            for frame in frames
        ]
        solved = solve_fixed_center_ptz_rotations(
            observations=rows,
            initial_cameras=initial,
            fixed_center=np.asarray(fixed_centers[str(view)]),
            reprojection_error_px=4.0,
            min_inliers_per_frame=12,
            min_successful_frames=3,
            rng_seed=0,
        )
        solver_rows[str(view)] = {
            key: value for key, value in solved.items() if key != "cameras"
        }
        if bool(solved.get("success", False)):
            for frame in solved.get("successful_frames", []):
                endpoints[(int(frame), str(view))] = np.asarray(
                    solved["cameras"][int(frame)].R, dtype=np.float64
                )

    keys = sorted(endpoints, key=lambda item: (item[0], TARGET_VIEWS.index(item[1])))
    candidate = {
        int(frame): dict(base_grid[int(frame)]) for frame in frames
    }


def _fit_direct_multi_ptz_candidate(
    *,
    component: Sequence[str],
    frames: Sequence[int],
    base_grid: Mapping[int, Mapping[str, CameraParams]],
    fixed_centers: Mapping[str, np.ndarray],
    fit_observations: Mapping[tuple[str, int, str], Mapping[str, Any]],
    temporal_observations: Mapping[tuple[str, int, str], Mapping[str, Any]],
    bundles: Mapping[int, Mapping[str, Any]],
    confidence: Mapping[tuple[int, str], float],
    optimize_shared_center: bool = False,
    center_trust_radii: Mapping[str, float] | None = None,
    use_joint_center_initialization: bool = False,
    direct_max_nfev: int = 10,
    direct_reprojection_weight: float = 1.0,
    direct_epipolar_weight: float = 1.0,
    direct_x_scale: str = "unit",
    direct_convergence_tolerance: float = 1e-8,
    use_temporal_fit_observations: bool = True,
    fixed_static_epipolar_observations: Mapping[
        tuple[str, int, str], Mapping[str, Any]
    ]
    | None = None,
    support_cameras: Mapping[str, CameraParams] | None = None,
    protected_nodes: set[tuple[int, str]] | None = None,
) -> tuple[dict[int, dict[str, CameraParams]], dict[str, Any]]:
    all_nodes = [
        (int(frame), str(view)) for frame in frames for view in component
    ]
    protected = set(protected_nodes or ())
    if protected - set(all_nodes):
        raise ValueError("protected nodes must belong to the fitted component")
    nodes = [node for node in all_nodes if node not in protected]
    if not nodes:
        return (
            {int(frame): dict(base_grid[int(frame)]) for frame in frames},
            {
                "success": False,
                "reason": "all_nodes_protected",
                "protected_nodes": [
                    f"{frame}:{view}" for frame, view in sorted(protected)
                ],
                "protected_node_count": len(protected),
                "optimized_node_count": 0,
            },
        )
    active = set(nodes)
    reprojection_observations = {
        node: _direct_fit_observation(
            spatial=fit_observations[(node[1], node[0], "fit")],
            temporal=temporal_observations[(node[1], node[0], "fit")],
            use_temporal=use_temporal_fit_observations,
        )
        for node in nodes
    }
    epipolar_observations: list[dict[str, Any]] = []
    for frame in frames:
        bundle = bundles[int(frame)]
        tracks = np.asarray(bundle["tracks"])[:, 0]
        visibility = np.asarray(bundle["visibility"], dtype=bool)[:, 0]
        views = list(bundle["views"])
        fit_ids = split_frame_local_track_ids(
            frame=int(frame),
            point_ids=np.arange(tracks.shape[1], dtype=np.int64),
        )["fit"]
        for left, right in combinations(component, 2):
            valid = (
                visibility[views.index(left), fit_ids]
                & visibility[views.index(right), fit_ids]
                & np.isfinite(tracks[views.index(left), fit_ids]).all(axis=1)
                & np.isfinite(tracks[views.index(right), fit_ids]).all(axis=1)
            )
            selected = fit_ids[valid]
            if not len(selected):
                continue
            left_node = (int(frame), str(left))
            right_node = (int(frame), str(right))
            if left_node not in active and right_node not in active:
                continue
            edge = {
                "left_points": tracks[views.index(left), selected],
                "right_points": tracks[views.index(right), selected],
            }
            if left_node in active:
                edge["left"] = left_node
            else:
                edge["left_camera"] = base_grid[int(frame)][str(left)]
            if right_node in active:
                edge["right"] = right_node
            else:
                edge["right_camera"] = base_grid[int(frame)][str(right)]
            epipolar_observations.append(edge)
    fixed_static_edge_count = 0
    fixed_static_point_count = 0
    if fixed_static_epipolar_observations is not None:
        if support_cameras is None:
            raise ValueError("fixed-static epipolar observations need support cameras")
        for frame, view in nodes:
            row = fixed_static_epipolar_observations.get((view, frame, "fit"))
            if row is None:
                continue
            static_view = str(row["static_view"])
            if static_view not in support_cameras:
                raise ValueError("fixed-static epipolar camera is unavailable")
            left_points = np.asarray(row["static_points"], dtype=np.float64)
            right_points = np.asarray(row["target_points"], dtype=np.float64)
            if not len(left_points):
                continue
            epipolar_observations.append(
                {
                    "left_camera": support_cameras[static_view],
                    "right": (int(frame), str(view)),
                    "left_points": left_points,
                    "right_points": right_points,
                    "source": "public_static_RoMa_fit",
                }
            )
            fixed_static_edge_count += 1
            fixed_static_point_count += int(len(left_points))
    direct_config = DirectRotationBAConfig(
            max_rotation_correction_deg=5.0,
            max_nfev=int(direct_max_nfev),
            robust_loss=os.environ.get("TRACK4D_CALIB_CLOSURE_LOSS", "huber"),
            robust_scale_px=3.0,
            reprojection_weight=float(direct_reprojection_weight),
            epipolar_weight=float(direct_epipolar_weight),
            x_scale=("jac" if direct_x_scale == "jac" else 1.0),
            convergence_tolerance=float(direct_convergence_tolerance),
    )
    solver_inputs = {
        "nodes": nodes,
        "base_cameras": {node: base_grid[node[0]][node[1]] for node in nodes},
        "fixed_centers": {node: fixed_centers[node[1]] for node in nodes},
        "reprojection_observations": reprojection_observations,
        "epipolar_observations": epipolar_observations,
        "confidence": {node: confidence[node] for node in nodes},
        "config": direct_config,
    }
    if optimize_shared_center:
        if center_trust_radii is None:
            raise ValueError("shared-center BA needs public trust radii")
        solver_inputs.update(
            {
                "center_group_by_node": {node: node[1] for node in nodes},
                "center_trust_radius_by_group": {
                    str(view): float(center_trust_radii[str(view)])
                    for view in sorted({node[1] for node in nodes})
                },
            }
        )
    static_start = solve_direct_rotation_component(**solver_inputs)

    endpoint_residuals: dict[tuple[int, str], np.ndarray] = {}
    endpoint_solver: dict[str, Any] = {}
    for view in component:
        active_frames = [
            int(frame) for frame in frames if (int(frame), str(view)) in active
        ]
        if len(active_frames) < 3:
            endpoint_solver[str(view)] = {
                "success": False,
                "reason": "fewer_than_three_unprotected_frames",
                "active_frames": active_frames,
            }
            continue
        initial = {
            int(frame): base_grid[int(frame)][str(view)] for frame in active_frames
        }
        observations = [
            reprojection_observations[(int(frame), str(view))]
            for frame in active_frames
        ]
        endpoint = solve_fixed_center_ptz_rotations(
            observations=observations,
            initial_cameras=initial,
            fixed_center=np.asarray(fixed_centers[str(view)]),
            reprojection_error_px=4.0,
            min_inliers_per_frame=12,
            min_successful_frames=3,
            rng_seed=0,
        )
        endpoint_solver[str(view)] = {
            key: value for key, value in endpoint.items() if key != "cameras"
        }
        if not bool(endpoint.get("success", False)):
            continue
        for frame in endpoint.get("successful_frames", []):
            node = (int(frame), str(view))
            endpoint_residuals[node] = Rotation.from_matrix(
                np.asarray(endpoint["cameras"][int(frame)].R)
                @ np.asarray(base_grid[int(frame)][str(view)].R).T
            ).as_rotvec()
    endpoint_start = solve_direct_rotation_component(
        **solver_inputs,
        initial_rotation_residuals=endpoint_residuals,
    )
    starts = {"static": static_start, "independent_endpoint": endpoint_start}
    joint_endpoint_solver: dict[str, Any] = {}
    joint_endpoint_residuals: dict[tuple[int, str], np.ndarray] = {}
    joint_center_residuals: dict[str, np.ndarray] = {}
    if optimize_shared_center and use_joint_center_initialization:
        for view in component:
            active_frames = [
                int(frame) for frame in frames if (int(frame), str(view)) in active
            ]
            if len(active_frames) < 3:
                joint_endpoint_solver[str(view)] = {
                    "success": False,
                    "reason": "fewer_than_three_unprotected_frames",
                    "active_frames": active_frames,
                }
                continue
            initial = {
                int(frame): base_grid[int(frame)][str(view)]
                for frame in active_frames
            }
            observations = [
                reprojection_observations[(int(frame), str(view))]
                for frame in active_frames
            ]
            joint_endpoint = solve_joint_ptz_shared_center(
                observations=observations,
                initial_cameras=initial,
                reprojection_error_px=4.0,
                min_inliers_per_frame=12,
                min_successful_frames=3,
                rng_seed=0,
                allow_initial_camera_bootstrap=True,
                allow_joint_refined_rotation_fallback=True,
            )
            joint_endpoint_solver[str(view)] = {
                key: value
                for key, value in joint_endpoint.items()
                if key != "cameras"
            }
            if not bool(joint_endpoint.get("success", False)):
                continue
            joint_center_residuals[str(view)] = np.asarray(
                joint_endpoint["center"], dtype=np.float64
            ) - np.asarray(fixed_centers[str(view)], dtype=np.float64)
            for frame, camera in joint_endpoint["cameras"].items():
                node = (int(frame), str(view))
                joint_endpoint_residuals[node] = Rotation.from_matrix(
                    np.asarray(camera.R)
                    @ np.asarray(base_grid[int(frame)][str(view)].R).T
                ).as_rotvec()
        if joint_center_residuals:
            starts["joint_shared_center_endpoint"] = (
                solve_direct_rotation_component(
                    **solver_inputs,
                    initial_rotation_residuals=joint_endpoint_residuals,
                    initial_center_residuals=joint_center_residuals,
                )
            )
    selected_start = min(
        starts,
        key=lambda name: float(starts[name]["least_squares"]["cost"]),
    )
    solved = starts[selected_start]
    candidate = {int(frame): dict(base_grid[int(frame)]) for frame in frames}
    for (frame, view), camera in solved["cameras"].items():
        candidate[int(frame)][str(view)] = camera
    diagnostics = {
        key: value
        for key, value in solved.items()
        if key not in {
            "cameras",
            "unconstrained_residuals",
            "unconstrained_center_residuals",
        }
    }
    diagnostics.update(
        {
            "reason": "ok" if solved["success"] else "direct_rotation_BA_failed",
            "parameterization": (
                "direct_per_node_SO3_plus_per_physical_PTZ_shared_center_residual"
                if optimize_shared_center
                else "direct_per_node_SO3_residual"
            ),
            "endpoint_required": False,
            "epipolar_edge_count": len(epipolar_observations),
            "fixed_static_epipolar_edge_count": int(fixed_static_edge_count),
            "fixed_static_epipolar_point_count": int(fixed_static_point_count),
            "selected_initialization": selected_start,
            "initialization_diagnostics": {
                name: result["least_squares"] for name, result in starts.items()
            },
            "independent_endpoint_solver": endpoint_solver,
            "endpoint_initialized_node_count": len(endpoint_residuals),
            "joint_shared_center_endpoint_solver": joint_endpoint_solver,
            "joint_endpoint_initialized_node_count": len(
                joint_endpoint_residuals
            ),
            "joint_endpoint_initialized_center_count": len(
                joint_center_residuals
            ),
            "protected_nodes": [
                f"{frame}:{view}" for frame, view in sorted(protected)
            ],
            "protected_node_count": len(protected),
            "optimized_node_count": len(nodes),
        }
    )
    return candidate, diagnostics


def _high_confidence_half_nodes(
    *,
    component: Sequence[str],
    frames: Sequence[int],
    confidence: Mapping[tuple[int, str], float],
) -> set[tuple[int, str]]:
    """Protect the higher-confidence half independently at each frame."""

    views = [str(view) for view in component]
    if not views:
        raise ValueError("component must be non-empty")
    count = (len(views) + 1) // 2
    protected: set[tuple[int, str]] = set()
    for raw_frame in frames:
        frame = int(raw_frame)
        rows = []
        for view in views:
            value = float(confidence[(frame, view)])
            if not np.isfinite(value):
                raise ValueError("Static+PnP confidence must be finite")
            rows.append((-value, view))
        protected.update((frame, view) for _score, view in sorted(rows)[:count])
    return protected


def _direct_geometry_accepts(
    *,
    confirm: Mapping[str, Any],
    final: Mapping[str, Any],
    temporal_confirm: Mapping[str, Any],
    temporal_final: Mapping[str, Any],
    use_noninferiority_guards: bool,
) -> bool:
    if not use_noninferiority_guards:
        return bool(
            confirm.get("commit")
            and final.get("commit")
            and temporal_confirm.get("commit")
            and temporal_final.get("commit")
        )
    return bool(
        confirm.get("PTZ_pair", {}).get("commit")
        and final.get("PTZ_pair", {}).get("commit")
        and confirm.get("static_noninferiority_guard", {}).get("guard")
        and final.get("static_noninferiority_guard", {}).get("guard")
        and temporal_confirm.get("noninferiority_guard", {}).get("guard")
        and temporal_final.get("noninferiority_guard", {}).get("guard")
    )


def _evaluate_direct_component_from_context(
    job: tuple[int, int, list[int], list[str]],
) -> tuple[
    dict[str, Any],
    dict[tuple[int, str], CameraParams],
    dict[tuple[int, str], CameraParams],
]:
    if _DIRECT_COMPONENT_CONTEXT is None:
        raise RuntimeError("direct component worker context is unset")
    component_index, spatial_index, frames, component = job
    context = _DIRECT_COMPONENT_CONTEXT
    if len(component) < 2:
        return (
            {
                "component_index": int(component_index),
                "spatial_component_index": int(spatial_index),
                "window": list(frames),
                "component": list(component),
                "commit": False,
                "reason": "singleton_has_no_PTZ_pair_geometry",
            },
            {},
            {},
        )
    protected_nodes = (
        _high_confidence_half_nodes(
            component=component,
            frames=frames,
            confidence=context["confidence"],
        )
        if bool(context.get("protect_high_confidence_half"))
        else set()
    )
    candidate, fit_result = _fit_direct_multi_ptz_candidate(
        component=component,
        frames=frames,
        base_grid=context["base_grid"],
        fixed_centers=context["fixed_centers"],
        fit_observations=context["spatial_observations"],
        temporal_observations=context["temporal_observations"],
        bundles=context["spatial_bundles"],
        confidence=context["confidence"],
        optimize_shared_center=bool(context["optimize_shared_center"]),
        center_trust_radii=context["center_trust_radii"],
        use_joint_center_initialization=bool(
            context["use_joint_center_initialization"]
        ),
        direct_max_nfev=int(context["direct_max_nfev"]),
        direct_reprojection_weight=float(context["direct_reprojection_weight"]),
        direct_epipolar_weight=float(context["direct_epipolar_weight"]),
        direct_x_scale=str(context["direct_x_scale"]),
        direct_convergence_tolerance=float(
            context["direct_convergence_tolerance"]
        ),
        use_temporal_fit_observations=bool(
            context["use_temporal_fit_observations"]
        ),
        fixed_static_epipolar_observations=context.get(
            "fixed_static_epipolar_observations"
        ),
        support_cameras=context.get("support_cameras"),
        protected_nodes=protected_nodes,
    )
    changed = [
        (int(frame), str(view))
        for frame in frames
        for view in component
        if not _camera_equal(
            context["base_grid"][int(frame)][str(view)],
            candidate[int(frame)][str(view)],
        )
    ]
    if not bool(fit_result.get("success", False)) or not changed:
        return (
            {
                "component_index": int(component_index),
                "spatial_component_index": int(spatial_index),
                "window": list(frames),
                "component": list(component),
                "commit": False,
                "reason": fit_result.get("reason", "candidate_failed"),
                "candidate": fit_result,
            },
            {},
            {},
        )
    confirm = _spatial_errors(
        bundles=context["spatial_bundles"],
        observations=context["spatial_observations"],
        component=component,
        frames=frames,
        role="confirm",
        base_grid=context["base_grid"],
        candidate_grid=candidate,
    )
    final = _spatial_errors(
        bundles=context["spatial_bundles"],
        observations=context["spatial_observations"],
        component=component,
        frames=frames,
        role="final",
        base_grid=context["base_grid"],
        candidate_grid=candidate,
    )
    temporal_confirm = _temporal_score(
        observations=context["temporal_observations"],
        component=component,
        frames=frames,
        role="confirm",
        base_grid=context["base_grid"],
        candidate_grid=candidate,
    )
    temporal_final = _temporal_score(
        observations=context["temporal_observations"],
        component=component,
        frames=frames,
        role="final",
        base_grid=context["base_grid"],
        candidate_grid=candidate,
    )
    commit = _direct_geometry_accepts(
        confirm=confirm,
        final=final,
        temporal_confirm=temporal_confirm,
        temporal_final=temporal_final,
        use_noninferiority_guards=bool(
            context["use_noninferiority_guards"]
        ),
    )
    row = {
        "component_index": int(component_index),
        "spatial_component_index": int(spatial_index),
        "window": list(frames),
        "component": list(component),
        "changed_nodes": [f"{frame}:{view}" for frame, view in changed],
        "candidate": fit_result,
        "confirm": confirm,
        "final": final,
        "temporal_confirm": temporal_confirm,
        "temporal_final": temporal_final,
        "commit": commit,
        "reason": (
            "all_independent_geometry_improved"
            if commit
            else "independent_geometry_rejected"
        ),
        "commit_policy": (
            "strict_PTZ_pair_plus_independent_static_and_temporal_noninferiority"
            if context["use_noninferiority_guards"]
            else "strict_all_geometry_improvement"
        ),
    }
    selected = (
        {(frame, view): candidate[frame][view] for frame, view in changed}
        if commit
        else {}
    )
    diagnostic_candidate = {
        (frame, view): candidate[frame][view] for frame, view in changed
    }
    return row, selected, diagnostic_candidate
    if not keys:
        return candidate, {
            "success": False,
            "reason": "no_fixed_center_rotation_endpoint",
            "solver": solver_rows,
        }

    def materialize(values: np.ndarray) -> dict[int, dict[str, CameraParams]]:
        output = {int(frame): dict(base_grid[int(frame)]) for frame in frames}
        for alpha, (frame, view) in zip(values, keys):
            output[frame][view] = materialize_candidate_pose_path(
                base_camera=base_grid[frame][view],
                proposal_rotation=endpoints[(frame, view)],
                fixed_center=fixed_centers[view],
                alpha=float(alpha),
                center_path_mode=center_path_mode,
            )
        return output

    def residuals(values: np.ndarray) -> np.ndarray:
        cameras = materialize(values)
        residual_parts: list[np.ndarray] = []
        for frame, view in keys:
            merged = _merge_observations(
                temporal_observations[(view, frame, "fit")],
                fit_observations[(view, frame, "fit")],
            )
            errors = _errors_for_observation(cameras[frame][view], merged)
            residual_parts.append(
                candidate_fit_residuals(
                    errors,
                    mode=fit_residual_mode,
                    threshold_px=4.0,
                )
            )
        for frame in frames:
            bundle = bundles[int(frame)]
            tracks = np.asarray(bundle["tracks"])[:, 0]
            visibility = np.asarray(bundle["visibility"], dtype=bool)[:, 0]
            views = list(bundle["views"])
            fit_ids = split_frame_local_track_ids(
                frame=int(frame), point_ids=np.arange(tracks.shape[1], dtype=np.int64)
            )["fit"]
            for left, right in combinations(component, 2):
                valid = visibility[views.index(left), fit_ids] & visibility[views.index(right), fit_ids]
                selected = fit_ids[valid]
                if not len(selected):
                    continue
                errors = sampson_distances_px(
                    target_camera=cameras[int(frame)][str(left)],
                    support_camera=cameras[int(frame)][str(right)],
                    target_points=tracks[views.index(left), selected],
                    support_points=tracks[views.index(right), selected],
                )
                residual_parts.append(
                    candidate_fit_residuals(
                        errors,
                        mode=fit_residual_mode,
                        threshold_px=4.0,
                    )
                )
        return np.concatenate(residual_parts) if residual_parts else np.full(1, 4.0)

    initial = np.ones(len(keys), dtype=np.float64)
    optimized = least_squares(
        residuals,
        initial,
        bounds=(np.zeros(len(keys)), np.ones(len(keys))),
        max_nfev=5,
        loss="huber",
        f_scale=3.0,
    )
    alpha = np.clip(np.asarray(optimized.x, dtype=np.float64), 0.0, 1.0)
    candidate = materialize(alpha)
    return candidate, {
        "success": bool(np.isfinite(alpha).all()),
        "reason": "ok" if np.isfinite(alpha).all() else "nonfinite_joint_alpha",
        "solver": solver_rows,
        "optimized_node_count": int(len(keys)),
        "alpha_by_node": {
            f"{frame}:{view}": float(value)
            for value, (frame, view) in zip(alpha, keys)
        },
        "least_squares": {
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "nfev": int(optimized.nfev),
            "cost": float(optimized.cost),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _SPATIAL_OBSERVATION_CONTEXT, _TEMPORAL_OBSERVATION_CONTEXT
    global _DIRECT_COMPONENT_CONTEXT
    started = time.perf_counter()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frames = list(range(int(args.num_frames)))
    if len(frames) <= 0 or len(frames) % 5:
        raise ValueError("num_frames must be a positive multiple of five")
    if args.arm not in {"B", "C", "D", "E"}:
        raise ValueError("arm must be B, C, D, or E")
    if bool(args.joint_center_initialization) and args.arm != "E":
        raise ValueError("joint center initialization is only valid for arm E")
    if int(args.workers) <= 0:
        raise ValueError("workers must be positive")
    if int(args.direct_max_nfev) <= 0:
        raise ValueError("direct max_nfev must be positive")
    if (
        not np.isfinite(float(args.direct_convergence_tolerance))
        or float(args.direct_convergence_tolerance) <= 0.0
    ):
        raise ValueError("direct convergence tolerance must be positive and finite")
    if bool(args.use_noninferiority_guards) and args.arm not in DIRECT_ARMS:
        raise ValueError("non-inferiority guards are only valid for direct arms")
    if bool(args.use_noninferiority_guards) and args.spatial_observation_cache is None:
        raise ValueError(
            "non-inferiority guards require an independent spatial observation cache"
        )
    if args.spatial_observation_cache is not None and args.arm not in DIRECT_ARMS:
        raise ValueError("external spatial observations are only valid for direct arms")
    if args.roma_static_epipolar_cache is not None and args.arm not in DIRECT_ARMS:
        raise ValueError("RoMa static epipolar evidence is only valid for direct arms")
    if bool(args.protect_high_confidence_half) and (
        args.arm != "E" or args.roma_static_epipolar_cache is None
    ):
        raise ValueError(
            "high-confidence protection requires arm E and a RoMa static cache"
        )
    if not bool(args.direct_temporal_fit) and args.arm not in DIRECT_ARMS:
        raise ValueError("the direct temporal-fit switch is only valid for direct arms")
    direct_weights = np.asarray(
        [args.direct_reprojection_weight, args.direct_epipolar_weight],
        dtype=np.float64,
    )
    if (
        not np.isfinite(direct_weights).all()
        or np.any(direct_weights < 0.0)
        or not np.any(direct_weights > 0.0)
    ):
        raise ValueError("direct observation weights must be finite, non-negative, and not all zero")

    runtime = Path(args.runtime_scene).resolve()
    runtime_manifest = json.loads((runtime / "runtime_manifest.json").read_text())
    if [str(view) for view in runtime_manifest["observed_short_views"]] != STATIC_VIEWS:
        raise RuntimeError("public static roster differs from No-Parent PTZ v1")
    if [str(view) for view in runtime_manifest["long_views"] if str(view) in TARGET_VIEWS] != TARGET_VIEWS:
        raise RuntimeError("dynamic training PTZ roster differs from No-Parent PTZ v1")

    support_cameras = CameraLoader(runtime / "cameras/static").load_cameras(STATIC_VIEWS)
    base_grid = _load_pose_grid(Path(args.base_pose_root), frames)
    dynamic_track_grid = _load_pose_grid(Path(args.dynamic_track_pose_root), frames)
    spatial_bundles = _load_spatial_bundles(Path(args.spatial_track_root), frames)
    temporal_bundle, temporal_track_hashes = _load_temporal_bundle_with_segment_adapter(
        Path(args.temporal_track_root).resolve(), frames
    )
    if list(temporal_bundle["views"]) != STATIC_VIEWS + TARGET_VIEWS:
        raise RuntimeError("temporal track view roster mismatch")
    if int(temporal_bundle["meta"].get("start_frame", -1)) != 0 or int(
        temporal_bundle["meta"].get("num_frames", -1)
    ) != len(frames):
        raise RuntimeError("temporal track frame range mismatch")

    spatial_track_hashes = _track_file_hashes(Path(args.spatial_track_root), frames)

    config = SameTimePnPConfig(
        min_support_views=2,
        support_reprojection_error_px=2.5,
        pnp_reprojection_error_px=4.0,
        min_pnp_inliers=12,
        rng_seed=0,
    )
    confidence, confidence_audit = load_static_pnp_confidence(
        Path(args.static_pnp_diagnostics),
        frames=frames,
        target_views=TARGET_VIEWS,
    )
    fixed_centers = {
        view: weighted_geometric_median(
            np.asarray([base_grid[frame][view].camera_center for frame in frames]),
            np.asarray([confidence[(frame, view)] for frame in frames]),
        )
        for view in TARGET_VIEWS
    }
    center_trust_radii = {
        view: float(
            np.median(
                [
                    np.linalg.norm(
                        base_grid[frame][view].camera_center - fixed_centers[view]
                    )
                    for frame in frames
                ]
            )
        )
        for view in TARGET_VIEWS
    }
    if any(
        not np.isfinite(radius) or radius <= 0.0
        for radius in center_trust_radii.values()
    ):
        raise RuntimeError("public Static+PnP center spread is invalid")
    roma_static_epipolar_cache_input = (
        Path(args.roma_static_epipolar_cache).resolve()
        if args.roma_static_epipolar_cache is not None
        else None
    )
    roma_static_epipolar_observations = None
    roma_static_cache_runtime = None
    if roma_static_epipolar_cache_input is not None:
        with roma_static_epipolar_cache_input.open("rb") as handle:
            roma_payload = pickle.load(handle)
        roma_static_epipolar_observations = (
            _validated_roma_static_epipolar_observations(
                roma_payload, frames=frames
            )
        )
        roma_static_cache_runtime = dict(roma_payload.get("runtime", {}))
    method_protocol = (
        DIRECT_ROMA_PROTECTED_PROTOCOL
        if bool(args.protect_high_confidence_half)
        else DIRECT_ROMA_STATIC_PROTOCOL
        if roma_static_epipolar_cache_input is not None
        else DIRECT_GUARD_PROTOCOL
        if bool(args.use_noninferiority_guards)
        else DIRECT_CENTER_INIT_PROTOCOL
        if args.arm == "E" and bool(args.joint_center_initialization)
        else DIRECT_CENTER_PROTOCOL
        if args.arm == "E"
        else (DIRECT_PROTOCOL if args.arm == "D" else PROTOCOL)
    )
    _write_json(
        output / "METHOD_FREEZE.json",
        {
            "protocol": method_protocol,
            "status": "frozen_before_public_geometry",
            "arm": args.arm,
            "single_variable": (
                (
                    "protect the higher-confidence half of Static+PnP nodes "
                    "inside each direct-BA frame/component"
                )
                if bool(args.protect_high_confidence_half)
                else (
                    "append frozen public-static RoMa fit epipolar edges while "
                    "keeping TRACK_BUNDLE candidate parameterization and verifier"
                )
                if roma_static_epipolar_cache_input is not None
                else (
                    "include or exclude frozen temporal reprojection observations "
                    "from direct candidate fit"
                )
                if bool(args.use_noninferiority_guards)
                else "DYNAMIC_TRACK candidate plus same-time spatial verifier"
                if args.arm == "B"
                else (
                    "same-time multi-PTZ alpha candidate with arm-B spatial verifier"
                    if args.arm == "C"
                    else (
                        "direct per-node SO3 residual BA replacing restricted alpha"
                        if args.arm == "D"
                        else (
                            "direct pose BA plus GT-free joint shared-center initialization"
                            if bool(args.joint_center_initialization)
                            else "direct SO3 BA plus public-spread-bounded per-PTZ shared-center residual"
                        )
                    )
                )
            ),
            "config": {
                "window_frames": 5,
                "support_reprojection_error_px": 2.5,
                "pnp_reprojection_error_px": 4.0,
                "min_pnp_inliers": 12,
                "max_points_per_node": 150,
                "PTZ_pair_min_shared_ids": 8,
                "joint_rotation_max_evaluations": 5,
                "fit_residual_mode": (
                    None if args.arm in DIRECT_ARMS else str(args.fit_residual_mode)
                ),
                "center_path_mode": (
                    None if args.arm in DIRECT_ARMS else str(args.center_path_mode)
                ),
                "parallel_observation_workers": int(args.workers),
                "observation_executor": str(args.observation_executor),
                "center_policy": "DYNAMIC_TRACK_static_pnp_confidence_weighted_per_physical_PTZ",
                "center_trust_radii": (
                    center_trust_radii if args.arm == "E" else None
                ),
                "center_trust_radius_policy": (
                    "median_public_Static_PnP_center_distance_to_fixed_center"
                    if args.arm == "E"
                    else None
                ),
                "joint_center_initialization": (
                    bool(args.joint_center_initialization)
                    if args.arm == "E"
                    else None
                ),
                "commit": "whole_spatiotemporal_component_or_exact_Static_fallback",
                "direct_temporal_fit_observations": (
                    bool(args.direct_temporal_fit)
                    if args.arm in DIRECT_ARMS
                    else None
                ),
                "high_confidence_node_protection": (
                    {
                        "enabled": True,
                        "scope": "each_frame_within_public_PTZ_component",
                        "protected_count": "ceil(component_size/2)",
                        "ordering": "descending_Static+PnP_confidence_then_view_id",
                        "protected_pose": "exact_Static+PnP",
                    }
                    if bool(args.protect_high_confidence_half)
                    else None
                ),
                "direct_verifier_policy": (
                    {
                        "PTZ_pair": "strict_pooled_pareto_improvement",
                        "independent_static": (
                            "zero_margin_one_sided_exact_sign_test_noninferiority"
                        ),
                        "temporal": (
                            "zero_margin_one_sided_exact_sign_test_noninferiority"
                        ),
                        "minimum_observations": 12,
                        "significance_level": 0.05,
                    }
                    if bool(args.use_noninferiority_guards)
                    else "strict_all_geometry_improvement"
                    if args.arm in DIRECT_ARMS
                    else None
                ),
                "direct_rotation_BA": (
                    {
                        "max_rotation_correction_deg": 5.0,
                        "max_nfev": int(args.direct_max_nfev),
                        "robust_loss": "huber",
                        "robust_scale_px": 3.0,
                        "reprojection_weight": float(
                            args.direct_reprojection_weight
                        ),
                        "epipolar_weight": float(args.direct_epipolar_weight),
                        "x_scale": str(args.direct_x_scale),
                        "convergence_tolerance": float(
                            args.direct_convergence_tolerance
                        ),
                        "signed_2D_reprojection": True,
                        "signed_normalized_epipolar": True,
                        "jac_sparsity": True,
                        "endpoint_required": False,
                    }
                    if args.arm in DIRECT_ARMS
                    else None
                ),
                "fixed_static_RoMa_fit": (
                    {
                        "cache_protocol": ROMA_STATIC_CACHE_PROTOCOL,
                        "roles_used_by_candidate": ["fit"],
                        "confirm_final_used_by_candidate": False,
                        "fixed_public_static_cameras": True,
                    }
                    if roma_static_epipolar_cache_input is not None
                    else None
                ),
            },
            "code": {
                "entrypoint": Path(__file__),
                "entrypoint_sha256": _sha256(Path(__file__)),
                "multi_ptz_module_sha256": _sha256(
                    PROJECT_ROOT / "track4d_calib/same_time_multi_ptz_closure.py"
                ),
                "direct_rotation_BA_module_sha256": (
                    _sha256(
                        PROJECT_ROOT / "track4d_calib/direct_multi_ptz_rotation_ba.py"
                    )
                    if args.arm in DIRECT_ARMS
                    else None
                ),
            },
            "selection_inputs": {
                "GT_used": False,
                "NVS_used": False,
                "test_RGB_or_cameras_used": False,
                "observation_cache": (
                    None
                    if args.observation_cache is None
                    else {
                        "path": Path(args.observation_cache).resolve(),
                        "sha256": _sha256(Path(args.observation_cache).resolve()),
                    }
                ),
                "independent_spatial_observation_cache": (
                    None
                    if args.spatial_observation_cache is None
                    else {
                        "path": Path(args.spatial_observation_cache).resolve(),
                        "sha256": _sha256(
                            Path(args.spatial_observation_cache).resolve()
                        ),
                        "required_protocol": DIRECT_STATIC_CACHE_PROTOCOL,
                    }
                ),
                "roma_static_epipolar_cache": (
                    None
                    if roma_static_epipolar_cache_input is None
                    else {
                        "path": roma_static_epipolar_cache_input,
                        "sha256": _sha256(roma_static_epipolar_cache_input),
                        "required_protocol": ROMA_STATIC_CACHE_PROTOCOL,
                    }
                ),
            },
            "spatial_track_files_sha256": spatial_track_hashes,
            "temporal_track_files_sha256": temporal_track_hashes,
        },
    )

    spatial_observations: dict[tuple[str, int, str], dict[str, Any]] = {}
    spatial_diagnostics: list[dict[str, Any]] = []
    temporal_observations: dict[tuple[str, int, str], dict[str, Any]] = {}
    spatial_seconds = 0.0
    temporal_seconds = 0.0
    observation_cache_input = (
        Path(args.observation_cache).resolve()
        if args.observation_cache is not None
        else None
    )
    spatial_observation_cache_input = (
        Path(args.spatial_observation_cache).resolve()
        if args.spatial_observation_cache is not None
        else None
    )
    if observation_cache_input is not None:
        with observation_cache_input.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("spatial_track_files_sha256") != spatial_track_hashes:
            raise RuntimeError("observation cache spatial-track hashes differ")
        if cached.get("temporal_track_files_sha256") != temporal_track_hashes:
            raise RuntimeError("observation cache temporal-track hashes differ")
        spatial_observations = cached["spatial_observations"]
        temporal_observations = cached["temporal_observations"]
        required_spatial = {
            (view, frame, role)
            for frame in frames
            for view in TARGET_VIEWS
            for role in (("confirm", "final") if args.arm == "B" else ("fit", "confirm", "final"))
        }
        if required_spatial - set(spatial_observations):
            raise RuntimeError("observation cache lacks required spatial observations")
        if args.arm in TEMPORAL_OBSERVATION_ARMS:
            required_temporal = {
                (view, frame, role)
                for frame in frames
                for view in TARGET_VIEWS
                for role in ("fit", "confirm", "final")
            }
            if required_temporal - set(temporal_observations):
                raise RuntimeError("observation cache lacks required temporal observations")
    else:
        spatial_roles = (
            ("confirm", "final")
            if args.arm == "B"
            else ("fit", "confirm", "final")
        )
        spatial_jobs = [
            (view, frame, role)
            for frame in frames
            for view in TARGET_VIEWS
            for role in spatial_roles
        ]
        _SPATIAL_OBSERVATION_CONTEXT = {
            "bundles": spatial_bundles,
            "support_cameras": support_cameras,
            "config": config,
        }
        spatial_started = time.perf_counter()
        with _observation_executor(str(args.observation_executor), int(args.workers)) as executor:
            for view, frame, role, observation, diagnostic in executor.map(
                _build_spatial_job_from_context, spatial_jobs
            ):
                spatial_observations[(view, frame, role)] = observation
                spatial_diagnostics.append(
                    {"view": view, "frame": frame, "role": role, **diagnostic}
                )
        spatial_seconds = time.perf_counter() - spatial_started
        _SPATIAL_OBSERVATION_CONTEXT = None

        temporal_seed_axis = seed_frame_by_point(
            temporal_bundle["meta"], np.asarray(temporal_bundle["tracks"]).shape[2]
        )
        if args.arm in TEMPORAL_OBSERVATION_ARMS:
            _TEMPORAL_OBSERVATION_CONTEXT = {
                "bundle": temporal_bundle,
                "support_cameras": support_cameras,
                "config": config,
                "seed_axis": temporal_seed_axis,
                "frames": frames,
                "base_grid": base_grid,
            }
            temporal_started = time.perf_counter()
            with _observation_executor(str(args.observation_executor), int(args.workers)) as executor:
                for view, split in executor.map(
                    _build_temporal_view_from_context, TARGET_VIEWS
                ):
                    for role in ("fit", "confirm", "final"):
                        for row in split[role]:
                            temporal_observations[(view, int(row["frame"]), role)] = row
            temporal_seconds = time.perf_counter() - temporal_started
            _TEMPORAL_OBSERVATION_CONTEXT = None

    required_direct_spatial = {
        (view, frame, role)
        for frame in frames
        for view in TARGET_VIEWS
        for role in ("fit", "confirm", "final")
    }
    if spatial_observation_cache_input is not None:
        with spatial_observation_cache_input.open("rb") as handle:
            independent_spatial_cache = pickle.load(handle)
        spatial_observations = _validated_direct_static_observations(
            independent_spatial_cache,
            required_keys=required_direct_spatial,
        )
    if args.arm in DIRECT_ARMS and required_direct_spatial - set(spatial_observations):
        raise RuntimeError("spatial observations lack direct fit/confirm/final roles")

    observation_cache_sha256 = (
        _sha256(observation_cache_input)
        if observation_cache_input is not None
        else None
    )
    if args.arm in DIRECT_ARMS and observation_cache_input is None:
        cache_path = output / "OBSERVATION_CACHE.pkl"
        temporary_cache = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
        with temporary_cache.open("wb") as handle:
            pickle.dump(
                {
                    "protocol": method_protocol,
                    "spatial_observations": spatial_observations,
                    "temporal_observations": temporal_observations,
                    "spatial_track_files_sha256": spatial_track_hashes,
                    "temporal_track_files_sha256": temporal_track_hashes,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temporary_cache, cache_path)
        observation_cache_sha256 = _sha256(cache_path)

    component_started = time.perf_counter()
    selected_grid = {frame: dict(base_grid[frame]) for frame in frames}
    diagnostic_candidate_grid = {
        frame: dict(base_grid[frame]) for frame in frames
    }
    component_rows: list[dict[str, Any]] = []
    if args.arm in DIRECT_ARMS:
        direct_jobs: list[tuple[int, int, list[int], list[str]]] = []
        for component_index, start in enumerate(range(0, len(frames), 5)):
            component_frames = list(range(start, start + 5))
            visibility_rows = [
                np.asarray(spatial_bundles[frame]["visibility"], dtype=bool)[:, 0]
                for frame in component_frames
            ]
            graph = target_components_from_window_visibility(
                visibility_by_frame=visibility_rows,
                view_ids=STATIC_VIEWS + TARGET_VIEWS,
                target_views=TARGET_VIEWS,
                min_shared_points=8,
            )
            direct_jobs.extend(
                (
                    int(component_index),
                    int(spatial_index),
                    component_frames,
                    list(component),
                )
                for spatial_index, component in enumerate(graph["components"])
            )
        _DIRECT_COMPONENT_CONTEXT = {
            "base_grid": base_grid,
            "fixed_centers": fixed_centers,
            "spatial_observations": spatial_observations,
            "temporal_observations": temporal_observations,
            "spatial_bundles": spatial_bundles,
            "confidence": confidence,
            "optimize_shared_center": args.arm == "E",
            "center_trust_radii": center_trust_radii,
            "use_joint_center_initialization": bool(
                args.joint_center_initialization
            ),
            "direct_max_nfev": int(args.direct_max_nfev),
            "direct_reprojection_weight": float(
                args.direct_reprojection_weight
            ),
            "direct_epipolar_weight": float(args.direct_epipolar_weight),
            "direct_x_scale": str(args.direct_x_scale),
            "direct_convergence_tolerance": float(
                args.direct_convergence_tolerance
            ),
            "use_temporal_fit_observations": bool(args.direct_temporal_fit),
            "protect_high_confidence_half": bool(
                args.protect_high_confidence_half
            ),
            "use_noninferiority_guards": bool(args.use_noninferiority_guards),
            "fixed_static_epipolar_observations": (
                roma_static_epipolar_observations
            ),
            "support_cameras": support_cameras,
        }
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(int(args.workers), max(len(direct_jobs), 1)),
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            for row, selected, diagnostic_candidate in executor.map(
                _evaluate_direct_component_from_context, direct_jobs
            ):
                component_rows.append(row)
                for (frame, view), camera in selected.items():
                    selected_grid[int(frame)][str(view)] = camera
                for (frame, view), camera in diagnostic_candidate.items():
                    diagnostic_candidate_grid[int(frame)][str(view)] = camera
        _DIRECT_COMPONENT_CONTEXT = None
    for component_index, start in enumerate(range(0, len(frames), 5)):
        component_frames = list(range(start, start + 5))
        visibility_rows = [
            np.asarray(spatial_bundles[frame]["visibility"], dtype=bool)[:, 0]
            for frame in component_frames
        ]
        graph = target_components_from_window_visibility(
            visibility_by_frame=visibility_rows,
            view_ids=STATIC_VIEWS + TARGET_VIEWS,
            target_views=TARGET_VIEWS,
            min_shared_points=8,
        )
        for spatial_index, component in enumerate(graph["components"]):
            if args.arm in DIRECT_ARMS:
                continue
            if len(component) < 2:
                component_rows.append(
                    {
                        "window": component_frames,
                        "component": component,
                        "commit": False,
                        "reason": "singleton_has_no_PTZ_pair_geometry",
                    }
                )
                continue
            if args.arm == "B":
                candidate = {
                    frame: {
                        view: dynamic_track_grid[frame][view]
                        for view in TARGET_VIEWS
                    }
                    for frame in component_frames
                }
                changed = [
                    (frame, view)
                    for frame in component_frames
                    for view in component
                    if not _camera_equal(base_grid[frame][view], candidate[frame][view])
                ]
                fit_result = {
                    "success": bool(changed),
                    "reason": "inherited_DYNAMIC_TRACK_candidate" if changed else "no_DYNAMIC_TRACK_change",
                    "changed_nodes": [f"{frame}:{view}" for frame, view in changed],
                }
            elif args.arm == "C":
                candidate, fit_result = _fit_multi_ptz_candidate(
                    component=component,
                    frames=component_frames,
                    base_grid=base_grid,
                    fixed_centers=fixed_centers,
                    fit_observations=spatial_observations,
                    temporal_observations=temporal_observations,
                    bundles=spatial_bundles,
                    fit_residual_mode=str(args.fit_residual_mode),
                    center_path_mode=str(args.center_path_mode),
                )
            else:
                candidate, fit_result = _fit_direct_multi_ptz_candidate(
                    component=component,
                    frames=component_frames,
                    base_grid=base_grid,
                    fixed_centers=fixed_centers,
                    fit_observations=spatial_observations,
                    temporal_observations=temporal_observations,
                    bundles=spatial_bundles,
                    confidence=confidence,
                )
                changed = [
                    (frame, view)
                    for frame in component_frames
                    for view in component
                    if not _camera_equal(base_grid[frame][view], candidate[frame][view])
                ]

            if not bool(fit_result.get("success", False)) or not changed:
                component_rows.append(
                    {
                        "window": component_frames,
                        "component": component,
                        "commit": False,
                        "reason": fit_result.get("reason", "candidate_failed"),
                        "candidate": fit_result,
                    }
                )
                continue
            confirm = _spatial_errors(
                bundles=spatial_bundles,
                observations=spatial_observations,
                component=component,
                frames=component_frames,
                role="confirm",
                base_grid=base_grid,
                candidate_grid=candidate,
            )
            final = _spatial_errors(
                bundles=spatial_bundles,
                observations=spatial_observations,
                component=component,
                frames=component_frames,
                role="final",
                base_grid=base_grid,
                candidate_grid=candidate,
            )
            temporal_confirm = (
                {"commit": True, "reason": "inherited_DYNAMIC_TRACK_temporal_verification"}
                if args.arm == "B"
                else _temporal_score(
                    observations=temporal_observations,
                    component=component,
                    frames=component_frames,
                    role="confirm",
                    base_grid=base_grid,
                    candidate_grid=candidate,
                )
            )
            temporal_final = (
                {"commit": True, "reason": "inherited_DYNAMIC_TRACK_temporal_verification"}
                if args.arm == "B"
                else _temporal_score(
                    observations=temporal_observations,
                    component=component,
                    frames=component_frames,
                    role="final",
                    base_grid=base_grid,
                    candidate_grid=candidate,
                )
            )
            commit = bool(
                confirm.get("commit")
                and final.get("commit")
                and temporal_confirm.get("commit")
                and temporal_final.get("commit")
            )
            if commit:
                for frame, view in changed:
                    selected_grid[frame][view] = candidate[frame][view]
            component_rows.append(
                {
                    "component_index": component_index,
                    "spatial_component_index": spatial_index,
                    "window": component_frames,
                    "component": component,
                    "changed_nodes": [f"{frame}:{view}" for frame, view in changed],
                    "candidate": fit_result,
                    "confirm": confirm,
                    "final": final,
                    "temporal_confirm": temporal_confirm,
                    "temporal_final": temporal_final,
                    "commit": commit,
                    "reason": (
                        "all_independent_geometry_improved"
                        if commit
                        else "independent_geometry_rejected"
                    ),
                }
            )
    component_seconds = time.perf_counter() - component_started

    pose_root = output / f"pose_arm_{args.arm}"
    committed_nodes = set()
    for row in component_rows:
        if row.get("commit"):
            committed_nodes.update(row.get("changed_nodes", []))
    fallback_total = 0
    fallback_equal = 0
    for frame in frames:
        destination = pose_root / f"f{frame:06d}" / "cameras_pred"
        save_cameras(destination, selected_grid[frame])
        replay = CameraLoader(destination).load_cameras(TARGET_VIEWS)
        for view in TARGET_VIEWS:
            if f"{frame}:{view}" in committed_nodes:
                continue
            fallback_total += 1
            fallback_equal += int(_camera_equal(replay[view], base_grid[frame][view]))
    if fallback_equal != fallback_total:
        raise RuntimeError("exact Static+PnP fallback audit failed")

    diagnostic_candidate_pose_root = None
    if bool(args.export_uncommitted_candidates):
        if args.arm not in DIRECT_ARMS:
            raise ValueError("diagnostic candidate export is only valid for direct arms")
        diagnostic_candidate_pose_root = output / "diagnostic_uncommitted_candidate_pose"
        for frame in frames:
            save_cameras(
                diagnostic_candidate_pose_root
                / f"f{frame:06d}"
                / "cameras_pred",
                diagnostic_candidate_grid[frame],
            )

    elapsed = time.perf_counter() - started
    summary = {
        "protocol": method_protocol,
        "arm": args.arm,
        "status": "complete_public_geometry",
        "pose_root": pose_root,
        "diagnostic_uncommitted_candidate_pose_root": (
            diagnostic_candidate_pose_root
        ),
        "component_count": int(len(component_rows)),
        "candidate_success_components": int(
            sum(bool(row.get("candidate", {}).get("success")) for row in component_rows)
        ),
        "commit_components": int(sum(bool(row.get("commit")) for row in component_rows)),
        "committed_nodes": int(len(committed_nodes)),
        "fallback_nodes": int(fallback_total),
        "exact_fallback_nodes": int(fallback_equal),
        "runtime_seconds": float(elapsed),
        "seconds_per_frame_all_PTZ": float(elapsed / 30.0),
        "parallel_observation_workers": int(args.workers),
        "observation_executor": str(args.observation_executor),
        "observation_cache_sha256": observation_cache_sha256,
        "observation_cache_input": observation_cache_input,
        "independent_spatial_observation_cache": spatial_observation_cache_input,
        "independent_spatial_observation_cache_sha256": (
            _sha256(spatial_observation_cache_input)
            if spatial_observation_cache_input is not None
            else None
        ),
        "direct_temporal_fit_observations": (
            bool(args.direct_temporal_fit) if args.arm in DIRECT_ARMS else None
        ),
        "use_noninferiority_guards": bool(args.use_noninferiority_guards),
        "roma_static_epipolar_cache": roma_static_epipolar_cache_input,
        "roma_static_epipolar_cache_sha256": (
            _sha256(roma_static_epipolar_cache_input)
            if roma_static_epipolar_cache_input is not None
            else None
        ),
        "roma_static_match_runtime": roma_static_cache_runtime,
        "calibration_seconds_per_frame_including_roma_inference": (
            float(elapsed / 30.0)
            + float(
                (roma_static_cache_runtime or {}).get(
                    "single_model_equivalent_inference_seconds_per_frame", 0.0
                )
            )
        ),
        "stage_runtime_seconds": {
            "spatial_observations": float(spatial_seconds),
            "temporal_observations": float(temporal_seconds),
            "component_fit_verify_commit": float(component_seconds),
        },
        "components": component_rows,
        "spatial_geometry": {
            "triangulated_count": [row["triangulated_count"] for row in spatial_diagnostics],
            "diagnostics": spatial_diagnostics,
        },
        "fixed_centers": fixed_centers,
        "center_trust_radii": center_trust_radii if args.arm == "E" else None,
        "lineage": {
            "runtime_scene": runtime,
            "runtime_manifest_sha256": _sha256(runtime / "runtime_manifest.json"),
            "base_pose_root": Path(args.base_pose_root).resolve(),
            "dynamic_track_pose_root": Path(args.dynamic_track_pose_root).resolve(),
            "spatial_track_root": Path(args.spatial_track_root).resolve(),
            "spatial_track_files_sha256": spatial_track_hashes,
            "temporal_track_root": Path(args.temporal_track_root).resolve(),
            "temporal_track_files_sha256": temporal_track_hashes,
            "independent_spatial_observation_cache": (
                spatial_observation_cache_input
            ),
            "independent_spatial_observation_cache_sha256": (
                _sha256(spatial_observation_cache_input)
                if spatial_observation_cache_input is not None
                else None
            ),
            "roma_static_epipolar_cache": roma_static_epipolar_cache_input,
            "roma_static_epipolar_cache_sha256": (
                _sha256(roma_static_epipolar_cache_input)
                if roma_static_epipolar_cache_input is not None
                else None
            ),
            "static_pnp_confidence": confidence_audit,
            "entrypoint_sha256": _sha256(Path(__file__)),
        },
        "attestation": {
            "GT_used": False,
            "NVS_used": False,
            "test_RGB_or_cameras_used": False,
            "fit_confirm_final_disjoint": True,
            "roma_fit_only_and_independent_verifier": bool(
                roma_static_epipolar_cache_input is None
                or args.spatial_observation_cache is not None
            ),
            "different_PTZ_centers_independent": True,
            "independent_static_cache_required_by_guard": bool(
                not args.use_noninferiority_guards
                or spatial_observation_cache_input is not None
            ),
            "exact_fallback": fallback_total == fallback_equal,
            "diagnostic_candidate_pose_is_not_deployable": bool(
                args.export_uncommitted_candidates
            ),
        },
    }
    _write_json(output / "PUBLIC_GEOMETRY_AUDIT.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["B", "C", "D", "E"], required=True)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Contiguous local frame count; defaults to the original 30-frame protocol.",
    )
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--base-pose-root", type=Path, required=True)
    parser.add_argument("--dynamic_track-pose-root", type=Path, required=True)
    parser.add_argument("--spatial-track-root", type=Path, required=True)
    parser.add_argument("--temporal-track-root", type=Path, required=True)
    parser.add_argument("--static-pnp-diagnostics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--observation-cache",
        type=Path,
        help="Reuse a frozen observation cache after exact track-hash validation.",
    )
    parser.add_argument(
        "--spatial-observation-cache",
        type=Path,
        help=(
            "Replace target-static observations with a frozen independent "
            "appended-static cache after protocol and prefix-attestation checks."
        ),
    )
    parser.add_argument(
        "--roma-static-epipolar-cache",
        type=Path,
        help=(
            "Append frozen fit-only public-static RoMa epipolar edges to direct BA; "
            "the existing independent confirm/final verifier remains unchanged."
        ),
    )
    parser.add_argument(
        "--protect-high-confidence-half",
        action="store_true",
        help=(
            "Keep the higher-confidence half of each frame/component exactly at "
            "Static+PnP while optimizing the lower-confidence nodes."
        ),
    )
    parser.add_argument(
        "--use-noninferiority-guards",
        action="store_true",
        help=(
            "Keep PTZ-pair verification strict while guarding independent static "
            "and temporal heldout evidence against significant degradation."
        ),
    )
    parser.add_argument(
        "--no-direct-temporal-fit",
        dest="direct_temporal_fit",
        action="store_false",
        help="Exclude temporal reprojection observations from direct candidate fit.",
    )
    parser.set_defaults(direct_temporal_fit=True)
    parser.add_argument("--direct-max-nfev", type=int, default=10)
    parser.add_argument("--direct-reprojection-weight", type=float, default=1.0)
    parser.add_argument("--direct-epipolar-weight", type=float, default=1.0)
    parser.add_argument(
        "--direct-x-scale",
        choices=("unit", "jac"),
        default="unit",
        help="Trust-region parameter scaling; unit preserves the frozen default.",
    )
    parser.add_argument(
        "--direct-convergence-tolerance",
        type=float,
        default=1e-8,
        help="Shared ftol/xtol/gtol for the direct least-squares solver.",
    )
    parser.add_argument(
        "--export-uncommitted-candidates",
        action="store_true",
        help=(
            "Export converged public fit candidates for post-freeze diagnostic "
            "pose evaluation; these poses are never used as committed output."
        ),
    )
    parser.add_argument(
        "--joint-center-initialization",
        action="store_true",
        help="Use GT-free joint shared-center PnP only as a direct-BA start.",
    )
    parser.add_argument(
        "--observation-executor",
        choices=("thread", "process"),
        default="thread",
    )
    parser.add_argument(
        "--fit-residual-mode",
        choices=("clipped", "huber_raw"),
        default="clipped",
        help="Candidate fit residual preparation; verification remains unchanged.",
    )
    parser.add_argument(
        "--center-path-mode",
        choices=("fixed", "interpolate"),
        default="fixed",
        help="Whether alpha also interpolates from the Static center.",
    )
    summary = run(parser.parse_args())
    print(json.dumps(_jsonable({key: value for key, value in summary.items() if key != "components"}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
