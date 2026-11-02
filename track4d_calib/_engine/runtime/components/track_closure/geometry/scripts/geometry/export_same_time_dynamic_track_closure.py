#!/usr/bin/env python3
"""Repair PTZ trajectories with frame-local 3D from public synchronized tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader, CameraParams, save_cameras
from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    triangulate_same_time_multiview_points,
)
from track4d_calib.track_joint_ptz_bundle import (
    _geometric_median,
    fit_confirm_refit_verify_joint_ptz_shared_center,
    solve_joint_ptz_shared_center,
    split_joint_ptz_track_groups,
)
from scripts.geometry.relay_manifest import (
    load_relay_manifests,
)
from scripts.geometry.export_static_pnp_inlier_seeded_track_closure import (
    _bundle_hashes,
    _load_bundle,
)
from scripts.geometry.run_track_closure import (
    build_node_observations,
)


PROTOCOL = "same-time-dynamic-track-shared-center-closure-v1"
TIME_POINT_STRIDE = 1_000_000
FORBIDDEN_PUBLIC_TOKENS = ("private", "gt_long", "hidden_parent", "heldout")


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
            if str(key) != "cameras"
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def seed_frame_by_point(meta: Mapping[str, Any], point_count: int) -> np.ndarray:
    seeds = [int(value) for value in meta.get("seed_frames", [])]
    counts = [int(value) for value in meta.get("num_points_by_seed", [])]
    if not seeds or len(seeds) != len(counts) or sum(counts) != int(point_count):
        raise ValueError("track bundle seed blocks do not cover the point axis")
    if seeds != sorted(set(seeds)) or any(count <= 0 for count in counts):
        raise ValueError("track bundle seed schedule is invalid")
    return np.repeat(np.asarray(seeds, dtype=np.int64), counts)


def point_ids_in_frame_window(
    seed_frames: np.ndarray,
    *,
    frame: int,
    radius: int,
) -> np.ndarray:
    values = np.asarray(seed_frames, dtype=np.int64).reshape(-1)
    if int(radius) < 0:
        raise ValueError("window radius must be non-negative")
    return np.flatnonzero(np.abs(values - int(frame)) <= int(radius)).astype(
        np.int64
    )


def make_time_local_observation(
    *,
    frame: int,
    physical_track_ids: np.ndarray,
    xyz: np.ndarray,
    image_points: np.ndarray,
) -> dict[str, Any]:
    physical = np.asarray(physical_track_ids, dtype=np.int64).reshape(-1)
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(physical) != len(points) or len(physical) != len(pixels):
        raise ValueError("track IDs, XYZ, and image points must have equal lengths")
    if np.any(physical < 0) or np.any(physical >= TIME_POINT_STRIDE):
        raise ValueError("physical track ID exceeds the time-local namespace")
    time_local_ids = (int(frame) + 1) * TIME_POINT_STRIDE + physical
    return {
        "frame": int(frame),
        "point_ids": time_local_ids,
        "validation_group_ids": physical.copy(),
        "xyz": points,
        "image_points": pixels,
    }


def _attest_public_bundle(
    *,
    bundle: Mapping[str, Any],
    runtime_scene: Path,
    static_views: Sequence[str],
    target_views: Sequence[str],
    frames: Sequence[int],
) -> dict[str, Any]:
    tracks = np.asarray(bundle["tracks"])
    visibility = np.asarray(bundle["visibility"])
    quality = bundle.get("track_quality")
    views = [str(view) for view in bundle["views"]]
    expected = [str(view) for view in static_views] + [
        str(view) for view in target_views
    ]
    if views != expected:
        raise ValueError("track bundle view axis differs from the frozen public split")
    if any(view in {"80", "85", "90", "95"} for view in views):
        raise ValueError("held-out test view leaked into the method track bundle")
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError("tracks must have shape V,T,N,2")
    if visibility.shape != tracks.shape[:-1]:
        raise ValueError("visibility shape differs from tracks")
    if quality is not None and np.asarray(quality).shape != visibility.shape:
        raise ValueError("track quality shape differs from visibility")
    if (
        tracks.shape[0] != len(views)
        or not frames
        or min(frames) < 0
        or max(frames) >= tracks.shape[1]
    ):
        raise ValueError("track bundle view/frame axes differ from the protocol")

    public_root = runtime_scene.resolve()
    path_checks: dict[str, bool] = {}
    for key in ("data_root", "images_root", "masks_root", "cameras_root"):
        raw = bundle["meta"].get(key)
        if raw is None:
            path_checks[key] = False
            continue
        resolved = Path(str(raw)).resolve()
        text = str(resolved).lower()
        path_checks[key] = bool(
            (resolved == public_root or public_root in resolved.parents)
            and not any(token in text for token in FORBIDDEN_PUBLIC_TOKENS)
        )
    if not all(path_checks.values()):
        raise ValueError("track bundle references a non-public method input")
    return {
        "views_exact": True,
        "heldout_views_absent": True,
        "path_checks": path_checks,
        "shape": list(tracks.shape),
        "tracker_backend": bundle["meta"].get("tracker_backend"),
        "mask_mode": bundle["meta"].get("mask_mode"),
    }


def build_time_local_observation(
    *,
    bundle: Mapping[str, Any],
    support_cameras: Mapping[str, CameraParams],
    static_views: Sequence[str],
    target_view: str,
    frame: int,
    allowed_point_ids: np.ndarray,
    config: SameTimePnPConfig,
    max_points: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support, target, raw = build_node_observations(
        tracks=bundle["tracks"],
        visibility=bundle["visibility"],
        track_quality=bundle.get("track_quality"),
        view_ids=bundle["views"],
        static_views=static_views,
        target_view=str(target_view),
        frame_local=int(frame),
        min_support_views=int(config.min_support_views),
        max_points=int(max_points),
        allowed_point_ids=np.asarray(allowed_point_ids, dtype=np.int64),
    )
    triangulated = triangulate_same_time_multiview_points(
        support_cameras=support_cameras,
        support_observations=support,
        target_observations=target,
        config=config,
    )
    physical_ids = np.asarray(triangulated["point_ids"], dtype=np.int64)
    observation = make_time_local_observation(
        frame=int(frame),
        physical_track_ids=physical_ids,
        xyz=np.asarray(triangulated["object_points"], dtype=np.float64),
        image_points=np.asarray(triangulated["image_points"], dtype=np.float64),
    )
    rows = list(triangulated.get("triangulation_rows", []))
    medians = np.asarray(
        [row["support_reprojection_median_px"] for row in rows],
        dtype=np.float64,
    )
    p90s = np.asarray(
        [row["support_reprojection_p90_px"] for row in rows],
        dtype=np.float64,
    )
    angles = np.asarray(
        [row["max_triangulation_angle_deg"] for row in rows],
        dtype=np.float64,
    )
    diagnostic = {
        "frame": int(frame),
        "view": str(target_view),
        "window_allowed_id_count": int(len(allowed_point_ids)),
        **raw,
        "triangulated_count": int(triangulated["triangulated_count"]),
        "support_observation_count": int(
            triangulated["support_observation_count"]
        ),
        "support_inlier_count": int(
            triangulated["support_observation_inlier_count"]
        ),
        "support_reprojection_median_px": (
            float(np.median(medians)) if len(medians) else None
        ),
        "support_reprojection_p90_px": (
            float(np.percentile(p90s, 90.0)) if len(p90s) else None
        ),
        "triangulation_angle_median_deg": (
            float(np.median(angles)) if len(angles) else None
        ),
        "triangulation_angle_p10_deg": (
            float(np.percentile(angles, 10.0)) if len(angles) else None
        ),
        "xyz_identity": "frame_local_X(physical_track_id,time)",
    }
    return observation, diagnostic


def _camera_equal(first: CameraParams, second: CameraParams) -> bool:
    return bool(
        first.view_id == second.view_id
        and first.H == second.H
        and first.W == second.W
        and np.array_equal(first.K, second.K)
        and np.array_equal(first.R, second.R)
        and np.array_equal(first.T, second.T)
        and np.array_equal(first.D, second.D)
    )


def _summarize_counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([int(row[key]) for row in rows], dtype=np.int64)
    return {
        "min": int(np.min(values)) if len(values) else 0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "max": int(np.max(values)) if len(values) else 0,
    }


def weighted_geometric_median(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    max_iterations: int = 64,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    confidence = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("points must have shape (N, 3) with N > 0")
    if confidence.shape != (len(values),):
        raise ValueError("weights must have shape (N,)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(confidence)):
        raise ValueError("points and weights must be finite")
    if np.any(confidence <= 0.0):
        raise ValueError("weights must be positive")
    estimate = np.average(values, axis=0, weights=confidence)
    for _iteration in range(int(max_iterations)):
        distances = np.linalg.norm(values - estimate.reshape(1, 3), axis=1)
        coincident = distances < 1e-12
        if np.any(coincident):
            return values[np.flatnonzero(coincident)[0]].copy()
        local_weights = confidence / distances
        updated = np.sum(values * local_weights[:, None], axis=0) / np.sum(
            local_weights
        )
        if np.linalg.norm(updated - estimate) < 1e-12:
            return updated
        estimate = updated
    return estimate


def load_static_pnp_confidence(
    path: Path,
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
) -> tuple[dict[tuple[int, str], float], dict[str, Any]]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("gt_used") is not False or payload.get("nvs_used") is not False:
        raise ValueError("Static+PnP diagnostics are not sealed GT/NVS-free")
    expected = {
        (int(frame), str(view)) for frame in frames for view in target_views
    }
    scores: dict[tuple[int, str], float] = {}
    for row in payload.get("rows", []):
        key = (int(row["frame"]), str(row["view"]))
        if key not in expected:
            continue
        if key in scores:
            raise ValueError(f"duplicate Static+PnP diagnostic node: {key}")
        solver = row.get("solver", {})
        ratio = float(solver.get("inlier_ratio", float("nan")))
        median = float(solver.get("reprojection_median_px", float("nan")))
        if (
            not bool(solver.get("success", False))
            or not np.isfinite(ratio)
            or not np.isfinite(median)
            or ratio <= 0.0
            or median <= 0.0
        ):
            raise ValueError(f"invalid Static+PnP confidence node: {key}")
        scores[key] = ratio / median
    if set(scores) != expected:
        raise ValueError("Static+PnP diagnostics do not cover the requested nodes")
    return scores, {
        "path": source,
        "sha256": _sha256(source),
        "node_count": int(len(scores)),
        "weight": "inlier_ratio_divided_by_reprojection_median_px",
        "new_tunable_numeric_hyperparameters": 0,
    }


def contiguous_frame_components(
    frames: Sequence[int], *, component_size: int
) -> list[list[int]]:
    ordered = [int(frame) for frame in frames]
    size = int(component_size)
    if size <= 0:
        raise ValueError("component_size must be positive")
    if not ordered or ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ValueError("frames must be contiguous")
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def estimate_trajectory_fit_shared_center(
    *,
    observations: Sequence[Mapping[str, Any]],
    initial_cameras: Mapping[int, CameraParams],
    reprojection_error_px: float,
    min_inliers_per_frame: int,
    min_successful_frames: int,
    rng_seed: int,
) -> dict[str, Any]:
    """Estimate one PTZ center from full-trajectory fit IDs only."""

    split = split_joint_ptz_track_groups(
        observations=observations,
        initial_cameras=initial_cameras,
    )
    if not bool(split["group_disjoint"]):
        return {
            "success": False,
            "reason": "trajectory_track_group_split_not_disjoint",
        }
    fit = solve_joint_ptz_shared_center(
        observations=split["fit"],
        initial_cameras=initial_cameras,
        reprojection_error_px=float(reprojection_error_px),
        min_inliers_per_frame=int(min_inliers_per_frame),
        min_successful_frames=int(min_successful_frames),
        rng_seed=int(rng_seed),
    )
    diagnostics = {key: value for key, value in fit.items() if key != "cameras"}
    return {
        **diagnostics,
        "fit_only": True,
        "fit_group_count": int(len(split["role_group_ids"]["fit"])),
        "confirm_group_count": int(len(split["role_group_ids"]["confirm"])),
        "final_group_count": int(len(split["role_group_ids"]["final"])),
        "group_disjoint": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    lineage = load_relay_manifests(args.relay_manifests)
    frames = sorted({int(frame) for frame in args.frames})
    target_views = [str(view) for view in args.target_views]
    all_target_views = [str(view) for view in lineage["pose_root_by_view"]]
    static_views = [str(view) for view in lineage["static_views"]]
    if not frames or frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError("frames must form one contiguous interval")
    if not target_views or not set(target_views).issubset(set(all_target_views)):
        raise ValueError("target views are not a subset of the frozen Static+PnP lineage")
    if int(args.window_radius) != 2:
        raise ValueError("v1 freezes the five-frame window radius to two")
    if int(args.component_size) != 5:
        raise ValueError("v1 freezes the atomic component size to five frames")
    frame_components = contiguous_frame_components(
        frames, component_size=int(args.component_size)
    )

    bundle = _load_bundle(args.track_root.resolve())
    bundle_attestation = _attest_public_bundle(
        bundle=bundle,
        runtime_scene=lineage["runtime_scene"],
        static_views=static_views,
        target_views=all_target_views,
        frames=frames,
    )
    seed_axis = seed_frame_by_point(
        bundle["meta"], np.asarray(bundle["tracks"]).shape[2]
    )
    support_cameras = CameraLoader(
        lineage["runtime_scene"] / "cameras/static"
    ).load_cameras(static_views)
    initial_by_view: dict[str, dict[int, CameraParams]] = {}
    for view in target_views:
        initial_by_view[view] = {
            frame: CameraLoader(
                lineage["pose_root_by_view"][view]
                / f"f{frame:06d}"
                / "cameras_pred"
            ).load_camera(view)
            for frame in frames
        }

    static_pnp_confidence: dict[tuple[int, str], float] | None = None
    static_pnp_confidence_audit: dict[str, Any] | None = None
    if (
        str(args.candidate_center_policy)
        == "static_pnp_confidence_weighted_center_bounded_rotation"
    ):
        if args.static_pnp_diagnostics is None:
            raise ValueError(
                "confidence-weighted center requires --static-pnp-diagnostics"
            )
        static_pnp_confidence, static_pnp_confidence_audit = (
            load_static_pnp_confidence(
                args.static_pnp_diagnostics,
                frames=frames,
                target_views=target_views,
            )
        )
    elif args.static_pnp_diagnostics is not None:
        raise ValueError(
            "--static-pnp-diagnostics is only valid for confidence-weighted center"
        )

    config = lineage["config"]
    all_diagnostics: list[dict[str, Any]] = []
    result_by_view: dict[str, dict[str, Any]] = {}
    committed_cameras: dict[str, dict[int, CameraParams]] = {}
    committed_nodes: set[tuple[int, str]] = set()
    for view in target_views:
        observations: list[dict[str, Any]] = []
        for frame in frames:
            allowed = point_ids_in_frame_window(
                seed_axis, frame=frame, radius=int(args.window_radius)
            )
            observation, diagnostic = build_time_local_observation(
                bundle=bundle,
                support_cameras=support_cameras,
                static_views=static_views,
                target_view=view,
                frame=frame,
                allowed_point_ids=allowed,
                config=config,
                max_points=int(args.max_points),
            )
            observations.append(observation)
            all_diagnostics.append(diagnostic)

        coverage_frames = sum(
            len(np.asarray(row["point_ids"])) >= int(config.min_pnp_inliers)
            for row in observations
        )
        selected = dict(initial_by_view[view])
        trajectory_center_fit: dict[str, Any] | None = None
        if (
            str(args.candidate_center_policy)
            == "trajectory_fit_shared_center_bounded_rotation"
        ):
            trajectory_center_fit = estimate_trajectory_fit_shared_center(
                observations=observations,
                initial_cameras=initial_by_view[view],
                reprojection_error_px=float(config.pnp_reprojection_error_px),
                min_inliers_per_frame=int(config.min_pnp_inliers),
                min_successful_frames=int(args.min_coverage_frames),
                rng_seed=int(config.rng_seed),
            )
            fixed_center = (
                np.asarray(trajectory_center_fit["center"], dtype=np.float64)
                if bool(trajectory_center_fit.get("success", False))
                else None
            )
        elif (
            str(args.candidate_center_policy)
            == "static_pnp_confidence_weighted_center_bounded_rotation"
        ):
            assert static_pnp_confidence is not None
            fixed_center = weighted_geometric_median(
                np.asarray(
                    [
                        initial_by_view[view][frame].camera_center
                        for frame in frames
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [static_pnp_confidence[(frame, view)] for frame in frames],
                    dtype=np.float64,
                ),
            )
        else:
            fixed_center = _geometric_median(
                np.asarray(
                    [
                        initial_by_view[view][frame].camera_center
                        for frame in frames
                    ],
                    dtype=np.float64,
                )
            )
        component_rows: list[dict[str, Any]] = []
        observations_by_frame = {
            int(row["frame"]): row for row in observations
        }
        for component_index, component_frames in enumerate(frame_components):
            component_observations = [
                observations_by_frame[frame] for frame in component_frames
            ]
            component_initial = {
                frame: initial_by_view[view][frame] for frame in component_frames
            }
            component_coverage = sum(
                len(np.asarray(row["point_ids"])) >= int(config.min_pnp_inliers)
                for row in component_observations
            )
            if (
                coverage_frames < int(args.min_coverage_frames)
                or component_coverage < int(args.min_component_coverage_frames)
            ):
                result: dict[str, Any] = {
                    "success": False,
                    "reason": "public_time_local_coverage_gate_failed",
                }
            elif fixed_center is None:
                result = {
                    "success": False,
                    "reason": "trajectory_fit_shared_center_failed",
                }
            else:
                result = fit_confirm_refit_verify_joint_ptz_shared_center(
                    observations=component_observations,
                    initial_cameras=component_initial,
                    reprojection_error_px=float(config.pnp_reprojection_error_px),
                    min_inliers_per_frame=int(config.min_pnp_inliers),
                    min_successful_frames=int(args.min_component_coverage_frames),
                    rng_seed=int(config.rng_seed) + int(component_index) * 1000,
                    verification_policy="window_component_pooled_pareto",
                    candidate_center_policy=(
                        "static_pnp_median_bounded_rotation"
                        if str(args.candidate_center_policy)
                        in {
                            "trajectory_fit_shared_center_bounded_rotation",
                            "static_pnp_confidence_weighted_center_bounded_rotation",
                        }
                        else str(args.candidate_center_policy)
                    ),
                    fixed_center_override=(
                        fixed_center
                        if str(args.candidate_center_policy) != "joint_fit"
                        else None
                    ),
                )
            confirmation = result.get("confirmation", {})
            final = result.get("final_verification", {})
            commit = bool(
                result.get("success", False)
                and confirmation.get("commit", False)
                and final.get("commit", False)
            )
            if commit:
                for frame, camera in result["cameras"].items():
                    selected[int(frame)] = camera
                    committed_nodes.add((int(frame), view))
            component_rows.append(
                {
                    "component_index": int(component_index),
                    "frames": component_frames,
                    "coverage_frames": int(component_coverage),
                    "candidate_success": bool(result.get("success", False)),
                    "candidate_reason": str(result.get("reason", "unknown")),
                    "commit": commit,
                    "commit_reason": (
                        "confirm_and_final_geometry_improved"
                        if commit
                        else str(
                            final.get(
                                "reason",
                                confirmation.get(
                                    "reason", result.get("reason", "unknown")
                                ),
                            )
                        )
                    ),
                    "result": result,
                }
            )
        committed_cameras[view] = selected
        result_by_view[view] = {
            "coverage_frames": int(coverage_frames),
            "fixed_center": fixed_center,
            "trajectory_center_fit": trajectory_center_fit,
            "candidate_success": any(
                row["candidate_success"] for row in component_rows
            ),
            "commit": any(row["commit"] for row in component_rows),
            "candidate_success_component_count": int(
                sum(row["candidate_success"] for row in component_rows)
            ),
            "commit_component_count": int(
                sum(row["commit"] for row in component_rows)
            ),
            "components": component_rows,
        }

    pose_root = args.output_root / "pose_track_same_time_dynamic_closure"
    fallback_equal = 0
    fallback_total = 0
    for frame in frames:
        cameras = {
            view: committed_cameras[view][frame] for view in target_views
        }
        destination = pose_root / f"f{frame:06d}" / "cameras_pred"
        save_cameras(destination, cameras)
        replay = CameraLoader(destination).load_cameras(target_views)
        for view in target_views:
            if (frame, view) in committed_nodes:
                continue
            fallback_total += 1
            fallback_equal += int(
                _camera_equal(replay[view], initial_by_view[view][frame])
            )
    if fallback_equal != fallback_total:
        raise RuntimeError("exact Static+PnP fallback replay audit failed")

    committed_views = [
        view for view in target_views if result_by_view[view]["commit"]
    ]
    committed_components = [
        {"view": view, "frames": row["frames"]}
        for view in target_views
        for row in result_by_view[view]["components"]
        if row["commit"]
    ]
    payload = {
        "protocol": PROTOCOL,
        "status": "public_geometry_frozen_before_GT_or_NVS",
        "frames": frames,
        "static_views": static_views,
        "target_views": target_views,
        "config": {
            **vars(config),
            "window_radius": int(args.window_radius),
            "max_points_per_frame": int(args.max_points),
            "min_coverage_frames": int(args.min_coverage_frames),
            "min_component_coverage_frames": int(
                args.min_component_coverage_frames
            ),
            "component_size": int(args.component_size),
            "candidate_center_policy": str(args.candidate_center_policy),
            "verification_policy": "window_component_pooled_pareto",
        },
        "summary": {
            "candidate_success_views": int(
                sum(row["candidate_success"] for row in result_by_view.values())
            ),
            "committed_views": committed_views,
            "committed_components": committed_components,
            "commit_count": int(len(committed_components)),
            "candidate_success_component_count": int(
                sum(
                    row["candidate_success_component_count"]
                    for row in result_by_view.values()
                )
            ),
            "fallback_view_count": int(len(target_views) - len(committed_views)),
            "triangulated_count": _summarize_counts(
                all_diagnostics, "triangulated_count"
            ),
            "runtime_seconds": float(time.perf_counter() - started),
            "seconds_per_frame_all_ptz": float(
                (time.perf_counter() - started) / max(len(frames), 1)
            ),
        },
        "summary_by_view": result_by_view,
        "node_geometry": all_diagnostics,
        "pose_root": pose_root,
        "lineage": {
            "runtime_scene": lineage["runtime_scene"],
            "track_root": args.track_root.resolve(),
            "track_hashes": _bundle_hashes(args.track_root.resolve()),
            "points_t0_sha256": _sha256(args.track_root / "points_t0.npy"),
            "relay_manifest_hashes": lineage["manifest_hashes"],
            "entrypoint": Path(__file__).resolve(),
            "entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "static_pnp_confidence": static_pnp_confidence_audit,
        },
        "attestation": {
            **bundle_attestation,
            "GT_used": False,
            "NVS_used": False,
            "test_RGB_or_cameras_used": False,
            "frame_local_X_not_shared_across_time": True,
            "physical_track_ID_role_persistent_across_time": True,
            "fit_confirm_final_physical_IDs_disjoint": True,
            "different_PTZ_centers_independent": True,
            "component_atomic_commit": True,
            "shared_center_estimated_from_complete_PTZ_initialization": True,
            "trajectory_fit_center_uses_fit_IDs_only": bool(
                str(args.candidate_center_policy)
                == "trajectory_fit_shared_center_bounded_rotation"
            ),
            "confidence_weighted_center_uses_public_Static_PnP_only": bool(
                str(args.candidate_center_policy)
                == "static_pnp_confidence_weighted_center_bounded_rotation"
            ),
            "fallback_camera_replay_exact": fallback_equal == fallback_total,
            "fallback_camera_count": int(fallback_total),
        },
    }
    _write_json(args.output_root / "PUBLIC_GEOMETRY_AUDIT.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--target-views", nargs="+", required=True)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=150)
    parser.add_argument("--min-coverage-frames", type=int, default=16)
    parser.add_argument("--component-size", type=int, default=5)
    parser.add_argument("--min-component-coverage-frames", type=int, default=3)
    parser.add_argument("--static-pnp-diagnostics", type=Path)
    parser.add_argument(
        "--candidate-center-policy",
        choices=(
            "joint_fit",
            "static_pnp_median_fixed",
            "static_pnp_median_bounded_rotation",
            "trajectory_fit_shared_center_bounded_rotation",
            "static_pnp_confidence_weighted_center_bounded_rotation",
        ),
        default="static_pnp_median_bounded_rotation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
