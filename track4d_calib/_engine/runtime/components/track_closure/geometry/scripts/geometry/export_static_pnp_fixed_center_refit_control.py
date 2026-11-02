#!/usr/bin/env python3
"""Refit Static+PnP rotations under one temporal center per physical camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader, CameraParams, save_cameras
from track4d_calib.fixed_center_pnp import solve_fixed_center_pnp
from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    solve_same_time_multiview_pnp,
)
from scripts.geometry.run_track_closure import (
    build_node_observations,
    camera_root_attestation,
)


PROTOCOL = "static-pnp-per-view-center-fixed-rotation-refit-control-v2"
SUPPORTED_BASE_PROTOCOLS = frozenset(
    {
        "hidden-parent-track-driven-same-time-full-pnp-v1",
        "static_refit_complete_static_pnp_stage_compat_v1",
    }
)
TRACK_FILENAMES = (
    "meta.json",
    "views.txt",
    "tracks.npy",
    "visibility.npy",
    "track_quality.npy",
)


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
            if str(key) != "camera"
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _load_attested_prediction(
    prediction_path: Path, attestation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction = Path(prediction_path).resolve()
    attestation = json.loads(Path(attestation_path).read_text(encoding="utf-8"))
    if Path(str(attestation.get("path", ""))).resolve() != prediction:
        raise ValueError("Static+PnP prediction attestation path mismatch")
    if str(attestation.get("sha256", "")) != _sha256(prediction):
        raise ValueError("Static+PnP prediction attestation hash mismatch")
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Static+PnP prediction must be a JSON object")
    return payload, attestation


def _config_from_prediction(prediction: Mapping[str, Any]) -> SameTimePnPConfig:
    row = dict(prediction["config"])
    return SameTimePnPConfig(
        min_support_views=int(row["min_support_views"]),
        min_support_baseline=float(row["min_support_baseline"]),
        min_triangulation_angle_deg=float(row["min_triangulation_angle_deg"]),
        support_reprojection_error_px=float(row["support_reprojection_error_px"]),
        pnp_reprojection_error_px=float(row["pnp_reprojection_error_px"]),
        min_pnp_inliers=int(row["min_pnp_inliers"]),
        pnp_iterations=int(row["pnp_iterations"]),
        pnp_confidence=float(row["pnp_confidence"]),
        rng_seed=int(row["rng_seed"]),
    )


def _load_camera_grid(
    root: Path,
    frames: Sequence[int],
    views: Sequence[str],
) -> dict[int, dict[str, CameraParams]]:
    return {
        int(frame): CameraLoader(
            Path(root) / f"f{int(frame):06d}" / "cameras_pred"
        ).load_cameras(views)
        for frame in frames
    }


def estimate_view_shared_centers(
    *,
    cameras_by_frame: Mapping[int, Mapping[str, CameraParams]],
    target_views: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Estimate one temporal center per physical camera without view pooling."""

    frames = sorted(int(frame) for frame in cameras_by_frame)
    views = [str(view) for view in target_views]
    if not frames or not views or len(views) != len(set(views)):
        raise ValueError("frames and unique target views are required")
    missing = [
        (frame, view)
        for frame in frames
        for view in views
        if view not in cameras_by_frame[frame]
    ]
    if missing:
        raise ValueError(f"missing target cameras: {missing[:8]}")
    shared_centers: dict[str, np.ndarray] = {}
    per_view: dict[str, dict[str, Any]] = {}
    for view in views:
        candidates = np.stack(
            [
                np.asarray(
                    cameras_by_frame[frame][view].camera_center,
                    dtype=np.float64,
                )
                for frame in frames
            ]
        )
        if not np.isfinite(candidates).all():
            raise ValueError(f"Static+PnP center pool is non-finite for view {view}")
        center = np.median(candidates, axis=0)
        distances = np.linalg.norm(candidates - center[None], axis=1)
        shared_centers[view] = center
        per_view[view] = {
            "support_count": int(len(candidates)),
            "center": center,
            "deviation_mean": float(np.mean(distances)),
            "deviation_median": float(np.median(distances)),
            "deviation_p90": float(np.percentile(distances, 90.0)),
            "deviation_max": float(np.max(distances)),
        }
    return shared_centers, {
        "support_count": int(len(frames) * len(views)),
        "frame_count": int(len(frames)),
        "view_count": int(len(views)),
        "per_view": per_view,
        "estimator": (
            "per-view componentwise median of independent per-frame PnP centers"
        ),
        "cross_view_center_pooling": False,
    }


def _validate_prediction_contract(prediction: Mapping[str, Any]) -> None:
    if prediction.get("protocol") not in SUPPORTED_BASE_PROTOCOLS:
        raise ValueError("unexpected Static+PnP protocol")
    if prediction.get("track_source_mode") != "independent_per_frame_bundle":
        raise ValueError("Static+PnP did not use independent per-frame bundles")
    if prediction.get("temporal_seed_policy") != "independent_same_frame":
        raise ValueError("Static+PnP did not use same-frame matching")
    if prediction.get("verify_commit_used") is not False:
        raise ValueError("Static+PnP unexpectedly used verify/commit")
    if prediction.get("gt_used") is not False or prediction.get("nvs_used") is not False:
        raise ValueError("Static+PnP prediction used a forbidden readout")


def _validate_bundle_hashes(
    bundle: Path, declared: Mapping[str, Any]
) -> dict[str, str]:
    actual = {name: _sha256(bundle / name) for name in TRACK_FILENAMES}
    if actual != dict(declared):
        raise RuntimeError(f"frozen per-frame matching bundle changed: {bundle}")
    return actual


def _same_intrinsics(left: CameraParams, right: CameraParams) -> bool:
    return bool(
        left.H == right.H
        and left.W == right.W
        and np.array_equal(left.K, right.K)
        and np.array_equal(left.D, right.D)
    )


def export(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    prediction_path = Path(args.base_predictions).resolve()
    prediction, attestation = _load_attested_prediction(
        prediction_path, Path(args.base_prediction_attestation)
    )
    _validate_prediction_contract(prediction)

    frames = [int(frame) for frame in prediction["frames"]]
    static_views = [str(view) for view in prediction["static_views"]]
    target_views = [str(view) for view in prediction["target_views"]]
    all_views = [*static_views, *target_views]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("Static+PnP frame roster is empty or duplicated")
    if not static_views or not target_views or len(all_views) != len(set(all_views)):
        raise ValueError("Static and target view rosters must be non-empty and disjoint")
    base_root = Path(str(prediction["pose_root"])).resolve()
    track_roots = [Path(str(path)).resolve() for path in prediction["track_roots"]]
    if len(track_roots) != len(frames):
        raise ValueError("Static+PnP does not declare one bundle per frame")
    row_by_node = {
        (int(row["frame"]), str(row["view"])): dict(row)
        for row in prediction["rows"]
    }
    expected_nodes = {(frame, view) for frame in frames for view in target_views}
    if set(row_by_node) != expected_nodes:
        raise ValueError("Static+PnP prediction rows are incomplete")
    if not all(bool(row_by_node[node]["solver"]["success"]) for node in expected_nodes):
        raise ValueError("control requires a complete successful Static+PnP baseline")

    base_grid = _load_camera_grid(base_root, frames, all_views)
    for frame in frames:
        frame_name = f"f{frame:06d}"
        actual = camera_root_attestation(base_root / frame_name / "cameras_pred")
        if actual != prediction["output_camera_hashes"][frame_name]:
            raise RuntimeError(f"frozen Static+PnP camera changed: {frame_name}")
    shared_centers, center_report = estimate_view_shared_centers(
        cameras_by_frame=base_grid,
        target_views=target_views,
    )
    config = _config_from_prediction(prediction)
    point_budget = int(prediction["fit_point_budget"])
    declared_bundle_hashes = dict(prediction["track_files_sha256"])

    output_root.mkdir(parents=True)
    pose_root = output_root / "pose_static_pnp_fixed_center_refit"
    node_rows: list[dict[str, Any]] = []
    changed_nodes: list[list[Any]] = []
    fallback_nodes: list[list[Any]] = []
    input_track_hashes: dict[str, dict[str, str]] = {}
    for frame, bundle in zip(frames, track_roots, strict=True):
        frame_name = f"f{frame:06d}"
        if bundle.name != frame_name:
            raise ValueError(f"bundle order/frame mismatch: {bundle} vs {frame_name}")
        bundle_key = str(bundle)
        input_track_hashes[frame_name] = _validate_bundle_hashes(
            bundle, dict(declared_bundle_hashes.get(bundle_key) or {})
        )
        meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
        bundle_views = (bundle / "views.txt").read_text(encoding="utf-8").split()
        if bundle_views != all_views:
            raise ValueError(f"per-frame matching roster differs: {frame_name}")
        if (
            int(meta["start_frame"]) != frame
            or int(meta["num_frames"]) != 1
            or str(meta.get("seed_builder")) != "spsg_target_star"
        ):
            raise ValueError(f"invalid independent same-frame bundle: {frame_name}")
        tracks = np.load(bundle / "tracks.npy", mmap_mode="r")
        visibility = np.load(bundle / "visibility.npy", mmap_mode="r")
        quality = np.load(bundle / "track_quality.npy", mmap_mode="r")
        base_cameras = base_grid[frame]
        support_cameras = {view: base_cameras[view] for view in static_views}
        selected = dict(base_cameras)
        for view in target_views:
            support_observations, target_observations, observation = build_node_observations(
                tracks=tracks,
                visibility=visibility,
                track_quality=quality,
                view_ids=bundle_views,
                static_views=static_views,
                target_view=view,
                frame_local=0,
                min_support_views=int(config.min_support_views),
                max_points=point_budget,
            )
            replay = solve_same_time_multiview_pnp(
                support_cameras=support_cameras,
                support_observations=support_observations,
                target_observations=target_observations,
                target_template=base_cameras[view],
                config=config,
            )
            if not bool(replay.get("success", False)):
                raise RuntimeError(f"Static+PnP replay failed: {frame_name}/{view}")
            replay_camera = replay["camera"]
            if not np.array_equal(replay_camera.R, base_cameras[view].R) or not np.array_equal(
                replay_camera.T, base_cameras[view].T
            ):
                raise RuntimeError(f"Static+PnP replay is not bit-exact: {frame_name}/{view}")
            refit = solve_fixed_center_pnp(
                object_points=replay["object_points"],
                image_points=replay["image_points"],
                target_template=base_cameras[view],
                fixed_center=shared_centers[view],
                reprojection_error_px=float(config.pnp_reprojection_error_px),
                min_inliers=int(config.min_pnp_inliers),
                iterations_count=int(config.pnp_iterations),
                rng_seed=int(config.rng_seed),
            )
            if bool(refit.get("success", False)):
                selected[view] = refit["camera"]
                changed_nodes.append([frame, view])
            else:
                fallback_nodes.append([frame, view])
            node_rows.append(
                {
                    "frame": frame,
                    "view": view,
                    "observation": observation,
                    "static_pnp_replay": {
                        "bit_exact": True,
                        "triangulated_count": int(replay["triangulated_count"]),
                        "inlier_count": int(replay["inlier_count"]),
                        "reprojection_median_px": float(replay["reprojection_median_px"]),
                        "reprojection_p90_px": float(replay["reprojection_p90_px"]),
                    },
                    "fixed_center_refit": refit,
                    "selected_source": (
                        "independent_matches_fixed_center_rotation_refit"
                        if bool(refit.get("success", False))
                        else "exact_static_pnp_fallback"
                    ),
                }
            )
        save_cameras(pose_root / frame_name / "cameras_pred", selected)

    reloaded = _load_camera_grid(pose_root, frames, all_views)
    for frame in frames:
        for view in all_views:
            before = base_grid[frame][view]
            after = reloaded[frame][view]
            if not _same_intrinsics(before, after):
                raise RuntimeError(f"control changed intrinsics: {(frame, view)}")
            if view in static_views and (
                not np.array_equal(before.R, after.R)
                or not np.array_equal(before.T, after.T)
            ):
                raise RuntimeError(f"control changed static camera: {(frame, view)}")

    frozen_inputs = output_root / "frozen_inputs"
    frozen_inputs.mkdir()
    frozen_prediction = frozen_inputs / "predictions_pre_gt.json"
    frozen_attestation = frozen_inputs / "prediction_attestation.json"
    shutil.copy2(prediction_path, frozen_prediction)
    shutil.copy2(Path(args.base_prediction_attestation), frozen_attestation)
    output_hashes = {
        f"f{frame:06d}": camera_root_attestation(
            pose_root / f"f{frame:06d}" / "cameras_pred"
        )
        for frame in frames
    }
    manifest = {
        "protocol": PROTOCOL,
        "status": "public_no_cross_frame_track_control_no_GT_no_NVS",
        "runtime_scene": prediction["runtime_scene"],
        "frames": frames,
        "static_views": static_views,
        "target_views": target_views,
        "base_pose_root": str(base_root),
        "pose_root": str(pose_root),
        "shared_centers": shared_centers,
        "center_report": center_report,
        "node_rows": node_rows,
        "changed_nodes": changed_nodes,
        "changed_node_count": int(len(changed_nodes)),
        "fallback_nodes": fallback_nodes,
        "fallback_node_count": int(len(fallback_nodes)),
        "input_camera_hashes": prediction["output_camera_hashes"],
        "input_track_hashes": input_track_hashes,
        "output_camera_hashes": output_hashes,
        "input_prediction": {
            "source": str(prediction_path),
            "source_sha256": _sha256(prediction_path),
            "attestation": attestation,
            "frozen": str(frozen_prediction),
            "frozen_sha256": _sha256(frozen_prediction),
        },
        "config": {
            "shared_center_estimator": "per-view componentwise median",
            "cross_view_center_pooling": False,
            "source_pnp_config": prediction["config"],
            "source_point_budget": point_budget,
            "fixed_center_rotation_ransac_reuses_source_threshold_minimum_iterations_seed": True,
            "new_tunable_numeric_hyperparameters": 0,
        },
        "fairness": {
            "GT_used": False,
            "NVS_used": False,
            "heldout_test_images_or_poses_used": False,
            "cross_frame_tracks_used": False,
            "independent_same_frame_matches_replayed_bit_exact": True,
            "same_2D_3D_correspondences_as_Static_PnP": True,
            "static_cameras_preserved_exactly": True,
            "failed_refits_preserve_exact_Static_PnP": True,
            "new_tunable_numeric_hyperparameters": 0,
        },
        "ablation_role": (
            "geometry-consistent no-cross-frame-track control: isolate a shared "
            "PTZ-center structural prior from temporal track closure"
        ),
        "code": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "solver": str(Path(sys.modules[solve_fixed_center_pnp.__module__].__file__).resolve()),
            "solver_sha256": _sha256(
                Path(sys.modules[solve_fixed_center_pnp.__module__].__file__).resolve()
            ),
        },
    }
    manifest_path = output_root / "fixed_center_refit_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        output_root / "prediction_attestation.json",
        {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
    )
    _write_json(
        output_root / "PUBLIC_OUTPUT_FREEZE.json",
        {
            "status": "frozen_before_pose_GT_or_NVS_readout",
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "pose_root": str(pose_root),
            "output_camera_hashes": output_hashes,
            "GT_used": False,
            "NVS_used": False,
        },
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--base-prediction-attestation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    manifest = export(parse_args())
    print(
        json.dumps(
            {
                "protocol": manifest["protocol"],
                "changed_node_count": manifest["changed_node_count"],
                "fallback_node_count": manifest["fallback_node_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
