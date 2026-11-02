#!/usr/bin/env python3
"""Repair failed same-time Static+PnP bands from tracked verified PnP inliers."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from track4d_calib.cotracker_query_tracking import track_image_queries
from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    score_same_time_multiview_poses,
)
from track4d_calib.spatiotemporal_rotation_closure import (
    refine_tracks_with_pyramidal_lk,
)
from track4d_calib.static_pnp_inlier_track_repair import (
    fit_cross_seed_track_candidate,
    recover_static_pnp_inlier_seed_bank,
    select_boundary_seed_frames,
)
from track4d_calib.io import resolve_frame_image
from track4d_calib.roma_query_relay import relay_bidirectional_query_points
from scripts.geometry.evaluate_track_pose_after_stage_a_seal import (
    _verify_stage_a_seal,
)
from scripts.geometry.export_static_pnp_fixed_center_refit_control import (
    estimate_view_shared_centers,
)
from scripts.geometry.export_public_ptz_joint_bundle_closure import (
    complete_base_camera_grid,
)
from scripts.geometry.export_public_ptz_track_center_closure import (
    _collect_successful_nodes,
    _load_attested_prediction,
    _load_successful_cameras,
    _verify_runtime_tree,
)
from scripts.geometry.run_track_closure import (
    build_node_observations,
    camera_root_attestation,
)


METHOD_BY_SEED_POLICY = {
    "pnp_inliers": "static_pnp_inlier_seeded_crossbank_track_band_closure_v1",
    "triangulated_static_candidates": (
        "static_triangulated_seeded_visibility_aware_crossbank_track_band_closure_v1"
    ),
}
REPLAY_ROTATION_TOLERANCE_DEG = 5e-3
REPLAY_CENTER_TOLERANCE = 5e-4
ROMA_MINIMUM_OVERLAP = 0.2
ROMA_MAXIMUM_CYCLE_ERROR_PX = 1.5
ROMA_SETTING = "precise"
COTRACKER_BACKENDS = frozenset(("cotracker", "cotracker_lk"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            if str(key) != "camera"
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


def _attested_target_views(
    *,
    requested_views: Sequence[str],
    sealed_views: Sequence[str],
    allow_subset: bool,
) -> list[str]:
    """Return the full roster used for seal and prediction attestation."""

    requested = [str(view) for view in requested_views]
    sealed = [str(view) for view in sealed_views]
    if len(sealed) != len(set(sealed)):
        raise ValueError("Stage-A seal target views contain duplicates")
    if not allow_subset:
        if requested != sealed:
            raise ValueError("target views do not match Stage-A seal")
        return sealed
    if not requested or not set(requested).issubset(set(sealed)):
        raise ValueError("target-view shard is not a non-empty Stage-A subset")
    return sealed


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


def _camera_replay_attestation(
    replayed: CameraParams,
    frozen: CameraParams,
) -> dict[str, Any]:
    intrinsics_equal = bool(
        replayed.view_id == frozen.view_id
        and replayed.H == frozen.H
        and replayed.W == frozen.W
        and np.array_equal(replayed.K, frozen.K)
        and np.array_equal(replayed.D, frozen.D)
    )
    relative = replayed.R @ frozen.R.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    center_error = float(
        np.linalg.norm(replayed.camera_center - frozen.camera_center)
    )
    equivalent = bool(
        intrinsics_equal
        and np.isfinite(rotation_deg)
        and np.isfinite(center_error)
        and rotation_deg <= REPLAY_ROTATION_TOLERANCE_DEG
        and center_error <= REPLAY_CENTER_TOLERANCE
    )
    return {
        "equivalent": equivalent,
        "intrinsics_equal": intrinsics_equal,
        "rotation_error_deg": rotation_deg,
        "center_error": center_error,
        "rotation_tolerance_deg": REPLAY_ROTATION_TOLERANCE_DEG,
        "center_tolerance": REPLAY_CENTER_TOLERANCE,
    }


def _load_bundle(root: Path) -> dict[str, Any]:
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    views = [
        line.strip()
        for line in (root / "views.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "root": root,
        "meta": meta,
        "views": views,
        "tracks": np.load(root / "tracks.npy", allow_pickle=False),
        "visibility": np.load(root / "visibility.npy", allow_pickle=False),
        "track_quality": (
            np.load(root / "track_quality.npy", allow_pickle=False)
            if (root / "track_quality.npy").is_file()
            else None
        ),
    }


def _bundle_hashes(root: Path) -> dict[str, str]:
    names = ["meta.json", "views.txt", "tracks.npy", "visibility.npy"]
    if (root / "track_quality.npy").is_file():
        names.append("track_quality.npy")
    return {name: _sha256(root / name) for name in names}


def _failure_bands(
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
    successful_nodes: set[tuple[int, str]],
) -> list[dict[str, Any]]:
    ordered = [int(frame) for frame in frames]
    frame_index = {frame: index for index, frame in enumerate(ordered)}
    bands: list[dict[str, Any]] = []
    for view in [str(value) for value in target_views]:
        failed = [frame for frame in ordered if (frame, view) not in successful_nodes]
        groups: list[list[int]] = []
        for frame in failed:
            if not groups or frame_index[frame] != frame_index[groups[-1][-1]] + 1:
                groups.append([frame])
            else:
                groups[-1].append(frame)
        for band in groups:
            successful_frames = [
                frame for frame in ordered if (frame, view) in successful_nodes
            ]
            bands.append(
                {
                    "view": view,
                    "frames": band,
                    "seed_frames": select_boundary_seed_frames(
                        successful_frames=successful_frames,
                        band_frames=band,
                        max_seed_banks=4,
                    ),
                }
            )
    return bands


def _prescribed_repair_bands(
    *,
    frames: Sequence[int],
    repair_frames: Sequence[int],
    target_views: Sequence[str],
    successful_nodes: set[tuple[int, str]],
    anchor_seed_frames: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Build one repair band per PTZ from successful outside anchors."""

    ordered = [int(frame) for frame in frames]
    repair = [int(frame) for frame in repair_frames]
    if repair != sorted(set(repair)):
        raise ValueError("repair_frames must be sorted and unique")
    if not repair or not set(repair).issubset(ordered):
        raise ValueError("repair_frames must be a non-empty subset of frames")
    frame_index = {frame: index for index, frame in enumerate(ordered)}
    if any(
        frame_index[right] != frame_index[left] + 1
        for left, right in zip(repair, repair[1:])
    ):
        raise ValueError("repair_frames must form one contiguous band")

    bands: list[dict[str, Any]] = []
    repair_set = set(repair)
    explicit_anchors = [int(frame) for frame in anchor_seed_frames]
    if explicit_anchors != sorted(set(explicit_anchors)):
        raise ValueError("anchor_seed_frames must be sorted and unique")
    if explicit_anchors and not set(explicit_anchors).issubset(ordered):
        raise ValueError("anchor_seed_frames must be a subset of frames")
    for raw_view in target_views:
        view = str(raw_view)
        successful_anchors = [
            frame
            for frame in ordered
            if frame not in repair_set and (frame, view) in successful_nodes
        ]
        if explicit_anchors:
            unavailable = [
                frame
                for frame in explicit_anchors
                if (frame, view) not in successful_nodes
            ]
            if unavailable:
                raise ValueError(
                    f"anchor seeds for view {view} are not successful: {unavailable}"
                )
            seed_frames = explicit_anchors
        else:
            seed_frames = select_boundary_seed_frames(
                successful_frames=successful_anchors,
                band_frames=repair,
                max_seed_banks=4,
            )
        if len(seed_frames) < 2:
            raise ValueError(
                f"prescribed repair band for view {view} lacks two successful anchors"
            )
        bands.append(
            {
                "view": view,
                "frames": repair,
                "seed_frames": seed_frames,
            }
        )
    return bands


def _active_seed_frames(
    *,
    seed_frames: Sequence[int],
    frame: int,
    max_track_distance: int,
) -> list[int]:
    """Exclude same-frame matcher evidence and retain only local track relays."""

    maximum = int(max_track_distance)
    if maximum < 0:
        raise ValueError("max_track_distance must be nonnegative")
    return [
        int(seed)
        for seed in seed_frames
        if int(seed) != int(frame)
        and (maximum == 0 or abs(int(seed) - int(frame)) <= maximum)
    ]


def _direct_lk_seed_bank(
    *, query_points: np.ndarray, frame_count: int, query_frame_index: int
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(query_points, dtype=np.float64).reshape(-1, 2)
    count = int(frame_count)
    query = int(query_frame_index)
    if count <= 0 or not 0 <= query < count:
        raise ValueError("query frame must be inside a non-empty track window")
    tracks = np.full((count, len(points), 2), np.nan, dtype=np.float64)
    visibility = np.zeros((count, len(points)), dtype=bool)
    tracks[query] = points
    visibility[query] = np.isfinite(points).all(axis=1)
    return tracks, visibility


def _load_roma_model(
    *,
    source_root: Path,
    weights: Path,
    device: str,
    setting: str | None = None,
    force_bidirectional: bool = False,
) -> tuple[Any, Any]:
    source = source_root.resolve()
    checkpoint = weights.resolve()
    if not (source / "romav2").is_dir():
        raise FileNotFoundError(f"RoMaV2 source root is invalid: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    import torch
    from romav2 import RoMaV2

    torch.set_float32_matmul_precision("highest")
    requested_device = torch.device(
        str(device) if torch.cuda.is_available() else "cpu"
    )
    original_loader = torch.hub.load_state_dict_from_url

    def load_local_weights(url: str, *args: Any, **kwargs: Any) -> Any:
        if Path(str(url)).name != "romav2.pt":
            return original_loader(url, *args, **kwargs)
        map_location = kwargs.get("map_location", "cpu")
        try:
            return torch.load(
                checkpoint, map_location=map_location, weights_only=True
            )
        except TypeError:
            return torch.load(checkpoint, map_location=map_location)

    torch.hub.load_state_dict_from_url = load_local_weights
    try:
        model = RoMaV2(
            RoMaV2.Cfg(
                setting=(ROMA_SETTING if setting is None else str(setting)),
                compile=False,
            )
        )
    finally:
        torch.hub.load_state_dict_from_url = original_loader
    if force_bidirectional:
        model.bidirectional = True
    model = model.to(requested_device)
    model.eval()
    return model, requested_device


def _relay_roma_seed_bank(
    *,
    query_points: np.ndarray,
    tracking_frames: Sequence[int],
    seed_frame: int,
    image_size: tuple[int, int],
    load_pair: Callable[[int, int], Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ordered = [int(frame) for frame in tracking_frames]
    seed = int(seed_frame)
    if seed not in ordered:
        raise ValueError("seed frame must be inside tracking_frames")
    points = np.asarray(query_points, dtype=np.float64).reshape(-1, 2)
    tracks = np.full((len(ordered), len(points), 2), np.nan, dtype=np.float64)
    visibility = np.zeros((len(ordered), len(points)), dtype=bool)
    seed_index = ordered.index(seed)
    tracks[seed_index] = points
    visibility[seed_index] = np.isfinite(points).all(axis=1)
    target_rows: list[dict[str, Any]] = []

    for target_index, target_frame in enumerate(ordered):
        if target_frame == seed:
            continue
        left, right = sorted((seed, target_frame))
        prediction = load_pair(left, right)
        if seed == left:
            forward = "AB"
            backward = "BA"
        else:
            forward = "BA"
            backward = "AB"
        required = (
            f"warp_{forward}",
            f"warp_{backward}",
            f"overlap_{forward}",
            f"overlap_{backward}",
        )
        missing = [key for key in required if prediction.get(key) is None]
        if missing:
            raise ValueError(f"bidirectional RoMa prediction is incomplete: {missing}")
        relay = relay_bidirectional_query_points(
            source_points_xy=points,
            warp_source_to_target=np.asarray(prediction[f"warp_{forward}"]),
            warp_target_to_source=np.asarray(prediction[f"warp_{backward}"]),
            overlap_source_to_target=np.asarray(
                prediction[f"overlap_{forward}"]
            ),
            overlap_target_to_source=np.asarray(
                prediction[f"overlap_{backward}"]
            ),
            source_image_size=image_size,
            target_image_size=image_size,
            minimum_overlap=ROMA_MINIMUM_OVERLAP,
            maximum_cycle_error_px=ROMA_MAXIMUM_CYCLE_ERROR_PX,
        )
        tracks[target_index] = relay["target_points"]
        visibility[target_index] = relay["valid"]
        target_rows.append(
            {
                "target_frame": int(target_frame),
                "pair": [int(left), int(right)],
                "direction": forward,
                **relay["diagnostics"],
            }
        )
    return tracks, visibility, {
        "backend": "roma_v2_bidirectional_query_cycle",
        "setting": ROMA_SETTING,
        "minimum_overlap": ROMA_MINIMUM_OVERLAP,
        "maximum_cycle_error_px": ROMA_MAXIMUM_CYCLE_ERROR_PX,
        "targets": target_rows,
    }


def _score_diagnostics(score: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
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
    )
    return {key: score[key] for key in keys if key in score}


def _independent_geometry_decision(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base = scores["base"]
    candidate = scores["candidate"]
    score_available = bool(base.get("success") and candidate.get("success"))
    truncated_gain = (
        float(base["truncated_reprojection_mean_px"])
        - float(candidate["truncated_reprojection_mean_px"])
        if score_available
        else None
    )
    inlier_nonregression = bool(
        score_available
        and int(candidate.get("inlier_count", 0)) >= int(base.get("inlier_count", 0))
    )
    accepted = bool(
        score_available
        and truncated_gain is not None
        and truncated_gain > 1e-12
        and inlier_nonregression
    )
    return {
        "accepted": accepted,
        "reason": (
            "independent_same_time_geometry_improved"
            if accepted
            else "independent_same_time_score_unavailable"
            if not score_available
            else "independent_same_time_geometry_not_improved"
        ),
        "truncated_reprojection_gain_px": truncated_gain,
        "inlier_nonregression": inlier_nonregression,
        "base": _score_diagnostics(base),
        "candidate": _score_diagnostics(candidate),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    runtime = args.runtime_scene.resolve()
    stage = args.stage_a_root.resolve()
    output = args.output_root.resolve()
    frames = [int(frame) for frame in args.frames]
    target_views = [str(view) for view in args.target_views]
    seed_point_policy = str(args.seed_point_policy)
    track_pose_model = str(args.track_pose_model)
    method = METHOD_BY_SEED_POLICY[seed_point_policy]
    if track_pose_model == "fixed_center_rotation":
        method = f"{method}_fixed_center_rotation_v1"
    if frames != sorted(set(frames)):
        raise ValueError("frames must be sorted and unique")
    if len(target_views) != len(set(target_views)):
        raise ValueError("target views must be unique")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)

    seal_payload = json.loads(args.stage_a_seal.read_text(encoding="utf-8"))
    selector_subdir = str(seal_payload["selector_subdir"])
    commit_policy = str(seal_payload["commit_policy"])
    sealed_target_views = [str(view) for view in seal_payload["target_views"]]
    attested_target_views = _attested_target_views(
        requested_views=target_views,
        sealed_views=sealed_target_views,
        allow_subset=bool(args.allow_sealed_target_view_subset),
    )
    seal_record = _verify_stage_a_seal(
        runtime_scene=runtime,
        stage_a_root=stage,
        stage_a_seal=args.stage_a_seal,
        selector_subdir=selector_subdir,
        expected_commit_policy=commit_policy,
        frames=frames,
        target_views=attested_target_views,
    )
    runtime_manifest, runtime_checksums = _verify_runtime_tree(runtime)
    prediction_path = stage / "base_same_frame_pnp" / "predictions_pre_gt.json"
    base_pose_root = stage / "base_same_frame_pnp" / "pose_track_same_time_pnp"
    prediction = _load_attested_prediction(
        attestation_path=stage / "base_same_frame_pnp" / "prediction_attestation.json",
        expected_prediction_path=prediction_path,
        expected_pose_root=base_pose_root,
        runtime=runtime,
        runtime_digest=str(runtime_checksums["tree_digest"]),
        frames=frames,
        target_views=attested_target_views,
        expected_track_mode="independent_per_frame_bundle",
    )
    static_views = [str(view) for view in prediction["static_views"]]
    row_by_node = {
        (int(row["frame"]), str(row["view"])): row for row in prediction["rows"]
    }
    successful_nodes = _collect_successful_nodes(prediction["rows"])
    raw_base_cameras = _load_successful_cameras(
        pose_root=base_pose_root,
        successful_nodes=successful_nodes,
        frames=frames,
        target_views=target_views,
    )
    base_cameras, base_completion = complete_base_camera_grid(
        frames=frames,
        target_views=target_views,
        base_cameras=raw_base_cameras,
    )
    shared_centers: dict[str, np.ndarray] = {}
    shared_center_report: dict[str, Any] | None = None
    if track_pose_model == "fixed_center_rotation":
        shared_centers, shared_center_report = estimate_view_shared_centers(
            cameras_by_frame={
                frame: {
                    view: base_cameras[(frame, view)] for view in target_views
                }
                for frame in frames
            },
            target_views=target_views,
        )
    prescribed_repair_frames = [int(frame) for frame in args.repair_frames]
    anchor_seed_frames = [int(frame) for frame in args.anchor_seed_frames]
    if anchor_seed_frames and not prescribed_repair_frames:
        raise ValueError("anchor_seed_frames require prescribed repair_frames")
    bands = (
        _prescribed_repair_bands(
            frames=frames,
            repair_frames=prescribed_repair_frames,
            target_views=target_views,
            successful_nodes=successful_nodes,
            anchor_seed_frames=anchor_seed_frames,
        )
        if prescribed_repair_frames
        else _failure_bands(
            frames=frames,
            target_views=target_views,
            successful_nodes=successful_nodes,
        )
    )
    if not bands:
        raise ValueError("Static+PnP has no failure band to repair")

    support_cameras = CameraLoader(runtime / "cameras/static").load_cameras(static_views)
    intrinsics = CameraIntrinsicsLoader(runtime / "cameras")
    templates = {view: intrinsics.load_camera_template(view) for view in target_views}
    config = SameTimePnPConfig(**prediction["config"])
    fit_budget = int(prediction["fit_point_budget"])

    bank_specs: dict[tuple[str, int], dict[str, Any]] = {}
    input_bundle_hashes: dict[str, dict[str, str]] = {}
    per_frame_matcher_root = (
        args.per_frame_matcher_root.resolve()
        if args.per_frame_matcher_root is not None
        else stage / "per_frame_matcher"
    )
    if not per_frame_matcher_root.is_dir():
        raise FileNotFoundError(per_frame_matcher_root)
    for band in bands:
        view = str(band["view"])
        for seed_frame in band["seed_frames"]:
            key = (view, int(seed_frame))
            if key in bank_specs:
                continue
            bundle_root = per_frame_matcher_root / f"f{int(seed_frame):06d}"
            bundle = _load_bundle(bundle_root)
            local = int(seed_frame) - int(bundle["meta"]["start_frame"])
            support_observations, target_observations, observation = build_node_observations(
                tracks=bundle["tracks"],
                visibility=bundle["visibility"],
                track_quality=bundle["track_quality"],
                view_ids=bundle["views"],
                static_views=static_views,
                target_view=view,
                frame_local=local,
                min_support_views=int(config.min_support_views),
                max_points=fit_budget,
            )
            bank = recover_static_pnp_inlier_seed_bank(
                seed_frame=int(seed_frame),
                target_view=view,
                support_cameras=support_cameras,
                support_observations=support_observations,
                target_observations=target_observations,
                target_template=templates[view],
                config=config,
                expected_solver=row_by_node[(int(seed_frame), view)]["solver"],
                seed_point_policy=seed_point_policy,
            )
            replay_attestation = _camera_replay_attestation(
                bank["camera"], raw_base_cameras[(int(seed_frame), view)]
            )
            if not replay_attestation["equivalent"]:
                raise ValueError(f"replayed camera differs from attested Static+PnP at {view}@{seed_frame}")
            bank_specs[key] = {
                **bank,
                "observation": observation,
                "replay_attestation": replay_attestation,
            }
            input_bundle_hashes[f"f{int(seed_frame):06d}"] = _bundle_hashes(bundle_root)

    track_backend = str(args.track_backend)
    cotracker_root = args.cotracker_root.resolve()
    predictor = None
    roma_model = None
    roma_pair_cache: OrderedDict[tuple[str, int, int], dict[str, np.ndarray]] = (
        OrderedDict()
    )
    roma_match_count = 0
    roma_cache_hit_count = 0
    if track_backend in COTRACKER_BACKENDS:
        if str(cotracker_root) not in sys.path:
            sys.path.insert(0, str(cotracker_root))
        from cotracker.predictor import CoTrackerPredictor

        predictor = CoTrackerPredictor(
            checkpoint=str(args.cotracker_checkpoint.resolve()),
            offline=True,
        ).to(str(args.device))
        predictor.eval()
    elif track_backend == "roma_cycle":
        roma_model, _roma_device = _load_roma_model(
            source_root=args.romav2_root,
            weights=args.romav2_weights,
            device=str(args.device),
        )

    def load_roma_pair(view: str, left: int, right: int) -> dict[str, np.ndarray]:
        nonlocal roma_match_count, roma_cache_hit_count
        if roma_model is None:
            raise RuntimeError("RoMa model was not initialized")
        key = (str(view), int(left), int(right))
        cached = roma_pair_cache.get(key)
        if cached is not None:
            roma_cache_hit_count += 1
            roma_pair_cache.move_to_end(key)
            return cached
        source_path = resolve_frame_image(runtime / "images" / str(view), frame=int(left))
        target_path = resolve_frame_image(runtime / "images" / str(view), frame=int(right))
        prediction = roma_model.match(str(source_path), str(target_path))
        arrays: dict[str, np.ndarray] = {}
        for direction in ("AB", "BA"):
            for field in ("warp", "overlap"):
                value = prediction.get(f"{field}_{direction}")
                if value is None:
                    raise ValueError(
                        f"RoMa {ROMA_SETTING} did not return {field}_{direction}"
                    )
                array = value[0].detach().to(device="cpu").numpy()
                if field == "overlap" and array.ndim == 3 and array.shape[-1] == 1:
                    array = array[..., 0]
                arrays[f"{field}_{direction}"] = np.asarray(array, dtype=np.float32)
        roma_match_count += 1
        roma_pair_cache[key] = arrays
        roma_pair_cache.move_to_end(key)
        while len(roma_pair_cache) > 4:
            roma_pair_cache.popitem(last=False)
        return arrays
    tracked_banks: dict[tuple[str, int], dict[str, Any]] = {}
    bank_rows: list[dict[str, Any]] = []
    tracks_root = output / "tracked_seed_banks"
    tracks_root.mkdir(parents=True, exist_ok=True)
    for (view, seed_frame), bank in sorted(bank_specs.items()):
        tracking_frames = [
            frame
            for frame in frames
            if int(args.max_track_distance) == 0
            or abs(int(frame) - int(seed_frame)) <= int(args.max_track_distance)
        ]
        image_paths = [
            resolve_frame_image(runtime / "images" / view, frame=frame)
            for frame in tracking_frames
        ]
        query_frame_index = tracking_frames.index(seed_frame)
        if track_backend == "lk":
            coarse_tracks, coarse_visibility = _direct_lk_seed_bank(
                query_points=np.asarray(bank["query_points"], dtype=np.float64),
                frame_count=len(tracking_frames),
                query_frame_index=query_frame_index,
            )
            local_tracks, local_visibility, lk_audit = (
                refine_tracks_with_pyramidal_lk(
                    image_paths=image_paths,
                    coarse_tracks=coarse_tracks,
                    coarse_visibility=coarse_visibility,
                    query_frame_index=query_frame_index,
                    forward_backward_threshold_px=float(args.lk_fb_threshold_px),
                )
            )
            tracker = {
                "backend": "direct_full_resolution_pyramidal_lk",
                "lk": lk_audit,
            }
        elif track_backend == "roma_cycle":
            local_tracks, local_visibility, tracker = _relay_roma_seed_bank(
                query_points=np.asarray(bank["query_points"], dtype=np.float64),
                tracking_frames=tracking_frames,
                seed_frame=seed_frame,
                image_size=(int(templates[view].H), int(templates[view].W)),
                load_pair=lambda left, right, current_view=view: load_roma_pair(
                    current_view, left, right
                ),
            )
        else:
            assert predictor is not None
            local_tracks, local_visibility, tracker = track_image_queries(
                predictor=predictor,
                image_paths=image_paths,
                query_points=np.asarray(bank["query_points"], dtype=np.float64),
                query_frame_index=query_frame_index,
                device=str(args.device),
            )
            if track_backend == "cotracker_lk":
                local_tracks, local_visibility, lk_audit = (
                    refine_tracks_with_pyramidal_lk(
                        image_paths=image_paths,
                        coarse_tracks=local_tracks,
                        coarse_visibility=local_visibility,
                        query_frame_index=query_frame_index,
                        forward_backward_threshold_px=float(args.lk_fb_threshold_px),
                    )
                )
                tracker = {**tracker, "lk_refinement": lk_audit}
        tracks = np.zeros(
            (len(frames), *local_tracks.shape[1:]), dtype=local_tracks.dtype
        )
        visibility = np.zeros(
            (len(frames), *local_visibility.shape[1:]), dtype=bool
        )
        for local_index, tracked_frame in enumerate(tracking_frames):
            full_index = frames.index(tracked_frame)
            tracks[full_index] = local_tracks[local_index]
            visibility[full_index] = local_visibility[local_index]
        source_ids = np.asarray(bank["source_point_ids"], dtype=np.int64)
        global_ids = (np.int64(seed_frame) << np.int64(32)) | source_ids
        path = tracks_root / f"view{view}_seed{seed_frame:06d}.npz"
        np.savez_compressed(
            path,
            seed_frame=np.asarray([seed_frame], dtype=np.int64),
            source_point_ids=source_ids,
            global_point_ids=global_ids,
            points3d=np.asarray(bank["points3d"], dtype=np.float64),
            tracks=tracks,
            visibility=visibility,
            frames=np.asarray(frames, dtype=np.int64),
        )
        tracked_banks[(view, seed_frame)] = {
            "seed_frame": seed_frame,
            "global_point_ids": global_ids,
            "source_point_ids": source_ids,
            "points3d": np.asarray(bank["points3d"], dtype=np.float64),
            "tracks": tracks,
            "visibility": visibility,
        }
        bank_rows.append(
            {
                "view": view,
                "seed_frame": seed_frame,
                "seed_inlier_count": int(len(source_ids)),
                "seed_point_count": int(len(source_ids)),
                "source_ransac_inlier_count": int(bank["solver"]["inlier_count"]),
                "seed_point_policy": seed_point_policy,
                "source_solver": bank["solver"],
                "source_observation": bank["observation"],
                "source_replay_attestation": bank["replay_attestation"],
                "tracker": tracker,
                "tracking_frames": tracking_frames,
                "track_file": str(path),
                "track_file_sha256": _sha256(path),
                "visible_count_by_frame": [int(np.sum(row)) for row in visibility],
            }
        )

    candidate_cameras: dict[tuple[int, str], CameraParams] = {}
    node_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    committed_nodes: set[tuple[int, str]] = set()
    for band_index, band in enumerate(bands):
        view = str(band["view"])
        seed_frames = [int(frame) for frame in band["seed_frames"]]
        band_node_rows: list[dict[str, Any]] = []
        for frame in [int(value) for value in band["frames"]]:
            frame_index = frames.index(frame)
            node_seed_frames = _active_seed_frames(
                seed_frames=seed_frames,
                frame=frame,
                max_track_distance=int(args.max_track_distance),
            )
            node_banks = []
            for seed_frame in node_seed_frames:
                bank = tracked_banks[(view, seed_frame)]
                node_banks.append(
                    {
                        "seed_frame": seed_frame,
                        "global_point_ids": bank["global_point_ids"],
                        "points3d": bank["points3d"],
                        "image_points": bank["tracks"][frame_index],
                        "visible": bank["visibility"][frame_index],
                    }
                )
            track_result = fit_cross_seed_track_candidate(
                banks=node_banks,
                base_camera=base_cameras[(frame, view)],
                camera_template=templates[view],
                reprojection_error_px=float(config.pnp_reprojection_error_px),
                min_inliers=int(config.min_pnp_inliers),
                rng_seed=int(config.rng_seed),
                pose_model=track_pose_model,
                fixed_center=shared_centers.get(view),
                iterations_count=int(config.pnp_iterations),
            )
            independent_decision = {
                "accepted": False,
                "reason": "track_candidate_unavailable",
            }
            if bool(track_result.get("camera") is not None):
                candidate_cameras[(frame, view)] = track_result["camera"]
            if bool(track_result.get("success")):
                bundle_root = per_frame_matcher_root / f"f{frame:06d}"
                bundle = _load_bundle(bundle_root)
                local = frame - int(bundle["meta"]["start_frame"])
                support_observations, target_observations, _observation = build_node_observations(
                    tracks=bundle["tracks"],
                    visibility=bundle["visibility"],
                    track_quality=bundle["track_quality"],
                    view_ids=bundle["views"],
                    static_views=static_views,
                    target_view=view,
                    frame_local=local,
                    min_support_views=int(config.min_support_views),
                    max_points=fit_budget,
                )
                scores = score_same_time_multiview_poses(
                    support_cameras=support_cameras,
                    support_observations=support_observations,
                    target_observations=target_observations,
                    target_cameras={
                        "base": base_cameras[(frame, view)],
                        "candidate": track_result["camera"],
                    },
                    config=config,
                )
                independent_decision = _independent_geometry_decision(scores)
                input_bundle_hashes[f"f{frame:06d}"] = _bundle_hashes(bundle_root)
            node_accepted = bool(
                track_result.get("success") and independent_decision["accepted"]
            )
            row = {
                "band_index": band_index,
                "frame": frame,
                "view": view,
                "seed_frames": node_seed_frames,
                "track_cross_seed": track_result,
                "independent_same_time_geometry": independent_decision,
                "node_accepted": node_accepted,
            }
            node_rows.append(row)
            band_node_rows.append(row)
        band_commit = bool(
            band_node_rows and all(bool(row["node_accepted"]) for row in band_node_rows)
        )
        if band_commit:
            committed_nodes.update(
                (int(row["frame"]), str(row["view"])) for row in band_node_rows
            )
        band_rows.append(
            {
                **band,
                "commit": band_commit,
                "reason": (
                    "all_nodes_cross_seed_and_independent_geometry_verified"
                    if band_commit
                    else "at_least_one_node_failed_geometry_verification"
                ),
                "accepted_node_count": int(
                    sum(bool(row["node_accepted"]) for row in band_node_rows)
                ),
                "node_count": int(len(band_node_rows)),
            }
        )

    baseline_root = output / "pose_static_pnp_completed"
    candidate_root = output / "pose_track_candidates"
    pose_root = output / "pose_track_band_verified"
    passthrough_checks: list[bool] = []
    for frame in frames:
        baseline_frame = dict(support_cameras)
        candidate_frame = dict(support_cameras)
        selected_frame = dict(support_cameras)
        for view in target_views:
            node = (frame, view)
            baseline_frame[view] = base_cameras[node]
            candidate_frame[view] = candidate_cameras.get(node, base_cameras[node])
            selected_frame[view] = (
                candidate_cameras[node]
                if node in committed_nodes
                else base_cameras[node]
            )
            if node not in committed_nodes:
                passthrough_checks.append(
                    _camera_equal(selected_frame[view], base_cameras[node])
                )
        save_cameras(
            baseline_root / f"f{frame:06d}" / "cameras_pred", baseline_frame
        )
        save_cameras(
            candidate_root / f"f{frame:06d}" / "cameras_pred", candidate_frame
        )
        save_cameras(pose_root / f"f{frame:06d}" / "cameras_pred", selected_frame)
    if not all(passthrough_checks):
        raise RuntimeError("a non-committed Static+PnP camera changed")
    if not prescribed_repair_frames and any(
        node in successful_nodes for node in committed_nodes
    ):
        raise RuntimeError("a successful Static+PnP node entered the commit set")

    manifest = {
        "protocol": method,
        "method": method,
        "runtime_scene": str(runtime),
        "runtime_tree_digest": runtime_checksums["tree_digest"],
        "runtime_protocol": runtime_manifest["protocol"],
        "stage_a_root": str(stage),
        "stage_a_seal": seal_record,
        "frames": frames,
        "static_views": static_views,
        "target_views": target_views,
        "sealed_target_views": attested_target_views,
        "base_pose_root": str(base_pose_root),
        "raw_static_pnp_success_count": int(len(successful_nodes)),
        "raw_static_pnp_failure_count": int(len(frames) * len(target_views) - len(successful_nodes)),
        "base_completion": base_completion,
        "repair_scope": {
            "mode": (
                "prescribed_successful_hard_band"
                if prescribed_repair_frames
                else "raw_static_pnp_failure_bands"
            ),
            "repair_frames": prescribed_repair_frames,
            "anchor_seed_frames": anchor_seed_frames,
            "max_track_distance": int(args.max_track_distance),
            "track_backend": track_backend,
            "track_pose_model": track_pose_model,
            "lk_forward_backward_threshold_px": float(args.lk_fb_threshold_px),
            "roma_query_cycle": (
                {
                    "setting": ROMA_SETTING,
                    "minimum_bidirectional_overlap": ROMA_MINIMUM_OVERLAP,
                    "maximum_cycle_error_px": ROMA_MAXIMUM_CYCLE_ERROR_PX,
                    "pair_match_count": int(roma_match_count),
                    "pair_cache_hit_count": int(roma_cache_hit_count),
                }
                if track_backend == "roma_cycle"
                else None
            ),
            "same_frame_seed_excluded_from_candidate": True,
            "selection_uses_gt_or_nvs": False,
        },
        "seed_policy": {
            "name": "nearest_successful_static_pnp_geometry_balanced_across_band_boundaries",
            "max_seed_banks": 4,
            "seed_point_policy": seed_point_policy,
            "seed_geometry": (
                "replayed robust static-view triangulation candidates at an "
                "attested successful Static+PnP node"
                if seed_point_policy == "triangulated_static_candidates"
                else "replayed aligned RANSAC inliers from attested Static+PnP"
            ),
            "new_scene_specific_numeric_hyperparameters": 0,
        },
        "fixed_center_model": {
            "enabled": track_pose_model == "fixed_center_rotation",
            "estimator": (
                "per-physical-PTZ componentwise median of all 30 frozen "
                "Static+PnP centers"
                if track_pose_model == "fixed_center_rotation"
                else None
            ),
            "cross_view_center_pooling": False,
            "shared_centers": shared_centers,
            "report": shared_center_report,
            "GT_used": False,
            "new_tunable_numeric_hyperparameters": 0,
        },
        "verification_policy": {
            "track": (
                "visibility-eligible leave-one-seed-bank-out median gain and "
                "nonnegative p90 gain; at least two eligible banks and strict majority"
            ),
            "independent": "unused same-time matcher truncated mean improves and inliers do not regress",
            "commit": "all nodes in a contiguous raw failure band must pass both gates",
            "pnp_rng_seed": int(config.rng_seed),
            "track_pose_model": track_pose_model,
        },
        "config": prediction["config"],
        "fit_point_budget": fit_budget,
        "bands": band_rows,
        "nodes": node_rows,
        "seed_banks": bank_rows,
        "input_per_frame_bundle_hashes": input_bundle_hashes,
        "per_frame_matcher_root": str(per_frame_matcher_root),
        "baseline_pose_root": str(baseline_root),
        "candidate_pose_root": str(candidate_root),
        "pose_root": str(pose_root),
        "committed_nodes": [
            {"frame": frame, "view": view} for frame, view in sorted(committed_nodes)
        ],
        "baseline_output_camera_hashes": {
            f"f{frame:06d}": camera_root_attestation(
                baseline_root / f"f{frame:06d}" / "cameras_pred"
            )
            for frame in frames
        },
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
            "strong_same_time_static_pnp_uses_all_available_static_views": True,
            "baseline_and_ours_start_from_same_completed_camera_grid": True,
            "successful_static_pnp_cameras_preserved_exactly": (
                all(node not in committed_nodes for node in successful_nodes)
                if not prescribed_repair_frames
                else None
            ),
            "commit_scope": (
                "prescribed_public_hard_band"
                if prescribed_repair_frames
                else "raw_static_pnp_failure_bands_only"
            ),
            "noncommitted_base_cameras_preserved_exactly": bool(all(passthrough_checks)),
            "tracking_seed_comes_only_from_attested_static_pnp_geometry": True,
            "target_view_shard_is_subset_of_fully_attested_roster": bool(
                set(target_views).issubset(set(attested_target_views))
            ),
            "independent_same_time_geometry_not_used_for_candidate_fit": True,
            "fixed_center_estimated_only_from_frozen_static_pnp": (
                track_pose_model == "fixed_center_rotation"
            ),
            "different_physical_ptz_centers_never_pooled": True,
        },
        "artifacts": {
            "cotracker_root": (
                str(cotracker_root) if track_backend in COTRACKER_BACKENDS else None
            ),
            "cotracker_checkpoint": (
                str(args.cotracker_checkpoint.resolve())
                if track_backend in COTRACKER_BACKENDS
                else None
            ),
            "cotracker_checkpoint_sha256": (
                _sha256(args.cotracker_checkpoint.resolve())
                if track_backend in COTRACKER_BACKENDS
                else None
            ),
            "romav2_root": (
                str(args.romav2_root.resolve())
                if track_backend == "roma_cycle"
                else None
            ),
            "romav2_weights": (
                str(args.romav2_weights.resolve())
                if track_backend == "roma_cycle"
                else None
            ),
            "romav2_weights_sha256": (
                _sha256(args.romav2_weights.resolve())
                if track_backend == "roma_cycle"
                else None
            ),
        },
        "runtime_seconds": float(time.perf_counter() - started_at),
        "runtime_seconds_per_frame": float(
            (time.perf_counter() - started_at) / max(1, len(frames))
        ),
        "code": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "seed_repair_module": str(
                (PROJECT_ROOT / "track4d_calib/static_pnp_inlier_track_repair.py").resolve()
            ),
            "seed_repair_module_sha256": _sha256(
                PROJECT_ROOT / "track4d_calib/static_pnp_inlier_track_repair.py"
            ),
            "same_time_pnp_sha256": _sha256(
                PROJECT_ROOT / "track4d_calib/same_time_multiview_pnp.py"
            ),
            "tracked_pnp_sha256": _sha256(
                PROJECT_ROOT / "track4d_calib/hloc_temporal_track_repair.py"
            ),
            "roma_query_relay_sha256": _sha256(
                PROJECT_ROOT / "track4d_calib/roma_query_relay.py"
            ),
            "fixed_center_pnp_sha256": _sha256(
                PROJECT_ROOT / "track4d_calib/fixed_center_pnp.py"
            ),
            "fixed_center_estimator_sha256": _sha256(
                PROJECT_ROOT
                / "scripts/geometry/export_static_pnp_fixed_center_refit_control.py"
            ),
            "cotracker_query_tracking_sha256": _sha256(
                PROJECT_ROOT / "track4d_calib/cotracker_query_tracking.py"
            ),
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
    parser.add_argument(
        "--allow-sealed-target-view-subset",
        action="store_true",
        help=(
            "Process a target-view shard after attesting the full target roster "
            "from the Stage-A seal and prediction payload."
        ),
    )
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--cotracker-root", type=Path, required=True)
    parser.add_argument("--cotracker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--track-backend",
        choices=("cotracker", "cotracker_lk", "lk", "roma_cycle"),
        default="cotracker",
    )
    parser.add_argument(
        "--romav2-root",
        type=Path,
        default=PROJECT_ROOT / "third_party/RoMaV2/src",
        help="RoMaV2 source root containing the romav2 package.",
    )
    parser.add_argument(
        "--romav2-weights",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/romav2.pt",
        help="Frozen local RoMaV2 checkpoint used by roma_cycle.",
    )
    parser.add_argument("--lk-fb-threshold-px", type=float, default=1.5)
    parser.add_argument(
        "--track-pose-model",
        choices=("full_pnp", "fixed_center_rotation"),
        default="full_pnp",
        help=(
            "Candidate pose model. fixed_center_rotation estimates one public "
            "temporal center per physical PTZ and solves only rotation."
        ),
    )
    parser.add_argument(
        "--repair-frames",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Opt-in frozen contiguous band to repair even when Static+PnP returned "
            "success. Empty preserves failure-band-only behavior."
        ),
    )
    parser.add_argument(
        "--anchor-seed-frames",
        nargs="*",
        type=int,
        default=[],
        help="Optional frozen successful Static+PnP anchors for a repair band.",
    )
    parser.add_argument(
        "--max-track-distance",
        type=int,
        default=0,
        help="Maximum frame distance from each anchor; zero keeps full tracks.",
    )
    parser.add_argument(
        "--per-frame-matcher-root",
        type=Path,
        help="Read-only matcher bundle root; defaults to STAGE_A/per_frame_matcher.",
    )
    parser.add_argument(
        "--seed-point-policy",
        choices=sorted(METHOD_BY_SEED_POLICY),
        default="pnp_inliers",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    payload = run(parse_args())
    print(
        json.dumps(
            {
                "pose_root": payload["pose_root"],
                "committed_nodes": payload["committed_nodes"],
                "bands": payload["bands"],
            },
            indent=2,
            sort_keys=True,
        )
    )
