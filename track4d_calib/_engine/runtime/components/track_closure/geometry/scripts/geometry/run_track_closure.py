#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from track4d_calib.pose_uncertainty import estimate_pnp_jackknife_stability
from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    pretriangulate_static_support_points,
    solve_pretriangulated_same_time_multiview_pnp,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def track_file_attestation(track_root: Path) -> dict[str, str]:
    names = ["meta.json", "views.txt", "tracks.npy", "visibility.npy"]
    if (track_root / "track_quality.npy").is_file():
        names.append("track_quality.npy")
    return {name: _sha256(track_root / name) for name in names}


def camera_root_attestation(camera_root: Path) -> dict[str, str]:
    """Hash the complete camera payload consumed by downstream reconstruction."""

    names = ["intri.yml", "extri.yml"]
    return {name: _sha256(camera_root / name) for name in names}


def frame_camera_directory(root: Path, frame: int) -> Path:
    tagged = root / f"f{frame:06d}" / "cameras_pred"
    if tagged.is_dir():
        return tagged
    plain = root / f"f{frame:06d}"
    if plain.is_dir():
        return plain
    raise FileNotFoundError(f"missing fallback cameras for frame {frame} under {root}")


def validate_runtime_track_views(
    *,
    bundle_views: Sequence[str],
    static_views: Sequence[str],
    target_views: Sequence[str],
) -> None:
    views = [str(view) for view in bundle_views]
    if len(views) != len(set(views)):
        raise ValueError("track bundle view IDs must be unique")
    expected = {str(view) for view in [*static_views, *target_views]}
    actual = set(views)
    undeclared = sorted(actual - expected)
    missing = sorted(expected - actual)
    if undeclared:
        raise ValueError(f"track bundle contains undeclared runtime view: {undeclared}")
    if missing:
        raise ValueError(f"track bundle is missing declared runtime view: {missing}")


def build_node_observations(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    static_views: Sequence[str],
    target_view: str,
    frame_local: int,
    min_support_views: int = 2,
    track_quality: np.ndarray | None = None,
    max_points: int = 0,
    allowed_point_ids: np.ndarray | None = None,
    allowed_point_range: tuple[int, int] | None = None,
    excluded_point_ranges: Sequence[tuple[int, int]] | None = None,
    balanced_point_ranges: Sequence[tuple[int, int]] | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, np.ndarray], dict[str, Any]]:
    track_array = np.asarray(tracks)
    visible_array = np.asarray(visibility, dtype=bool)
    if track_array.ndim != 4 or track_array.shape[-1] != 2:
        raise ValueError(f"tracks must have shape (V,T,N,2), got {track_array.shape}")
    if visible_array.shape != track_array.shape[:-1]:
        raise ValueError("visibility shape does not match tracks")
    views = [str(view) for view in view_ids]
    view_to_index = {view: index for index, view in enumerate(views)}
    target = str(target_view)
    if target not in view_to_index:
        raise KeyError(f"target view {target} is absent from track bundle")
    legal_static = [str(view) for view in static_views if str(view) in view_to_index]
    target_index = view_to_index[target]
    target_valid = (
        visible_array[target_index, frame_local]
        & np.isfinite(track_array[target_index, frame_local]).all(axis=1)
    )
    target_ids = np.flatnonzero(target_valid)
    allowed_count: int | None = None
    if allowed_point_ids is not None:
        allowed_ids = np.asarray(allowed_point_ids, dtype=np.int64).reshape(-1)
        if len(np.unique(allowed_ids)) != len(allowed_ids):
            raise ValueError("allowed_point_ids must be unique")
        if np.any(allowed_ids < 0) or np.any(allowed_ids >= track_array.shape[2]):
            raise ValueError("allowed_point_ids lies outside the track point axis")
        allowed_mask = np.zeros(track_array.shape[2], dtype=bool)
        allowed_mask[allowed_ids] = True
        target_ids = target_ids[allowed_mask[target_ids]]
        allowed_count = int(len(allowed_ids))
    if allowed_point_range is not None:
        point_start, point_stop = (int(value) for value in allowed_point_range)
        if not (0 <= point_start <= point_stop <= track_array.shape[2]):
            raise ValueError(
                f"invalid allowed point range {(point_start, point_stop)} for "
                f"{track_array.shape[2]} points"
            )
        target_ids = target_ids[
            (target_ids >= int(point_start)) & (target_ids < int(point_stop))
        ]
    if excluded_point_ranges:
        keep = np.ones(len(target_ids), dtype=bool)
        for raw_start, raw_stop in excluded_point_ranges:
            point_start, point_stop = int(raw_start), int(raw_stop)
            if not (0 <= point_start <= point_stop <= track_array.shape[2]):
                raise ValueError(
                    f"invalid excluded point range {(point_start, point_stop)}"
                )
            keep &= ~(
                (target_ids >= point_start) & (target_ids < point_stop)
            )
        target_ids = target_ids[keep]
    candidates: list[tuple[float, int, dict[str, np.ndarray], np.ndarray]] = []
    for point_id in target_ids.tolist():
        support: dict[str, np.ndarray] = {}
        quality_values: list[float] = []
        for view in legal_static:
            view_index = view_to_index[view]
            point = np.asarray(
                track_array[view_index, frame_local, point_id],
                dtype=np.float64,
            ).reshape(2)
            if not bool(visible_array[view_index, frame_local, point_id]):
                continue
            if not np.isfinite(point).all():
                continue
            support[view] = point
            if track_quality is not None:
                value = float(track_quality[view_index, frame_local, point_id])
                if math.isfinite(value):
                    quality_values.append(value)
        if len(support) < max(2, int(min_support_views)):
            continue
        target_point = np.asarray(
            track_array[target_index, frame_local, point_id],
            dtype=np.float64,
        ).reshape(2)
        if track_quality is not None:
            target_quality = float(track_quality[target_index, frame_local, point_id])
            if math.isfinite(target_quality):
                quality_values.append(target_quality)
        quality = float(np.mean(quality_values)) if quality_values else 1.0
        score = float(len(support)) + quality
        candidates.append((score, int(point_id), support, target_point))

    candidates.sort(key=lambda row: (-row[0], row[1]))
    selected_count_by_point_range: dict[str, int] | None = None
    if balanced_point_ranges:
        ranges = [(int(start), int(stop)) for start, stop in balanced_point_ranges]
        if any(start < 0 or stop < start or stop > track_array.shape[2] for start, stop in ranges):
            raise ValueError("balanced point range lies outside the track point axis")
        buckets: list[list[tuple[float, int, dict[str, np.ndarray], np.ndarray]]] = [
            [] for _range in ranges
        ]
        for candidate in candidates:
            memberships = [
                index
                for index, (start, stop) in enumerate(ranges)
                if start <= int(candidate[1]) < stop
            ]
            if len(memberships) != 1:
                raise ValueError("balanced point ranges must uniquely cover candidates")
            buckets[memberships[0]].append(candidate)
        budget = int(max_points) if int(max_points) > 0 else len(candidates)
        balanced: list[tuple[float, int, dict[str, np.ndarray], np.ndarray]] = []
        offsets = [0 for _bucket in buckets]
        while len(balanced) < budget:
            added = False
            for index, bucket in enumerate(buckets):
                if offsets[index] >= len(bucket) or len(balanced) >= budget:
                    continue
                balanced.append(bucket[offsets[index]])
                offsets[index] += 1
                added = True
            if not added:
                break
        candidates = balanced
        selected_count_by_point_range = {
            f"{start}:{stop}": int(offsets[index])
            for index, (start, stop) in enumerate(ranges)
        }
    elif int(max_points) > 0:
        candidates = candidates[: int(max_points)]
    support_output = {point_id: support for _score, point_id, support, _target in candidates}
    target_output = {point_id: target_point for _score, point_id, _support, target_point in candidates}
    support_counts = [len(support) for support in support_output.values()]
    diagnostics = {
        "target_visible_count": int(len(target_ids)),
        "enough_support_count": int(len(support_output)),
        "selected_count": int(len(support_output)),
        "support_count_mean": float(np.mean(support_counts)) if support_counts else 0.0,
    }
    if selected_count_by_point_range is not None:
        diagnostics["selected_count_by_point_range"] = selected_count_by_point_range
    if allowed_count is not None:
        diagnostics["allowed_point_count"] = allowed_count
    return support_output, target_output, diagnostics


def merge_support_observations_for_frame(
    observations_by_target: Mapping[
        str, tuple[Mapping[int, Mapping[str, np.ndarray]], Mapping[int, np.ndarray]]
    ],
) -> dict[int, dict[str, np.ndarray]]:
    """Merge target-local selections into one target-agnostic track cache."""

    merged: dict[int, dict[str, np.ndarray]] = {}
    for target_view in sorted(observations_by_target):
        support_observations, _target_observations = observations_by_target[target_view]
        for raw_point_id, support in support_observations.items():
            point_id = int(raw_point_id)
            normalized = {
                str(view): np.asarray(point, dtype=np.float64).reshape(2)
                for view, point in support.items()
            }
            if point_id not in merged:
                merged[point_id] = normalized
                continue
            if set(merged[point_id]) != set(normalized):
                raise ValueError(
                    f"point {point_id} has target-dependent static support views"
                )
            for view in normalized:
                if not np.array_equal(merged[point_id][view], normalized[view]):
                    raise ValueError(
                        f"point {point_id} has target-dependent observation in {view}"
                    )
    return merged


def _rotation_error_deg(predicted: np.ndarray, reference: np.ndarray) -> float:
    delta = np.asarray(predicted, dtype=np.float64) @ np.asarray(reference, dtype=np.float64).T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _parse_views(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def resolve_runtime_static_views(
    observed_views: Sequence[str],
    requested_views: Sequence[str] | None,
) -> list[str]:
    observed = [str(view) for view in observed_views]
    if requested_views is None:
        return observed
    requested = [str(view) for view in requested_views]
    if len(requested) != len(set(requested)):
        raise ValueError("requested static views must be unique")
    missing = sorted(set(requested) - set(observed))
    if missing:
        raise ValueError(f"requested static views are not observed: {missing}")
    if not requested:
        raise ValueError("requested static views must be non-empty")
    return requested


def _parse_frames(value: str, *, start: int, count: int) -> list[int]:
    if not value.strip():
        return list(range(start, start + count))
    return sorted({int(item) for item in value.replace(",", " ").split()})


def seed_point_ranges(
    *,
    meta: Mapping[str, Any],
    total_points: int,
) -> list[dict[str, Any]]:
    """Decode the immutable contiguous point bank assigned to each seed frame."""

    seed_frames = [int(value) for value in meta.get("seed_frames", [])]
    if not seed_frames:
        seed_frames = [int(meta.get("seed_frame", meta.get("start_frame", 0)))]
    point_counts = [int(value) for value in meta.get("num_points_by_seed", [])]
    if not point_counts and len(seed_frames) == 1:
        point_counts = [int(total_points)]
    if len(seed_frames) != len(point_counts):
        raise ValueError("seed_frames and num_points_by_seed must have equal length")
    if any(count < 0 for count in point_counts) or sum(point_counts) != int(total_points):
        raise ValueError("num_points_by_seed must partition the track point axis")

    output: list[dict[str, Any]] = []
    point_start = 0
    for seed_frame, point_count in zip(seed_frames, point_counts):
        point_stop = point_start + int(point_count)
        output.append(
            {
                "seed_frame": int(seed_frame),
                "point_range": (int(point_start), int(point_stop)),
            }
        )
        point_start = point_stop
    return output


def exact_seed_point_ranges(
    *,
    meta: Mapping[str, Any],
    frame: int,
    total_points: int,
) -> list[tuple[int, int]]:
    """Return point ranges seeded at exactly the query frame."""

    return [
        tuple(int(value) for value in row["point_range"])
        for row in seed_point_ranges(meta=meta, total_points=total_points)
        if int(row["seed_frame"]) == int(frame)
    ]


def nearest_seed_point_range(
    *,
    meta: Mapping[str, Any],
    frame: int,
    total_points: int,
) -> tuple[tuple[int, int], int]:
    """Return the contiguous point bank belonging to the nearest track seed."""

    ranges = seed_point_ranges(meta=meta, total_points=total_points)
    selected = min(
        ranges,
        key=lambda row: (
            abs(int(row["seed_frame"]) - int(frame)),
            int(row["seed_frame"]),
        ),
    )
    return tuple(selected["point_range"]), int(selected["seed_frame"])


def verify_multiseed_inlier_support(
    *,
    inlier_point_ids: np.ndarray,
    point_ranges: Sequence[Mapping[str, Any]],
    min_seed_banks: int = 2,
    min_inliers_per_seed: int = 4,
) -> dict[str, Any]:
    """Require PnP inliers from independent temporal seed point banks."""

    required_banks = int(min_seed_banks)
    required_inliers = int(min_inliers_per_seed)
    if required_banks < 1 or required_inliers < 1:
        raise ValueError("multiseed verification minima must be positive")
    point_ids = np.unique(np.asarray(inlier_point_ids, dtype=np.int64).reshape(-1))
    counts: dict[str, int] = {}
    assigned = np.zeros(len(point_ids), dtype=bool)
    for row in point_ranges:
        seed_frame = int(row["seed_frame"])
        point_start, point_stop = (int(value) for value in row["point_range"])
        if point_start < 0 or point_stop < point_start:
            raise ValueError(f"invalid seed point range {(point_start, point_stop)}")
        membership = (point_ids >= point_start) & (point_ids < point_stop)
        if np.any(assigned & membership):
            raise ValueError("seed point ranges overlap")
        assigned |= membership
        counts[str(seed_frame)] = int(np.count_nonzero(membership))
    qualified = [
        int(seed_frame)
        for seed_frame, count in counts.items()
        if int(count) >= required_inliers
    ]
    unassigned_count = int(np.count_nonzero(~assigned))
    verified = len(qualified) >= required_banks and unassigned_count == 0
    return {
        "verified": bool(verified),
        "reason": (
            "ok"
            if verified
            else "insufficient_independent_seed_bank_inliers"
        ),
        "min_seed_banks": required_banks,
        "min_inliers_per_seed": required_inliers,
        "qualified_seed_bank_count": int(len(qualified)),
        "qualified_seed_frames": qualified,
        "inlier_count_by_seed": counts,
        "unassigned_inlier_count": unassigned_count,
    }


def _result_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_success",
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
    ]
    return {key: result[key] for key in keys if key in result}


def attach_deferred_pnp_jackknife_audits(
    *,
    prediction_rows: list[dict[str, Any]],
    audit_jobs: Sequence[Mapping[str, Any]],
    estimator: Any = estimate_pnp_jackknife_stability,
) -> None:
    """Run diagnostic PnP fits only after all exported poses are solved.

    OpenCV exposes a process-global RANSAC RNG. A separate diagnostic pass prevents
    those extra fits from changing a later base PnP solve.
    """

    seen: set[int] = set()
    for job in audit_jobs:
        row_index = int(job["prediction_row_index"])
        if row_index in seen:
            raise ValueError(f"duplicate PnP jackknife row index: {row_index}")
        if row_index < 0 or row_index >= len(prediction_rows):
            raise IndexError(f"PnP jackknife row index is out of range: {row_index}")
        if prediction_rows[row_index].get("pnp_jackknife") is not None:
            raise ValueError(f"PnP jackknife row is already populated: {row_index}")
        arguments = job.get("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("PnP jackknife job arguments are missing")
        prediction_rows[row_index]["pnp_jackknife"] = estimator(**dict(arguments))
        seen.add(row_index)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track-driven same-time multiview triangulation and full-pose PnP."
    )
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--track-root", type=Path)
    parser.add_argument("--per-frame-track-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--static-views",
        type=_parse_views,
        help="explicit observed-static subset consumed by the track bundle",
    )
    parser.add_argument("--target-views", type=_parse_views, required=True)
    parser.add_argument("--frames", default="")
    parser.add_argument("--sealed-reference", type=Path)
    parser.add_argument(
        "--fallback-pose-root",
        type=Path,
        help=(
            "Public safe pose tree copied exactly when a track-PnP proposal "
            "fails its geometric acceptance gate."
        ),
    )
    parser.add_argument("--min-support-views", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--support-reprojection-px", type=float, default=2.5)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=0.5)
    parser.add_argument("--pnp-reprojection-px", type=float, default=4.0)
    parser.add_argument("--min-pnp-inliers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--audit-pnp-jackknife",
        action="store_true",
        help=(
            "Audit deterministic three-fold PnP pose stability without changing "
            "the exported full-data pose."
        ),
    )
    parser.add_argument(
        "--exclude-exact-seed-bank",
        action="store_true",
        help=(
            "Exclude points initialized at the query frame so the exported "
            "candidate is supported by temporal propagation only."
        ),
    )
    parser.add_argument(
        "--temporal-seed-policy",
        choices=["nearest", "all", "verified_all"],
        default="nearest",
        help=(
            "Point-bank policy for merged multi-seed temporal tracks. "
            "verified_all commits pooled PnP only when its inliers span at least "
            "two independent seed banks."
        ),
    )
    args = parser.parse_args()

    runtime = args.runtime_scene.resolve()
    output = args.output_root.resolve()
    runtime_manifest = json.loads((runtime / "runtime_manifest.json").read_text(encoding="utf-8"))
    static_views = resolve_runtime_static_views(
        runtime_manifest["observed_short_views"],
        args.static_views,
    )
    legal_targets = {str(view) for view in runtime_manifest["long_views"]}
    target_views = [str(view) for view in args.target_views]
    if set(target_views) - legal_targets:
        raise ValueError("target views are absent from runtime manifest")

    if (args.track_root is None) == (args.per_frame_track_root is None):
        raise ValueError("provide exactly one of --track-root or --per-frame-track-root")

    def load_track_bundle(path: Path) -> dict[str, Any]:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        views = (path / "views.txt").read_text(encoding="utf-8").split()
        if set(views) - set(runtime_manifest["runtime_views"]):
            raise ValueError(f"track bundle {path} contains a non-runtime view")
        validate_runtime_track_views(
            bundle_views=views,
            static_views=static_views,
            target_views=target_views,
        )
        quality_path = path / "track_quality.npy"
        return {
            "root": path,
            "meta": meta,
            "views": views,
            "tracks": np.load(path / "tracks.npy", mmap_mode="r"),
            "visibility": np.load(path / "visibility.npy", mmap_mode="r"),
            "track_quality": (
                np.load(quality_path, mmap_mode="r") if quality_path.is_file() else None
            ),
        }

    if args.track_root is not None:
        shared_bundle = load_track_bundle(args.track_root.resolve())
        start_frame = int(shared_bundle["meta"]["start_frame"])
        num_frames = int(shared_bundle["meta"]["num_frames"])
        frames = _parse_frames(args.frames, start=start_frame, count=num_frames)
        if any(frame < start_frame or frame >= start_frame + num_frames for frame in frames):
            raise ValueError("requested frame lies outside track bundle")
        bundle_for_frame = {frame: shared_bundle for frame in frames}
        track_source_mode = "shared_temporal_bundle"
        if str(args.temporal_seed_policy) == "nearest":
            temporal_selection_by_frame = {
                frame: nearest_seed_point_range(
                    meta=shared_bundle["meta"],
                    frame=frame,
                    total_points=int(shared_bundle["tracks"].shape[2]),
                )
                for frame in frames
            }
        else:
            temporal_selection_by_frame = {
                frame: (None, None) for frame in frames
            }
        temporal_point_ranges = seed_point_ranges(
            meta=shared_bundle["meta"],
            total_points=int(shared_bundle["tracks"].shape[2]),
        )
    else:
        per_frame_root = args.per_frame_track_root.resolve()
        if args.frames.strip():
            frames = _parse_frames(args.frames, start=0, count=0)
        else:
            frames = sorted(
                int(path.name[1:])
                for path in per_frame_root.glob("f[0-9][0-9][0-9][0-9][0-9][0-9]")
                if (path / "meta.json").is_file()
            )
        if not frames:
            raise ValueError("no per-frame track bundles were selected")
        bundle_for_frame = {
            frame: load_track_bundle(per_frame_root / f"f{frame:06d}")
            for frame in frames
        }
        for frame, bundle in bundle_for_frame.items():
            if int(bundle["meta"]["start_frame"]) != frame or int(bundle["meta"]["num_frames"]) != 1:
                raise ValueError(f"frame {frame} is not backed by a one-frame matcher bundle")
        track_source_mode = "independent_per_frame_bundle"
        temporal_selection_by_frame = {
            frame: (None, frame) for frame in frames
        }
        temporal_point_ranges = []

    if bool(args.exclude_exact_seed_bank) and track_source_mode != "shared_temporal_bundle":
        raise ValueError(
            "--exclude-exact-seed-bank requires a shared temporal track bundle"
        )

    support_cameras = CameraLoader(runtime / "cameras/static").load_cameras(static_views)
    intrinsics = CameraIntrinsicsLoader(runtime / "cameras")
    templates = {view: intrinsics.load_camera_template(view) for view in target_views}
    fallback_root = (
        args.fallback_pose_root.resolve()
        if args.fallback_pose_root is not None
        else None
    )
    config = SameTimePnPConfig(
        min_support_views=int(args.min_support_views),
        min_triangulation_angle_deg=float(args.min_triangulation_angle_deg),
        support_reprojection_error_px=float(args.support_reprojection_px),
        pnp_reprojection_error_px=float(args.pnp_reprojection_px),
        min_pnp_inliers=int(args.min_pnp_inliers),
        rng_seed=int(args.seed),
    )

    prediction_rows: list[dict[str, Any]] = []
    pnp_jackknife_jobs: list[dict[str, Any]] = []
    triangulation_cache_rows: list[dict[str, Any]] = []
    pose_root = output / "pose_track_same_time_pnp"
    for frame in frames:
        bundle = bundle_for_frame[frame]
        frame_local = int(frame - int(bundle["meta"]["start_frame"]))
        allowed_point_range, selected_seed_frame = temporal_selection_by_frame[frame]
        excluded_point_ranges = (
            exact_seed_point_ranges(
                meta=bundle["meta"],
                frame=frame,
                total_points=int(bundle["tracks"].shape[2]),
            )
            if bool(args.exclude_exact_seed_bank)
            else []
        )
        frame_cameras: dict[str, CameraParams] = dict(support_cameras)
        fallback_cameras = (
            CameraLoader(frame_camera_directory(fallback_root, frame)).load_cameras(
                target_views
            )
            if fallback_root is not None
            else {}
        )
        node_inputs: dict[
            str,
            tuple[
                dict[int, dict[str, np.ndarray]],
                dict[int, np.ndarray],
                dict[str, Any],
            ],
        ] = {}
        for view in target_views:
            support_observations, target_observations, observation_diag = build_node_observations(
                tracks=bundle["tracks"],
                visibility=bundle["visibility"],
                track_quality=bundle["track_quality"],
                view_ids=bundle["views"],
                static_views=static_views,
                target_view=view,
                frame_local=frame_local,
                min_support_views=int(args.min_support_views),
                max_points=int(args.max_points),
                allowed_point_range=allowed_point_range,
                excluded_point_ranges=excluded_point_ranges,
                balanced_point_ranges=(
                    [tuple(row["point_range"]) for row in temporal_point_ranges]
                    if track_source_mode == "shared_temporal_bundle"
                    and str(args.temporal_seed_policy) == "verified_all"
                    else None
                ),
            )
            observation_diag.update(
                {
                    "temporal_seed_policy": (
                        str(args.temporal_seed_policy)
                        if track_source_mode == "shared_temporal_bundle"
                        else "independent_same_frame"
                    ),
                    "selected_seed_frame": selected_seed_frame,
                    "selected_point_range": (
                        list(allowed_point_range)
                        if allowed_point_range is not None
                        else None
                    ),
                    "excluded_exact_seed_point_ranges": [
                        list(point_range) for point_range in excluded_point_ranges
                    ],
                }
            )
            node_inputs[view] = (
                support_observations,
                target_observations,
                observation_diag,
            )

        merged_support = merge_support_observations_for_frame(
            {
                view: (node_inputs[view][0], node_inputs[view][1])
                for view in target_views
            }
        )
        triangulation_started = time.perf_counter()
        pretriangulated = pretriangulate_static_support_points(
            support_cameras=support_cameras,
            support_observations=merged_support,
            config=config,
        )
        triangulation_seconds = float(time.perf_counter() - triangulation_started)
        selected_total = int(
            sum(len(node_inputs[view][0]) for view in target_views)
        )
        unique_total = int(len(merged_support))
        triangulation_cache_rows.append(
            {
                "frame": int(frame),
                "target_selected_point_count_sum": selected_total,
                "unique_static_track_count": unique_total,
                "avoided_retriangulation_count": int(selected_total - unique_total),
                "reuse_factor": float(selected_total / max(unique_total, 1)),
                "triangulated_point_count": int(
                    pretriangulated["triangulated_point_count"]
                ),
                "seconds": triangulation_seconds,
            }
        )

        for view in target_views:
            support_observations, target_observations, observation_diag = node_inputs[view]
            result = solve_pretriangulated_same_time_multiview_pnp(
                pretriangulated=pretriangulated,
                support_observations=support_observations,
                target_observations=target_observations,
                target_template=templates[view],
                config=config,
            )
            pnp_jackknife_arguments = None
            if bool(args.audit_pnp_jackknife) and bool(result["success"]):
                pnp_jackknife_arguments = {
                    "point_ids": np.asarray(result["point_ids"], dtype=np.int64).copy(),
                    "points3d": np.asarray(
                        result["object_points"], dtype=np.float64
                    ).copy(),
                    "image_points": np.asarray(
                        result["image_points"], dtype=np.float64
                    ).copy(),
                    "intrinsic": np.asarray(templates[view].K, dtype=np.float64).copy(),
                    "distortion": np.asarray(templates[view].D, dtype=np.float64).copy(),
                    "reference_rotation": np.asarray(
                        result["camera"].R, dtype=np.float64
                    ).copy(),
                    "reference_translation": np.asarray(
                        result["camera"].T, dtype=np.float64
                    ).copy(),
                    "num_folds": 3,
                    "reprojection_error_px": float(config.pnp_reprojection_error_px),
                    "min_inliers": int(config.min_pnp_inliers),
                    "iterations_count": int(config.pnp_iterations),
                    "confidence": float(config.pnp_confidence),
                    "rng_seed": int(config.rng_seed),
                }
            verification: dict[str, Any] | None = None
            if (
                track_source_mode == "shared_temporal_bundle"
                and str(args.temporal_seed_policy) == "verified_all"
            ):
                candidate_success = bool(result["success"])
                if candidate_success:
                    verification = verify_multiseed_inlier_support(
                        inlier_point_ids=np.asarray(result["inlier_point_ids"]),
                        point_ranges=temporal_point_ranges,
                    )
                else:
                    verification = {
                        "verified": False,
                        "reason": "candidate_solver_failed",
                        "min_seed_banks": 2,
                        "min_inliers_per_seed": 4,
                        "qualified_seed_bank_count": 0,
                        "qualified_seed_frames": [],
                        "inlier_count_by_seed": {
                            str(row["seed_frame"]): 0
                            for row in temporal_point_ranges
                        },
                        "unassigned_inlier_count": 0,
                    }
                result = dict(result)
                result["candidate_success"] = candidate_success
                result["success"] = bool(
                    candidate_success and verification["verified"]
                )
                if candidate_success and not result["success"]:
                    result["reason"] = str(verification["reason"])
            if bool(result["success"]):
                frame_cameras[view] = result["camera"]
                pose_source = "track_pnp_geometry_commit"
            elif fallback_root is not None:
                frame_cameras[view] = fallback_cameras[view]
                pose_source = "exact_public_safe_fallback"
            else:
                pose_source = "missing_after_failed_track_pnp"
            prediction_row_index = len(prediction_rows)
            prediction_rows.append(
                {
                    "frame": int(frame),
                    "view": view,
                    "pose_source": pose_source,
                    "observation": observation_diag,
                    "solver": _result_diagnostics(result),
                    "pnp_jackknife": None,
                    "verification": verification,
                }
            )
            if pnp_jackknife_arguments is not None:
                pnp_jackknife_jobs.append(
                    {
                        "prediction_row_index": prediction_row_index,
                        "arguments": pnp_jackknife_arguments,
                    }
                )
        save_cameras(pose_root / f"f{frame:06d}" / "cameras_pred", frame_cameras)

    attach_deferred_pnp_jackknife_audits(
        prediction_rows=prediction_rows,
        audit_jobs=pnp_jackknife_jobs,
    )

    unique_track_roots = sorted(
        {str(bundle["root"].resolve()) for bundle in bundle_for_frame.values()}
    )
    track_file_attestations = {}
    for root_text in unique_track_roots:
        root = Path(root_text)
        track_file_attestations[root_text] = track_file_attestation(root)
    prediction_manifest = {
        "protocol": "hidden-parent-track-driven-same-time-full-pnp-v1",
        "runtime_scene": str(runtime),
        "runtime_tree_digest": json.loads(
            (runtime / "checksums.sha256.json").read_text(encoding="utf-8")
        )["tree_digest"],
        "track_source_mode": track_source_mode,
        "track_roots": unique_track_roots,
        "track_files_sha256": track_file_attestations,
        "pose_root": str(pose_root.resolve()),
        "failed_track_pnp_fallback": {
            "enabled": fallback_root is not None,
            "pose_root": str(fallback_root) if fallback_root is not None else None,
            "selection_uses_GT_or_NVS": False,
            "committed_track_pnp_nodes": int(
                sum(row["pose_source"] == "track_pnp_geometry_commit" for row in prediction_rows)
            ),
            "exact_fallback_nodes": int(
                sum(row["pose_source"] == "exact_public_safe_fallback" for row in prediction_rows)
            ),
        },
        "output_camera_hashes": {
            f"f{frame:06d}": camera_root_attestation(
                pose_root / f"f{frame:06d}" / "cameras_pred"
            )
            for frame in frames
        },
        "static_views": static_views,
        "target_views": target_views,
        "frames": frames,
        "config": vars(config),
        "fit_point_budget": int(args.max_points),
        "pnp_jackknife_audit": {
            "enabled": bool(args.audit_pnp_jackknife),
            "num_folds": 3,
            "fold_policy": (
                "balanced_stable_point_id_hash_rank_leave_one_fold_out"
            ),
            "changes_exported_pose": False,
            "gt_used": False,
            "nvs_used": False,
        },
        "exclude_exact_seed_bank": bool(args.exclude_exact_seed_bank),
        "temporal_seed_policy": (
            str(args.temporal_seed_policy)
            if track_source_mode == "shared_temporal_bundle"
            else "independent_same_frame"
        ),
        "verify_commit_used": bool(
            track_source_mode == "shared_temporal_bundle"
            and str(args.temporal_seed_policy) == "verified_all"
        ),
        "verification_protocol": (
            "independent-temporal-seed-bank-inlier-support-v1"
            if track_source_mode == "shared_temporal_bundle"
            and str(args.temporal_seed_policy) == "verified_all"
            else None
        ),
        "gt_used": False,
        "nvs_used": False,
        "triangulation_cache": {
            "enabled": True,
            "scope": "one shared static-track cache per frame for all PTZ targets",
            "changes_correspondences_or_geometry_gates": False,
            "total_target_selected_point_count": int(
                sum(row["target_selected_point_count_sum"] for row in triangulation_cache_rows)
            ),
            "total_unique_static_track_count": int(
                sum(row["unique_static_track_count"] for row in triangulation_cache_rows)
            ),
            "total_avoided_retriangulation_count": int(
                sum(row["avoided_retriangulation_count"] for row in triangulation_cache_rows)
            ),
            "seconds": float(sum(row["seconds"] for row in triangulation_cache_rows)),
            "rows": triangulation_cache_rows,
        },
        "rows": prediction_rows,
    }
    prediction_path = output / "predictions_pre_gt.json"
    _write_json(prediction_path, prediction_manifest)
    attestation = {
        "path": str(prediction_path),
        "sha256": _sha256(prediction_path),
    }
    _write_json(output / "prediction_attestation.json", attestation)

    if args.sealed_reference is None:
        print(json.dumps(attestation, indent=2))
        return 0

    with np.load(args.sealed_reference, allow_pickle=True) as reference:
        reference_views = [str(view) for view in reference["long_views"]]
        reference_index = {view: index for index, view in enumerate(reference_views)}
        reference_r = np.asarray(reference["R"], dtype=np.float64)
        reference_t = np.asarray(reference["T"], dtype=np.float64)
    evaluation_rows = []
    for row in prediction_rows:
        if not bool(row["solver"].get("success", False)):
            evaluation_rows.append({**row, "rotation_error_deg": None, "center_error": None})
            continue
        frame = int(row["frame"])
        view = str(row["view"])
        predicted = CameraLoader(
            pose_root / f"f{frame:06d}" / "cameras_pred"
        ).load_camera(view)
        view_index = reference_index[view]
        gt_r = reference_r[frame, view_index]
        gt_t = reference_t[frame, view_index]
        gt_center = -gt_r.T @ gt_t
        evaluation_rows.append(
            {
                **row,
                "rotation_error_deg": _rotation_error_deg(predicted.R, gt_r),
                "center_error": float(np.linalg.norm(predicted.camera_center - gt_center)),
            }
        )
    successful = [row for row in evaluation_rows if row["rotation_error_deg"] is not None]
    evaluation = {
        "protocol": "hidden-parent-track-driven-same-time-full-pnp-v1",
        "prediction_attestation": attestation,
        "reference_path": str(args.sealed_reference.resolve()),
        "reference_sha256": _sha256(args.sealed_reference),
        "success_rate": float(len(successful) / max(len(evaluation_rows), 1)),
        "rotation_error_deg": _stats([row["rotation_error_deg"] for row in successful]),
        "center_error": _stats([row["center_error"] for row in successful]),
        "rows": evaluation_rows,
    }
    _write_json(output / "evaluation.json", evaluation)
    print(json.dumps({key: value for key, value in evaluation.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
