from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from track4d_calib.cameras import CameraParams
from track4d_calib.triangulation import triangulate_point_dlt


@dataclass(frozen=True)
class CrossViewTrackVerifierConfig:
    min_context_views: int = 3
    context_inlier_px: float = 3.0
    focus_inlier_px: float = 4.0
    max_pair_hypotheses: int = 24


def deduplicate_track_components(
    points_t0: np.ndarray,
    visibility_t0: np.ndarray,
    *,
    quantization_px: float = 1e-3,
) -> np.ndarray:
    """Return the first row for each repeated cross-view seed component."""

    points = np.asarray(points_t0, dtype=np.float64)
    visibility = np.asarray(visibility_t0, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"points_t0 must have shape (V,N,2), got {points.shape}")
    if visibility.shape != points.shape[:2]:
        raise ValueError(f"visibility_t0 shape mismatch: {visibility.shape} vs {points.shape[:2]}")
    step = float(quantization_px)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("quantization_px must be finite and positive")

    finite = np.isfinite(points).all(axis=-1)
    valid = visibility & finite
    safe = np.where(valid[..., None], points, 0.0)
    quantized = np.rint(safe / step).astype(np.int64)
    selected: list[int] = []
    seen: set[bytes] = set()
    for point_id in range(points.shape[1]):
        signature = valid[:, point_id].tobytes() + quantized[:, point_id, :].tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(int(point_id))
    return np.asarray(selected, dtype=np.int64)


def deduplicate_multiseed_track_components(
    *,
    points_seed_by_view: np.ndarray,
    visibility: np.ndarray,
    seed_frames: Sequence[int],
    start_frame: int,
    point_counts: Sequence[int],
) -> np.ndarray:
    """Deduplicate each seed group using observations at its own seed frame."""

    points = np.asarray(points_seed_by_view, dtype=np.float64)
    visible = np.asarray(visibility, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"points_seed_by_view must have shape (V,N,2), got {points.shape}")
    if (
        visible.ndim != 3
        or visible.shape[0] != points.shape[0]
        or visible.shape[2] != points.shape[1]
    ):
        raise ValueError(
            "visibility must have shape (V,T,N) matching points_seed_by_view; "
            f"got {visible.shape} vs {points.shape}"
        )
    frames = [int(frame) for frame in seed_frames]
    counts = [int(count) for count in point_counts]
    if len(frames) != len(counts):
        raise ValueError("seed_frames and point_counts must have the same length")
    if any(count < 0 for count in counts) or sum(counts) != int(points.shape[1]):
        raise ValueError("point_counts must be non-negative and sum to the track point dimension")

    selected_groups: list[np.ndarray] = []
    offset = 0
    for seed_frame, count in zip(frames, counts):
        local_frame = int(seed_frame) - int(start_frame)
        if local_frame < 0 or local_frame >= int(visible.shape[1]):
            raise ValueError(
                f"seed frame {seed_frame} is outside the visibility sequence starting at {start_frame}"
            )
        end = offset + int(count)
        if count > 0:
            local_ids = deduplicate_track_components(
                points[:, offset:end, :],
                visible[:, local_frame, offset:end],
            )
            selected_groups.append(local_ids + int(offset))
        offset = end
    if not selected_groups:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(selected_groups).astype(np.int64, copy=False)


def _projection_matrix(camera: CameraParams) -> np.ndarray:
    return np.asarray(camera.K, dtype=np.float64) @ np.hstack(
        [
            np.asarray(camera.R, dtype=np.float64).reshape(3, 3),
            np.asarray(camera.T, dtype=np.float64).reshape(3, 1),
        ]
    )


def _undistort_pixel(camera: CameraParams, uv: np.ndarray) -> np.ndarray:
    point = np.asarray(uv, dtype=np.float64).reshape(1, 1, 2)
    distortion = np.asarray(camera.D, dtype=np.float64).reshape(-1)
    return cv2.undistortPoints(point, camera.K, distortion, P=camera.K).reshape(2)


def _project_point(camera: CameraParams, point: np.ndarray) -> tuple[np.ndarray | None, float]:
    xyz = np.asarray(point, dtype=np.float64).reshape(3)
    xyz_camera = np.asarray(camera.R, dtype=np.float64).reshape(3, 3) @ xyz + np.asarray(
        camera.T, dtype=np.float64
    ).reshape(3)
    depth = float(xyz_camera[2])
    if not np.isfinite(depth) or depth <= 1e-8:
        return None, depth
    rotation_vector, _ = cv2.Rodrigues(np.asarray(camera.R, dtype=np.float64).reshape(3, 3))
    projected, _ = cv2.projectPoints(
        xyz.reshape(1, 3),
        rotation_vector,
        np.asarray(camera.T, dtype=np.float64).reshape(3, 1),
        np.asarray(camera.K, dtype=np.float64).reshape(3, 3),
        np.asarray(camera.D, dtype=np.float64).reshape(-1),
    )
    uv = projected.reshape(2)
    if not np.isfinite(uv).all():
        return None, depth
    return uv, depth


def _reprojection_errors(
    point: np.ndarray,
    observations: Sequence[tuple[str, CameraParams, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.full((len(observations),), np.inf, dtype=np.float64)
    valid = np.zeros((len(observations),), dtype=bool)
    for index, (_view, camera, observed_uv) in enumerate(observations):
        projected_uv, depth = _project_point(camera, point)
        if projected_uv is None or not np.isfinite(depth) or depth <= 1e-8:
            continue
        errors[index] = float(np.linalg.norm(projected_uv - np.asarray(observed_uv, dtype=np.float64).reshape(2)))
        valid[index] = np.isfinite(errors[index])
    return errors, valid


def _triangulate_observations(
    observations: Sequence[tuple[str, CameraParams, np.ndarray]],
) -> np.ndarray | None:
    projections = [_projection_matrix(camera) for _view, camera, _uv in observations]
    pixels = [_undistort_pixel(camera, uv) for _view, camera, uv in observations]
    return triangulate_point_dlt(projections, pixels)


def _robust_triangulate(
    observations: Sequence[tuple[str, CameraParams, np.ndarray]],
    *,
    config: CrossViewTrackVerifierConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    min_views = max(2, int(config.min_context_views))
    if len(observations) < min_views:
        return None
    pair_rows = []
    for first, second in combinations(range(len(observations)), 2):
        center_first = np.asarray(observations[first][1].camera_center, dtype=np.float64).reshape(3)
        center_second = np.asarray(observations[second][1].camera_center, dtype=np.float64).reshape(3)
        baseline = float(np.linalg.norm(center_first - center_second))
        pair_rows.append((baseline, first, second))
    pair_rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    max_hypotheses = max(1, int(config.max_pair_hypotheses))

    best_point: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    best_score: tuple[int, float, float] | None = None
    for baseline, first, second in pair_rows[:max_hypotheses]:
        point = _triangulate_observations([observations[first], observations[second]])
        if point is None or not np.isfinite(point).all():
            continue
        errors, valid = _reprojection_errors(point, observations)
        inliers = valid & (errors <= float(config.context_inlier_px))
        count = int(np.count_nonzero(inliers))
        if count < min_views:
            continue
        median_error = float(np.median(errors[inliers]))
        score = (count, -median_error, float(baseline))
        if best_score is None or score > best_score:
            best_score = score
            best_point = np.asarray(point, dtype=np.float64).reshape(3)
            best_inliers = inliers
    if best_point is None or best_inliers is None:
        return None

    for _iteration in range(2):
        inlier_observations = [obs for obs, keep in zip(observations, best_inliers) if bool(keep)]
        refined = _triangulate_observations(inlier_observations)
        if refined is None or not np.isfinite(refined).all():
            break
        errors, valid = _reprojection_errors(refined, observations)
        refined_inliers = valid & (errors <= float(config.context_inlier_px))
        if int(np.count_nonzero(refined_inliers)) < min_views:
            break
        best_point = np.asarray(refined, dtype=np.float64).reshape(3)
        best_inliers = refined_inliers

    final_errors, final_valid = _reprojection_errors(best_point, observations)
    final_inliers = final_valid & (final_errors <= float(config.context_inlier_px))
    if int(np.count_nonzero(final_inliers)) < min_views:
        return None
    return best_point, final_errors, final_inliers


def build_context_point_bank(
    *,
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    cameras_by_view: Mapping[str, CameraParams],
    context_views: Sequence[str],
    point_ids: np.ndarray | None,
    config: CrossViewTrackVerifierConfig,
) -> dict[str, Any]:
    tracks = np.asarray(tracks_frame, dtype=np.float64)
    visibility = np.asarray(visibility_frame, dtype=bool)
    views = [str(view) for view in view_ids]
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks_frame must have shape (V,N,2), got {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(f"visibility_frame shape mismatch: {visibility.shape} vs {tracks.shape[:2]}")
    if len(views) != tracks.shape[0]:
        raise ValueError(f"view_ids length mismatch: {len(views)} vs {tracks.shape[0]}")
    view_to_index = {view: index for index, view in enumerate(views)}
    contexts = [str(view) for view in context_views if str(view) in view_to_index and str(view) in cameras_by_view]
    ids = (
        np.arange(tracks.shape[1], dtype=np.int64)
        if point_ids is None
        else np.asarray(point_ids, dtype=np.int64).reshape(-1)
    )

    kept_ids: list[int] = []
    xyz: list[np.ndarray] = []
    inlier_counts: list[int] = []
    observation_counts: list[int] = []
    context_medians: list[float] = []
    context_p90s: list[float] = []
    for point_id in ids.tolist():
        if point_id < 0 or point_id >= tracks.shape[1]:
            continue
        observations: list[tuple[str, CameraParams, np.ndarray]] = []
        for view in contexts:
            view_index = int(view_to_index[view])
            if not bool(visibility[view_index, point_id]):
                continue
            uv = np.asarray(tracks[view_index, point_id], dtype=np.float64).reshape(2)
            if not np.isfinite(uv).all():
                continue
            observations.append((view, cameras_by_view[view], uv))
        triangulated = _robust_triangulate(observations, config=config)
        if triangulated is None:
            continue
        point, errors, inliers = triangulated
        inlier_errors = errors[inliers]
        kept_ids.append(int(point_id))
        xyz.append(point)
        inlier_counts.append(int(np.count_nonzero(inliers)))
        observation_counts.append(int(len(observations)))
        context_medians.append(float(np.median(inlier_errors)))
        context_p90s.append(float(np.percentile(inlier_errors, 90.0)))

    return {
        "num_points_considered": int(ids.size),
        "num_points_valid": int(len(kept_ids)),
        "point_ids": np.asarray(kept_ids, dtype=np.int64),
        "xyz": np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        "context_inlier_count": np.asarray(inlier_counts, dtype=np.int64),
        "context_observation_count": np.asarray(observation_counts, dtype=np.int64),
        "context_reproj_median_px": np.asarray(context_medians, dtype=np.float64),
        "context_reproj_p90_px": np.asarray(context_p90s, dtype=np.float64),
        "context_views": contexts,
    }


def stable_point_id_validation_groups(point_ids: np.ndarray) -> np.ndarray:
    """Hash structured track IDs into deterministic cross-fit groups."""

    values = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    if np.any(values < 0):
        raise ValueError("point IDs must be non-negative")
    mask = (1 << 64) - 1
    output = []
    for value in values.tolist():
        mixed = (int(value) + 0x9E3779B97F4A7C15) & mask
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
        mixed ^= mixed >> 31
        output.append(mixed & ((1 << 63) - 1))
    return np.asarray(output, dtype=np.int64)


def split_point_bank_for_validation(
    point_bank: Mapping[str, Any],
    *,
    num_folds: int = 3,
    validation_fold: int = 0,
) -> dict[str, Any]:
    """Split fixed landmarks so focus-PnP fitting cannot reuse verify tracks."""

    folds = int(num_folds)
    heldout = int(validation_fold)
    if folds < 2:
        raise ValueError("num_folds must be at least two")
    if heldout < 0 or heldout >= folds:
        raise ValueError("validation_fold must be inside [0, num_folds)")
    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    has_explicit_groups = "validation_group_ids" in point_bank
    validation_group_ids = (
        np.asarray(point_bank["validation_group_ids"], dtype=np.int64).reshape(-1)
        if has_explicit_groups
        else stable_point_id_validation_groups(point_ids)
    )
    if len(validation_group_ids) != len(point_ids):
        raise ValueError("validation_group_ids and point_ids have different lengths")
    validation_mask = np.mod(validation_group_ids, folds) == heldout
    fit_mask = ~validation_mask

    def subset(mask: np.ndarray) -> dict[str, Any]:
        output: dict[str, Any] = {}
        source = dict(point_bank)
        source["validation_group_ids"] = validation_group_ids
        for key, value in source.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == point_ids.size:
                output[key] = value[mask]
            else:
                output[key] = value
        output["num_points_valid"] = int(np.count_nonzero(mask))
        return output

    return {
        "protocol": (
            "explicit_validation_group_id_modulo"
            if has_explicit_groups
            else "stable_point_id_splitmix64_modulo"
        ),
        "num_folds": folds,
        "validation_fold": heldout,
        "fit": subset(fit_mask),
        "validation": subset(validation_mask),
        "fit_count": int(np.count_nonzero(fit_mask)),
        "validation_count": int(np.count_nonzero(validation_mask)),
    }


def _summarize_errors(errors: np.ndarray, *, inlier_px: float) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "reproj_mean_px": float("inf"),
            "reproj_median_px": float("inf"),
            "reproj_p90_px": float("inf"),
            "reproj_max_px": float("inf"),
            "inlier_ratio": 0.0,
        }
    return {
        "count": int(values.size),
        "reproj_mean_px": float(np.mean(values)),
        "reproj_median_px": float(np.median(values)),
        "reproj_p90_px": float(np.percentile(values, 90.0)),
        "reproj_max_px": float(np.max(values)),
        "inlier_ratio": float(np.mean(values <= float(inlier_px))),
    }


def compare_focus_cameras(
    *,
    point_bank: Mapping[str, Any],
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    focus_view: str,
    base_camera: CameraParams,
    candidate_camera: CameraParams,
    config: CrossViewTrackVerifierConfig,
) -> dict[str, Any]:
    tracks = np.asarray(tracks_frame, dtype=np.float64)
    visibility = np.asarray(visibility_frame, dtype=bool)
    views = [str(view) for view in view_ids]
    focus = str(focus_view)
    if focus not in views:
        raise KeyError(f"focus_view {focus!r} is not in view_ids")
    focus_index = int(views.index(focus))
    point_ids = np.asarray(point_bank.get("point_ids", []), dtype=np.int64).reshape(-1)
    xyz = np.asarray(point_bank.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    if point_ids.size != xyz.shape[0]:
        raise ValueError("point_bank point_ids and xyz have different lengths")

    paired_ids: list[int] = []
    base_errors: list[float] = []
    candidate_errors: list[float] = []
    for point_id, point in zip(point_ids.tolist(), xyz):
        if point_id < 0 or point_id >= tracks.shape[1] or not bool(visibility[focus_index, point_id]):
            continue
        observed = np.asarray(tracks[focus_index, point_id], dtype=np.float64).reshape(2)
        if not np.isfinite(observed).all():
            continue
        base_uv, base_depth = _project_point(base_camera, point)
        candidate_uv, candidate_depth = _project_point(candidate_camera, point)
        if (
            base_uv is None
            or candidate_uv is None
            or not np.isfinite(base_depth)
            or not np.isfinite(candidate_depth)
            or base_depth <= 1e-8
            or candidate_depth <= 1e-8
        ):
            continue
        paired_ids.append(int(point_id))
        base_errors.append(float(np.linalg.norm(base_uv - observed)))
        candidate_errors.append(float(np.linalg.norm(candidate_uv - observed)))

    base_arr = np.asarray(base_errors, dtype=np.float64)
    candidate_arr = np.asarray(candidate_errors, dtype=np.float64)
    return {
        "focus_view": focus,
        "context_point_count": int(point_ids.size),
        "paired_count": int(base_arr.size),
        "point_ids": np.asarray(paired_ids, dtype=np.int64),
        "base_errors_px": base_arr,
        "candidate_errors_px": candidate_arr,
        "base": _summarize_errors(base_arr, inlier_px=float(config.focus_inlier_px)),
        "candidate": _summarize_errors(candidate_arr, inlier_px=float(config.focus_inlier_px)),
        "gain_mean_px": float(np.mean(base_arr) - np.mean(candidate_arr)) if base_arr.size else float("nan"),
        "gain_median_px": float(np.median(base_arr) - np.median(candidate_arr)) if base_arr.size else float("nan"),
        "gain_p90_px": (
            float(np.percentile(base_arr, 90.0) - np.percentile(candidate_arr, 90.0))
            if base_arr.size
            else float("nan")
        ),
    }


def combine_focus_camera_comparisons(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    config: CrossViewTrackVerifierConfig,
) -> dict[str, Any]:
    """Aggregate disjoint cross-fit residuals without refitting on verify tracks."""

    if not comparisons:
        raise ValueError("at least one focus comparison is required")
    focus_views = {str(comparison["focus_view"]) for comparison in comparisons}
    if len(focus_views) != 1:
        raise ValueError(f"focus comparisons use different views: {sorted(focus_views)}")
    base_errors = np.concatenate(
        [np.asarray(comparison["base_errors_px"], dtype=np.float64) for comparison in comparisons]
    )
    candidate_errors = np.concatenate(
        [np.asarray(comparison["candidate_errors_px"], dtype=np.float64) for comparison in comparisons]
    )
    point_ids = np.concatenate(
        [np.asarray(comparison["point_ids"], dtype=np.int64) for comparison in comparisons]
    )
    if base_errors.shape != candidate_errors.shape or base_errors.shape != point_ids.shape:
        raise ValueError("cross-fit comparison arrays have incompatible shapes")
    return {
        "focus_view": next(iter(focus_views)),
        "context_point_count": int(
            sum(int(comparison["context_point_count"]) for comparison in comparisons)
        ),
        "paired_count": int(base_errors.size),
        "point_ids": point_ids,
        "base_errors_px": base_errors,
        "candidate_errors_px": candidate_errors,
        "base": _summarize_errors(base_errors, inlier_px=float(config.focus_inlier_px)),
        "candidate": _summarize_errors(candidate_errors, inlier_px=float(config.focus_inlier_px)),
        "gain_mean_px": float(np.mean(base_errors) - np.mean(candidate_errors)),
        "gain_median_px": float(np.median(base_errors) - np.median(candidate_errors)),
        "gain_p90_px": float(
            np.percentile(base_errors, 90.0) - np.percentile(candidate_errors, 90.0)
        ),
    }
