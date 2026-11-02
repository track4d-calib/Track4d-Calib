from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from track4d_calib.track_crossview_verify import (
    CrossViewTrackVerifierConfig,
    build_context_point_bank,
)


def _project(camera: Any, points: np.ndarray) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(np.asarray(camera.R, dtype=np.float64))
    projected, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        rotation_vector,
        np.asarray(camera.T, dtype=np.float64).reshape(3),
        np.asarray(camera.K, dtype=np.float64).reshape(3, 3),
        np.asarray(camera.D, dtype=np.float64).reshape(-1),
    )
    return projected.reshape(-1, 2)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(np.max(array)),
    }


def evaluate_fixed_static_landmark_identity(
    *,
    tracks_frame: np.ndarray,
    visibility_frame: np.ndarray,
    view_ids: Sequence[str],
    cameras_by_view: Mapping[str, Any],
    context_views: Sequence[str],
    point_ids: np.ndarray,
    landmark_xyz: np.ndarray,
    reprojection_gate_px: float,
    min_context_views: int = 2,
) -> dict[str, Any]:
    """Keep fixed landmarks supported by same-time public-static observations."""

    tracks = np.asarray(tracks_frame, dtype=np.float64)
    visibility = np.asarray(visibility_frame, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError("tracks_frame must have shape (V,N,2)")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError("visibility_frame does not match tracks_frame")
    views = [str(view) for view in view_ids]
    if len(views) != tracks.shape[0] or len(views) != len(set(views)):
        raise ValueError("view_ids must uniquely match the track view axis")
    view_index = {view: index for index, view in enumerate(views)}
    contexts = [str(view) for view in context_views]
    if len(contexts) != len(set(contexts)) or any(
        view not in view_index or view not in cameras_by_view for view in contexts
    ):
        raise ValueError("context_views must uniquely identify available cameras")

    ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    xyz = np.asarray(landmark_xyz, dtype=np.float64).reshape(-1, 3)
    if len(ids) != len(xyz) or len(ids) != len(np.unique(ids)):
        raise ValueError("point_ids and landmark_xyz must align with unique IDs")
    if np.any(ids < 0) or np.any(ids >= tracks.shape[1]):
        raise ValueError("point_ids lies outside the track point axis")
    if not np.isfinite(xyz).all():
        raise ValueError("landmark_xyz must be finite")
    gate = float(reprojection_gate_px)
    minimum = int(min_context_views)
    if not np.isfinite(gate) or gate <= 0.0:
        raise ValueError("reprojection_gate_px must be finite and positive")
    if minimum < 2:
        raise ValueError("min_context_views must be at least two")

    inlier_counts = np.zeros(len(ids), dtype=np.int64)
    observation_counts = np.zeros(len(ids), dtype=np.int64)
    errors_by_point: list[list[float]] = [[] for _ in ids]
    inlier_errors_by_point: list[list[float]] = [[] for _ in ids]
    for view in contexts:
        index = view_index[view]
        observed = visibility[index, ids] & np.isfinite(tracks[index, ids]).all(axis=1)
        if not np.any(observed):
            continue
        projected = _project(cameras_by_view[view], xyz[observed])
        errors = np.linalg.norm(projected - tracks[index, ids[observed]], axis=1)
        observed_indices = np.flatnonzero(observed)
        observation_counts[observed_indices] += 1
        for local_index, error in zip(observed_indices.tolist(), errors.tolist()):
            errors_by_point[local_index].append(float(error))
            if np.isfinite(error) and error <= gate:
                inlier_counts[local_index] += 1
                inlier_errors_by_point[local_index].append(float(error))

    accepted = inlier_counts >= minimum
    all_errors = [error for values in errors_by_point for error in values]
    accepted_inlier_errors = [
        error
        for keep, values in zip(accepted.tolist(), inlier_errors_by_point)
        if keep
        for error in values
    ]
    point_rows = []
    for index, point_id in enumerate(ids.tolist()):
        point_rows.append(
            {
                "point_id": int(point_id),
                "observation_count": int(observation_counts[index]),
                "inlier_count": int(inlier_counts[index]),
                "accepted": bool(accepted[index]),
                "reprojection_px": _summary(errors_by_point[index]),
                "inlier_reprojection_px": _summary(inlier_errors_by_point[index]),
            }
        )
    return {
        "accepted_point_ids": ids[accepted],
        "accepted_count": int(np.count_nonzero(accepted)),
        "input_count": int(len(ids)),
        "reprojection_gate_px": gate,
        "min_context_views": minimum,
        "all_observation_reprojection_px": _summary(all_errors),
        "accepted_inlier_reprojection_px": _summary(accepted_inlier_errors),
        "point_rows": point_rows,
        "gt_fields_used": [],
        "nvs_fields_used": [],
    }


def select_temporally_static_track_ids(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    cameras_by_frame: Mapping[int, Mapping[str, Any]],
    context_views: Sequence[str],
    point_ids_by_seed: Mapping[int, np.ndarray],
    track_start_frame: int,
    sample_frames: Sequence[int],
    config: CrossViewTrackVerifierConfig,
    reprojection_gate_px: float,
    min_support_frames: int = 2,
) -> dict[str, Any]:
    """Keep seed-map points that remain static in registered context cameras."""

    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must have shape (V,T,N,2), got {tracks.shape}")
    if visibility.shape != tracks.shape[:-1]:
        raise ValueError(
            f"visibility shape {visibility.shape} does not match tracks {tracks.shape}"
        )
    ordered_views = [str(view) for view in view_ids]
    view_index = {view: index for index, view in enumerate(ordered_views)}
    context = [str(view) for view in context_views]
    if any(view not in view_index for view in context):
        missing = sorted(set(context) - set(view_index))
        raise ValueError(f"context views are absent from tracks: {missing}")
    samples = sorted({int(frame) for frame in sample_frames})
    if not samples:
        raise ValueError("sample_frames must not be empty")
    required_support = min(int(min_support_frames), len(samples))
    if required_support <= 0:
        raise ValueError("min_support_frames must be positive")

    selected: set[int] = set()
    seed_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for seed_frame_value, seed_point_ids_value in sorted(point_ids_by_seed.items()):
        seed_frame = int(seed_frame_value)
        seed_local = seed_frame - int(track_start_frame)
        if seed_local < 0 or seed_local >= tracks.shape[1]:
            raise IndexError(f"seed frame {seed_frame} is outside the track bundle")
        if seed_frame not in cameras_by_frame:
            raise KeyError(f"missing registered cameras for seed frame {seed_frame}")
        seed_point_ids = np.asarray(seed_point_ids_value, dtype=np.int64).reshape(-1)
        bank = build_context_point_bank(
            tracks_frame=tracks[:, seed_local],
            visibility_frame=visibility[:, seed_local],
            view_ids=ordered_views,
            cameras_by_view=cameras_by_frame[seed_frame],
            context_views=context,
            point_ids=seed_point_ids,
            config=config,
        )
        xyz_by_id = {
            int(point_id): np.asarray(xyz, dtype=np.float64).reshape(3)
            for point_id, xyz in zip(bank["point_ids"], bank["xyz"])
        }
        static_scores: list[float] = []
        rejected_dynamic = 0
        rejected_support = 0
        for point_id, xyz in xyz_by_id.items():
            errors: list[float] = []
            supporting_frames = 0
            for frame in samples:
                local = int(frame - int(track_start_frame))
                if local < 0 or local >= tracks.shape[1]:
                    raise IndexError(f"sample frame {frame} is outside the track bundle")
                if frame not in cameras_by_frame:
                    raise KeyError(f"missing registered cameras for sample frame {frame}")
                frame_supported = False
                for view in context:
                    index = view_index[view]
                    if not visibility[index, local, point_id]:
                        continue
                    projected = _project(
                        cameras_by_frame[frame][view],
                        xyz.reshape(1, 3),
                    )[0]
                    errors.append(
                        float(
                            np.linalg.norm(
                                projected - tracks[index, local, point_id]
                            )
                        )
                    )
                    frame_supported = True
                supporting_frames += int(frame_supported)
            score = float(np.median(errors)) if errors else float("inf")
            keep = (
                supporting_frames >= required_support
                and score <= float(reprojection_gate_px)
            )
            if keep:
                selected.add(int(point_id))
                static_scores.append(score)
            elif supporting_frames < required_support:
                rejected_support += 1
            else:
                rejected_dynamic += 1
            point_rows.append(
                {
                    "seed_frame": seed_frame,
                    "point_id": int(point_id),
                    "supporting_frames": int(supporting_frames),
                    "temporal_static_reprojection_median_px": score,
                    "selected": bool(keep),
                }
            )
        seed_rows.append(
            {
                "seed_frame": seed_frame,
                "input_point_ids": int(len(seed_point_ids)),
                "triangulated_points": int(bank["num_points_valid"]),
                "temporally_static_points": int(len(static_scores)),
                "rejected_dynamic_points": int(rejected_dynamic),
                "rejected_support_points": int(rejected_support),
                "temporal_static_reprojection_px": _summary(static_scores),
            }
        )

    preselected_ids = np.asarray(sorted(selected), dtype=np.int64)
    reference_frame = samples[0]
    reference_cameras = cameras_by_frame[reference_frame]
    for frame in samples[1:]:
        for view in context:
            reference = reference_cameras[view]
            candidate = cameras_by_frame[frame][view]
            if not all(
                np.array_equal(
                    np.asarray(getattr(reference, field)),
                    np.asarray(getattr(candidate, field)),
                )
                for field in ("K", "R", "T", "D")
            ):
                raise ValueError(
                    "temporal-median landmarks require fixed public-static cameras"
                )

    pooled_tracks = np.zeros((len(ordered_views), tracks.shape[2], 2), dtype=np.float64)
    pooled_visibility = np.zeros((len(ordered_views), tracks.shape[2]), dtype=bool)
    sample_locals = [frame - int(track_start_frame) for frame in samples]
    for view in context:
        index = view_index[view]
        for point_id in preselected_ids.tolist():
            coordinates = np.asarray(
                [
                    tracks[index, local, point_id]
                    for local in sample_locals
                    if visibility[index, local, point_id]
                    and np.isfinite(tracks[index, local, point_id]).all()
                ],
                dtype=np.float64,
            ).reshape(-1, 2)
            if not len(coordinates):
                continue
            pooled_tracks[index, point_id] = np.median(coordinates, axis=0)
            pooled_visibility[index, point_id] = True

    pooled_bank = build_context_point_bank(
        tracks_frame=pooled_tracks,
        visibility_frame=pooled_visibility,
        view_ids=ordered_views,
        cameras_by_view=reference_cameras,
        context_views=context,
        point_ids=preselected_ids,
        config=config,
    )
    selected_ids = np.asarray(pooled_bank["point_ids"], dtype=np.int64)
    selected_landmark_xyz = np.asarray(pooled_bank["xyz"], dtype=np.float64).reshape(-1, 3)
    context_indices = [view_index[view] for view in context]
    selected_pooled_static_tracks = np.asarray(
        pooled_tracks[np.ix_(context_indices, selected_ids)], dtype=np.float64
    )
    selected_pooled_static_visibility = np.asarray(
        pooled_visibility[np.ix_(context_indices, selected_ids)], dtype=bool
    )
    final_selected = set(selected_ids.tolist())
    for row in point_rows:
        row["seed_static_selected"] = bool(row["selected"])
        row["selected"] = int(row["point_id"]) in final_selected
    return {
        "policy": "seed_static_map_multiframe_context_reprojection_median",
        "landmark_policy": (
            "per_static_view_temporal_median_then_robust_multiview_triangulation"
        ),
        "reprojection_gate_px": float(reprojection_gate_px),
        "min_support_frames": int(required_support),
        "sample_frames": samples,
        "context_views": context,
        "selected_point_ids": selected_ids,
        "selected_landmark_xyz": selected_landmark_xyz,
        "selected_pooled_static_tracks": selected_pooled_static_tracks,
        "selected_pooled_static_visibility": selected_pooled_static_visibility,
        "pooled_context_inlier_count": np.asarray(
            pooled_bank["context_inlier_count"], dtype=np.int64
        ),
        "pooled_context_observation_count": np.asarray(
            pooled_bank["context_observation_count"], dtype=np.int64
        ),
        "pooled_context_reproj_median_px": np.asarray(
            pooled_bank["context_reproj_median_px"], dtype=np.float64
        ),
        "pooled_context_reproj_p90_px": np.asarray(
            pooled_bank["context_reproj_p90_px"], dtype=np.float64
        ),
        "selected_count": int(len(selected_ids)),
        "seed_static_preselected_count": int(len(preselected_ids)),
        "pooled_landmark_reprojection_px": _summary(
            np.asarray(pooled_bank["context_reproj_median_px"], dtype=np.float64)
        ),
        "seed_rows": seed_rows,
        "point_rows": point_rows,
        "gt_fields_used": [],
        "nvs_fields_used": [],
    }
