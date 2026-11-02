from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def decide_dual_geometry_commit(
    *,
    track_crossfit_accepted: bool,
    common_motion_accepted: bool,
    direct_map_accepted: bool,
    uniform_same_time_epipolar_accepted: bool,
) -> dict[str, Any]:
    """Combine mandatory track/common gates with two independent evidence channels."""

    accepted_channels = []
    if bool(direct_map_accepted):
        accepted_channels.append("independent_static_map")
    if bool(uniform_same_time_epipolar_accepted):
        accepted_channels.append("uniform_same_time_multiview")
    if not bool(track_crossfit_accepted):
        reason = "track_crossfit_rejected"
    elif not bool(common_motion_accepted):
        reason = "common_motion_rejected"
    elif not accepted_channels:
        reason = "no_independent_geometry_channel_confirmed"
    else:
        reason = "dual_geometry_commit_verified"
    accepted = bool(
        track_crossfit_accepted
        and common_motion_accepted
        and accepted_channels
    )
    return {
        "accepted": accepted,
        "reason": reason,
        "accepted_channels": accepted_channels,
        "track_crossfit_accepted": bool(track_crossfit_accepted),
        "common_motion_accepted": bool(common_motion_accepted),
        "direct_map_accepted": bool(direct_map_accepted),
        "uniform_same_time_epipolar_accepted": bool(
            uniform_same_time_epipolar_accepted
        ),
    }


def make_clean_boundary_seed_proposals(
    *,
    risk_nodes: set[tuple[int, str]],
    frames: Sequence[int],
    views: Sequence[str],
) -> dict[str, Any]:
    """Assign each contiguous risk band its immediate clean temporal boundaries."""

    ordered_frames = sorted({int(frame) for frame in frames})
    frame_to_index = {frame: index for index, frame in enumerate(ordered_frames)}
    allowed_views = {str(view) for view in views}
    normalized = {
        (int(frame), str(view))
        for frame, view in risk_nodes
        if int(frame) in frame_to_index and str(view) in allowed_views
    }
    seeds_by_node: dict[tuple[int, str], list[int]] = {}
    bands: list[dict[str, Any]] = []
    skipped_bands: list[dict[str, Any]] = []
    for view in sorted(allowed_views):
        risk_frames = [
            frame for frame in ordered_frames if (frame, view) in normalized
        ]
        frame_bands: list[list[int]] = []
        for frame in risk_frames:
            if (
                not frame_bands
                or frame_to_index[frame] != frame_to_index[frame_bands[-1][-1]] + 1
            ):
                frame_bands.append([frame])
            else:
                frame_bands[-1].append(frame)

        for band_frames in frame_bands:
            first_index = frame_to_index[band_frames[0]]
            last_index = frame_to_index[band_frames[-1]]
            seed_frames = []
            if first_index > 0:
                left = ordered_frames[first_index - 1]
                if (left, view) not in normalized:
                    seed_frames.append(left)
            if last_index + 1 < len(ordered_frames):
                right = ordered_frames[last_index + 1]
                if (right, view) not in normalized:
                    seed_frames.append(right)
            row = {
                "view": view,
                "frames": [int(frame) for frame in band_frames],
                "seed_frames": [int(frame) for frame in seed_frames],
            }
            if seed_frames:
                bands.append(row)
                for frame in band_frames:
                    seeds_by_node[(int(frame), view)] = list(seed_frames)
            else:
                skipped_bands.append({**row, "reason": "no_clean_temporal_boundary"})
    return {
        "policy": "immediate_clean_boundaries_of_each_contiguous_risk_band",
        "bands": bands,
        "skipped_bands": skipped_bands,
        "seeds_by_node": seeds_by_node,
    }


def deterministic_crossfit_masks(point_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split point IDs into stable, balanced fit and verification subsets."""

    point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    unique_ids = np.unique(point_ids)
    fit_ids = set(int(point_id) for point_id in unique_ids[::2])
    fit = np.asarray([int(point_id) in fit_ids for point_id in point_ids], dtype=bool)
    return fit, ~fit


def summarize_track_reprojection(
    *,
    camera: Any,
    points3d: np.ndarray,
    image_points: np.ndarray,
    visible: np.ndarray,
) -> dict[str, Any]:
    """Measure one camera against a specified tracked 2D-3D observation subset."""

    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    visible = np.asarray(visible, dtype=bool).reshape(-1)
    if not (len(points3d) == len(image_points) == len(visible)):
        raise ValueError("points3d, image_points, and visible must have equal length")
    valid = visible & np.isfinite(points3d).all(axis=1) & np.isfinite(image_points).all(axis=1)
    if not np.any(valid):
        return {
            "point_count": 0,
            "mean_px": float("inf"),
            "median_px": float("inf"),
            "p90_px": float("inf"),
        }
    rotation_vector, _ = cv2.Rodrigues(
        np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
    )
    projected, _ = cv2.projectPoints(
        points3d[valid],
        rotation_vector,
        np.asarray(camera.T, dtype=np.float64).reshape(3),
        np.asarray(camera.K, dtype=np.float64).reshape(3, 3),
        np.asarray(camera.D, dtype=np.float64).reshape(-1),
    )
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - image_points[valid],
        axis=1,
    )
    return {
        "point_count": int(len(errors)),
        "mean_px": float(np.mean(errors)),
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90.0)),
    }


def verify_crossfit_track_candidate(
    *,
    base_camera: Any,
    candidate_camera: Any,
    points3d: np.ndarray,
    image_points: np.ndarray,
    visible: np.ndarray,
    verify_mask: np.ndarray,
    min_points: int,
) -> dict[str, Any]:
    """Verify a fit-subset pose on deterministic held-out tracked points."""

    visible = np.asarray(visible, dtype=bool).reshape(-1)
    verify_mask = np.asarray(verify_mask, dtype=bool).reshape(-1)
    if len(visible) != len(verify_mask):
        raise ValueError("visible and verify_mask must have equal length")
    heldout_visible = visible & verify_mask
    base = summarize_track_reprojection(
        camera=base_camera,
        points3d=points3d,
        image_points=image_points,
        visible=heldout_visible,
    )
    candidate = summarize_track_reprojection(
        camera=candidate_camera,
        points3d=points3d,
        image_points=image_points,
        visible=heldout_visible,
    )
    enough_points = int(candidate["point_count"]) >= int(min_points)
    median_gain = float(base["median_px"] - candidate["median_px"])
    accepted = bool(enough_points and median_gain > 1e-12)
    return {
        "accepted": accepted,
        "reason": (
            "heldout_track_median_gain"
            if accepted
            else "insufficient_heldout_points"
            if not enough_points
            else "heldout_track_median_not_improved"
        ),
        "min_points": int(min_points),
        "base": base,
        "candidate": candidate,
        "median_gain_px": median_gain,
        "median_ratio": float(
            candidate["median_px"] / max(base["median_px"], 1e-12)
        ),
    }


def verify_cross_seed_candidate(
    *,
    base_camera: Any,
    candidate_camera: Any,
    fit_point_ids: np.ndarray,
    verify_point_ids: np.ndarray,
    verify_points3d: np.ndarray,
    verify_image_points: np.ndarray,
    verify_visible: np.ndarray,
    min_points: int,
) -> dict[str, Any]:
    """Verify one seed's pose on another seed's non-overlapping map tracks."""

    fit_ids = np.asarray(fit_point_ids, dtype=np.int64).reshape(-1)
    verify_ids = np.asarray(verify_point_ids, dtype=np.int64).reshape(-1)
    points3d = np.asarray(verify_points3d, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(verify_image_points, dtype=np.float64).reshape(-1, 2)
    visible = np.asarray(verify_visible, dtype=bool).reshape(-1)
    if not (
        len(verify_ids) == len(points3d) == len(image_points) == len(visible)
    ):
        raise ValueError("cross-seed verification arrays must have equal length")
    if len(np.unique(verify_ids)) != len(verify_ids):
        raise ValueError("verify_point_ids must be unique")

    exclusive = ~np.isin(verify_ids, fit_ids)
    exclusive_visible = visible & exclusive
    base = summarize_track_reprojection(
        camera=base_camera,
        points3d=points3d,
        image_points=image_points,
        visible=exclusive_visible,
    )
    candidate = summarize_track_reprojection(
        camera=candidate_camera,
        points3d=points3d,
        image_points=image_points,
        visible=exclusive_visible,
    )
    enough_points = int(candidate["point_count"]) >= int(min_points)
    median_gain = float(base["median_px"] - candidate["median_px"])
    p90_gain = float(base["p90_px"] - candidate["p90_px"])
    accepted = bool(enough_points and median_gain > 1e-12 and p90_gain >= -1e-12)
    return {
        "accepted": accepted,
        "reason": (
            "independent_seed_median_and_p90_gain"
            if accepted
            else "insufficient_independent_seed_points"
            if not enough_points
            else "independent_seed_geometry_regression"
        ),
        "min_points": int(min_points),
        "exclusive_point_count": int(np.sum(exclusive_visible)),
        "excluded_shared_point_count": int(np.sum(~exclusive)),
        "base": base,
        "candidate": candidate,
        "median_gain_px": median_gain,
        "p90_gain_px": p90_gain,
    }


def find_reference_colmap_pose(
    *,
    images_by_id: Mapping[int, Any],
    ref_view: str,
    qvec_to_rotation: Any,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Read one reference camera pose from a COLMAP image mapping."""

    wanted = f"view{ref_view}"
    matches = [image for image in images_by_id.values() if Path(image.name).stem == wanted]
    if len(matches) != 1:
        raise ValueError(
            f"expected one COLMAP reference image for {wanted}, found {len(matches)}"
        )
    image = matches[0]
    return (
        np.asarray(qvec_to_rotation(image.qvec), dtype=np.float64).reshape(3, 3),
        np.asarray(image.tvec, dtype=np.float64).reshape(3),
        str(image.name),
    )


def normalize_pose_to_reference(
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    reference_rotation: np.ndarray,
    reference_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a raw COLMAP world-to-camera pose in the reference-camera gauge."""

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    reference_rotation = np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)
    reference_translation = np.asarray(reference_translation, dtype=np.float64).reshape(3)
    normalized_rotation = rotation @ reference_rotation.T
    normalized_translation = translation - normalized_rotation @ reference_translation
    return normalized_rotation, normalized_translation


def _resolve_query(
    localizations: Mapping[str, Any],
    *,
    frame: int,
    view: str,
) -> str | None:
    stem = f"q_f{int(frame):06d}_view{view}"
    matches = [query for query in localizations if Path(query).stem == stem]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"ambiguous HLoc query for {stem}: {sorted(matches)}")
    return matches[0]


def _unique_inlier_geometry(
    *,
    record: Mapping[str, Any],
    points3d_by_id: Mapping[int, Any],
) -> dict[str, Any]:
    image_points = np.asarray(record["keypoints_query"], dtype=np.float64).reshape(-1, 2)
    point_ids = np.asarray(record["points3D_ids"], dtype=np.int64).reshape(-1)
    inlier_mask = np.asarray(record["PnP_ret"]["inlier_mask"], dtype=bool).reshape(-1)
    if not (len(image_points) == len(point_ids) == len(inlier_mask)):
        raise ValueError("HLoc keypoints, point IDs, and inlier mask have different lengths")
    selected_indices = np.flatnonzero(inlier_mask)
    unique_indices: list[int] = []
    seen: set[int] = set()
    for index in selected_indices.tolist():
        point_id = int(point_ids[index])
        if point_id in seen:
            continue
        if point_id not in points3d_by_id:
            raise KeyError(f"HLoc references missing point3D id {point_id}")
        seen.add(point_id)
        unique_indices.append(int(index))
    unique_ids = point_ids[unique_indices]
    points3d = np.stack(
        [
            np.asarray(
                getattr(points3d_by_id[int(point_id)], "xyz", points3d_by_id[int(point_id)]),
                dtype=np.float64,
            ).reshape(3)
            for point_id in unique_ids
        ]
    ) if len(unique_ids) else np.empty((0, 3), dtype=np.float64)
    return {
        "raw_inlier_count": int(np.sum(inlier_mask)),
        "point_ids": unique_ids.astype(np.int64),
        "points3d": points3d,
        "image_points": image_points[unique_indices],
        "unique_inlier_count": int(len(unique_ids)),
    }


def select_best_hloc_seed_geometry(
    *,
    localizations: Mapping[str, Any],
    points3d_by_id: Mapping[int, Any],
    frames: Sequence[int],
    view: str,
) -> dict[str, Any]:
    """Choose the frame with the most unique HLoc PnP inlier map points."""

    candidates: list[dict[str, Any]] = []
    for frame_value in sorted({int(frame) for frame in frames}):
        query = _resolve_query(
            localizations,
            frame=frame_value,
            view=str(view),
        )
        if query is None:
            continue
        geometry = _unique_inlier_geometry(
            record=localizations[query],
            points3d_by_id=points3d_by_id,
        )
        candidates.append(
            {
                "seed_frame": int(frame_value),
                "query": str(query),
                **geometry,
            }
        )
    if not candidates:
        raise ValueError(f"no HLoc localization records found for view {view}")
    return max(
        candidates,
        key=lambda row: (
            int(row["unique_inlier_count"]),
            -int(row["seed_frame"]),
        ),
    )


def select_bracketing_hloc_seed_geometries(
    *,
    localizations: Mapping[str, Any],
    points3d_by_id: Mapping[int, Any],
    frames: Sequence[int],
    band_frames: Sequence[int],
    view: str,
) -> dict[str, Any]:
    """Choose nearby high-support HLoc seeds on both sides of a risk band."""

    ordered_frames = sorted({int(frame) for frame in frames})
    band = sorted({int(frame) for frame in band_frames})
    if not ordered_frames or not band:
        raise ValueError("frames and band_frames must not be empty")
    if not set(band).issubset(ordered_frames):
        raise ValueError("band_frames must be a subset of frames")
    band_start, band_end = band[0], band[-1]

    candidates: list[dict[str, Any]] = []
    for frame in ordered_frames:
        if band_start <= frame <= band_end:
            continue
        query = _resolve_query(localizations, frame=frame, view=str(view))
        if query is None:
            continue
        geometry = _unique_inlier_geometry(
            record=localizations[query],
            points3d_by_id=points3d_by_id,
        )
        candidates.append(
            {
                "seed_frame": int(frame),
                "query": str(query),
                **geometry,
            }
        )

    def choose(side: str) -> dict[str, Any] | None:
        if side == "left":
            side_rows = [row for row in candidates if int(row["seed_frame"]) < band_start]
            boundary = band_start
        else:
            side_rows = [row for row in candidates if int(row["seed_frame"]) > band_end]
            boundary = band_end
        if not side_rows:
            return None
        support_median = float(
            np.median([int(row["unique_inlier_count"]) for row in side_rows])
        )
        eligible = [
            row
            for row in side_rows
            if int(row["unique_inlier_count"]) >= support_median
        ]
        selected = min(
            eligible,
            key=lambda row: (
                abs(int(row["seed_frame"]) - boundary),
                -int(row["unique_inlier_count"]),
                int(row["seed_frame"]),
            ),
        )
        return {
            **selected,
            "side": side,
            "side_support_median": support_median,
            "side_candidate_count": int(len(side_rows)),
            "side_eligible_count": int(len(eligible)),
        }

    return {
        "selection_policy": (
            "nearest_on_each_side_among_frames_at_or_above_side_median_unique_inliers"
        ),
        "view": str(view),
        "band_frames": band,
        "left": choose("left"),
        "right": choose("right"),
    }


def solve_tracked_map_pnp(
    *,
    points3d: np.ndarray,
    image_points: np.ndarray,
    visible: np.ndarray,
    camera_template: Any,
    reprojection_error_px: float,
    min_inliers: int,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Solve a camera from temporally tracked observations of anchor-map points."""

    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    visible = np.asarray(visible, dtype=bool).reshape(-1)
    if not (len(points3d) == len(image_points) == len(visible)):
        raise ValueError("points3d, image_points, and visible must have equal length")
    valid = visible & np.isfinite(points3d).all(axis=1) & np.isfinite(image_points).all(axis=1)
    object_points = points3d[valid]
    pixels = image_points[valid]
    required = max(int(min_inliers), 6)
    if len(object_points) < required:
        return {
            "success": False,
            "reason": "insufficient_correspondences",
            "point_count": int(len(object_points)),
            "inlier_count": 0,
        }
    cv2.setRNGSeed(int(rng_seed))
    success, rotation_vector, translation, inlier_indices = cv2.solvePnPRansac(
        object_points,
        pixels,
        np.asarray(camera_template.K, dtype=np.float64),
        np.asarray(camera_template.D, dtype=np.float64),
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=1000,
        reprojectionError=float(reprojection_error_px),
        confidence=0.999,
    )
    inlier_count = 0 if inlier_indices is None else int(len(inlier_indices))
    if not success or inlier_count < required:
        return {
            "success": False,
            "reason": "pnp_ransac_failed_or_too_few_inliers",
            "point_count": int(len(object_points)),
            "inlier_count": int(inlier_count),
        }
    inlier_indices = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
    try:
        rotation_vector, translation = cv2.solvePnPRefineLM(
            object_points[inlier_indices],
            pixels[inlier_indices],
            np.asarray(camera_template.K, dtype=np.float64),
            np.asarray(camera_template.D, dtype=np.float64),
            rotation_vector,
            translation,
        )
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        np.asarray(camera_template.K, dtype=np.float64),
        np.asarray(camera_template.D, dtype=np.float64),
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    camera = replace(
        camera_template,
        R=np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        T=translation,
    )
    return {
        "success": True,
        "reason": "ok",
        "camera": camera,
        "point_count": int(len(object_points)),
        "inlier_count": int(inlier_count),
        "inlier_ratio": float(inlier_count / len(object_points)),
        "reprojection_mean_px": float(np.mean(errors)),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_p90_px": float(np.percentile(errors, 90.0)),
        "inlier_indices": inlier_indices,
    }
