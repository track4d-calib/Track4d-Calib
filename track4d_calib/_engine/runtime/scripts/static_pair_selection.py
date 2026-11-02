"""Choose a stable static baseline using fresh fit correspondences.

Camera labels convey no azimuth or dataset identity. The existing static
triangulator and its acceptance gates decide whether a support pair is useful.
Equal capped point counts need not imply equal depth information. Prefer
triangulation angle and agreement with other already-matched static cameras.
Selection consumes no confirm/final IDs and performs no extra image matching.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _depth_information(result, pair, destinations, cameras, config):
    """Scale-free stereo conditioning, downweighted by cross-static residuals.

    sin(angle)^2 is an angular depth-information proxy, not a calibrated pose
    covariance. Other static views use only these same fit IDs for ranking;
    they are not a substitute for the downstream independent final fold.
    """
    import cv2

    ids = np.asarray(result["point_ids"], dtype=np.int64)
    xyz = np.asarray(result["object_points"], dtype=np.float64)
    angles = np.asarray([row["max_triangulation_angle_deg"] for row in result["triangulation_rows"]])
    information = np.sin(np.deg2rad(angles)) ** 2
    errors = [[] for _ in ids]
    by_view = {}
    invalid = 0
    for view in sorted(destinations):
        if view in pair or view not in cameras or not len(ids):
            continue
        cam = cameras[view]
        # A duplicate optical center is not an independent depth witness.
        if any(np.linalg.norm(cam.camera_center - cameras[endpoint].camera_center)
               < config.min_support_baseline for endpoint in pair):
            continue
        positions = [i for i, point in enumerate(ids) if int(point) in destinations[view]]
        if not positions:
            continue
        points = xyz[positions]
        rotation, _ = cv2.Rodrigues(np.asarray(cam.R, dtype=np.float64))
        projected, _ = cv2.projectPoints(points, rotation, np.asarray(cam.T, dtype=np.float64),
                                         np.asarray(cam.K, dtype=np.float64), np.asarray(cam.D, dtype=np.float64))
        observed = np.asarray([destinations[view][int(ids[i])] for i in positions])
        residual = np.linalg.norm(projected.reshape(-1, 2) - observed, axis=1)
        depth = (points @ np.asarray(cam.R).T + np.asarray(cam.T).reshape(3))[:, 2]
        bad = ~np.isfinite(residual) | (depth <= 0)
        residual[bad] = np.inf
        invalid += int(bad.sum())
        by_view[view] = len(positions)
        for i, value in zip(positions, residual):
            errors[i].append(float(value))
    supported = np.asarray([bool(values) for values in errors], dtype=bool)
    per_point = np.asarray([float(np.median(values)) for values in errors if values])
    # Use the existing triangulation tolerance as the residual scale. No
    # scene-specific threshold or PSNR-dependent adjustment is introduced.
    weights = 1. / (1. + np.square(per_point / config.support_reprojection_error_px))
    crossed = float(np.sum(information[supported] * weights))
    conditioning = float(information.sum())
    median_error = float(np.median(per_point)) if len(per_point) else None
    return {
        "conditioning_information": conditioning,
        "cross_static_depth_information": crossed,
        "fit_depth_information": crossed if supported.sum() >= config.min_pnp_inliers else conditioning,
        "triangulation_angle_median_deg": float(np.median(angles)) if len(angles) else None,
        "cross_static_fit_points": int(supported.sum()),
        "cross_static_median_error_px": median_error if median_error is None or np.isfinite(median_error) else None,
        "cross_static_observations_by_view": by_view,
        "invalid_cross_static_projections": invalid,
    }


def choose_second_static(
    *, first: Mapping[str, Any], eligible: Sequence[Mapping[str, Any]],
    matches: Mapping[str, Mapping[str, Any]], queries: np.ndarray,
    support_cameras: Mapping[str, Any], fit_ids: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from track4d_calib.same_time_multiview_pnp import (
        SameTimePnPConfig, triangulate_same_time_multiview_points,
    )

    left = str(first["static_view"])
    fit_set = set(map(int, fit_ids))
    destinations = {
        str(view): {int(point): xy for point, xy in zip(match["point_ids"], match["target_points"])
                    if int(point) in fit_set}
        for view, match in matches.items()
    }
    left_xy = destinations[left]
    config = SameTimePnPConfig()
    rows = []
    for candidate in eligible:
        right = str(candidate["static_view"])
        if right == left:
            continue
        right_xy = destinations[right]
        shared = sorted(fit_set & set(left_xy) & set(right_xy))
        # Same deterministic fit-point capacity as the existing geometry.
        indices = np.linspace(0, len(shared) - 1, min(150, len(shared))).round().astype(int)
        selected = [shared[index] for index in indices]
        result = triangulate_same_time_multiview_points(
            support_cameras={left: support_cameras[left], right: support_cameras[right]},
            support_observations={point: {left: left_xy[point], right: right_xy[point]}
                                  for point in selected},
            target_observations={point: queries[point] for point in selected},
            config=config,
        )
        rows.append({"static_view": right, "shared_fit_points": len(shared),
                     "sampled_fit_points": len(selected),
                     "triangulated_fit_points": int(result["triangulated_count"]),
                     **_depth_information(result, (left, right), destinations, support_cameras, config)})
    counts = {row["static_view"]: row["triangulated_fit_points"] for row in rows}
    valid = [row for row in eligible if counts.get(str(row["static_view"]), 0) >= 12]
    if not valid:
        raise RuntimeError("no second static support passes fresh fit-track triangulation")
    by_view = {row["static_view"]: row for row in rows}
    crossed = [row for row in valid if by_view[str(row["static_view"])]["cross_static_fit_points"] >= config.min_pnp_inliers]
    # If no third static has coverage, retain the existing two-support
    # contract, explicitly without claiming cross-static verification.
    ranked = crossed or valid
    second = min(ranked, key=lambda row: (
        -by_view[str(row["static_view"])]["fit_depth_information"],
        -counts[str(row["static_view"])], -float(row["overlap_median"]),
        -int(row["accepted_count"]), str(row["static_view"]),
    ))
    return dict(second), {
        "policy": "fit_cross_static_weighted_stereo_information_v1",
        "ranking_evidence": "cross_static_fit_tracks" if crossed else "conditioning_only_no_covered_third_static",
        "information_proxy": "sum(sin(triangulation_angle)^2 / (1 + (cross_static_median_error / existing_support_tolerance)^2))",
        "cross_static_role": "fit-only pair ranking, not final-fold verification",
        "candidate_pairs": rows,
        "maximum_fit_points_per_pair": 150,
        "triangulation_gates_unchanged": True,
        "confirm_or_final_IDs_used": False,
        "extra_matching_pairs": 0,
        "camera_ID_azimuth_assumptions": False,
    }
