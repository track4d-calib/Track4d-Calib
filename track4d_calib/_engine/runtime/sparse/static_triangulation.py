"""Batch pair hypotheses for RGB queries observed by a fixed static rig."""

from itertools import combinations

import cv2
import numpy as np


def triangulate_queries(cameras, mappings, allowed, config):
    """Return query IDs and points using the scalar robust-triangulation rule.

    Pair hypotheses are scored in batches. The winning inlier set uses the same
    DLT refinement and final checks as the single-query implementation.
    """
    from track4d_calib.same_time_multiview_pnp import _project, _triangulate_normalized

    counts = np.zeros(len(allowed), dtype=int)
    for mapping in mappings:
        counts[list(mapping)] += 1
    minimum = max(2, int(config.min_support_views))
    query_ids = np.flatnonzero(allowed & (counts >= minimum))
    if not len(query_ids):
        return np.empty(0, dtype=int), np.empty((0, 3))
    count, support_count = len(query_ids), len(cameras)
    observed = np.zeros((support_count, count, 2))
    present = np.zeros((support_count, count), dtype=bool)
    projections, normalized, rays, centers, vectors = [], [], [], [], []
    for index, (camera, mapping) in enumerate(zip(cameras, mappings)):
        for slot, query in enumerate(query_ids):
            if query in mapping:
                observed[index, slot] = mapping[query]
                present[index, slot] = True
        rotation = np.asarray(camera.R, dtype=float).reshape(3, 3)
        translation = np.asarray(camera.T, dtype=float).reshape(3, 1)
        points = cv2.undistortPoints(observed[index].reshape(-1, 1, 2),
            np.asarray(camera.K, dtype=float), np.asarray(camera.D, dtype=float).reshape(-1)).reshape(-1, 2)
        directions = np.c_[points, np.ones(count)] @ rotation
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        projections.append(np.hstack([rotation, translation]))
        normalized.append(points)
        rays.append(directions)
        centers.append(camera.camera_center)
        vectors.append(cv2.Rodrigues(rotation)[0])

    best_count = np.zeros(count, dtype=int)
    best_error = np.full(count, np.inf)
    best_angle = np.zeros(count)
    best_inliers = np.zeros((count, support_count), dtype=bool)
    pair_angles = {}
    for first, second in combinations(range(support_count), 2):
        if np.linalg.norm(centers[first] - centers[second]) < config.min_support_baseline:
            continue
        angle = np.degrees(np.arccos(np.clip(np.sum(rays[first] * rays[second], axis=1), -1., 1.)))
        pair_angles[first, second] = angle
        valid = present[first] & present[second] & (angle >= config.min_triangulation_angle_deg)
        slots = np.flatnonzero(valid)
        if not len(slots):
            continue
        matrix = np.empty((len(slots), 4, 4))
        for offset, support in enumerate((first, second)):
            xy, projection = normalized[support][slots], projections[support]
            matrix[:, 2 * offset] = xy[:, 0, None] * projection[2] - projection[0]
            matrix[:, 2 * offset + 1] = xy[:, 1, None] * projection[2] - projection[1]
        try:
            homogeneous = np.linalg.svd(matrix)[2][:, -1]
        except np.linalg.LinAlgError:
            homogeneous = np.full((len(slots), 4), np.nan)
            for index, value in enumerate(matrix):
                try:
                    homogeneous[index] = np.linalg.svd(value)[2][-1]
                except np.linalg.LinAlgError:
                    pass
        finite = np.isfinite(homogeneous).all(1) & (np.abs(homogeneous[:, 3]) >= 1e-12)
        slots, homogeneous = slots[finite], homogeneous[finite]
        if not len(slots):
            continue
        points = homogeneous[:, :3] / homogeneous[:, 3, None]
        errors = np.full((len(slots), support_count), np.inf)
        inliers = np.zeros(errors.shape, dtype=bool)
        for support, camera in enumerate(cameras):
            projected = cv2.projectPoints(points, vectors[support], camera.T, camera.K, camera.D)[0].reshape(-1, 2)
            errors[:, support] = np.linalg.norm(projected - observed[support, slots], axis=1)
            depths = points @ camera.R[2] + np.asarray(camera.T).reshape(3)[2]
            inliers[:, support] = (present[support, slots] & (depths > 1e-8)
                & np.isfinite(errors[:, support]) & (errors[:, support] <= config.support_reprojection_error_px))
        sizes = inliers.sum(1)
        enough = sizes >= minimum
        slots, sizes, inliers, errors = slots[enough], sizes[enough], inliers[enough], errors[enough]
        if not len(slots):
            continue
        # Ignore invalid observations when taking the median, including even counts.
        errors[~inliers] = np.inf
        sorted_errors = np.sort(errors, axis=1)
        median = (sorted_errors[np.arange(len(slots)), (sizes - 1) // 2]
                  + sorted_errors[np.arange(len(slots)), sizes // 2]) * .5
        tied = sizes == best_count[slots]
        better = (sizes > best_count[slots]) | (tied & (median < best_error[slots]))
        better |= tied & (median == best_error[slots]) & (angle[slots] > best_angle[slots])
        selected = slots[better]
        best_count[selected], best_error[selected] = sizes[better], median[better]
        best_angle[selected], best_inliers[selected] = angle[selected], inliers[better]

    kept, points = [], []
    for slot in np.flatnonzero(best_count >= minimum):
        query = query_ids[slot]
        observations = [(str(i), camera, mappings[i][query]) for i, camera in enumerate(cameras)
                        if best_inliers[slot, i]]
        point = _triangulate_normalized(observations)
        if point is None or not np.isfinite(point).all():
            continue
        final = []
        for support, camera in enumerate(cameras):
            if not present[support, slot]:
                continue
            pixel, depth = _project(camera, point)
            error = np.linalg.norm(pixel - observed[support, slot])
            if depth > 1e-8 and np.isfinite(error) and error <= config.support_reprojection_error_px:
                final.append(support)
        if len(final) < minimum:
            continue
        max_angle = max((pair_angles.get(pair, np.zeros(count))[slot] for pair in combinations(final, 2)), default=0.)
        if max_angle < config.min_triangulation_angle_deg:
            continue
        kept.append(query)
        points.append(point)
    return np.asarray(kept, dtype=int), np.asarray(points).reshape(-1, 3)
