from __future__ import annotations

from typing import Any

import numpy as np


def pixel_to_normalized(points_xy: np.ndarray, *, height: int, width: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2).copy()
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("image dimensions must be positive")
    points[:, 0] = points[:, 0] / float(width) * 2.0 - 1.0
    points[:, 1] = points[:, 1] / float(height) * 2.0 - 1.0
    return points


def normalized_to_pixel(points_xy: np.ndarray, *, height: int, width: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2).copy()
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("image dimensions must be positive")
    points[:, 0] = (points[:, 0] + 1.0) * 0.5 * float(width)
    points[:, 1] = (points[:, 1] + 1.0) * 0.5 * float(height)
    return points


def bilinear_sample_map(
    values: np.ndarray, query_xy_normalized: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    queries = np.asarray(query_xy_normalized, dtype=np.float64).reshape(-1, 2)
    scalar = array.ndim == 2
    if scalar:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError("values must have shape (H, W) or (H, W, C)")

    height, width, _channels = array.shape
    x = (queries[:, 0] + 1.0) * 0.5 * float(width) - 0.5
    y = (queries[:, 1] + 1.0) * 0.5 * float(height) - 0.5
    valid = (
        np.isfinite(queries).all(axis=1)
        & (x >= 0.0)
        & (x <= float(width - 1))
        & (y >= 0.0)
        & (y <= float(height - 1))
    )

    x0 = np.floor(np.where(np.isfinite(x), x, 0.0)).astype(np.int64)
    y0 = np.floor(np.where(np.isfinite(y), y, 0.0)).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    sampled = (
        array[y0c, x0c] * wa[:, None]
        + array[y1c, x0c] * wb[:, None]
        + array[y0c, x1c] * wc[:, None]
        + array[y1c, x1c] * wd[:, None]
    )
    sampled[~valid] = np.nan
    if scalar:
        sampled = sampled[:, 0]
    return sampled, valid


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def relay_bidirectional_query_points(
    *,
    source_points_xy: np.ndarray,
    warp_source_to_target: np.ndarray,
    warp_target_to_source: np.ndarray,
    overlap_source_to_target: np.ndarray,
    overlap_target_to_source: np.ndarray,
    source_image_size: tuple[int, int],
    target_image_size: tuple[int, int],
    minimum_overlap: float,
    maximum_cycle_error_px: float,
) -> dict[str, Any]:
    """Map fixed source queries to a target image and reject failed appearance cycles."""

    if not 0.0 <= float(minimum_overlap) <= 1.0:
        raise ValueError("minimum_overlap must be in [0, 1]")
    if float(maximum_cycle_error_px) <= 0.0:
        raise ValueError("maximum_cycle_error_px must be positive")
    source_height, source_width = (int(value) for value in source_image_size)
    target_height, target_width = (int(value) for value in target_image_size)
    source_points = np.asarray(source_points_xy, dtype=np.float64).reshape(-1, 2)
    source_normalized = pixel_to_normalized(
        source_points, height=source_height, width=source_width
    )

    target_normalized, forward_sample_valid = bilinear_sample_map(
        warp_source_to_target, source_normalized
    )
    forward_overlap, forward_overlap_valid = bilinear_sample_map(
        overlap_source_to_target, source_normalized
    )
    back_normalized, backward_sample_valid = bilinear_sample_map(
        warp_target_to_source, target_normalized
    )
    backward_overlap, backward_overlap_valid = bilinear_sample_map(
        overlap_target_to_source, target_normalized
    )

    target_points = normalized_to_pixel(
        target_normalized, height=target_height, width=target_width
    )
    back_points = normalized_to_pixel(
        back_normalized, height=source_height, width=source_width
    )
    cycle_error_px = np.linalg.norm(back_points - source_points, axis=1)
    target_in_bounds = (
        np.isfinite(target_points).all(axis=1)
        & (target_points[:, 0] >= 0.0)
        & (target_points[:, 0] < float(target_width))
        & (target_points[:, 1] >= 0.0)
        & (target_points[:, 1] < float(target_height))
    )
    valid = (
        np.isfinite(source_points).all(axis=1)
        & forward_sample_valid
        & forward_overlap_valid
        & backward_sample_valid
        & backward_overlap_valid
        & target_in_bounds
        & np.isfinite(cycle_error_px)
        & (np.asarray(forward_overlap) >= float(minimum_overlap))
        & (np.asarray(backward_overlap) >= float(minimum_overlap))
        & (cycle_error_px <= float(maximum_cycle_error_px))
    )
    filtered_points = target_points.copy()
    filtered_points[~valid] = np.nan
    return {
        "target_points": filtered_points,
        "valid": valid,
        "cycle_error_px": cycle_error_px,
        "forward_overlap": np.asarray(forward_overlap, dtype=np.float64),
        "backward_overlap": np.asarray(backward_overlap, dtype=np.float64),
        "diagnostics": {
            "query_count": int(len(source_points)),
            "accepted_count": int(np.count_nonzero(valid)),
            "minimum_overlap": float(minimum_overlap),
            "maximum_cycle_error_px": float(maximum_cycle_error_px),
            "cycle_error_px": _summary(cycle_error_px),
            "forward_overlap": _summary(np.asarray(forward_overlap)),
            "backward_overlap": _summary(np.asarray(backward_overlap)),
        },
    }
