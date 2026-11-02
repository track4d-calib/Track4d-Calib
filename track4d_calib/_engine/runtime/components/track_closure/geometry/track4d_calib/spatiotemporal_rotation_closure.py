from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def detect_temporal_seed_points(
    image_path: Path,
    *,
    max_points: int,
) -> np.ndarray:
    """Detect deterministic target-view corners without cross-view matching."""

    path = Path(image_path)
    if int(max_points) <= 0:
        raise ValueError("max_points must be positive")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"failed to load temporal seed image: {path}")
    cv2.setRNGSeed(0)
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=int(max_points),
        qualityLevel=0.01,
        minDistance=6.0,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(corners[:, 0], dtype=np.float64)


def build_target_only_lk_seed_bank(
    *,
    image_paths: list[Path],
    query_frame_index: int,
    max_points: int,
    forward_backward_threshold_px: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Track one target view from target-only corners over a short window."""

    paths = [Path(path) for path in image_paths]
    query_index = int(query_frame_index)
    if not paths:
        raise ValueError("image_paths must be non-empty")
    if not 0 <= query_index < len(paths):
        raise IndexError("query_frame_index is outside the image sequence")
    points = detect_temporal_seed_points(
        paths[query_index],
        max_points=int(max_points),
    )
    coarse = np.full((len(paths), len(points), 2), np.nan, dtype=np.float64)
    coarse_visibility = np.zeros((len(paths), len(points)), dtype=bool)
    coarse[query_index] = points
    coarse_visibility[query_index] = True
    tracks, visibility, audit = refine_tracks_with_pyramidal_lk(
        image_paths=paths,
        coarse_tracks=coarse,
        coarse_visibility=coarse_visibility,
        query_frame_index=query_index,
        forward_backward_threshold_px=float(forward_backward_threshold_px),
    )
    return tracks, visibility, {
        **audit,
        "seed_source": "target_only_gftt",
        "detected_point_count": int(len(points)),
        "max_points": int(max_points),
    }


def uniform_track_seed_frames(frames: list[int], *, stride: int) -> list[int]:
    """Choose a fixed-stride keyframe schedule and always cover the last frame."""

    ordered = [int(frame) for frame in frames]
    if not ordered or ordered != sorted(set(ordered)):
        raise ValueError("frames must be a non-empty sorted unique sequence")
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    seeds = ordered[:: int(stride)]
    if seeds[-1] != ordered[-1]:
        seeds.append(ordered[-1])
    return seeds


def bidirectional_seed_windows(
    *, frames: list[int], seed_frames: list[int]
) -> dict[int, list[int]]:
    """Cover each inter-keyframe segment from both verified endpoint seeds."""

    ordered = [int(frame) for frame in frames]
    seeds = sorted({int(frame) for frame in seed_frames})
    if not ordered or ordered != sorted(set(ordered)):
        raise ValueError("frames must be a non-empty sorted unique sequence")
    if not seeds or not set(seeds).issubset(ordered):
        raise ValueError("seed_frames must be a non-empty subset of frames")
    windows: dict[int, list[int]] = {}
    for index, seed in enumerate(seeds):
        left = seeds[index - 1] if index > 0 else seed
        right = seeds[index + 1] if index + 1 < len(seeds) else seed
        windows[seed] = [frame for frame in ordered if left <= frame <= right]
    return windows


def refine_tracks_with_pyramidal_lk(
    *,
    image_paths: list[Path],
    coarse_tracks: np.ndarray,
    coarse_visibility: np.ndarray,
    query_frame_index: int,
    forward_backward_threshold_px: float = 1.5,
    allow_invisible_coarse_hint: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Refine short-window coarse tracks at full image resolution."""

    paths = [Path(path) for path in image_paths]
    coarse = _tracks(coarse_tracks, name="coarse_tracks")
    coarse_visible = _visibility(
        coarse_visibility,
        name="coarse_visibility",
    )
    if not paths or len(paths) != coarse.shape[0]:
        raise ValueError("image_paths and coarse tracks must have equal non-zero length")
    if coarse_visible.shape != coarse.shape[:2]:
        raise ValueError("coarse visibility shape does not match tracks")
    query_index = int(query_frame_index)
    if not 0 <= query_index < len(paths):
        raise IndexError("query_frame_index is outside the image sequence")
    if float(forward_backward_threshold_px) <= 0.0:
        raise ValueError("forward_backward_threshold_px must be positive")

    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(image is None for image in images):
        missing = [str(path) for path, image in zip(paths, images) if image is None]
        raise FileNotFoundError(f"failed to load LK images: {missing}")
    height, width = images[0].shape
    if any(image.shape != (height, width) for image in images):
        raise ValueError("all LK images must have the same resolution")

    refined = coarse.copy()
    visible = coarse_visible & np.isfinite(coarse).all(axis=2)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    attempted_without_coarse_hint = 0
    accepted_without_coarse_hint = 0
    attempted_invisible_coarse_hint = 0
    accepted_invisible_coarse_hint = 0

    def propagate(current_index: int, next_index: int) -> None:
        nonlocal attempted_without_coarse_hint, accepted_without_coarse_hint
        nonlocal attempted_invisible_coarse_hint, accepted_invisible_coarse_hint
        active = visible[current_index] & np.isfinite(refined[current_index]).all(axis=1)
        point_ids = np.flatnonzero(active)
        visible[next_index] = False
        if not len(point_ids):
            return
        current_points = (
            refined[current_index, point_ids]
            .astype(np.float32)
            .reshape(-1, 1, 2)
            .copy()
        )
        coarse_next = coarse[next_index, point_ids]
        finite_coarse_hint = (
            np.isfinite(coarse_next).all(axis=1)
            & (coarse_next[:, 0] >= 0.0)
            & (coarse_next[:, 0] <= float(width - 1))
            & (coarse_next[:, 1] >= 0.0)
            & (coarse_next[:, 1] <= float(height - 1))
        )
        visible_coarse_hint = coarse_visible[next_index, point_ids]
        has_coarse_hint = finite_coarse_hint & (
            visible_coarse_hint
            | bool(allow_invisible_coarse_hint)
        )
        invisible_coarse_hint = has_coarse_hint & ~visible_coarse_hint
        initial_next = current_points.copy()
        initial_next[has_coarse_hint, 0] = coarse_next[has_coarse_hint].astype(
            np.float32
        )
        attempted_without_coarse_hint += int(np.count_nonzero(~has_coarse_hint))
        attempted_invisible_coarse_hint += int(
            np.count_nonzero(invisible_coarse_hint)
        )
        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            images[current_index],
            images[next_index],
            current_points,
            initial_next,
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if next_points is None or forward_status is None:
            return
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            images[next_index],
            images[current_index],
            next_points,
            np.ascontiguousarray(current_points.copy()),
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if backward_points is None or backward_status is None:
            return
        next_xy = next_points[:, 0].astype(np.float64)
        backward_xy = backward_points[:, 0].astype(np.float64)
        current_xy = current_points[:, 0].astype(np.float64)
        forward_backward_error = np.linalg.norm(backward_xy - current_xy, axis=1)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(next_xy).all(axis=1)
            & np.isfinite(backward_xy).all(axis=1)
            & (forward_backward_error <= float(forward_backward_threshold_px))
            & (next_xy[:, 0] >= 0.0)
            & (next_xy[:, 0] <= float(width - 1))
            & (next_xy[:, 1] >= 0.0)
            & (next_xy[:, 1] <= float(height - 1))
        )
        accepted_ids = point_ids[valid]
        refined[next_index, accepted_ids] = next_xy[valid]
        visible[next_index, accepted_ids] = True
        accepted_without_coarse_hint += int(
            np.count_nonzero(valid & ~has_coarse_hint)
        )
        accepted_invisible_coarse_hint += int(
            np.count_nonzero(valid & invisible_coarse_hint)
        )

    for next_index in range(query_index + 1, len(paths)):
        propagate(next_index - 1, next_index)
    for next_index in range(query_index - 1, -1, -1):
        propagate(next_index + 1, next_index)

    return refined, visible, {
        "backend": "full-resolution pyramidal LK with FB check",
        "source_hw": [int(height), int(width)],
        "query_frame_index": query_index,
        "forward_backward_threshold_px": float(forward_backward_threshold_px),
        "window_size": [21, 21],
        "max_level": 3,
        "coarse_visibility_required_for_propagation": False,
        "allow_invisible_coarse_hint": bool(allow_invisible_coarse_hint),
        "attempted_without_coarse_hint": int(attempted_without_coarse_hint),
        "accepted_without_coarse_hint": int(accepted_without_coarse_hint),
        "attempted_invisible_coarse_hint": int(
            attempted_invisible_coarse_hint
        ),
        "accepted_invisible_coarse_hint": int(accepted_invisible_coarse_hint),
    }


def _tracks(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"{name} must have shape (T, N, 2)")
    return array


def _visibility(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (T, N)")
    return array


def visible_track_correspondences(
    *,
    source_tracks: np.ndarray,
    source_visibility: np.ndarray,
    target_tracks: np.ndarray,
    target_visibility: np.ndarray,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return identity-preserving source/target track pairs at one frame."""

    source = _tracks(source_tracks, name="source_tracks")
    target = _tracks(target_tracks, name="target_tracks")
    source_visible = _visibility(source_visibility, name="source_visibility")
    target_visible = _visibility(target_visibility, name="target_visibility")
    if source.shape != target.shape:
        raise ValueError("source and target tracks must have the same shape")
    if source_visible.shape != source.shape[:2]:
        raise ValueError("source visibility shape does not match source tracks")
    if target_visible.shape != target.shape[:2]:
        raise ValueError("target visibility shape does not match target tracks")
    index = int(frame_index)
    if not 0 <= index < source.shape[0]:
        raise IndexError("frame_index is outside the track sequence")
    visible = (
        source_visible[index]
        & target_visible[index]
        & np.isfinite(source[index]).all(axis=1)
        & np.isfinite(target[index]).all(axis=1)
    )
    point_ids = np.flatnonzero(visible).astype(np.int64)
    return source[index, visible], target[index, visible], point_ids


def _correspondence_pair(
    value: tuple[np.ndarray, np.ndarray], *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(value[0], dtype=np.float64).reshape(-1, 2)
    target = np.asarray(value[1], dtype=np.float64).reshape(-1, 2)
    if len(source) != len(target):
        raise ValueError(f"{name} source and target counts differ")
    return source, target


def combine_correspondence_sets(
    values: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool any number of independently seeded correspondence sets."""

    if not values:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty.copy()
    pairs = [
        _correspondence_pair(value, name=f"correspondence_sets[{index}]")
        for index, value in enumerate(values)
    ]
    source = np.concatenate([pair[0] for pair in pairs], axis=0)
    target = np.concatenate([pair[1] for pair in pairs], axis=0)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    return source[finite], target[finite]


def combine_correspondences(
    *,
    direct: tuple[np.ndarray, np.ndarray],
    tracked: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool independent same-time matches and propagated temporal tracks."""

    return combine_correspondence_sets([direct, tracked])
