from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def track_image_queries(
    *,
    predictor: Any,
    image_paths: Sequence[Path],
    query_points: np.ndarray,
    query_frame_index: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Track explicit source-pixel queries through one ordered image sequence."""

    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("image_paths must not be empty")
    if not 0 <= int(query_frame_index) < len(paths):
        raise IndexError("query_frame_index is outside the image sequence")
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
    if not len(query_points):
        raise ValueError("query_points must not be empty")

    frames = [_load_rgb(path) for path in paths]
    source_h, source_w = frames[0].shape[:2]
    if any(frame.shape[:2] != (source_h, source_w) for frame in frames):
        raise ValueError("all query-tracking images must have the same resolution")
    interp_h, interp_w = int(predictor.interp_shape[0]), int(predictor.interp_shape[1])
    resized = [
        cv2.resize(frame, (interp_w, interp_h), interpolation=cv2.INTER_LINEAR)
        if frame.shape[:2] != (interp_h, interp_w)
        else frame
        for frame in frames
    ]
    scale_x = (interp_w - 1) / max(source_w - 1, 1)
    scale_y = (interp_h - 1) / max(source_h - 1, 1)
    query_small = query_points.copy()
    query_small[:, 0] *= scale_x
    query_small[:, 1] *= scale_y
    queries = torch.zeros(
        (1, len(query_small), 3),
        dtype=torch.float32,
        device=device,
    )
    queries[0, :, 0] = float(query_frame_index)
    queries[0, :, 1:] = torch.from_numpy(query_small).to(device)
    video = (
        torch.from_numpy(np.stack(resized))
        .permute(0, 3, 1, 2)[None]
        .float()
        .to(device)
    )
    with torch.no_grad():
        predicted_tracks, predicted_visibility = predictor(
            video,
            queries=queries,
            grid_size=0,
            backward_tracking=True,
        )
    tracks = predicted_tracks[0].detach().cpu().numpy().astype(np.float64)
    visibility_raw = predicted_visibility[0].detach().cpu().numpy()
    if visibility_raw.ndim == 3 and visibility_raw.shape[-1] == 1:
        visibility_raw = visibility_raw[..., 0]
    visibility = np.asarray(visibility_raw).astype(bool)
    tracks[..., 0] *= max(source_w - 1, 1) / max(interp_w - 1, 1)
    tracks[..., 1] *= max(source_h - 1, 1) / max(interp_h - 1, 1)
    tracks[int(query_frame_index)] = query_points
    visibility[int(query_frame_index)] = True
    in_bounds = (
        np.isfinite(tracks).all(axis=2)
        & (tracks[..., 0] >= 0.0)
        & (tracks[..., 0] <= float(source_w - 1))
        & (tracks[..., 1] >= 0.0)
        & (tracks[..., 1] <= float(source_h - 1))
    )
    visibility &= in_bounds
    return tracks, visibility, {
        "source_hw": [int(source_h), int(source_w)],
        "tracker_hw": [int(interp_h), int(interp_w)],
        "query_frame_index": int(query_frame_index),
        "query_count": int(len(query_points)),
        "backward_tracking": True,
    }
