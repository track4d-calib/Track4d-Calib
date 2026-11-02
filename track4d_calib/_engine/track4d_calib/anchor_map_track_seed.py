from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _quality_key(
    point_id: int,
    *,
    point_error_by_id: Mapping[int, float],
    point_track_length_by_id: Mapping[int, int],
) -> tuple[float, int, int]:
    error = float(point_error_by_id.get(int(point_id), float("inf")))
    if not np.isfinite(error):
        error = float("inf")
    track_length = int(point_track_length_by_id.get(int(point_id), 0))
    return error, -track_length, int(point_id)


def _normalize_observations(
    observations: Mapping[int, np.ndarray],
    *,
    points3d_xyz_by_id: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    normalized: dict[int, np.ndarray] = {}
    for point_id_raw, xy_raw in observations.items():
        point_id = int(point_id_raw)
        if point_id not in points3d_xyz_by_id:
            continue
        xy = np.asarray(xy_raw, dtype=np.float32).reshape(-1)
        xyz = np.asarray(points3d_xyz_by_id[point_id], dtype=np.float64).reshape(-1)
        if xy.shape != (2,) or xyz.shape != (3,):
            continue
        if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(xyz)):
            continue
        normalized[point_id] = xy
    return normalized


def build_anchor_map_track_seed(
    *,
    observations_by_view: Mapping[str, Mapping[int, np.ndarray]],
    points3d_xyz_by_id: Mapping[int, np.ndarray],
    views: Sequence[str],
    ref_view: str,
    max_points_per_view: int,
    point_error_by_id: Mapping[int, float] | None = None,
    point_track_length_by_id: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Pack anchor-map observations into the tracker's shared point-ID layout."""

    view_order = [str(view) for view in views]
    if len(view_order) != len(set(view_order)):
        raise ValueError("views must be unique")
    if str(ref_view) not in view_order:
        raise ValueError(f"ref_view={ref_view} is not present in views")

    error_by_id = point_error_by_id or {}
    track_length_by_id = point_track_length_by_id or {}
    selected_by_view: dict[str, dict[int, np.ndarray]] = {}
    contribution_counts: dict[str, int] = {}

    for view in view_order:
        normalized = _normalize_observations(
            observations_by_view.get(view, {}),
            points3d_xyz_by_id=points3d_xyz_by_id,
        )
        point_ids = sorted(
            normalized,
            key=lambda point_id: _quality_key(
                point_id,
                point_error_by_id=error_by_id,
                point_track_length_by_id=track_length_by_id,
            ),
        )
        if int(max_points_per_view) > 0:
            point_ids = point_ids[: int(max_points_per_view)]
        selected_by_view[view] = {
            point_id: normalized[point_id] for point_id in point_ids
        }
        contribution_counts[view] = int(len(point_ids))

    point3d_ids = np.asarray(
        sorted(
            {
                point_id
                for observations in selected_by_view.values()
                for point_id in observations
            }
        ),
        dtype=np.int64,
    )
    if point3d_ids.size == 0:
        raise RuntimeError(
            "No valid anchor-map observations remain for the requested views"
        )

    point_index = {
        int(point_id): index for index, point_id in enumerate(point3d_ids.tolist())
    }
    num_points = int(point3d_ids.shape[0])
    xy_by_view: dict[str, np.ndarray] = {}
    valid_by_view: dict[str, np.ndarray] = {}
    for view in view_order:
        xy = np.zeros((num_points, 2), dtype=np.float32)
        valid = np.zeros((num_points,), dtype=bool)
        for point_id, observation_xy in selected_by_view[view].items():
            index = point_index[int(point_id)]
            xy[index] = observation_xy
            valid[index] = True
        xy_by_view[view] = xy
        valid_by_view[view] = valid

    xyz = np.stack(
        [
            np.asarray(
                points3d_xyz_by_id[int(point_id)], dtype=np.float64
            ).reshape(3)
            for point_id in point3d_ids
        ],
        axis=0,
    )
    ref_key = str(ref_view)
    other_views = [view for view in view_order if view != ref_key]

    return {
        "ref_xy": xy_by_view[ref_key],
        "ref_valid": valid_by_view[ref_key],
        "xy_by_view": {view: xy_by_view[view] for view in other_views},
        "valid_by_view": {view: valid_by_view[view] for view in other_views},
        "point3D_ids": point3d_ids,
        "points3D_xyz": xyz,
        "diag": {
            "landmark_selected_total": num_points,
            "selected_contribution_by_view": contribution_counts,
            "anchor_observations_by_view": contribution_counts,
            "max_points_per_view": int(max_points_per_view),
        },
    }


def _resolve_image_view_id(image_name: str, views: Sequence[str]) -> str | None:
    requested = [str(view) for view in views]
    stem = Path(str(image_name)).stem
    candidates = [stem]
    if stem.lower().startswith("view"):
        candidates.append(stem[4:])

    for candidate in candidates:
        if candidate in requested:
            return candidate

    numeric_to_view: dict[int, str] = {}
    ambiguous: set[int] = set()
    for view in requested:
        if not view.isdigit():
            continue
        key = int(view)
        if key in numeric_to_view and numeric_to_view[key] != view:
            ambiguous.add(key)
        else:
            numeric_to_view[key] = view
    for candidate in candidates:
        if candidate.isdigit():
            key = int(candidate)
            if key not in ambiguous and key in numeric_to_view:
                return numeric_to_view[key]
    return None


def _point_is_in_mask(xy: np.ndarray, mask: np.ndarray) -> bool:
    x = int(np.rint(float(xy[0])))
    y = int(np.rint(float(xy[1])))
    return (
        0 <= y < int(mask.shape[0])
        and 0 <= x < int(mask.shape[1])
        and bool(mask[y, x])
    )


def _track_length(track: Any) -> int:
    length = getattr(track, "length", None)
    if callable(length):
        return int(length())
    elements = getattr(track, "elements", [])
    return int(len(elements))


def build_anchor_map_track_seed_from_reconstruction(
    reconstruction: Any,
    *,
    views: Sequence[str],
    ref_view: str,
    max_points_per_view: int,
    masks_by_view: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    view_order = [str(view) for view in views]
    observations_by_view: dict[str, dict[int, np.ndarray]] = {
        view: {} for view in view_order
    }
    masks = {
        str(view): np.asarray(mask) for view, mask in (masks_by_view or {}).items()
    }

    for image in reconstruction.images.values():
        view = _resolve_image_view_id(str(image.name), view_order)
        if view is None:
            continue
        for point2d in image.points2D:
            if not bool(point2d.has_point3D()):
                continue
            point_id = int(point2d.point3D_id)
            xy = np.asarray(point2d.xy, dtype=np.float32).reshape(-1)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                continue
            if view in masks and not _point_is_in_mask(xy, masks[view]):
                continue
            observations_by_view[view][point_id] = xy

    points3d_xyz_by_id: dict[int, np.ndarray] = {}
    point_error_by_id: dict[int, float] = {}
    point_track_length_by_id: dict[int, int] = {}
    for point_id_raw, point3d in reconstruction.points3D.items():
        point_id = int(point_id_raw)
        points3d_xyz_by_id[point_id] = np.asarray(
            point3d.xyz, dtype=np.float64
        ).reshape(3)
        point_error_by_id[point_id] = float(point3d.error)
        point_track_length_by_id[point_id] = _track_length(point3d.track)

    return build_anchor_map_track_seed(
        observations_by_view=observations_by_view,
        points3d_xyz_by_id=points3d_xyz_by_id,
        views=view_order,
        ref_view=str(ref_view),
        max_points_per_view=int(max_points_per_view),
        point_error_by_id=point_error_by_id,
        point_track_length_by_id=point_track_length_by_id,
    )


def load_anchor_map_track_seed(
    model_dir: Path,
    *,
    views: Sequence[str],
    ref_view: str,
    max_points_per_view: int,
    masks_by_view: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    try:
        import pycolmap
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("pycolmap is required for seed_builder=anchor_map") from exc

    model_path = Path(model_dir)
    required = [
        model_path / name for name in ("cameras.bin", "images.bin", "points3D.bin")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Invalid anchor map model; missing: {missing}")
    reconstruction = pycolmap.Reconstruction(str(model_path))
    return build_anchor_map_track_seed_from_reconstruction(
        reconstruction,
        views=views,
        ref_view=ref_view,
        max_points_per_view=max_points_per_view,
        masks_by_view=masks_by_view,
    )
