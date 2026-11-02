from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from track4d_calib.cameras import CameraParams


def triangulate_point_dlt(
    proj_matrices: List[np.ndarray],
    pts_2d: List[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Triangulate a single point with Direct Linear Transform (DLT).

    Args:
        proj_matrices: list of (3, 4) projection matrices.
        pts_2d: list of (2,) pixel coordinates aligned with proj_matrices.

    Returns:
        (3,) 3D point in world coordinates, or None if ill-conditioned.
    """
    if len(proj_matrices) != len(pts_2d):
        raise ValueError("proj_matrices and pts_2d must have the same length")
    n = len(proj_matrices)
    if n < 2:
        return None

    A = np.zeros((2 * n, 4), dtype=np.float64)
    for i, (P, pt) in enumerate(zip(proj_matrices, pts_2d)):
        x, y = float(pt[0]), float(pt[1])
        A[2 * i + 0] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]

    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    X = vt[-1]
    if abs(float(X[3])) < 1e-12:
        return None
    return (X[:3] / X[3]).astype(np.float64)


def _camera_to_P(cam: Union[CameraParams, Dict[str, np.ndarray]]) -> np.ndarray:
    if isinstance(cam, CameraParams):
        return cam.P
    # Support lightweight dict input in tests / callers.
    K = np.asarray(cam["K"], dtype=np.float64)
    R = np.asarray(cam["R"], dtype=np.float64)
    T = np.asarray(cam["T"], dtype=np.float64).reshape(3)
    return K @ np.hstack([R, T.reshape(3, 1)])


def triangulate_point_multi_view(
    *,
    cameras: Dict[str, Union[CameraParams, Dict[str, np.ndarray]]],
    points_2d: Dict[str, np.ndarray],
    visibility: Dict[str, bool],
    reprojection_error_threshold: float,
    min_views_for_triangulation: int = 2,
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Triangulate a point from multiple views with reprojection-error rejection.

    Returns:
        point_3d, mean_reprojection_error, num_views_used
    """
    visible_views = [vid for vid in cameras.keys() if bool(visibility.get(vid, False))]
    if len(visible_views) < int(min_views_for_triangulation):
        return None, float("inf"), len(visible_views)

    proj = []
    pts = []
    for vid in visible_views:
        if vid not in points_2d:
            continue
        proj.append(_camera_to_P(cameras[vid]))
        pts.append(np.asarray(points_2d[vid], dtype=np.float64).reshape(2))

    if len(proj) < int(min_views_for_triangulation):
        return None, float("inf"), len(proj)

    X = triangulate_point_dlt(proj, pts)
    if X is None:
        return None, float("inf"), len(proj)

    # Mean reprojection error over used views.
    total = 0.0
    for P, pt in zip(proj, pts):
        x = P @ np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
        if x[2] <= 1e-12:
            return None, float("inf"), len(proj)
        uv = x[:2] / x[2]
        total += float(np.linalg.norm(uv - pt))
    mean_err = total / float(len(proj))

    if mean_err > float(reprojection_error_threshold):
        return None, mean_err, len(proj)

    return X, mean_err, len(proj)
