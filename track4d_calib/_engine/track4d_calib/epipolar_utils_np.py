from __future__ import annotations

from typing import Optional

import numpy as np


def _as_np64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _skew(t: np.ndarray) -> np.ndarray:
    t = _as_np64(t).reshape(3)
    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
    return np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]], dtype=np.float64)


def essential_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build an essential matrix E = [t]_x R from a relative pose (cam2_from_cam1).

    Notes:
      - E is defined up to scale; we normalize t for numerical stability.
      - If ||t|| is ~0, we return a zero matrix.
    """
    R = _as_np64(R).reshape(3, 3)
    t = _as_np64(t).reshape(3)
    tn = float(np.linalg.norm(t))
    if tn < 1e-12:
        return np.zeros((3, 3), dtype=np.float64)
    t = t / tn
    return _skew(t) @ R


def sampson_distance(E: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute Sampson distance for normalized correspondences.

    Args:
        E: (3,3) essential matrix (scale-invariant for Sampson distance).
        p1: (N,2) normalized points in camera 1.
        p2: (N,2) normalized points in camera 2.
    Returns:
        (N,) distances (float64).
    """
    E = _as_np64(E).reshape(3, 3)
    p1 = _as_np64(p1).reshape(-1, 2)
    p2 = _as_np64(p2).reshape(-1, 2)
    if p1.shape != p2.shape:
        raise ValueError("p1 and p2 must have the same shape")

    n = int(p1.shape[0])
    x1 = np.concatenate([p1, np.ones((n, 1), dtype=np.float64)], axis=1)  # (N,3)
    x2 = np.concatenate([p2, np.ones((n, 1), dtype=np.float64)], axis=1)  # (N,3)

    Ex1 = (E @ x1.T).T  # (N,3)
    Etx2 = (E.T @ x2.T).T  # (N,3)
    x2tEx1 = np.sum(x2 * Ex1, axis=1)  # (N,)

    denom = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    denom = np.clip(denom, 1e-12, None)
    return (x2tEx1**2) / denom


def mean_sampson_distance(E: np.ndarray, p1: np.ndarray, p2: np.ndarray, *, mask: Optional[np.ndarray] = None) -> float:
    d = sampson_distance(E, p1, p2)
    if mask is not None:
        mask = np.asarray(mask).reshape(-1).astype(bool)
        if mask.shape[0] != d.shape[0]:
            raise ValueError("mask must have same length as points")
        d = d[mask]
    if d.size == 0:
        return float("inf")
    return float(np.mean(d))


def should_accept_pose_update(
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    R_init: np.ndarray,
    t_init: np.ndarray,
    R_cand: np.ndarray,
    t_cand: np.ndarray,
    mask: Optional[np.ndarray] = None,
    improvement_ratio: float = 0.0,
) -> bool:
    """
    Decide whether to replace an initial relative pose with a candidate pose.

    We accept the candidate only if it reduces mean Sampson distance by at least
    `improvement_ratio` (relative).
    """
    E_init = essential_from_Rt(R_init, t_init)
    E_cand = essential_from_Rt(R_cand, t_cand)
    m_init = mean_sampson_distance(E_init, p1, p2, mask=mask)
    m_cand = mean_sampson_distance(E_cand, p1, p2, mask=mask)

    if not np.isfinite(m_cand):
        return False
    if not np.isfinite(m_init):
        return True
    if m_init <= 1e-12:
        # Init is already (near) perfect on the given correspondences.
        return False
    return m_cand < m_init * (1.0 - float(improvement_ratio))

