from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_np64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def colmap_camera_to_opencv_KD(camera) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a pycolmap.Camera to OpenCV-style (K, D) used by Track4D-Calib.

    Notes:
      - Track4D-Calib uses D as 5 coeffs: (k1,k2,p1,p2,k3).
      - For COLMAP models without some coefficients, missing entries are set to 0.
    """
    model = getattr(camera.model, "name", str(camera.model))
    model = str(model)
    p = _as_np64(camera.params).reshape(-1)

    if model in ("PINHOLE",):
        fx, fy, cx, cy = map(float, p[:4])
        D = np.zeros((5,), dtype=np.float64)
    elif model in ("SIMPLE_PINHOLE",):
        f, cx, cy = map(float, p[:3])
        fx, fy = f, f
        D = np.zeros((5,), dtype=np.float64)
    elif model in ("SIMPLE_RADIAL",):
        f, cx, cy, k = map(float, p[:4])
        fx, fy = f, f
        D = np.array([k, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model in ("RADIAL",):
        f, cx, cy, k1, k2 = map(float, p[:5])
        fx, fy = f, f
        D = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model in ("OPENCV",):
        fx, fy, cx, cy, k1, k2, p1, p2 = map(float, p[:8])
        D = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    elif model in ("FULL_OPENCV",):
        # fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6
        fx, fy, cx, cy = map(float, p[:4])
        k1, k2, p1, p2, k3 = map(float, p[4:9])
        D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    else:
        # Fallback: use calibration matrix for K, and zero distortion.
        K = _as_np64(camera.calibration_matrix()).reshape(3, 3)
        return K, np.zeros((5,), dtype=np.float64)

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, D


def normalize_cameras_to_ref(
    *,
    R_all: np.ndarray,
    t_all: np.ndarray,
    ref_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize a set of world->cam extrinsics so that the reference becomes identity.

    Args:
        R_all: (V,3,3) world->cam rotations.
        t_all: (V,3) world->cam translations.
        ref_idx: index of the reference camera in 0..V-1.
    Returns:
        (R_new, t_new) with same shapes. For ref: R=I, t=0.
    """
    R_all = _as_np64(R_all).reshape(-1, 3, 3)
    t_all = _as_np64(t_all).reshape(-1, 3)
    ref_idx = int(ref_idx)
    if not (0 <= ref_idx < R_all.shape[0]):
        raise ValueError("ref_idx out of range")

    R_ref = R_all[ref_idx]
    t_ref = t_all[ref_idx]

    R_new = R_all @ R_ref.T
    # t_i' = t_i - R_i R_ref^T t_ref
    t_new = t_all - (R_new @ t_ref.reshape(3, 1)).reshape(-1, 3)
    return R_new, t_new

