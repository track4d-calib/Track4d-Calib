"""Verify geometric improvements without increasing high-frequency center motion."""

import numpy as np

if __package__:
    from .camera_motion import spline_basis
    from .center_trajectory import verify_trajectory
else:
    from camera_motion import spline_basis
    from center_trajectory import verify_trajectory


def center_high_frequency(centers, frame_ids, scale, knot_spacing=10.):
    """Measure center motion outside the continuous trajectory model."""
    centers = np.asarray(centers, dtype=float)
    if (centers.shape != (len(frame_ids), 3) or not np.isfinite(centers).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("Finite camera centers and a positive rig scale are required")
    basis = spline_basis(frame_ids, knot_spacing)
    residual = centers - basis @ np.linalg.lstsq(basis, centers, rcond=None)[0]
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))) / scale)


def verify_camera_motion(before, after, before_centers, after_centers, frame_ids, scale):
    """Require held-out geometry gains and non-increasing center jitter.

    The center statistic has no displacement or velocity bound. Constant,
    translating and smoothly curving trajectories are represented by the same
    low-frequency basis. Rotations retain the geometric verification criteria.
    """
    result = verify_trajectory(before, after)
    old = center_high_frequency(before_centers, frame_ids, scale)
    new = center_high_frequency(after_centers, frame_ids, scale)
    motion = {"before_fraction": old, "after_fraction": new,
              "knot_spacing": 10., "numerical_tolerance": 1e-10,
              "accepted": bool(new <= old + 1e-10)}
    return {**result, "geometry_accepted": result["accepted"], "center_motion": motion,
            "accepted": bool(result["accepted"] and motion["accepted"])}
