"""Fit camera motion constraints using the same triangulated track observations."""

import json
from pathlib import Path

import numpy as np


def prepare_centers(root: Path, config: dict) -> dict:
    """Estimate one reference center per camera from the initial pose sequence."""
    if config["center_radius"] is None:
        return {}
    from track4d_calib.cameras import CameraLoader

    centers = {view: [] for view in config["dynamic_views"]}
    for frame in range(config["frames"]):
        loader = CameraLoader(root / f"initial_poses/f{frame:06d}/cameras_pred")
        for view, camera in loader.load_cameras(config["dynamic_views"]).items():
            centers[view].append((-camera.R.T @ camera.T).reshape(3))
    references = {view: np.median(values, axis=0).tolist() for view, values in centers.items()}
    (root / "motion_centers.json").write_text(json.dumps(references, indent=2) + "\n")
    return references


def solve_constrained_pose(points, pixels, template, center, radius):
    """Use the existing fixed/bounded-center solvers with track-derived 3D points."""
    from track4d_calib.fixed_center_pnp import solve_fixed_center_pnp, solve_near_fixed_center_pnp

    arguments = dict(
        object_points=points, image_points=pixels, target_template=template,
        fixed_center=np.asarray(center, dtype=float), reprojection_error_px=4.0,
        min_inliers=12, iterations_count=2000, rng_seed=0,
    )
    if radius == 0:
        return solve_fixed_center_pnp(**arguments)
    return solve_near_fixed_center_pnp(**arguments, center_radius_m=radius)
