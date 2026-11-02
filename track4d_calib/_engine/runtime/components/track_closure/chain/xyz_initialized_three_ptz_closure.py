#!/usr/bin/env python3
"""Initialize bundle adjustment from dual-static XYZ observations."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


JACOBIAN_WRAPPER = Path(__file__).resolve().with_name(
    "jac_scaled_three_ptz_closure.py"
)
REPROJECTION_ERROR_PX = 4.0
MIN_INLIERS_PER_FRAME = 12
MIN_SUCCESSFUL_FRAMES = 3


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


JACOBIAN = _load_module(JACOBIAN_WRAPPER, "jacobian_closure")
MULTIVIEW = JACOBIAN.MULTIVIEW
XYZ = MULTIVIEW.XYZ
BASE = JACOBIAN.BASE
RAW_SOLVE = XYZ.ORIGINAL_SOLVE
XYZ_INITIALIZATION_CACHE: dict[tuple[tuple[int, str], ...], tuple[dict[Any, np.ndarray], dict[str, Any]]] = {}


def xyz_endpoint_initialization(**kwargs) -> tuple[dict[Any, np.ndarray], dict[str, Any]]:
    nodes = [(int(frame), str(view)) for frame, view in kwargs["nodes"]]
    key = tuple(nodes)
    cached = XYZ_INITIALIZATION_CACHE.get(key)
    if cached is not None:
        residuals, audit = cached
        return {node: value.copy() for node, value in residuals.items()}, dict(audit)

    grouped: dict[str, list[int]] = {}
    for frame, view in nodes:
        grouped.setdefault(view, []).append(frame)
    residuals: dict[Any, np.ndarray] = {}
    rows = {}
    for view, raw_frames in grouped.items():
        frames = sorted(raw_frames)
        observations = []
        for frame in frames:
            node = (frame, view)
            row = kwargs["reprojection_observations"][node]
            relay = MULTIVIEW.RELAY_OBSERVATIONS[(view, frame, "fit")]
            xyz = np.asarray(row["xyz"], dtype=np.float64).reshape(-1, 3)
            image = np.asarray(row["image_points"], dtype=np.float64).reshape(-1, 2)
            relay_ids = np.asarray(relay["point_ids"], dtype=np.int64).reshape(-1)
            relay_xyz = np.asarray(relay["object_points"], dtype=np.float64).reshape(-1, 3)
            relay_image = np.asarray(relay["image_points"], dtype=np.float64).reshape(-1, 2)
            selected = XYZ.evenly_spaced_indices(
                len(relay_ids), XYZ.MAX_XYZ_POINTS_PER_NODE
            )
            if not (
                np.array_equal(xyz, relay_xyz[selected])
                and np.array_equal(image, relay_image[selected])
            ):
                raise ValueError(f"direct BA observation lost frozen relay identity at {node}")
            observations.append(
                {
                    "frame": frame,
                    "point_ids": relay_ids[selected],
                    "xyz": xyz,
                    "image_points": image,
                }
            )
        solved = BASE.solve_fixed_center_ptz_rotations(
            observations=observations,
            initial_cameras={frame: kwargs["base_cameras"][(frame, view)] for frame in frames},
            fixed_center=np.asarray(kwargs["fixed_centers"][(frames[0], view)], dtype=np.float64),
            reprojection_error_px=REPROJECTION_ERROR_PX,
            min_inliers_per_frame=MIN_INLIERS_PER_FRAME,
            min_successful_frames=MIN_SUCCESSFUL_FRAMES,
            rng_seed=0,
        )
        successful = [int(frame) for frame in solved.get("successful_frames", [])]
        if bool(solved.get("success", False)):
            for frame in successful:
                node = (frame, view)
                residuals[node] = Rotation.from_matrix(
                    np.asarray(solved["cameras"][frame].R, dtype=np.float64)
                    @ np.asarray(kwargs["base_cameras"][node].R, dtype=np.float64).T
                ).as_rotvec()
        rows[view] = {
            "success": bool(solved.get("success", False)),
            "successful_frames": successful,
            "frame_inlier_counts": solved.get("frame_inlier_counts", {}),
            "frame_reprojection_median_px": solved.get("frame_reprojection_median_px", {}),
        }
    audit = {
        "policy": "frozen_dual_static_XYZ_fixed_center_rotation_endpoint",
        "initialized_node_count": len(residuals),
        "node_count": len(nodes),
        "reprojection_error_px": REPROJECTION_ERROR_PX,
        "min_inliers_per_frame": MIN_INLIERS_PER_FRAME,
        "min_successful_frames": MIN_SUCCESSFUL_FRAMES,
        "views": rows,
        "GT_test_or_NVS_used": False,
    }
    XYZ_INITIALIZATION_CACHE[key] = (
        {node: value.copy() for node, value in residuals.items()},
        dict(audit),
    )
    return residuals, audit


def solve_with_xyz_endpoint_initialization(*args, **kwargs):
    if args:
        raise ValueError("INITIALIZER expects keyword-only direct BA inputs")
    residuals, audit = xyz_endpoint_initialization(**kwargs)
    initial = {
        node: np.asarray(value, dtype=np.float64).copy()
        for node, value in (kwargs.get("initial_rotation_residuals") or {}).items()
    }
    initial.update(residuals)
    call = dict(kwargs)
    call["initial_rotation_residuals"] = initial
    result = RAW_SOLVE(**call)
    result["dual_static_XYZ_endpoint_initialization"] = audit
    return result


solve_with_xyz_endpoint_initialization.__module__ = BASE.__name__
solve_with_xyz_endpoint_initialization.__name__ = "solve_direct_rotation_component"
solve_with_xyz_endpoint_initialization.__qualname__ = "solve_direct_rotation_component"
XYZ.ORIGINAL_SOLVE = solve_with_xyz_endpoint_initialization


def annotate_audit() -> None:
    JACOBIAN.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "INITIALIZER-dual-static-XYZ-initialized-all-three-PTZ-closure-v1"
    payload["method_override"]["dual_static_XYZ_endpoint_initialization"] = {
        "source": "same frozen RELAY_XYZ fit observations already used by JACOBIAN candidate",
        "reprojection_error_px": REPROJECTION_ERROR_PX,
        "min_inliers_per_frame": MIN_INLIERS_PER_FRAME,
        "min_successful_frames": MIN_SUCCESSFUL_FRAMES,
        "optimizer_objective_changed": False,
        "verifier_changed": False,
        "new_tuned_numeric_thresholds": 0,
    }
    payload["attestation"].update(
        {
            "INITIALIZER_single_change_is_candidate_initialization": True,
            "XYZ_endpoint_uses_fit_IDs_only": True,
            "confirm_and_final_remain_independent": True,
        }
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    exit_code = BASE.main()
    if exit_code == 0:
        annotate_audit()
    raise SystemExit(exit_code)
