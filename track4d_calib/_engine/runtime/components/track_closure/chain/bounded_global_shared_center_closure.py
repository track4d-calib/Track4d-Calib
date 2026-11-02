#!/usr/bin/env python3
"""Fit shared reference centers and optimize bounded camera motion."""

from __future__ import annotations

import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


INITIALIZER_WRAPPER = Path(__file__).resolve().with_name(
    "xyz_initialized_three_ptz_closure.py"
)
from relay_contract import TARGET_VIEWS as TARGETS
_NUM_FRAMES = int(os.environ.get("TRACK4D_CALIB_NUM_FRAMES", "30"))
if _NUM_FRAMES <= 0 or _NUM_FRAMES % 5:
    raise ValueError("TRACK4D_CALIB_NUM_FRAMES must be a positive multiple of five")
FRAMES = tuple(range(_NUM_FRAMES))
MAX_POINTS = 150


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INITIALIZER = _load_module(INITIALIZER_WRAPPER, "initializer_global_center_parent")
MULTIVIEW = INITIALIZER.MULTIVIEW
XYZ = INITIALIZER.XYZ
BASE = INITIALIZER.BASE
RAW_SOLVE = XYZ.ORIGINAL_SOLVE
GLOBAL_CENTERS: dict[str, np.ndarray] = {}
GLOBAL_CENTER_AUDIT: dict[str, Any] = {}
GLOBAL_CENTER_PRECOMPUTE_SECONDS = 0.0
GLOBAL_CENTER_CPU_SECONDS = 0.0


def bounded_center(
    anchor: np.ndarray, proposal: np.ndarray, radius: float
) -> tuple[np.ndarray, bool]:
    base = np.asarray(anchor, dtype=np.float64).reshape(3)
    value = np.asarray(proposal, dtype=np.float64).reshape(3)
    limit = float(radius)
    if not np.isfinite(base).all() or not np.isfinite(value).all():
        raise ValueError("center inputs must be finite")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("center trust radius must be finite and positive")
    delta = value - base
    distance = float(np.linalg.norm(delta))
    if distance <= limit:
        return value.copy(), False
    return base + delta * (limit / distance), True


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _fit_observation(view: str, frame: int) -> dict[str, Any]:
    row = MULTIVIEW.RELAY_OBSERVATIONS[(str(view), int(frame), "fit")]
    point_ids = np.asarray(row["point_ids"], dtype=np.int64).reshape(-1)
    selected = XYZ.evenly_spaced_indices(len(point_ids), MAX_POINTS)
    return {
        "frame": int(frame),
        "point_ids": point_ids[selected],
        "xyz": np.asarray(row["object_points"], dtype=np.float64)[selected],
        "image_points": np.asarray(row["image_points"], dtype=np.float64)[selected],
    }


def _solve_center_job(job: tuple[str, list[dict[str, Any]], dict[int, Any]]) -> dict[str, Any]:
    view, observations, initial = job
    started = time.perf_counter()
    solved = BASE.solve_joint_ptz_shared_center(
        observations=observations,
        initial_cameras=initial,
        reprojection_error_px=4.0,
        min_inliers_per_frame=12,
        min_successful_frames=3,
        rng_seed=0,
        allow_initial_camera_bootstrap=True,
        allow_joint_refined_rotation_fallback=True,
    )
    return {
        "view": str(view),
        "seconds": float(time.perf_counter() - started),
        "success": bool(solved.get("success", False)),
        "reason": str(solved.get("reason", "unknown")),
        "center": (
            np.asarray(solved["center"], dtype=np.float64)
            if solved.get("success", False)
            else None
        ),
        "input_frame_count": int(solved.get("input_frame_count", 0)),
        "successful_frame_count": int(solved.get("successful_frame_count", 0)),
    }


def prepare_global_centers() -> dict[str, Any]:
    global GLOBAL_CENTERS, GLOBAL_CENTER_AUDIT
    global GLOBAL_CENTER_PRECOMPUTE_SECONDS, GLOBAL_CENTER_CPU_SECONDS

    base_grid = BASE._load_pose_grid(Path(_argument("--base-pose-root")), FRAMES)
    confidence, _ = BASE.load_static_pnp_confidence(
        Path(_argument("--static-pnp-diagnostics")),
        frames=FRAMES,
        target_views=TARGETS,
    )
    anchors = {
        view: BASE.weighted_geometric_median(
            np.asarray([base_grid[frame][view].camera_center for frame in FRAMES]),
            np.asarray([confidence[(frame, view)] for frame in FRAMES]),
        )
        for view in TARGETS
    }
    radii = {
        view: float(
            np.median(
                [
                    np.linalg.norm(base_grid[frame][view].camera_center - anchors[view])
                    for frame in FRAMES
                ]
            )
        )
        for view in TARGETS
    }
    if any(not math.isfinite(radius) or radius <= 0.0 for radius in radii.values()):
        raise RuntimeError("public Static+PnP center spread is invalid")
    jobs = [
        (
            view,
            [_fit_observation(view, frame) for frame in FRAMES],
            {frame: base_grid[frame][view] for frame in FRAMES},
        )
        for view in TARGETS
    ]
    workers = min(len(jobs), max(1, int(os.cpu_count() or 1)))
    started = time.perf_counter()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        solved_rows = pool.map(_solve_center_job, jobs)
    GLOBAL_CENTER_PRECOMPUTE_SECONDS = float(time.perf_counter() - started)
    GLOBAL_CENTER_CPU_SECONDS = float(sum(row["seconds"] for row in solved_rows))

    centers = {}
    rows = {}
    for row in solved_rows:
        view = str(row["view"])
        anchor = anchors[view]
        proposal = row["center"] if row["success"] else anchor
        center, clipped = bounded_center(anchor, proposal, radii[view])
        centers[view] = center
        rows[view] = {
            key: value
            for key, value in row.items()
            if key not in {"center"}
        }
        rows[view].update(
            {
                "anchor": anchor.tolist(),
                "raw_proposal": np.asarray(proposal).tolist(),
                "bounded_center": center.tolist(),
                "trust_radius": radii[view],
                "raw_shift": float(np.linalg.norm(np.asarray(proposal) - anchor)),
                "bounded_shift": float(np.linalg.norm(center - anchor)),
                "clipped_to_existing_trust_radius": bool(clipped),
            }
        )
    GLOBAL_CENTERS = centers
    GLOBAL_CENTER_AUDIT = {
        "policy": "fit-only-30-frame-shared-center-bounded-by-Static-center-spread-v1",
        "parallel_workers": workers,
        "wall_seconds": GLOBAL_CENTER_PRECOMPUTE_SECONDS,
        "CPU_seconds": GLOBAL_CENTER_CPU_SECONDS,
        "fit_role_only": True,
        "confirm_or_final_used": False,
        "reprojection_error_px": 4.0,
        "min_inliers_per_frame": 12,
        "min_successful_frames": 3,
        "rng_seed": 0,
        "views": rows,
    }
    return GLOBAL_CENTER_AUDIT


def solve_with_global_shared_centers(*args, **kwargs):
    if args:
        raise ValueError("BOUNDED expects keyword-only direct BA inputs")
    if set(TARGETS) - set(GLOBAL_CENTERS):
        raise RuntimeError("global shared centers were not prepared")
    call = dict(kwargs)
    call["fixed_centers"] = {
        (int(frame), str(view)): np.asarray(GLOBAL_CENTERS[str(view)]).copy()
        for frame, view in kwargs["nodes"]
    }
    result = RAW_SOLVE(**call)
    result["global_shared_center"] = {
        "policy": GLOBAL_CENTER_AUDIT.get("policy"),
        "centers": {
            view: np.asarray(GLOBAL_CENTERS[view]).tolist()
            for view in sorted({str(node[1]) for node in kwargs["nodes"]})
        },
    }
    return result


solve_with_global_shared_centers.__module__ = BASE.__name__
solve_with_global_shared_centers.__name__ = "solve_direct_rotation_component"
solve_with_global_shared_centers.__qualname__ = "solve_direct_rotation_component"
XYZ.ORIGINAL_SOLVE = solve_with_global_shared_centers


def annotate_audit() -> None:
    INITIALIZER.annotate_audit()
    output_root = Path(_argument("--output-root"))
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "BOUNDED-bounded-global-shared-center-all-three-PTZ-closure-v1"
    payload["method_override"]["bounded_global_shared_center"] = {
        "single_change_from_INITIALIZER": True,
        "source": "same frozen RELAY_XYZ fit IDs and frame-local XYZ",
        "one_center_per_physical_PTZ": True,
        "different_PTZ_centers_independent": True,
        "trust_radius_policy": "existing median Static+PnP center spread",
        "candidate_rotation_BA_changed": False,
        "verifier_changed": False,
        "new_tuned_numeric_thresholds": 0,
        "precompute": GLOBAL_CENTER_AUDIT,
    }
    payload["optimized_global_centers"] = {
        view: center.tolist() for view, center in GLOBAL_CENTERS.items()
    }
    payload["global_center_precompute_wall_seconds"] = GLOBAL_CENTER_PRECOMPUTE_SECONDS
    payload["global_center_precompute_CPU_seconds"] = GLOBAL_CENTER_CPU_SECONDS
    payload["seconds_per_frame_all_PTZ"] = float(
        payload["seconds_per_frame_all_PTZ"]
        + GLOBAL_CENTER_PRECOMPUTE_SECONDS / float(_NUM_FRAMES)
    )
    payload["calibration_seconds_per_frame_including_roma_inference"] = float(
        payload["calibration_seconds_per_frame_including_roma_inference"]
        + GLOBAL_CENTER_PRECOMPUTE_SECONDS / float(_NUM_FRAMES)
    )
    payload["attestation"].update(
        {
            "BOUNDED_single_change_is_fit_only_global_shared_center": True,
            "global_center_uses_no_confirm_or_final_IDs": True,
            "global_center_uses_no_GT_test_or_NVS": True,
            "different_physical_PTZ_centers_remain_independent": True,
        }
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    prepare_global_centers()
    exit_code = BASE.main()
    if exit_code == 0:
        annotate_audit()
    raise SystemExit(exit_code)
