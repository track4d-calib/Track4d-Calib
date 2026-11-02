#!/usr/bin/env python3
"""Optimize camera geometry against dual-static XYZ observations."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

import numpy as np

from relay_contract import validate_runtime_xyz_cache_payload


PARTITION_WRAPPER = Path(__file__).resolve().with_name(
    "support_partitioned_guarded_closure.py"
)
XYZ_CACHE = Path(os.environ["TRACK4D_CALIB_RELAY_CACHE"]).expanduser().resolve()
MAX_XYZ_POINTS_PER_NODE = 150


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARTITION = _load_module(PARTITION_WRAPPER, "partition_closure")
BASE = PARTITION.BASE
ORIGINAL_SOLVE = BASE.solve_direct_rotation_component
ORIGINAL_EVALUATE = PARTITION.CROSSFIT.CONFIRM.evaluate_confirm_protected_component

with XYZ_CACHE.open("rb") as handle:
    XYZ_PAYLOAD = pickle.load(handle)
XYZ_OBSERVATIONS = XYZ_PAYLOAD["observations"]


def validate_xyz_cache() -> dict[str, Any]:
    return validate_runtime_xyz_cache_payload(XYZ_PAYLOAD)


XYZ_CACHE_AUDIT = validate_xyz_cache()


def evenly_spaced_indices(count: int, maximum: int) -> np.ndarray:
    count = int(count)
    maximum = int(maximum)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum).round().astype(np.int64)


def solve_with_dual_static_xyz(*args, **kwargs):
    """Replace representative-node fit XYZ with two-static triangulation."""
    observations = {
        (int(node[0]), str(node[1])): dict(row)
        for node, row in kwargs["reprojection_observations"].items()
    }
    replaced = []
    for node in [(int(frame), str(view)) for frame, view in kwargs["nodes"]]:
        row = XYZ_OBSERVATIONS.get((node[1], node[0], "fit"))
        if row is None:
            continue
        xyz = np.asarray(row["object_points"], dtype=np.float64)
        image = np.asarray(row["image_points"], dtype=np.float64)
        selected = evenly_spaced_indices(len(xyz), MAX_XYZ_POINTS_PER_NODE)
        observations[node] = {
            "xyz": xyz[selected],
            "image_points": image[selected],
        }
        replaced.append((node[0], node[1], len(selected), len(xyz)))
    call = dict(kwargs)
    call["reprojection_observations"] = observations
    result = ORIGINAL_SOLVE(*args, **call)
    result["dual_static_target_query_XYZ"] = {
        "replaced_node_count": len(replaced),
        "selected_point_count": int(sum(row[2] for row in replaced)),
        "available_point_count": int(sum(row[3] for row in replaced)),
        "max_points_per_node": MAX_XYZ_POINTS_PER_NODE,
        "selection": "deterministic_evenly_spaced_query_ID_order",
        "nodes": [f"{frame}:{view}" for frame, view, _selected, _available in replaced],
    }
    return result


solve_with_dual_static_xyz.__module__ = BASE.__name__
solve_with_dual_static_xyz.__name__ = "solve_direct_rotation_component"
solve_with_dual_static_xyz.__qualname__ = "solve_direct_rotation_component"
BASE.solve_direct_rotation_component = solve_with_dual_static_xyz


def reprojection_errors(camera: Any, row: Mapping[str, Any]) -> np.ndarray:
    xyz = np.asarray(row["object_points"], dtype=np.float64).reshape(-1, 3)
    image = np.asarray(row["image_points"], dtype=np.float64).reshape(-1, 2)
    camera_points = (np.asarray(camera.R) @ xyz.T).T + np.asarray(camera.T)
    projected = (np.asarray(camera.K) @ camera_points.T).T
    depth = projected[:, 2]
    safe = np.where(np.abs(depth) > 1e-9, depth, np.nan)
    pixels = projected[:, :2] / safe[:, None]
    errors = np.linalg.norm(pixels - image, axis=1)
    errors[depth <= 1e-9] = np.inf
    return errors


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(np.max(array)),
    }


def dual_static_xyz_score(
    candidate: Mapping[int, Mapping[str, Any]],
    *,
    component: list[str],
    frames: list[int],
    role: str,
) -> dict[str, Any]:
    if role not in {"confirm", "final"}:
        raise ValueError("dual-static XYZ guard only accepts heldout roles")
    context = BASE._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")
    target_views = sorted(
        view
        for view in component
        if all((str(view), int(frame), role) in XYZ_OBSERVATIONS for frame in frames)
    )
    if len(target_views) != 1:
        return {
            "accepted": False,
            "reason": "component_does_not_have_exactly_one_dual_static_XYZ_target",
            "target_views": target_views,
        }
    base_all = []
    candidate_all = []
    gain_all = []
    rows = []
    for frame in frames:
        for view in target_views:
            observation = XYZ_OBSERVATIONS[(str(view), int(frame), role)]
            base_errors = reprojection_errors(
                context["base_grid"][int(frame)][str(view)], observation
            )
            candidate_errors = reprojection_errors(
                candidate[int(frame)][str(view)], observation
            )
            if not np.isfinite(base_errors).all() or not np.isfinite(candidate_errors).all():
                return {
                    "accepted": False,
                    "reason": "dual_static_XYZ_nonpositive_depth_or_nonfinite_projection",
                    "node": f"{frame}:{view}:{role}",
                }
            gains = np.log1p(base_errors) - np.log1p(candidate_errors)
            base_all.append(base_errors)
            candidate_all.append(candidate_errors)
            gain_all.append(gains)
            rows.append(
                {
                    "frame": int(frame),
                    "view": str(view),
                    "count": int(len(gains)),
                    "mean_log_gain": float(np.mean(gains)),
                    "median_log_gain": float(np.median(gains)),
                }
            )
    base_values = np.concatenate(base_all)
    candidate_values = np.concatenate(candidate_all)
    gain_values = np.concatenate(gain_all)
    node_gains = np.asarray([row["median_log_gain"] for row in rows])
    metrics = {
        "mean_log_gain": float(np.mean(gain_values)),
        "median_log_gain": float(np.median(gain_values)),
    }
    wins = int(np.count_nonzero(node_gains > 0.0))
    losses = int(np.count_nonzero(node_gains < 0.0))
    checks = {name: bool(value > 0.0) for name, value in metrics.items()}
    checks["node_wins_exceed_losses"] = bool(wins > losses)
    accepted = bool(all(checks.values()))
    return {
        "policy": "dual_static_target_query_XYZ_paired_log_gain_guard_v1",
        "accepted": accepted,
        "reason": (
            "dual_static_XYZ_heldout_geometry_improved"
            if accepted
            else "dual_static_XYZ_heldout_geometry_not_improved"
        ),
        "role": role,
        "target_views": target_views,
        "base": _summary(base_values),
        "candidate": _summary(candidate_values),
        **metrics,
        "node_median_log_gain": _summary(node_gains),
        "node_wins": wins,
        "node_losses": losses,
        "checks": checks,
        "new_tuned_numeric_thresholds": 0,
        "rows": rows,
    }


def evaluate_dual_static_xyz_guarded_component(job):
    row, selected, diagnostic = ORIGINAL_EVALUATE(job)
    if not row.get("commit"):
        return row, selected, diagnostic
    context = BASE._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")
    frames = [int(frame) for frame in row["window"]]
    component = [str(view) for view in row["component"]]
    candidate = {frame: dict(context["base_grid"][frame]) for frame in frames}
    for (frame, view), camera in selected.items():
        candidate[int(frame)][str(view)] = camera
    confirm = dual_static_xyz_score(
        candidate, component=component, frames=frames, role="confirm"
    )
    final = dual_static_xyz_score(
        candidate, component=component, frames=frames, role="final"
    )
    accepted = bool(confirm["accepted"] and final["accepted"])
    row["dual_static_XYZ_crossfit_guard"] = {
        "accepted": accepted,
        "confirm": confirm,
        "final": final,
        "fit_confirm_final_target_query_IDs_disjoint": True,
        "candidate_modified": False,
    }
    row["commit_policy"] += "_plus_dual_static_XYZ_confirm_final"
    if accepted:
        return row, selected, diagnostic
    row["commit"] = False
    row["reason"] = "dual_static_XYZ_crossfit_rejected"
    return row, {}, diagnostic


evaluate_dual_static_xyz_guarded_component.__module__ = BASE.__name__
evaluate_dual_static_xyz_guarded_component.__name__ = "_evaluate_direct_component_from_context"
evaluate_dual_static_xyz_guarded_component.__qualname__ = "_evaluate_direct_component_from_context"
BASE._evaluate_direct_component_from_context = evaluate_dual_static_xyz_guarded_component


def annotate_audit() -> None:
    PARTITION.CROSSFIT.CONFIRM.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "XYZ-support-partitioned-dual-static-target-query-XYZ-v1"
    payload["method_override"]["support_partitioned_components"] = {
        "max_component_views": PARTITION.MAX_COMPONENT_VIEWS,
        "selection_evidence": "summed public same-time shared-track counts",
        "numeric_geometry_thresholds_changed": False,
    }
    payload["method_override"]["dual_static_target_query_XYZ"] = {
        "cache": str(XYZ_CACHE),
        "candidate_role": "fit",
        "verifier_roles": ["confirm", "final"],
        "replace_representative_node_fit_XYZ": True,
        "max_points_per_node": MAX_XYZ_POINTS_PER_NODE,
        "numeric_geometry_thresholds_changed": False,
        "new_tuned_numeric_thresholds": 0,
        **XYZ_CACHE_AUDIT,
    }
    inference = float(
        XYZ_PAYLOAD["runtime"]["single_model_equivalent_inference_seconds_per_frame"]
    )
    payload["dual_static_XYZ_match_runtime"] = XYZ_PAYLOAD["runtime"]
    payload["calibration_seconds_per_frame_including_roma_inference"] = float(
        payload["seconds_per_frame_all_PTZ"] + inference
    )
    payload["attestation"].update(
        {
            "component_partition_uses_no_GT_test_or_NVS": True,
            "dual_static_XYZ_uses_public_train_cameras_only": True,
            "dual_static_XYZ_is_frame_local": True,
            "dual_static_XYZ_fit_confirm_final_IDs_disjoint": True,
            "dual_static_XYZ_confirm_final_unused_by_candidate": True,
            "legacy_single_static_RoMa_cache_unused": True,
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
    annotate_audit()
    raise SystemExit(exit_code)
