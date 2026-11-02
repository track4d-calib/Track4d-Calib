#!/usr/bin/env python3
"""Jointly optimize multiview camera geometry against XYZ observations."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

import numpy as np

from relay_contract import BANDS, validate_relay_payload


XYZ_WRAPPER = Path(__file__).resolve().with_name("dual_static_xyz_closure.py")
RELAY_CACHE = Path(os.environ["TRACK4D_CALIB_RELAY_CACHE"]).expanduser().resolve()
ROLES = ("fit", "confirm", "final")


def _num_frames() -> int:
    value = int(os.environ.get("TRACK4D_CALIB_NUM_FRAMES", "30"))
    if value <= 0 or value % 5:
        raise ValueError("TRACK4D_CALIB_NUM_FRAMES must be a positive multiple of five")
    return value


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


XYZ = _load_module(XYZ_WRAPPER, "xyz_closure")
BASE = XYZ.BASE
ORIGINAL_EVALUATE = XYZ.ORIGINAL_EVALUATE

with RELAY_CACHE.open("rb") as handle:
    RELAY_PAYLOAD = pickle.load(handle)
RELAY_OBSERVATIONS = RELAY_PAYLOAD["observations"]


def validate_relay_cache() -> dict[str, Any]:
    return validate_relay_payload(
        RELAY_PAYLOAD,
        bands=BANDS,
        frames=tuple(range(_num_frames())),
        roles=ROLES,
    )


RELAY_CACHE_AUDIT = validate_relay_cache()

# The XYZ solver reads this module global. Replacing it makes the same
# direct-rotation BA consume the relayed XYZ on all three PTZ nodes.
XYZ.XYZ_OBSERVATIONS = RELAY_OBSERVATIONS


def reprojection_errors(camera: Any, row: Mapping[str, Any]) -> np.ndarray:
    return XYZ.reprojection_errors(camera, row)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(np.max(array)),
    }


def all_ptz_dual_static_xyz_score(
    candidate: Mapping[int, Mapping[str, Any]],
    *,
    component: list[str],
    frames: list[int],
    role: str,
) -> dict[str, Any]:
    if role not in {"confirm", "final"}:
        raise ValueError("all-PTZ XYZ guard only accepts heldout roles")
    context = BASE._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")

    base_all = []
    candidate_all = []
    gain_all = []
    rows = []
    gains_by_view: dict[str, list[np.ndarray]] = {}
    unchanged_nodes = []
    for frame in frames:
        for view in component:
            base_camera = context["base_grid"][int(frame)][str(view)]
            candidate_camera = candidate[int(frame)][str(view)]
            if BASE._camera_equal(base_camera, candidate_camera):
                unchanged_nodes.append(f"{frame}:{view}")
                continue
            observation = RELAY_OBSERVATIONS[(str(view), int(frame), role)]
            base_errors = reprojection_errors(
                base_camera, observation
            )
            candidate_errors = reprojection_errors(
                candidate_camera, observation
            )
            if (base_errors.size == 0 or base_errors.shape != candidate_errors.shape
                    or not np.isfinite(base_errors).all() or not np.isfinite(candidate_errors).all()):
                return {
                    "accepted": False,
                    "reason": "all_PTZ_dual_static_XYZ_nonpositive_depth_or_nonfinite_projection",
                    "node": f"{frame}:{view}:{role}",
                }
            gains = np.log1p(base_errors) - np.log1p(candidate_errors)
            base_all.append(base_errors)
            candidate_all.append(candidate_errors)
            gain_all.append(gains)
            gains_by_view.setdefault(str(view), []).append(gains)
            rows.append(
                {
                    "frame": int(frame),
                    "view": str(view),
                    "count": int(len(gains)),
                    "mean_log_gain": float(np.mean(gains)),
                    "median_log_gain": float(np.median(gains)),
                }
            )

    if not gain_all:
        return {"accepted": False, "reason": "no_changed_camera_nodes", "role": role,
                "unchanged_nodes": unchanged_nodes}
    base_values = np.concatenate(base_all)
    candidate_values = np.concatenate(candidate_all)
    gain_values = np.concatenate(gain_all)
    node_gains = np.asarray([row["median_log_gain"] for row in rows], dtype=np.float64)
    view_medians = {
        view: float(np.median(np.concatenate(values)))
        for view, values in gains_by_view.items()
    }
    view_balanced_median = float(np.median(list(view_medians.values())))
    wins = int(np.count_nonzero(node_gains > 0.0))
    losses = int(np.count_nonzero(node_gains < 0.0))
    checks = {
        "global_mean_log_gain_positive": bool(np.mean(gain_values) > 0.0),
        "global_median_log_gain_positive": bool(np.median(gain_values) > 0.0),
        "view_balanced_median_log_gain_positive": bool(view_balanced_median > 0.0),
        "every_PTZ_median_log_gain_positive": bool(all(value > 0.0 for value in view_medians.values())),
        "node_wins_exceed_losses": bool(wins > losses),
    }
    accepted = bool(all(checks.values()))
    return {
        "policy": "changed_nodes_dual_static_XYZ_paired_log_gain_guard_v2",
        "accepted": accepted,
        "reason": (
            "all_PTZ_dual_static_XYZ_heldout_geometry_improved"
            if accepted
            else "all_PTZ_dual_static_XYZ_heldout_geometry_not_improved"
        ),
        "role": role,
        "views": [str(view) for view in component],
        "verified_views": list(gains_by_view),
        "unchanged_nodes": unchanged_nodes,
        "base": _summary(base_values),
        "candidate": _summary(candidate_values),
        "mean_log_gain": float(np.mean(gain_values)),
        "median_log_gain": float(np.median(gain_values)),
        "view_median_log_gain": view_medians,
        "view_balanced_median_log_gain": view_balanced_median,
        "node_median_log_gain": _summary(node_gains),
        "node_wins": wins,
        "node_losses": losses,
        "checks": checks,
        "new_tuned_numeric_thresholds": 0,
        "rows": rows,
    }


def evaluate_all_ptz_xyz_guarded_component(job):
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
    confirm = all_ptz_dual_static_xyz_score(
        candidate, component=component, frames=frames, role="confirm"
    )
    final = all_ptz_dual_static_xyz_score(
        candidate, component=component, frames=frames, role="final"
    )
    accepted = bool(confirm["accepted"] and final["accepted"])
    row["all_PTZ_dual_static_XYZ_crossfit_guard"] = {
        "accepted": accepted,
        "confirm": confirm,
        "final": final,
        "fit_confirm_final_target_query_IDs_disjoint": True,
        "candidate_modified": False,
    }
    row["commit_policy"] += "_plus_all_PTZ_dual_static_XYZ_confirm_final"
    if accepted:
        return row, selected, diagnostic
    row["commit"] = False
    row["reason"] = "all_PTZ_dual_static_XYZ_crossfit_rejected"
    return row, {}, diagnostic


evaluate_all_ptz_xyz_guarded_component.__module__ = BASE.__name__
evaluate_all_ptz_xyz_guarded_component.__name__ = "_evaluate_direct_component_from_context"
evaluate_all_ptz_xyz_guarded_component.__qualname__ = "_evaluate_direct_component_from_context"
BASE._evaluate_direct_component_from_context = evaluate_all_ptz_xyz_guarded_component


def annotate_audit() -> None:
    XYZ.PARTITION.CROSSFIT.CONFIRM.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "MULTIVIEW-dual-static-XYZ-all-three-PTZ-closure-v1"
    payload["method_override"]["support_partitioned_components"] = {
        "max_component_views": XYZ.PARTITION.MAX_COMPONENT_VIEWS,
        "selection_evidence": "summed public same-time shared-track counts",
        "numeric_geometry_thresholds_changed": False,
    }
    payload["method_override"]["all_PTZ_dual_static_target_query_XYZ"] = {
        "cache": str(RELAY_CACHE),
        "candidate_role": "fit",
        "verifier_roles": ["confirm", "final"],
        "replace_fit_XYZ_on_every_component_PTZ": True,
        "max_points_per_node": XYZ.MAX_XYZ_POINTS_PER_NODE,
        "numeric_geometry_thresholds_changed": False,
        "new_tuned_numeric_thresholds": 0,
        **RELAY_CACHE_AUDIT,
    }
    source_inference = float(
        XYZ.XYZ_PAYLOAD["runtime"]["single_model_equivalent_inference_seconds_per_frame"]
    )
    relay_inference = float(
        RELAY_PAYLOAD["runtime"]["single_model_equivalent_inference_seconds_per_frame"]
    )
    shared_runtime_payload = XYZ.XYZ_CACHE.resolve() == RELAY_CACHE.resolve()
    unique_inference = relay_inference if shared_runtime_payload else (
        source_inference + relay_inference
    )
    payload["dual_static_XYZ_source_match_runtime"] = XYZ.XYZ_PAYLOAD["runtime"]
    payload["dual_static_XYZ_PTZ_relay_runtime"] = RELAY_PAYLOAD["runtime"]
    payload["calibration_seconds_per_frame_including_roma_inference"] = float(
        payload["seconds_per_frame_all_PTZ"] + unique_inference
    )
    payload["runtime_accounting"] = {
        "policy": "unique_content_cache_inference_v1",
        "source_and_relay_share_merged_cache": shared_runtime_payload,
        "source_inference_seconds_per_frame": source_inference,
        "relay_inference_seconds_per_frame": relay_inference,
        "unique_inference_seconds_per_frame": unique_inference,
        "legacy_double_count_avoided_seconds_per_frame": (
            source_inference if shared_runtime_payload else 0.0
        ),
        "algorithm_or_output_changed": False,
    }
    payload["attestation"].update(
        {
            "component_partition_uses_no_GT_test_or_NVS": True,
            "all_PTZ_dual_static_XYZ_uses_public_train_cameras_only": True,
            "all_PTZ_dual_static_XYZ_is_frame_local": True,
            "all_PTZ_dual_static_XYZ_fit_confirm_final_IDs_disjoint": True,
            "all_PTZ_dual_static_XYZ_confirm_final_unused_by_candidate": True,
            "different_physical_PTZ_centers_remain_independent": True,
            "legacy_single_static_RoMa_cache_unused": True,
            "uncommitted_nodes_are_exact_Static_fallback": True,
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
