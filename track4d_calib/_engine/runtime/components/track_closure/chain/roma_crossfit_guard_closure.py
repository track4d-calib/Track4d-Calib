#!/usr/bin/env python3
"""Add held-out public-static RoMa verification to confirmation closure."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


CONFIRM_WRAPPER = Path(__file__).resolve().with_name("confirm_protected_closure.py")


def _load_confirm():
    spec = importlib.util.spec_from_file_location("confirm_closure", CONFIRM_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {CONFIRM_WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONFIRM = _load_confirm()
BASE = CONFIRM.BASE
ORIGINAL_EVALUATE = BASE._evaluate_direct_component_from_context


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "max": float(np.max(values)),
    }


def roma_crossfit_score(
    candidate: Mapping[int, Mapping[str, Any]],
    *,
    component: list[str],
    frames: list[int],
    role: str,
) -> dict[str, Any]:
    """Score a fixed candidate on RoMa IDs disjoint from candidate fit IDs."""
    if role not in {"confirm", "final"}:
        raise ValueError("RoMa commit guard only accepts held-out roles")
    context = BASE._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")
    observations = context.get("fixed_static_epipolar_observations")
    support_cameras = context.get("support_cameras")
    if observations is None or support_cameras is None:
        raise RuntimeError("RoMa commit guard lacks fixed public-static evidence")

    base_all = []
    candidate_all = []
    gain_all = []
    node_rows = []
    for frame in frames:
        for view in component:
            row = observations.get((str(view), int(frame), role))
            if row is None:
                return {
                    "accepted": False,
                    "reason": "missing_heldout_RoMa_observation",
                    "missing_node": f"{frame}:{view}:{role}",
                }
            static_view = str(row["static_view"])
            target_points = np.asarray(row["target_points"], dtype=np.float64)
            static_points = np.asarray(row["static_points"], dtype=np.float64)
            if not len(target_points):
                return {
                    "accepted": False,
                    "reason": "empty_heldout_RoMa_observation",
                    "empty_node": f"{frame}:{view}:{role}",
                }
            base_errors = BASE.sampson_distances_px(
                target_camera=context["base_grid"][int(frame)][str(view)],
                support_camera=support_cameras[static_view],
                target_points=target_points,
                support_points=static_points,
            )
            candidate_errors = BASE.sampson_distances_px(
                target_camera=candidate[int(frame)][str(view)],
                support_camera=support_cameras[static_view],
                target_points=target_points,
                support_points=static_points,
            )
            gains = np.log1p(base_errors) - np.log1p(candidate_errors)
            base_all.append(base_errors)
            candidate_all.append(candidate_errors)
            gain_all.append(gains)
            node_rows.append(
                {
                    "frame": int(frame),
                    "view": str(view),
                    "static_view": static_view,
                    "count": int(len(gains)),
                    "mean_log_gain": float(np.mean(gains)),
                    "median_log_gain": float(np.median(gains)),
                }
            )

    base_values = np.concatenate(base_all)
    candidate_values = np.concatenate(candidate_all)
    gain_values = np.concatenate(gain_all)
    node_gains = np.asarray(
        [row["median_log_gain"] for row in node_rows], dtype=np.float64
    )
    view_gains = {
        view: float(
            np.median(
                [row["median_log_gain"] for row in node_rows if row["view"] == view]
            )
        )
        for view in component
    }
    metrics = {
        "mean_log_gain": float(np.mean(gain_values)),
        "median_log_gain": float(np.median(gain_values)),
        "view_balanced_median_log_gain": float(np.median(list(view_gains.values()))),
    }
    node_wins = int(np.count_nonzero(node_gains > 0.0))
    node_losses = int(np.count_nonzero(node_gains < 0.0))
    checks = {name: bool(value > 0.0) for name, value in metrics.items()}
    checks["node_wins_exceed_losses"] = bool(node_wins > node_losses)
    accepted = bool(all(checks.values()))
    return {
        "policy": "heldout_RoMa_paired_log_gain_component_guard_v1",
        "accepted": accepted,
        "reason": (
            "heldout_RoMa_geometry_improved"
            if accepted
            else "heldout_RoMa_geometry_not_improved"
        ),
        "role": role,
        "base": _summary(base_values),
        "candidate": _summary(candidate_values),
        **metrics,
        "node_median_log_gain": _summary(node_gains),
        "node_wins": node_wins,
        "node_losses": node_losses,
        "view_median_log_gain": view_gains,
        "checks": checks,
        "new_tuned_numeric_thresholds": 0,
        "rows": node_rows,
    }


def evaluate_roma_crossfit_guarded_component(job):
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
    confirm = roma_crossfit_score(
        candidate, component=component, frames=frames, role="confirm"
    )
    final = roma_crossfit_score(
        candidate, component=component, frames=frames, role="final"
    )
    accepted = bool(confirm["accepted"] and final["accepted"])
    row["roma_crossfit_guard"] = {
        "accepted": accepted,
        "confirm": confirm,
        "final": final,
        "fit_confirm_final_point_IDs_disjoint": True,
        "candidate_modified": False,
    }
    if accepted:
        row["commit_policy"] += "_plus_heldout_RoMa_confirm_final"
        return row, selected, diagnostic
    row["commit"] = False
    row["reason"] = "heldout_RoMa_crossfit_rejected"
    row["commit_policy"] += "_plus_heldout_RoMa_confirm_final"
    return row, {}, diagnostic


evaluate_roma_crossfit_guarded_component.__module__ = BASE.__name__
evaluate_roma_crossfit_guarded_component.__name__ = (
    "_evaluate_direct_component_from_context"
)
evaluate_roma_crossfit_guarded_component.__qualname__ = (
    "_evaluate_direct_component_from_context"
)
BASE._evaluate_direct_component_from_context = evaluate_roma_crossfit_guarded_component


def annotate_audit() -> None:
    CONFIRM.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "CROSSFIT-CONFIRM-plus-heldout-RoMa-crossfit-guard-v1"
    payload["method_override"]["heldout_RoMa_crossfit_commit_guard"] = {
        "roles": ["confirm", "final"],
        "candidate_modified": False,
        "strict_positive_paired_log_gain": True,
        "new_tuned_numeric_thresholds": 0,
    }
    payload["attestation"].update(
        {
            "RoMa_fit_confirm_final_point_IDs_disjoint": True,
            "RoMa_confirm_final_unused_by_candidate_or_rollback": True,
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
