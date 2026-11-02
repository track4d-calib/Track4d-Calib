#!/usr/bin/env python3
"""Run fixed-center direct BA with confirm rollback and final verification."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


BASE_ENTRY = (
    Path(__file__).resolve().parents[1]
    / "geometry/scripts/geometry/export_same_time_multi_ptz_closure.py"
)


def _load_base():
    spec = importlib.util.spec_from_file_location("component_closure_base", BASE_ENTRY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BASE_ENTRY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
ORIGINAL_EVALUATE = BASE._evaluate_direct_component_from_context


def _relative_mean(score: Mapping[str, Any]) -> float | None:
    if "base" not in score or "candidate" not in score:
        return None
    base = float(score["base"]["clipped_mean_px"])
    candidate = float(score["candidate"]["clipped_mean_px"])
    if not math.isfinite(base) or not math.isfinite(candidate):
        return None
    return candidate / max(base, 1e-12)


def confirm_objective(
    spatial: Mapping[str, Any], temporal: Mapping[str, Any]
) -> float:
    """Equal-weight relative error over the three geometry families."""
    values = [
        value
        for value in (
            _relative_mean(spatial["PTZ_pair"]),
            _relative_mean(spatial["static"]),
            _relative_mean(temporal),
        )
        if value is not None
    ]
    return float(sum(values) / len(values)) if values else math.inf


def _scores(candidate, *, component, frames, role):
    context = BASE._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")
    spatial = BASE._spatial_errors(
        bundles=context["spatial_bundles"],
        observations=context["spatial_observations"],
        component=component,
        frames=frames,
        role=role,
        base_grid=context["base_grid"],
        candidate_grid=candidate,
    )
    temporal = BASE._temporal_score(
        observations=context["temporal_observations"],
        component=component,
        frames=frames,
        role=role,
        base_grid=context["base_grid"],
        candidate_grid=candidate,
    )
    return spatial, temporal


def _final_accepts(spatial: Mapping[str, Any], temporal: Mapping[str, Any]) -> bool:
    return bool(
        spatial.get("PTZ_pair", {}).get("commit")
        and spatial.get("static_noninferiority_guard", {}).get("guard")
        and temporal.get("noninferiority_guard", {}).get("guard")
    )


def evaluate_confirm_protected_component(job):
    row, selected, diagnostic = ORIGINAL_EVALUATE(job)
    if row.get("commit") or not row.get("candidate", {}).get("success") or not diagnostic:
        return row, selected, diagnostic

    context = BASE._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")
    frames = [int(frame) for frame in row["window"]]
    component = [str(view) for view in row["component"]]
    candidate = {frame: dict(context["base_grid"][frame]) for frame in frames}
    for (frame, view), camera in diagnostic.items():
        candidate[int(frame)][str(view)] = camera

    confirm, temporal_confirm = _scores(
        candidate, component=component, frames=frames, role="confirm"
    )
    objective_before = confirm_objective(confirm, temporal_confirm)
    objective = objective_before
    rolled_back = []
    for frame, view in sorted(diagnostic):
        trial = {current: dict(candidate[current]) for current in frames}
        trial[int(frame)][str(view)] = context["base_grid"][int(frame)][str(view)]
        trial_spatial, trial_temporal = _scores(
            trial, component=component, frames=frames, role="confirm"
        )
        trial_objective = confirm_objective(trial_spatial, trial_temporal)
        if trial_objective < objective - 1e-12:
            candidate = trial
            confirm = trial_spatial
            temporal_confirm = trial_temporal
            objective = trial_objective
            rolled_back.append((int(frame), str(view)))

    changed = [
        (frame, view)
        for frame in frames
        for view in component
        if not BASE._camera_equal(
            context["base_grid"][frame][view], candidate[frame][view]
        )
    ]
    if not changed:
        row.update(
            {
                "commit": False,
                "reason": "confirm_protection_rolled_back_all_nodes",
                "confirm": confirm,
                "temporal_confirm": temporal_confirm,
                "confirm_protection": {
                    "objective_before": objective_before,
                    "objective_after": objective,
                    "rolled_back_nodes": [f"{frame}:{view}" for frame, view in rolled_back],
                    "remaining_changed_nodes": [],
                },
            }
        )
        return row, {}, {}

    final, temporal_final = _scores(
        candidate, component=component, frames=frames, role="final"
    )
    commit = _final_accepts(final, temporal_final)
    row.update(
        {
            "changed_nodes": [f"{frame}:{view}" for frame, view in changed],
            "confirm": confirm,
            "final": final,
            "temporal_confirm": temporal_confirm,
            "temporal_final": temporal_final,
            "commit": commit,
            "reason": (
                "confirm_selected_and_final_geometry_accepted"
                if commit
                else "confirm_selected_but_final_geometry_rejected"
            ),
            "commit_policy": (
                "confirm_selects_node_rollback_then_final_PTZ_pair_strict_plus_"
                "static_and_temporal_noninferiority"
            ),
            "confirm_protection": {
                "objective": (
                    "sum of candidate/base clipped means for PTZ-pair, static, temporal"
                ),
                "objective_before": objective_before,
                "objective_after": objective,
                "rolled_back_nodes": [f"{frame}:{view}" for frame, view in rolled_back],
                "remaining_changed_nodes": [f"{frame}:{view}" for frame, view in changed],
                "confirm_used_for_selection_not_verification": True,
                "final_used_only_after_selection": True,
            },
        }
    )
    output = {(frame, view): candidate[frame][view] for frame, view in changed}
    return row, output if commit else {}, output


evaluate_confirm_protected_component.__module__ = BASE.__name__
evaluate_confirm_protected_component.__name__ = "_evaluate_direct_component_from_context"
evaluate_confirm_protected_component.__qualname__ = "_evaluate_direct_component_from_context"
BASE._evaluate_direct_component_from_context = evaluate_confirm_protected_component


def annotate_audit() -> None:
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "CONFIRM-global-fixed-center-direct-rotation-BA-v1"
    payload["method_override"] = {
        "scope": "rejected converged TRACK_CENTER candidates only",
        "fit_role": "unchanged TRACK_CENTER candidate optimization",
        "confirm_role": "deterministic node rollback selection",
        "final_role": "independent commit verification",
        "confirm_final_physical_IDs_disjoint": True,
        "thresholds_changed": False,
        "protect_high_confidence_half": False,
        "all_component_nodes_are_direct_BA_variables": True,
        "per_component_center_residual": False,
        "global_confidence_weighted_center_per_physical_PTZ": True,
    }
    payload["attestation"].update(
        {
            "confirm_used_for_selection_not_commit_verification": True,
            "final_independent_of_fit_and_confirm_selection_IDs": True,
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
