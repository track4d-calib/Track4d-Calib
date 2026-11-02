"""Select node updates using independent track-family verification."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


SCORE_KEY = "dual_static_xyz_confirm_selection"
OBJECTIVE = "equal_family_PTZ_pair_ratio_and_view_balanced_dual_static_mean_log_ratio"


def confirm_xyz_summary(module, candidate, *, component, frames) -> dict[str, Any]:
    """Summarize paired confirm errors, giving each physical camera equal weight."""
    context = module.BASE._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")
    rows = {}
    for view in component:
        view = str(view)
        base_errors, candidate_errors = [], []
        for frame in frames:
            frame = int(frame)
            observation = module.RELAY_OBSERVATIONS.get((view, frame, "confirm"))
            if observation is None:
                return {"valid": False, "reason": "missing_confirm_observation"}
            base = np.asarray(module.reprojection_errors(
                context["base_grid"][frame][view], observation), dtype=np.float64)
            trial = np.asarray(module.reprojection_errors(
                candidate[frame][view], observation), dtype=np.float64)
            if (base.ndim != 1 or base.size == 0 or base.shape != trial.shape
                    or not np.isfinite(base).all() or not np.isfinite(trial).all()
                    or np.any(base < 0) or np.any(trial < 0)):
                return {"valid": False, "reason": "invalid_confirm_projection"}
            base_errors.append(base)
            candidate_errors.append(trial)
        if not base_errors:
            return {"valid": False, "reason": "empty_confirm_frames"}
        base = np.concatenate(base_errors)
        trial = np.concatenate(candidate_errors)
        rows[view] = {
            "count": int(base.size),
            "base_mean_log_error": float(np.log1p(base).mean()),
            "candidate_mean_log_error": float(np.log1p(trial).mean()),
        }
    return {"valid": bool(rows), "role": "confirm", "views": rows}


def confirm_objective(spatial: Mapping[str, Any], temporal: Mapping[str, Any]) -> float:
    """Equal-family normalized error; unavailable active evidence fails closed."""
    del temporal
    try:
        pair = spatial["PTZ_pair"]
        base = float(pair["base"]["clipped_mean_px"])
        trial = float(pair["candidate"]["clipped_mean_px"])
        xyz = spatial[SCORE_KEY]
        if not xyz["valid"] or not xyz["views"]:
            return math.inf
        ratios = []
        for row in xyz["views"].values():
            a = float(row["base_mean_log_error"])
            b = float(row["candidate_mean_log_error"])
            if row["count"] <= 0 or not all(math.isfinite(x) and x >= 0 for x in (a, b)):
                return math.inf
            ratios.append(b / max(a, 1e-12))
        if not all(math.isfinite(x) and x >= 0 for x in (base, trial)):
            return math.inf
        return 0.5 * (trial / max(base, 1e-12) + sum(ratios) / len(ratios))
    except (KeyError, TypeError, ValueError):
        return math.inf


def install_confirm_selection(closure) -> None:
    """Replace selection only; preserve the candidate solver and final gates."""
    selector = closure.CONFIRM
    if getattr(selector, "_calibration_active_confirm_installed", False):
        return
    xyz = closure.VERIFIER.MULTIVIEW
    original_scores = selector._scores

    def scores(candidate, *, component, frames, role):
        spatial, temporal = original_scores(
            candidate, component=component, frames=frames, role=role)
        if role == "confirm":
            spatial = dict(spatial)
            spatial[SCORE_KEY] = confirm_xyz_summary(
                xyz, candidate, component=component, frames=frames)
        return spatial, temporal

    selector._scores = scores
    selector.confirm_objective = confirm_objective
    selector._calibration_active_confirm_installed = True


def annotate_selection_audit(payload: dict[str, Any]) -> None:
    """Record the selection change without relabeling inherited diagnostics."""
    payload["method_override"]["calibration_active_confirm_selection"] = {
        "objective": OBJECTIVE,
        "selection_roles": ["confirm"],
        "selection_families": ["PTZ_pair", "dual_static_XYZ"],
        "legacy_static_temporal_are_diagnostics_only": True,
        "fit_optimizer_unchanged": True,
        "final_verifier_and_thresholds_unchanged": True,
        "additional_matching_pairs": 0,
    }
    for row in payload.get("components", []):
        if "confirm_protection" in row:
            row["confirm_protection"]["objective"] = OBJECTIVE
