"""Select a fit-supported rotation reference for local bundle adjustment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


POLICY = "rebase_saturated_local_rotation_chart_at_current_fit_endpoint_v1"


def rebase_saturated_fit_endpoints(
    kwargs: Mapping[str, Any], fit_residuals: Mapping[Any, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move only the local rotation reference; never mutate the input pose grid."""
    maximum = np.deg2rad(float(kwargs["config"].max_rotation_correction_deg))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("rotation chart radius must be finite and positive")
    call = dict(kwargs)
    base = dict(kwargs["base_cameras"])
    initial = dict(kwargs.get("initial_rotation_residuals") or {})
    changed = []
    for node in kwargs["nodes"]:
        if node not in fit_residuals:
            continue
        residual = np.asarray(fit_residuals[node], dtype=np.float64).reshape(3)
        if not np.isfinite(residual).all():
            raise ValueError("fit rotation endpoint must be finite")
        magnitude = float(np.linalg.norm(residual))
        if magnitude < maximum:
            continue
        original = base[node]
        rotation = Rotation.from_rotvec(residual).as_matrix() @ original.R
        # Translation here only preserves the reference camera center. The
        # actual BA center still comes from the original fixed/bounded model.
        base[node] = replace(original, R=rotation, T=-rotation @ original.camera_center)
        initial[node] = np.zeros(3, dtype=np.float64)
        changed.append({"node": str(node), "fit_endpoint_correction_deg":
                        float(np.degrees(magnitude))})
    if changed:
        call["base_cameras"] = base
        call["initial_rotation_residuals"] = initial
    return call, {
        "policy": POLICY,
        "rebased_nodes": changed,
        "rebased_node_count": len(changed),
        "local_rotation_radius_deg": float(np.degrees(maximum)),
        "reference_evidence": "current_run_dual_static_XYZ_fit_only_endpoint",
        "rotation_prior_reference_changed_only_for_saturated_nodes": True,
        "center_reference_and_bound_unchanged": True,
        "confirm_final_or_GT_used_for_reference": False,
    }


def install_rotation_reference(closure) -> None:
    """Hook after fit initialization, before its residual is silently clipped."""
    endpoint = closure.VERIFIER.UNBOUNDED.BOUNDED.INITIALIZER
    if getattr(endpoint, "_rotation_reference_installed", False):
        return
    raw_solve = endpoint.RAW_SOLVE

    def solve(**kwargs):
        # This is the same cached fit endpoint used by the inherited wrapper,
        # not another fit, another seed, or a confirm-selected proposal.
        fit_residuals, _ = endpoint.xyz_endpoint_initialization(**kwargs)
        call, audit = rebase_saturated_fit_endpoints(kwargs, fit_residuals)
        result = raw_solve(**call)
        result["rotation_reference"] = audit
        return result

    endpoint.RAW_SOLVE = solve
    endpoint._rotation_reference_installed = True


def annotate_rotation_reference(payload: dict[str, Any]) -> None:
    payload["method_override"]["rotation_reference"] = {
        "policy": POLICY,
        "selection": "successful_fit_endpoint_outside_existing_local_rotation_chart",
        "candidate_reference_changed": True,
        "local_rotation_radius_and_center_bound_unchanged": True,
        "fit_observations_and_BA_budget_unchanged": True,
        "confirm_selection_and_final_verifier_unchanged": True,
        "additional_matching_pairs": 0,
    }
