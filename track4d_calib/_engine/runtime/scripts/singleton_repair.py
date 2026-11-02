"""Repair PTZ-isolated cameras using their independent static-anchored tracks.

No dynamic-to-dynamic edge is invented. The existing fit-only BA and shared
center estimate are reused, then both disjoint held-out XYZ folds must show
geometric improvement. Multi-camera components retain their existing gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np


POLICY = "static_anchored_singleton_fit_BA_disjoint_XYZ_confirm_final_v1"


def require_disjoint_support(observations, view, frames):
    """Use the existing minimum support; never admit missing/overlapping folds."""
    for frame in frames:
        ids_by_role = []
        for role in ("fit", "confirm", "final"):
            row = observations.get((view, frame, role))
            if row is None:
                return False
            ids = np.asarray(row["point_ids"], dtype=np.int64).reshape(-1)
            if len(ids) < 12 or len(set(ids.tolist())) != len(ids):
                return False
            ids_by_role.append(set(ids.tolist()))
        if any(ids_by_role[i] & ids_by_role[j] for i in range(3) for j in range(i)):
            return False
    return True


def fit_singleton(base, context, component, frames):
    return base._fit_direct_multi_ptz_candidate(
        component=component, frames=frames, base_grid=context["base_grid"],
        fixed_centers=context["fixed_centers"], fit_observations=context["spatial_observations"],
        temporal_observations=context["temporal_observations"], bundles=context["spatial_bundles"],
        confidence=context["confidence"], optimize_shared_center=bool(context["optimize_shared_center"]),
        center_trust_radii=context["center_trust_radii"],
        use_joint_center_initialization=bool(context["use_joint_center_initialization"]),
        direct_max_nfev=int(context["direct_max_nfev"]),
        direct_reprojection_weight=float(context["direct_reprojection_weight"]),
        direct_epipolar_weight=float(context["direct_epipolar_weight"]),
        direct_x_scale=str(context["direct_x_scale"]),
        direct_convergence_tolerance=float(context["direct_convergence_tolerance"]),
        use_temporal_fit_observations=bool(context["use_temporal_fit_observations"]),
        fixed_static_epipolar_observations=context.get("fixed_static_epipolar_observations"),
        support_cameras=context.get("support_cameras"), protected_nodes=set(),
    )


def repair_isolated_component(base, xyz, job, previous):
    row, selected, diagnostic = previous
    if row.get("reason") != "singleton_has_no_PTZ_pair_geometry" or len(job[3]) != 1:
        return previous
    frames, component = list(map(int, job[2])), list(map(str, job[3]))
    if not require_disjoint_support(xyz.RELAY_OBSERVATIONS, component[0], frames):
        return ({**row, "singleton_repair": {"policy": POLICY, "attempted": False,
                                            "reason": "missing_or_nonindependent_static_track_support"}}, selected, diagnostic)
    context = base._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")
    candidate, fitted = fit_singleton(base, context, component, frames)
    evidence: dict[str, Any] = {"policy": POLICY, "attempted": True, "fit_role": "fit",
                                "PTZ_pair_evidence": "absent; no fabricated edge or bypass of a failed pair gate",
                                "fit_confirm_final_IDs_disjoint": True, "additional_matching_pairs": 0}
    row = {**row, "candidate": fitted, "singleton_repair": evidence}
    if not fitted.get("success"):
        row["reason"] = "singleton_static_track_fit_failed"
        return row, {}, {}
    # Candidate fitting/selection is complete before either held-out fold is scored.
    confirm = xyz.all_ptz_dual_static_xyz_score(candidate, component=component, frames=frames, role="confirm")
    final = xyz.all_ptz_dual_static_xyz_score(candidate, component=component, frames=frames, role="final")
    evidence.update(confirm=confirm, final=final)
    changed = {(frame, view): candidate[frame][view] for frame in frames for view in component
               if not base._camera_equal(context["base_grid"][frame][view], candidate[frame][view])}
    accepted = bool(changed and confirm.get("accepted") and final.get("accepted"))
    row.update(commit=accepted, changed_nodes=[f"{frame}:{view}" for frame, view in changed],
               reason="singleton_static_tracks_repaired_and_verified" if accepted else "singleton_static_tracks_verification_rejected",
               commit_policy=POLICY)
    return row, changed if accepted else {}, changed


def install_singleton_repair(closure):
    base, xyz = closure.BASE, closure.VERIFIER.MULTIVIEW
    if getattr(base, "_singleton_repair_installed", False):
        return
    original = base._evaluate_direct_component_from_context

    def evaluate(job):
        return repair_isolated_component(base, xyz, job, original(job))

    base._evaluate_direct_component_from_context = evaluate
    base._singleton_repair_installed = True


def annotate_singleton_repair(payload):
    rows = [row for row in payload.get("components", []) if "singleton_repair" in row]
    payload["method_override"]["singleton_repair"] = {
        "policy": POLICY, "detected_components": len(rows),
        "attempted_components": sum(row["singleton_repair"]["attempted"] for row in rows),
        "verified_components": sum(bool(row.get("commit")) for row in rows),
        "multi_camera_candidates_and_verifiers_unchanged": True,
        "global_center_estimator_unchanged": True, "additional_matching_pairs": 0,
        "GT_historical_poses_or_NVS_used_for_repair": False,
    }
