"""Enforce the declared motion bound on retained as well as optimized nodes.

Only out-of-bound nodes are repaired. References and disjoint observations
come from the current run. A failed repair aborts publication, never restores
an out-of-bound pose under a successful seal. No dataset-specific policy.
"""

from typing import Any
import json
from pathlib import Path

import numpy as np


POLICY = "all_nodes_center_bound_with_feasible_confirm_selection_and_fit_only_repair_v2"
TOLERANCE = 1e-8  # Floating-point serialization tolerance, in SfM units.
SELECTION_FEASIBILITY_KEY = "calibration_confirm_center_feasible"


def center_offset(camera, reference):
    value = np.asarray(reference, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("invalid current-run reference center")
    center = np.asarray(camera.camera_center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("invalid camera center")
    return float(np.linalg.norm(center - value))


def validate_radius(radius):
    if radius is None:
        return None
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("center radius must be finite and nonnegative")
    return radius


def install_center_feasible_selection(closure, radius):
    """Do not let confirm rollback leave the declared feasible pose set.

    The existing fit candidate, confirm objective, and final geometry gates
    are unchanged. Feasibility is determined only by the declared motion bound
    and current-run fit-derived centers, never by held-out residual thresholds.
    """
    radius = validate_radius(radius)
    selector = closure.CONFIRM
    if getattr(selector, "_calibration_center_feasible_selection_installed", False):
        return
    original_scores, original_objective = selector._scores, selector.confirm_objective
    center_module = closure.VERIFIER.UNBOUNDED.BOUNDED

    def scores(candidate, *, component, frames, role):
        spatial, temporal = original_scores(candidate, component=component, frames=frames, role=role)
        if role == "confirm":
            spatial = dict(spatial)
            spatial[SELECTION_FEASIBILITY_KEY] = all(
                center_offset(candidate[int(f)][str(v)], center_module.GLOBAL_CENTERS[str(v)])
                <= radius + TOLERANCE for f in frames for v in component)
        return spatial, temporal

    def objective(spatial, temporal):
        if spatial.get(SELECTION_FEASIBILITY_KEY) is not True:
            return float("inf")
        return original_objective(spatial, temporal)

    selector._scores = scores
    selector.confirm_objective = objective
    selector._calibration_center_feasible_selection_installed = True


def feasible_center(camera, reference, radius):
    radius = validate_radius(radius)
    offset = center_offset(camera, reference)
    reference = np.asarray(reference, dtype=float)
    if radius == 0:
        return reference.copy()
    if offset <= radius:
        return np.asarray(camera.camera_center).copy()
    return reference + (np.asarray(camera.camera_center) - reference) * (radius / offset)


def audit_grid(grid, references, radius, frames, dynamic_views):
    """Inspect every declared node; missing/invalid data are hard errors."""
    radius = validate_radius(radius)
    frames, views = list(frames), list(dynamic_views)
    if not frames or not views or len(set(frames)) != len(frames) or len(set(views)) != len(views):
        raise ValueError("frame/view rosters must be nonempty and unique")
    if radius is None:
        for frame in frames:
            for view in views:
                # Free translation still requires every serialized pose to be finite.
                center_offset(grid[frame][view], np.zeros(3))
        return {"policy": "complete_finite_pose_grid_without_center_motion_bound",
                "radius": None, "bound_applied": False, "checked_nodes": len(frames)*len(views),
                "maximum_offset": None, "violations": [], "passed": True}
    if set(views) != set(references):
        raise ValueError("reference center roster mismatch")
    rows = []
    maximum = 0.0
    for frame in frames:
        for view in views:
            camera = grid[frame][view]  # Missing cameras must not be skipped.
            offset = center_offset(camera, references[view])
            maximum = max(maximum, offset)
            if offset > radius + TOLERANCE:
                rows.append({"frame": int(frame), "view": str(view), "offset": offset})
    return {"policy": POLICY, "radius": radius, "units": "current_run_static_SfM_coordinate_units",
        "tolerance": TOLERANCE, "checked_nodes": len(frames)*len(views),
        "maximum_offset": maximum, "violations": rows, "passed": not rows}


def pooled_roles(spatial, temporal, view, frame):
    """Reuse the existing static-triangulated tracks and their disjoint folds."""
    roles = {role: {"point_ids": [], "xyz": [], "image_points": []}
             for role in ("fit", "confirm", "final")}
    seen = {}
    seen_groups = {}
    for family, source in (("spatial", spatial), ("temporal", temporal)):
        for role, output in roles.items():
            row = source.get((view, frame, role))
            if row is None:
                raise ValueError("missing current-run observation fold")
            ids = np.asarray(row["point_ids"], dtype=np.int64).reshape(-1)
            xyz = np.asarray(row["xyz"], dtype=float)
            pixels = np.asarray(row["image_points"], dtype=float)
            if xyz.shape != (len(ids), 3) or pixels.shape != (len(ids), 2):
                raise ValueError("observation array shapes disagree")
            if not np.isfinite(xyz).all() or not np.isfinite(pixels).all():
                raise ValueError("nonfinite observation; cannot silently filter")
            if len(set(ids.tolist())) != len(ids):
                raise ValueError("duplicate physical track IDs")
            groups = np.asarray(row.get("validation_group_ids", ids), dtype=np.int64).reshape(-1)
            if len(groups) != len(ids):
                raise ValueError("validation group count mismatch")
            for ident in groups:
                key = (family, int(ident))
                if key in seen_groups and seen_groups[key] != role:
                    raise ValueError("fit/confirm/final validation group overlap")
                seen_groups[key] = role
            for ident in ids:
                key = (family, int(ident))
                if key in seen:
                    raise ValueError("fit/confirm/final physical track overlap")
                seen[key] = role
            output["point_ids"].extend((family, int(ident)) for ident in ids)
            output["xyz"].extend(xyz)
            output["image_points"].extend(pixels)
    for row in roles.values():
        row["xyz"] = np.asarray(row["xyz"], dtype=float).reshape(-1, 3)
        row["image_points"] = np.asarray(row["image_points"], dtype=float).reshape(-1, 2)
    return roles


def projection_errors(camera, observations):
    import cv2

    xyz = np.asarray(observations["xyz"])
    points = (np.asarray(camera.R) @ xyz.T).T + np.asarray(camera.T).reshape(3)
    if np.any(points[:, 2] <= 1e-8) or not np.isfinite(points).all():
        return None
    rvec, _ = cv2.Rodrigues(np.asarray(camera.R, dtype=float))
    pixels, _ = cv2.projectPoints(xyz, rvec, np.asarray(camera.T, dtype=float),
                                 np.asarray(camera.K, dtype=float), np.asarray(camera.D, dtype=float))
    pixels = pixels.reshape(-1, 2)
    if not np.isfinite(pixels).all():
        return None
    return np.linalg.norm(pixels - observations["image_points"], axis=1)


def verify_repair(original, candidate, roles):
    """Both held-out folds must improve over the actual retained pose."""
    scores = {}
    for role in ("confirm", "final"):
        row = roles[role]
        if len(row["point_ids"]) < 12:
            scores[role] = {"accepted": False, "count": len(row["point_ids"]),
                           "reason": "insufficient_support_or_invalid_depth"}
            continue
        before, after = projection_errors(original, row), projection_errors(candidate, row)
        if before is None or after is None:
            scores[role] = {"accepted": False, "count": len(row["point_ids"]),
                            "reason": "insufficient_support_or_invalid_depth"}
            continue
        gain = np.log1p(before) - np.log1p(after)
        scores[role] = {"accepted": bool(np.mean(gain) > 0 and np.median(gain) > 0),
            "count": len(gain), "mean_log_gain": float(np.mean(gain)),
            "median_log_gain": float(np.median(gain)),
            "before_median_px": float(np.median(before)), "after_median_px": float(np.median(after)),
            "before_p90_px": float(np.percentile(before, 90)), "after_p90_px": float(np.percentile(after, 90))}
    return {"accepted": all(row["accepted"] for row in scores.values()), "folds": scores,
            "reference": "actual retained pose, not an artificially projected baseline",
            "selection_or_fit_uses_heldout": False}


def fit_rotation(camera, reference, radius, fit, *, frame):
    """Use the repository's existing robust fixed-center rotation solver."""
    from track4d_calib.track_pnp_repair import solve_focus_track_rotation

    center = feasible_center(camera, reference, radius)
    count = len(fit["point_ids"])
    result = solve_focus_track_rotation(
        point_bank={"point_ids": np.arange(count), "xyz": fit["xyz"]},
        tracks_frame=fit["image_points"].reshape(1, count, 2),
        visibility_frame=np.ones((1, count), dtype=bool), view_ids=[camera.view_id],
        focus_view=camera.view_id, camera_template=camera, fixed_center=center,
        reprojection_error_px=4.0, min_inliers=12, max_evaluations=100,
        direction_ransac_prefilter=True, iterations_count=512, confidence=.999,
        rng_seed=int(frame))
    return result.get("camera") if result.get("success") else None


def repair_node(camera, reference, radius, roles, *, frame, fitter=fit_rotation):
    radius = validate_radius(radius)
    before = center_offset(camera, reference)
    report: dict[str, Any] = {"offset_before": before, "radius": radius,
                              "attempted": False, "accepted": False}
    if before <= radius + TOLERANCE:
        return camera, {**report, "accepted": True, "reason": "already_feasible_unchanged"}
    if set(roles) != {"fit", "confirm", "final"}:
        raise ValueError("repair requires all three disjoint roles")
    if min(len(roles[r]["point_ids"]) for r in roles) < 12:
        return None, {**report, "reason": "insufficient_disjoint_track_support",
                      "counts": {r: len(row["point_ids"]) for r, row in roles.items()}}
    candidate = fitter(camera, reference, radius, roles["fit"], frame=frame)
    report["attempted"] = True
    if candidate is None:
        return None, {**report, "reason": "constrained_rotation_fit_failed"}
    after = center_offset(candidate, reference)
    if after > radius + TOLERANCE:
        raise RuntimeError("repair solver violated the center bound")
    if not np.array_equal(candidate.K, camera.K) or not np.array_equal(candidate.D, camera.D):
        raise RuntimeError("repair changed intrinsics")
    verified = verify_repair(camera, candidate, roles)
    report.update(accepted=verified["accepted"], offset_after=after, verification=verified,
                  reason="track_verified_center_repair" if verified["accepted"] else "heldout_geometry_rejected")
    return candidate if verified["accepted"] else None, report


def enforce_component(closure, radius, job, previous):
    """A rejected candidate may retain a base pose only if that pose is feasible."""
    row, selected, diagnostic = previous
    base = closure.BASE
    context = base._DIRECT_COMPONENT_CONTEXT
    if context is None:
        raise RuntimeError("direct component worker context is unset")
    references = closure.VERIFIER.UNBOUNDED.BOUNDED.GLOBAL_CENTERS
    frames, views = list(map(int, job[2])), list(map(str, job[3]))
    candidate = {f: dict(context["base_grid"][f]) for f in frames}
    for (f, v), camera in selected.items():
        candidate[f][v] = camera
    repaired, evidence, failed = {}, {}, []
    for f in frames:
        for v in views:
            camera = candidate[f][v]
            if center_offset(camera, references[v]) <= radius + TOLERANCE:
                continue
            roles = pooled_roles(context["spatial_observations"],
                                 context["temporal_observations"], v, f)
            trial, report = repair_node(camera, references[v], radius, roles, frame=f)
            evidence[f"{f}:{v}"] = report
            if trial is None:
                failed.append(f"{f}:{v}")
            else:
                repaired[f, v] = trial
                candidate[f][v] = trial
    if failed:
        # No out-of-bound retained node is emitted, even if other nodes repaired.
        raise RuntimeError(json.dumps({"center_contract_failed": failed, "radius": radius,
                                       "repairs": evidence, "publication_aborted": True}))
    if not repaired:
        return previous
    # A repaired node can change dynamic-to-dynamic geometry. Recheck the
    # complete component, not just the repaired camera's static projections.
    checks = {}
    for role in ("confirm", "final"):
        if len(views) > 1:
            spatial, temporal = closure.CONFIRM._scores(candidate, component=views, frames=frames, role=role)
            checks[f"PTZ_pair_{role}"] = bool(closure.CONFIRM._final_accepts(spatial, temporal))
        # Low relay support is a missing-evidence precheck, not a failed
        # geometric gate. Only that path may use the alternate static tracks.
        if row.get("reason") != "insufficient_relay_support_exact_Static_fallback":
            score = closure.VERIFIER.MULTIVIEW.all_ptz_dual_static_xyz_score(
                candidate, component=views, frames=frames, role=role)
            checks[f"dual_static_{role}"] = bool(score.get("accepted"))
    if not all(checks.values()):
        raise RuntimeError(json.dumps({"center_repair_component_verification_failed": checks,
                                       "publication_aborted": True}))
    result = {**selected, **repaired}
    row = {**row, "commit": True, "reason": "track_verified_center_bound_repair",
           "candidate_before_center_repair": row.get("candidate"),
           "candidate": {"success": True, "reason": "fit_only_center_repair_independently_verified"},
           "changed_nodes": [f"{f}:{v}" for f, v in sorted(result)],
           "center_contract_repair": {"policy": POLICY, "nodes": evidence,
                                      "component_checks": checks, "additional_matching_pairs": 0}}
    return row, result, {**diagnostic, **repaired}


def install_final_center_contract(closure, radius):
    """Install AFTER the relay-support guard so skipped jobs cannot bypass it."""
    radius = validate_radius(radius)
    if radius is None:
        return  # Independent geometric selection and final gates remain active.
    install_center_feasible_selection(closure, radius)
    base = closure.BASE
    if getattr(base, "_final_center_contract_installed", False):
        return
    original = base._evaluate_direct_component_from_context

    def evaluate(job):
        return enforce_component(closure, radius, job, original(job))

    # The existing fork executor pickles the callback by the base module name.
    evaluate.__module__ = base.__name__
    evaluate.__name__ = "_evaluate_direct_component_from_context"
    evaluate.__qualname__ = "_evaluate_direct_component_from_context"
    base._evaluate_direct_component_from_context = evaluate
    base._final_center_contract_installed = True


def audit_pose_root(pose_root, references, radius, frames, dynamic_views):
    """Read serialized poses, not candidate metadata or only the NVS prefix."""
    from track4d_calib.cameras import CameraLoader

    frames, views = list(frames), list(dynamic_views)
    grid = {f: CameraLoader(Path(pose_root) / f"f{f:06d}" / "cameras_pred").load_cameras(views)
            for f in frames}
    return audit_grid(grid, references, radius, frames, views)


def require_final_center_contract(pose_root, rig, frames):
    """Independent publication gate; missing/mismatched evidence fails closed."""
    pose_root = Path(pose_root).resolve()
    payload = json.loads((pose_root.parent / "PUBLIC_GEOMETRY_AUDIT.json").read_text())
    if payload.get("rig_contract") != rig:
        raise RuntimeError("output rig differs from the declared motion configuration")
    references = payload["optimized_global_centers"]
    report = audit_pose_root(pose_root, references, rig["center_motion_radius"],
                             frames, rig["dynamic_views"])
    if not report["passed"]:
        raise RuntimeError(json.dumps({"final_center_contract_failed": report,
                                       "publication_aborted": True}))
    return report
