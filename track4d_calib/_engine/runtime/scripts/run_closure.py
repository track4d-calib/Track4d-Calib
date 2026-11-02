#!/usr/bin/env python3
"""Adapt center motion and fit-supported rotation references for verified BA.

Zero center radius remains rotation-only. Positive radii add local center
residuals; None permits independent unbounded translation. Fit-derived rotation
endpoints outside the old PnP chart become the new local BA reference.
Confirm/final tracks never estimate those references or fit the candidate.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from rig_contract import ENVIRONMENT_KEY, center_solver_kwargs, load_rig_contract
    from confirm_selection import annotate_selection_audit, install_confirm_selection
    from rotation_reference import annotate_rotation_reference, install_rotation_reference
    from singleton_repair import annotate_singleton_repair, install_singleton_repair
    from final_center import audit_pose_root, install_final_center_contract
    from window_relay import CONFIG_ENV, refresh_low_support, validate_windowed_relay
except ModuleNotFoundError:
    from scripts.rig_contract import ENVIRONMENT_KEY, center_solver_kwargs, load_rig_contract
    from scripts.confirm_selection import annotate_selection_audit, install_confirm_selection
    from scripts.rotation_reference import annotate_rotation_reference, install_rotation_reference
    from scripts.singleton_repair import annotate_singleton_repair, install_singleton_repair
    from scripts.final_center import audit_pose_root, install_final_center_contract
    from scripts.window_relay import CONFIG_ENV, refresh_low_support, validate_windowed_relay


METHOD = Path(__file__).resolve().parents[1] / "components/track_closure"
_CENTER_MODULE: Any = None
_RADIUS = 0.0


def solve_free_candidate(module, **kwargs):
    """Fit per-node SE3 endpoints and refine with the common track BA objective."""
    from track4d_calib.track_pnp_repair import solve_focus_track_pnp

    cameras = dict(kwargs["base_cameras"])
    initialized = []
    for node in kwargs["nodes"]:
        row = kwargs["reprojection_observations"][node]  # Fit-only relay observations.
        count = len(row["xyz"])
        fitted = solve_focus_track_pnp(
            point_bank={"point_ids": np.arange(count), "xyz": row["xyz"]},
            tracks_frame=np.asarray(row["image_points"]).reshape(1, count, 2),
            visibility_frame=np.ones((1, count), dtype=bool), view_ids=[str(node[1])],
            focus_view=str(node[1]), camera_template=cameras[node],
            reprojection_error_px=4.0, min_inliers=12, iterations_count=1000,
            confidence=.999, rng_seed=0)
        if fitted["success"]:
            cameras[node] = fitted["camera"]
            initialized.append(str(node))
    call = dict(kwargs)
    call.update(base_cameras=cameras,
                fixed_centers={node: cameras[node].camera_center.copy() for node in kwargs["nodes"]},
                initial_rotation_residuals={node: np.zeros(3) for node in kwargs["nodes"]},
                initial_center_residuals=None)
    call.update(center_solver_kwargs(kwargs["nodes"], None))
    # The same raw BA, before the fixed-center endpoint wrappers. All independent
    # confirm/final gates remain installed on the enclosing component evaluator.
    result = module.INITIALIZER.RAW_SOLVE(**call)
    result["calibration_free_centers"] = {
        "radius": None, "translation_regularizer": False,
        "endpoint": "current_run_fit_only_full_SE3_PnP",
        "initialized_nodes": initialized, "independent_geometry_verification_required": True,
    }
    return result


def solve_bounded_candidate(**kwargs):
    """Inject only bounded center parameters; preserve the fit observations."""
    if _CENTER_MODULE is None:
        raise RuntimeError("center adapter was not configured")
    module = _CENTER_MODULE
    if _RADIUS is None:
        return solve_free_candidate(module, **kwargs)
    if _RADIUS == 0:
        return module.solve_with_global_shared_centers(**kwargs)
    call = dict(kwargs)
    call["fixed_centers"] = {
        (int(frame), str(view)): np.asarray(module.GLOBAL_CENTERS[str(view)]).copy()
        for frame, view in kwargs["nodes"]
    }
    call.update(center_solver_kwargs(kwargs["nodes"], _RADIUS))
    result = module.RAW_SOLVE(**call)
    offsets = [
        float(np.linalg.norm(camera.camera_center - call["fixed_centers"][node]))
        for node, camera in result["cameras"].items()
    ]
    maximum = max(offsets, default=0.0)
    if not np.isfinite(maximum) or maximum > _RADIUS + 1e-9:
        raise RuntimeError("candidate BA violated the declared center bound")
    result["calibration_center_constraint"] = {
        "radius": _RADIUS,
        "units": "current_run_static_SfM_coordinate_units",
        "reference": "current_run_fit_only_global_center_per_camera",
        "maximum_candidate_offset": maximum,
        "candidate_bound_passed": True,
        "requires_independent_confirm_and_final_before_commit": True,
    }
    return result


def install_center_adapter(closure, radius: float | None) -> None:
    global _CENTER_MODULE, _RADIUS
    _CENTER_MODULE = closure.VERIFIER.UNBOUNDED.BOUNDED
    _RADIUS = None if radius is None else float(radius)
    if _RADIUS == 0:
        return
    # The process executor forks the fully prepared closure context. Give the
    # callback the inherited callable's stable identity for worker pickling.
    base = closure.BASE
    solve_bounded_candidate.__module__ = base.__name__
    solve_bounded_candidate.__name__ = "solve_direct_rotation_component"
    solve_bounded_candidate.__qualname__ = "solve_direct_rotation_component"
    _CENTER_MODULE.XYZ.ORIGINAL_SOLVE = solve_bounded_candidate
    if _RADIUS is None:
        def prepare_free_centers():
            _CENTER_MODULE.GLOBAL_CENTERS = {}
            _CENTER_MODULE.GLOBAL_CENTER_AUDIT = {
                "policy": "independent_per_node_centers_no_shared_center_estimation", "views": {}}
            return _CENTER_MODULE.GLOBAL_CENTER_AUDIT
        _CENTER_MODULE.prepare_global_centers = prepare_free_centers


def main() -> int:
    rig = load_rig_contract(Path(os.environ[ENVIRONMENT_KEY]))
    sys.path.insert(0, str(METHOD))
    path = METHOD / "complete_closure.py"
    spec = importlib.util.spec_from_file_location("calibration_verified_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load transfer implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_load = module._load_module

    def load(path, name):
        loaded = original_load(path, name)
        if Path(path).resolve() == module.CLOSURE_WRAPPER.resolve():
            install_center_adapter(loaded, rig["center_motion_radius"])
            install_confirm_selection(loaded)
            if rig["center_motion_radius"] is not None:
                install_rotation_reference(loaded)
            install_singleton_repair(loaded)
        return loaded

    original_install_relay = module.install_scene_relay
    original_validate = module.validate_relay_payload
    module.validate_relay_payload = lambda payload, **kw: validate_windowed_relay(
        payload, original=original_validate, **kw)
    repair_audit = {}

    def install_relay(closure, relay_cache):
        result = original_install_relay(closure, relay_cache)
        config_path = os.environ.get(CONFIG_ENV)
        if config_path:
            center_module = closure.VERIFIER.UNBOUNDED.BOUNDED
            prepare = center_module.prepare_global_centers

            def prepare_and_refresh():
                output = Path(sys.argv[sys.argv.index("--output-root") + 1])
                repair_audit.update(refresh_low_support(
                    Path(config_path), relay_cache, output.with_name(output.name + "_relay_support_repair"),
                    bands=rig["relay_bands"], frames=range(int(os.environ["TRACK4D_CALIB_NUM_FRAMES"])),
                    validator=original_validate))
                if repair_audit["jobs"]:
                    result.update(original_install_relay(closure, Path(repair_audit["relay"])))
                    # Both solver aliases now consume the same repaired payload;
                    # keep the runtime identity in sync to avoid double counting.
                    center_module.INITIALIZER.MULTIVIEW.XYZ.XYZ_CACHE = Path(repair_audit["relay"])
                # Estimate centers from the same fit relay used by downstream
                # BA. Freezing them before refresh leaves repaired windows
                # constrained by the incomplete observation bank they replaced.
                centers = prepare()
                repair_audit["centers_estimated_after_refresh"] = rig["center_motion_radius"] is not None
                # Install last because reinstalling the relay replaces its guard.
                install_final_center_contract(closure, rig["center_motion_radius"])
                return centers

            center_module.prepare_global_centers = prepare_and_refresh
        else:
            # Standalone geometry diagnostics may have no GPU repair service.
            install_final_center_contract(closure, rig["center_motion_radius"])
        return result

    original_annotate = module.annotate_audit

    def annotate(*args):
        original_annotate(*args)
        output = Path(sys.argv[sys.argv.index("--output-root") + 1])
        audit_path = output / "PUBLIC_GEOMETRY_AUDIT.json"
        payload = json.loads(audit_path.read_text())
        payload["method_version"] = "TRACK4D_CALIB"
        payload["closure_loss"] = os.environ.get("TRACK4D_CALIB_CLOSURE_LOSS", "huber")
        payload["rig_contract"] = rig
        payload["window_relay_repair"] = repair_audit
        payload["method_override"]["scene_local_transfer"].update({
            "candidate_parameterization_changed": _RADIUS != 0,
            "center_policy_changed": True,
            "candidate_rotation_reference_policy_changed": True,
            "low_relay_support_forces_exact_component_fallback": False,
            "low_relay_support_fallback_requires_feasible_centers": True,
        })
        payload["method_override"]["calibration_bounded_node_centers"] = {
            "enabled": _RADIUS is not None and _RADIUS > 0,
            "radius": _RADIUS,
            "base_centers_are_current_run_fit_only": True,
            "existing_independent_verifiers_unchanged": True,
            "retained_node_center_repair_has_additional_disjoint_track_verification": True,
            "same_run_rejected_candidates_keep_exact_base_pose_only_if_feasible": True,
            "bound_applies_to_all_final_nodes": _RADIUS is not None,
            "confirm_rollback_must_be_center_feasible": True,
            "failed_center_repair_aborts_publication": True,
        }
        annotate_selection_audit(payload)
        if _RADIUS is not None:
            annotate_rotation_reference(payload)
        else:
            for key in ("unbounded_global_shared_center", "dual_static_XYZ_endpoint_initialization"):
                payload["method_override"].pop(key, None)
            payload["method_override"]["calibration_free_centers"] = {
                "per_node_translation_unbounded": True, "translation_regularizer": False,
                "shared_center_estimation": False, "fixed_center_Layer1_repair": False,
                "endpoint": "fit_only_full_SE3_PnP", "confirm_and_final_verifiers_unchanged": True}
            payload["attestation"]["UNBOUNDED_single_change_from_BOUNDED_is_center_clipping_disabled"] = False
            payload["attestation"]["INITIALIZER_single_change_is_candidate_initialization"] = False
        annotate_singleton_repair(payload)
        report = audit_pose_root(output / "pose_arm_D", payload["optimized_global_centers"],
                                 rig["center_motion_radius"],
                                 range(int(os.environ["TRACK4D_CALIB_NUM_FRAMES"])), rig["dynamic_views"])
        payload["final_center_contract"] = report
        if not report["passed"]:
            raise RuntimeError(json.dumps({"final_center_contract_failed": report}))
        temporary = audit_path.with_name(f".{audit_path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, audit_path)

    module._load_module = load
    module.install_scene_relay = install_relay
    module.annotate_audit = annotate
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
