#!/usr/bin/env python3
"""Run track_closure with scene-local observations and isolated weak components."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

from relay_contract import validate_relay_payload


VERSION_ROOT = Path(__file__).resolve().parent
CLOSURE_WRAPPER = VERSION_ROOT / "chain/complete_dual_static_verifier.py"

_ORIGINAL_COMPONENT_EVALUATOR = None
_INSUFFICIENT_RELAY_KEYS: set[tuple[str, int, str]] = set()


def _num_frames() -> int:
    value = int(os.environ.get("TRACK4D_CALIB_NUM_FRAMES", "30"))
    if value <= 0 or value % 5:
        raise ValueError("TRACK4D_CALIB_NUM_FRAMES must be a positive multiple of five")
    return value


def _adapt_runtime_schema(runtime: dict[str, Any]) -> str:
    """Expose the per-frame view of a full-sequence relay cache."""

    legacy_key = "single_model_equivalent_seconds_per_frame"
    expected_key = "single_model_equivalent_inference_seconds_per_frame"
    if expected_key in runtime:
        return "already_current"
    if legacy_key in runtime:
        runtime[expected_key] = float(runtime[legacy_key])
        return "legacy_key"
    rows = list(runtime.get("single120_segment_runtime_rows", []))
    if not rows:
        return "missing"
    frame_counts = []
    summed_seconds = 0.0
    for row in rows:
        per_frame = float(row[expected_key])
        total = float(row["summed_inference_seconds"])
        if not per_frame > 0.0 or not total >= 0.0:
            raise ValueError("invalid segment runtime row")
        inferred_frames = int(round(total / per_frame))
        if abs(total - per_frame * inferred_frames) > 1e-5:
            raise ValueError("segment runtime row has inconsistent totals")
        frame_counts.append(inferred_frames)
        summed_seconds += total
    if sum(frame_counts) != _num_frames():
        raise ValueError("segment runtime rows do not cover the configured frames")
    per_frame = summed_seconds / float(_num_frames())
    runtime["summed_inference_seconds"] = summed_seconds
    runtime[legacy_key] = per_frame
    runtime[expected_key] = per_frame
    runtime["model_load_excluded"] = all(
        bool(row.get("model_load_excluded")) for row in rows
    )
    runtime["single120_segment_frame_counts"] = frame_counts
    return "single120_segment_rows"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_scene_relay(
    multiview: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    bands = tuple(tuple(map(str, band)) for band in multiview.BANDS)
    roles = tuple(map(str, multiview.ROLES))
    return validate_relay_payload(
        payload,
        bands=bands,
        frames=tuple(range(_num_frames())),
        roles=roles,
    )


def evaluate_component_with_relay_support_guard(job):
    if _ORIGINAL_COMPONENT_EVALUATOR is None:
        raise RuntimeError("component evaluator was not installed")
    component_index, spatial_index, raw_frames, raw_component = job
    component = [str(view) for view in raw_component]
    frames = [int(frame) for frame in raw_frames]
    blocked = sorted(
        key
        for key in _INSUFFICIENT_RELAY_KEYS
        if key[0] in component and key[1] in frames
    )
    if not blocked:
        return _ORIGINAL_COMPONENT_EVALUATOR(job)
    return (
        {
            "component_index": int(component_index),
            "spatial_component_index": int(spatial_index),
            "window": frames,
            "component": component,
            "commit": False,
            "reason": "insufficient_relay_support_exact_Static_fallback",
            "relay_support_guard": {
                "accepted": False,
                "minimum_points": 12,
                "blocked_node_roles": [list(key) for key in blocked],
                "scope": "whole_5frame_3PTZ_component",
                "candidate_evaluation_skipped": True,
                "numeric_thresholds_changed": False,
            },
        },
        {},
        {},
    )


def install_scene_relay(closure: Any, relay_cache: Path) -> dict[str, Any]:
    global _ORIGINAL_COMPONENT_EVALUATOR, _INSUFFICIENT_RELAY_KEYS
    with relay_cache.open("rb") as handle:
        payload = pickle.load(handle)
    runtime = payload.setdefault("runtime", {})
    runtime_schema_adaptation = _adapt_runtime_schema(runtime)
    observations = payload["observations"]
    verifier = closure.VERIFIER
    unbounded = verifier.UNBOUNDED
    bounded = unbounded.BOUNDED
    initializer = bounded.INITIALIZER
    multiview = initializer.MULTIVIEW
    xyz = multiview.XYZ
    multiview.RELAY_CACHE = relay_cache
    multiview.RELAY_PAYLOAD = payload
    multiview.RELAY_OBSERVATIONS = observations
    # Preserve the selected source identity through runtime revalidation.
    multiview.RELAY_CACHE_AUDIT = _validate_scene_relay(multiview, payload)
    _INSUFFICIENT_RELAY_KEYS = {
        (str(view), int(frame), str(role))
        for view, frame, role in multiview.RELAY_CACHE_AUDIT["below_minimum_keys"]
    }
    xyz.XYZ_PAYLOAD = payload
    xyz.XYZ_OBSERVATIONS = observations
    initializer.XYZ_INITIALIZATION_CACHE.clear()
    if _ORIGINAL_COMPONENT_EVALUATOR is None:
        _ORIGINAL_COMPONENT_EVALUATOR = closure.BASE._evaluate_direct_component_from_context
    evaluate_component_with_relay_support_guard.__module__ = closure.BASE.__name__
    evaluate_component_with_relay_support_guard.__name__ = (
        "_evaluate_direct_component_from_context"
    )
    evaluate_component_with_relay_support_guard.__qualname__ = (
        "_evaluate_direct_component_from_context"
    )
    closure.BASE._evaluate_direct_component_from_context = (
        evaluate_component_with_relay_support_guard
    )
    return {
        "observation_count": len(observations),
        "relay_cache_audit": multiview.RELAY_CACHE_AUDIT,
        "old_scene_cache_references_replaced": True,
        "runtime_key_schema_adapted": runtime_schema_adaptation != "already_current",
        "runtime_key_schema_adaptation_source": runtime_schema_adaptation,
    }


def annotate_audit(closure: Any, relay_audit: dict[str, Any], scene: str) -> None:
    closure.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "track_closure-track-driven-component-local-fallback-closure-v1"
    payload["method_version"] = "track_closure"
    payload["method_override"]["scene_local_transfer"] = {
        "single_change_from_CLOSURE": "scene and sealed scene-local inputs only",
        "candidate_parameterization_changed": False,
        "solver_changed": False,
        "numeric_thresholds_changed": False,
        "center_policy_changed": False,
        "confirm_rollback_changed": False,
        "dual_static_verifier_changed": False,
        "legal_StaticPnP_null_confidence_is_zero_evidence": True,
        "low_relay_support_forces_exact_component_fallback": True,
        "low_relay_support_is_classified_before_candidate_evaluation": True,
        "preflight_runtime_share_relay_validator": True,
        "legacy_observation_caches_used": False,
        **relay_audit,
    }
    payload["attestation"].update(
        {
            "scene": scene,
            "runtime_GT_used": False,
            "runtime_test_or_NVS_used": False,
            "runtime_reference_pose_used": False,
            "all_method_observations_scene_local": True,
        }
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    relay_cache = Path(os.environ["TRACK4D_CALIB_RELAY_CACHE"]).expanduser().absolute()
    scene = os.environ["TRACK4D_CALIB_SCENE"]
    if not relay_cache.is_file() or "/" in scene or ".." in scene:
        raise RuntimeError((relay_cache, scene))
    closure = _load_module(CLOSURE_WRAPPER, f"closure_scene_transfer_{scene}")
    relay_audit = install_scene_relay(closure, relay_cache)
    nullsafe = _load_module(
        VERSION_ROOT / "chain/select_band_static_pair_nullsafe.py",
        f"pair_fallback_nullsafe_confidence_{scene}",
    )
    closure.VERIFIER.UNBOUNDED.BOUNDED.BASE.load_static_pnp_confidence = (
        nullsafe.load_static_pnp_confidence_positive_nullsafe
    )
    closure.VERIFIER.UNBOUNDED.BOUNDED.prepare_global_centers()
    if "--resume-annotation" in sys.argv:
        sys.argv.remove("--resume-annotation")
        annotate_audit(closure, relay_audit, scene)
        return 0
    exit_code = closure.BASE.main()
    if exit_code == 0:
        annotate_audit(closure, relay_audit, scene)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
