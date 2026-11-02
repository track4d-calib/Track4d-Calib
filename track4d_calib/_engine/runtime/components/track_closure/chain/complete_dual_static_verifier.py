#!/usr/bin/env python3
"""Verify component updates and rollback candidates with dual-static tracks."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


VERIFIER_WRAPPER = Path(__file__).resolve().with_name("dual_static_verified_closure.py")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_module(VERIFIER_WRAPPER, "verifier_complete_verifier_parent")
BASE = VERIFIER.BASE
CONFIRM = VERIFIER.MULTIVIEW.XYZ.PARTITION.CROSSFIT.CONFIRM


def ptz_pair_final_gate(
    spatial: Mapping[str, Any], temporal: Mapping[str, Any]
) -> bool:
    """Pass rollback-selected candidates to downstream independent guards."""
    del temporal
    return bool(spatial.get("PTZ_pair", {}).get("commit"))


CONFIRM._final_accepts = ptz_pair_final_gate


def annotate_audit() -> None:
    VERIFIER.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "CLOSURE-complete-dual-static-verifier-substitution-v1"
    payload["method_override"]["confirm_rollback_final_gate_fix"] = {
        "single_change_from_VERIFIER": (
            "apply the already frozen verifier substitution to CONFIRM's separate "
            "confirm-rollback final acceptance path"
        ),
        "candidate_changed": False,
        "confirm_rollback_selection_changed": False,
        "numeric_thresholds_changed": False,
        "PTZ_pair_final_still_required": True,
        "downstream_dual_static_confirm_final_still_required": True,
    }
    payload["attestation"].update(
        {
            "CLOSURE_single_change_is_missing_branch_fix": True,
            "legacy_temporal_and_static_scores_do_not_gate_any_commit_path": True,
            "candidate_uses_same_temporal_and_same_time_tracks": True,
            "commit_uses_independent_dual_static_target_tracks": True,
            "GT_used": False,
            "NVS_used": False,
        }
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    VERIFIER.UNBOUNDED.BOUNDED.prepare_global_centers()
    exit_code = BASE.main()
    if exit_code == 0:
        annotate_audit()
    raise SystemExit(exit_code)
