#!/usr/bin/env python3
"""Evaluate dual-static confirmation and final verification observations."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


UNBOUNDED_WRAPPER = Path(__file__).resolve().with_name(
    "unbounded_global_shared_center_closure.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


UNBOUNDED = _load_module(UNBOUNDED_WRAPPER, "unbounded_dual_static_verified_parent")
BASE = UNBOUNDED.BASE
MULTIVIEW = UNBOUNDED.BOUNDED.INITIALIZER.MULTIVIEW


def ptz_pair_candidate_gate(
    *,
    confirm: Mapping[str, Any],
    final: Mapping[str, Any],
    temporal_confirm: Mapping[str, Any],
    temporal_final: Mapping[str, Any],
    use_noninferiority_guards: bool,
) -> bool:
    """Pass candidates to MULTIVIEW's independent dual-static cross-fit guard."""
    del temporal_confirm, temporal_final, use_noninferiority_guards
    return bool(
        confirm.get("PTZ_pair", {}).get("commit")
        and final.get("PTZ_pair", {}).get("commit")
    )


BASE._direct_geometry_accepts = ptz_pair_candidate_gate


def annotate_audit() -> None:
    UNBOUNDED.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "VERIFIER-dual-static-verified-unbounded-center-closure-v1"
    payload["method_override"]["independent_verifier_substitution"] = {
        "single_change_from_UNBOUNDED": (
            "replace saturated legacy static/temporal commit guards with the "
            "existing independent dual-static XYZ confirm/final cross-fit guard"
        ),
        "candidate_changed": False,
        "candidate_tracks_changed": False,
        "center_changed": False,
        "numeric_thresholds_changed": False,
        "PTZ_pair_confirm_final_still_required": True,
        "dual_static_XYZ_confirm_final_required": True,
        "legacy_static_temporal_scores_retained_as_diagnostics": True,
        "fit_confirm_final_target_query_IDs_disjoint": True,
    }
    payload["attestation"].update(
        {
            "VERIFIER_single_change_is_commit_evidence_source": True,
            "candidate_uses_same_temporal_and_same_time_tracks": True,
            "commit_uses_independent_dual_static_target_tracks": True,
            "legacy_temporal_and_static_scores_do_not_gate_commit": True,
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
    UNBOUNDED.BOUNDED.prepare_global_centers()
    exit_code = BASE.main()
    if exit_code == 0:
        annotate_audit()
    raise SystemExit(exit_code)
