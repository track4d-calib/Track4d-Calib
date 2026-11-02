#!/usr/bin/env python3
"""Estimate shared reference centers in the reconstructed coordinate scale."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


BOUNDED_WRAPPER = Path(__file__).resolve().with_name(
    "bounded_global_shared_center_closure.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BOUNDED = _load_module(BOUNDED_WRAPPER, "bounded_unbounded_parent")
BASE = BOUNDED.BASE


def unbounded_center(
    anchor: np.ndarray, proposal: np.ndarray, radius: float
) -> tuple[np.ndarray, bool]:
    base = np.asarray(anchor, dtype=np.float64).reshape(3)
    value = np.asarray(proposal, dtype=np.float64).reshape(3)
    limit = float(radius)
    if not np.isfinite(base).all() or not np.isfinite(value).all():
        raise ValueError("center inputs must be finite")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("diagnostic Static center spread must be finite and positive")
    return value.copy(), False


BOUNDED.bounded_center = unbounded_center


def annotate_audit() -> None:
    BOUNDED.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "UNBOUNDED-unbounded-global-shared-center-all-three-PTZ-closure-v1"
    previous = payload["method_override"].pop("bounded_global_shared_center")
    precompute = previous["precompute"]
    precompute["policy"] = "fit-only-30-frame-unbounded-shared-center-v1"
    for row in precompute["views"].values():
        row["trust_radius_applied"] = False
    payload["method_override"]["unbounded_global_shared_center"] = {
        "single_change_from_BOUNDED": "do not clip the fit-only shared-center proposal",
        "source": previous["source"],
        "one_center_per_physical_PTZ": True,
        "different_PTZ_centers_independent": True,
        "candidate_rotation_BA_changed": False,
        "verifier_changed": False,
        "new_tuned_numeric_thresholds": 0,
        "precompute": precompute,
    }
    payload["attestation"].pop(
        "BOUNDED_single_change_is_fit_only_global_shared_center", None
    )
    payload["attestation"].update(
        {
            "UNBOUNDED_single_change_from_BOUNDED_is_center_clipping_disabled": True,
            "global_center_uses_no_confirm_or_final_IDs": True,
            "global_center_uses_no_GT_test_or_NVS": True,
        }
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    BOUNDED.prepare_global_centers()
    BOUNDED.GLOBAL_CENTER_AUDIT["policy"] = "fit-only-30-frame-unbounded-shared-center-v1"
    exit_code = BASE.main()
    if exit_code == 0:
        annotate_audit()
    raise SystemExit(exit_code)
