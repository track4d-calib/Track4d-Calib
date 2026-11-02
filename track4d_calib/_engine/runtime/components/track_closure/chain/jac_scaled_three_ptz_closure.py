#!/usr/bin/env python3
"""Apply Jacobian scaling to the bundle-adjustment trust region."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


MULTIVIEW_WRAPPER = Path(__file__).resolve().with_name(
    "dual_static_xyz_three_ptz_closure.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MULTIVIEW = _load_module(MULTIVIEW_WRAPPER, "multiview_closure")
BASE = MULTIVIEW.BASE


def annotate_audit() -> None:
    MULTIVIEW.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "JACOBIAN-jac-scaled-dual-static-XYZ-all-three-PTZ-closure-v1"
    payload["method_override"]["direct_x_scale"] = {
        "value": "jac",
        "objective_changed": False,
        "max_nfev_changed": False,
        "numeric_geometry_thresholds_changed": False,
    }
    payload["attestation"]["JACOBIAN_single_change_is_solver_parameter_scaling"] = True
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    exit_code = BASE.main()
    if exit_code == 0:
        annotate_audit()
    raise SystemExit(exit_code)
