#!/usr/bin/env python3
"""Select static support pairs with explicit missing-score handling."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ORIGINAL_SELECTOR = Path(__file__).resolve().with_name("select_band_static_pair.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_static_pnp_confidence_nullsafe(
    path: Path,
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
) -> tuple[dict[tuple[int, str], float], dict[str, Any]]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("gt_used") is not False or payload.get("nvs_used") is not False:
        raise ValueError("Static+PnP diagnostics are not sealed GT/NVS-free")
    expected = {
        (int(frame), str(view)) for frame in frames for view in target_views
    }
    scores: dict[tuple[int, str], float] = {}
    invalid_nodes: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        key = (int(row["frame"]), str(row["view"]))
        if key not in expected:
            continue
        if key in scores:
            raise ValueError(f"duplicate Static+PnP diagnostic node: {key}")
        solver = row.get("solver") or {}
        ratio_value = solver.get("inlier_ratio")
        median_value = solver.get("reprojection_median_px")
        try:
            ratio = float(ratio_value)
            median = float(median_value)
        except (TypeError, ValueError):
            ratio = float("nan")
            median = float("nan")
        valid = (
            bool(solver.get("success", False))
            and np.isfinite(ratio)
            and np.isfinite(median)
            and ratio > 0.0
            and median > 0.0
        )
        scores[key] = ratio / median if valid else 0.0
        if not valid:
            invalid_nodes.append(
                {
                    "frame": key[0],
                    "view": key[1],
                    "policy": "zero_confidence_for_legal_StaticPnP_fallback",
                }
            )
    if set(scores) != expected:
        missing = sorted(expected - set(scores))
        raise ValueError(f"Static+PnP diagnostics do not cover requested nodes: {missing}")
    return scores, {
        "path": str(source),
        "sha256": sha256(source),
        "node_count": len(scores),
        "valid_node_count": len(scores) - len(invalid_nodes),
        "zero_confidence_fallback_count": len(invalid_nodes),
        "zero_confidence_fallback_nodes": invalid_nodes,
        "weight": "inlier_ratio_divided_by_reprojection_median_px_else_zero",
        "new_tunable_numeric_hyperparameters": 0,
    }


def load_static_pnp_confidence_positive_nullsafe(
    path: Path,
    *,
    frames: Sequence[int],
    target_views: Sequence[str],
) -> tuple[dict[tuple[int, str], float], dict[str, Any]]:
    """Adapt zero evidence to an API that requires strictly positive weights."""
    scores, audit = load_static_pnp_confidence_nullsafe(
        path, frames=frames, target_views=target_views
    )
    numerical_zero = float(np.finfo(np.float64).tiny)
    positive = {
        key: (value if value > 0.0 else numerical_zero)
        for key, value in scores.items()
    }
    audit = dict(audit)
    audit.update(
        {
            "weight": (
                "inlier_ratio_divided_by_reprojection_median_px_else_"
                "machine_tiny_positive_zero"
            ),
            "machine_tiny_positive_zero": numerical_zero,
        }
    )
    return positive, audit


def main() -> int:
    selector = load_module(ORIGINAL_SELECTOR, "pair_fallback_original_band_selector")
    original_loader = selector.load_module

    def patched_loader(path: Path, name: str):
        module = original_loader(path, name)
        if Path(path).resolve() == Path(selector.ANCHOR_BUILDER).resolve():
            module.load_static_pnp_confidence = load_static_pnp_confidence_nullsafe
        return module

    selector.load_module = patched_loader
    return int(selector.main())


if __name__ == "__main__":
    raise SystemExit(main())
