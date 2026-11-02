#!/usr/bin/env python3
"""Select one PTZ anchor and two public statics from distinct physical azimuths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image


ANCHOR_BUILDER = Path(__file__).resolve().with_name(
    "build_second_roma_component_anchor_cache.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STATIC_VIEWS = tuple(
    "00 02 03 05 06 08 09 11 12 14 15 17 18 20 21 23 "
    "24 26 27 29 30 32 33 35 36 38 39 41 42 44 45 47".split()
)
_DECLARED_RIG = bool(os.environ.get("TRACK4D_CALIB_RIG_CONTRACT"))
if _DECLARED_RIG:
    from relay_contract import STATIC_VIEWS


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def physical_azimuth(view: str) -> int:
    return int(view) // 3


def camera_center(camera) -> np.ndarray:
    return -np.asarray(camera.R, dtype=np.float64).T @ np.asarray(
        camera.T, dtype=np.float64
    )


def parse_tokens(value: str) -> list[str]:
    return [token for token in value.replace(",", " ").split() if token]


def main() -> int:
    ANCHOR = load_module(ANCHOR_BUILDER, "pair_selection_anchor")
    CameraLoader = ANCHOR.CameraLoader
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--base-pose-root", type=Path, required=True)
    parser.add_argument("--static-pnp-diagnostics", type=Path, required=True)
    parser.add_argument("--component", type=parse_tokens, required=True)
    parser.add_argument("--frames", type=parse_tokens, required=True)
    parser.add_argument("--romav2-root", type=Path, required=True)
    parser.add_argument("--romav2-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    component = [str(view) for view in args.component]
    frames = [int(frame) for frame in args.frames]
    runtime = args.runtime_scene.resolve()
    confidence, confidence_audit = ANCHOR.load_static_pnp_confidence(
        args.static_pnp_diagnostics, frames=frames, target_views=component
    )
    anchor = ANCHOR.choose_anchor(
        component=component, frames=frames, confidence=confidence
    )
    base = {
        frame: CameraLoader(
            args.base_pose_root / f"f{frame:06d}" / "cameras_pred"
        ).load_camera(anchor)
        for frame in frames
    }
    static_cameras = CameraLoader(runtime / "cameras/static").load_cameras(
        STATIC_VIEWS
    )
    nearest = ANCHOR.nearest_static_views(
        anchor_cameras=base, static_cameras=static_cameras, count=8
    )
    with Image.open(runtime / "images" / anchor / f"{args.selection_frame:06d}.png") as image:
        width, height = image.size
    queries = ANCHOR.query_grid(height=height, width=width, stride=8)
    fit_ids = None
    if _DECLARED_RIG:
        fit_ids = ANCHOR.split_frame_local_track_ids(
            frame=int(args.selection_frame), point_ids=np.arange(len(queries), dtype=np.int64)
        )["fit"]
    started = time.perf_counter()
    model, _device = ANCHOR._load_roma_model(
        source_root=args.romav2_root,
        weights=args.romav2_weights,
        device=args.device,
        setting="fast",
        force_bidirectional=True,
    )
    model_load_seconds = time.perf_counter() - started
    rows = []
    matches = {}
    inference_seconds = 0.0
    for static_view in nearest:
        match = ANCHOR.match_queries(
            model=model,
            static_image=runtime / "images" / (anchor if _DECLARED_RIG else static_view) / f"{args.selection_frame:06d}.png",
            target_image=runtime / "images" / (static_view if _DECLARED_RIG else anchor) / f"{args.selection_frame:06d}.png",
            queries=queries,
            height=height,
            width=width,
            minimum_overlap=0.2,
            maximum_cycle_error_px=1.5,
        )
        overlap = np.minimum(match["forward_overlap"], match["backward_overlap"])
        matches[static_view] = match
        selected = (np.isin(match["point_ids"], fit_ids) if _DECLARED_RIG
                    else np.ones(len(match["point_ids"]), dtype=bool))
        overlap = overlap[selected]
        cycle = np.asarray(match["cycle_error_px"])[selected]
        inference_seconds += float(match["inference_seconds"])
        rows.append(
            {
                "static_view": static_view,
                "physical_azimuth": None if _DECLARED_RIG else physical_azimuth(static_view),
                "accepted_count": int(np.count_nonzero(selected)),
                "overlap_median": float(np.median(overlap)) if len(overlap) else None,
                "cycle_error_median_px": float(np.median(cycle)) if len(cycle) else None,
            }
        )
    eligible = [row for row in rows if row["accepted_count"] >= 12]
    if len(eligible) < 2:
        raise RuntimeError("fewer than two public static candidates have 12 matches")
    score = lambda row: (-float(row["overlap_median"]), -int(row["accepted_count"]), str(row["static_view"]))
    first = min([row for row in eligible if row["static_view"] in nearest[:4]], key=score)
    selection_audit = None
    if _DECLARED_RIG:
        source = Path(__file__).resolve().parents[3] / "scripts/static_pair_selection.py"
        selection = load_module(source, "calibration_fit_static_pair_selection")
        second, selection_audit = selection.choose_second_static(
            first=first, eligible=eligible, matches=matches, queries=queries,
            support_cameras=static_cameras, fit_ids=fit_ids,
        )
    else:
        second_pool = [
            row for row in eligible
            if row["static_view"] != first["static_view"]
            and row["physical_azimuth"] != first["physical_azimuth"]
        ]
        if not second_pool:
            raise RuntimeError("no second static camera from a distinct physical azimuth")
        second = min(second_pool, key=score)
    c0 = camera_center(static_cameras[first["static_view"]])
    c1 = camera_center(static_cameras[second["static_view"]])
    payload = {
        "protocol": "PAIR_SELECTION-public-conditioned-dual-static-pair-selection-v1",
        "status": "complete_public_selection",
        "component": component,
        "anchor_target": anchor,
        "selected_static_views": [first["static_view"], second["static_view"]],
        "selected_static_physical_azimuths": [first["physical_azimuth"], second["physical_azimuth"]],
        "selected_center_baseline": float(np.linalg.norm(c0 - c1)),
        "candidate_rows": rows,
        "fit_geometry_selection": selection_audit,
        "confidence_audit": confidence_audit,
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
        },
        "config": {
            "nearest_candidates": 8,
            "first_selection_pool": 4,
            "minimum_overlap": 0.2,
            "maximum_cycle_error_px": 1.5,
            "different_physical_azimuth_required": not _DECLARED_RIG,
            "nondegenerate_fit_triangulation_required": _DECLARED_RIG,
            "selection_frame": args.selection_frame,
        },
        "attestation": {
            "GT_used": False,
            "NVS_used": False,
            "test_RGB_or_cameras_used": False,
            "reference_pose_used": False,
        },
    }
    write_json(args.output, payload)
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
