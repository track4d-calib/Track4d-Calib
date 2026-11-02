#!/usr/bin/env python3
"""Relay dual-static frame-local XYZ to same-band public PTZ views."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image


ANCHOR_BUILDER = Path(__file__).resolve().with_name(
    "build_second_roma_component_anchor_cache.py"
)
PROTOCOL = "dual-static-frame-local-XYZ-same-band-PTZ-relay-v1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANCHOR = load_module(ANCHOR_BUILDER, "anchor_builder")

from relay_contract import validate_relay_component


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(ANCHOR.jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def destination_by_id(match: Mapping[str, Any]) -> dict[int, np.ndarray]:
    ids = np.asarray(match["point_ids"], dtype=np.int64)
    points = np.asarray(match["target_points"], dtype=np.float64)
    if len(ids) != len(points) or len(np.unique(ids)) != len(ids):
        raise ValueError("relay query IDs are not one-to-one")
    return {int(point_id): point for point_id, point in zip(ids, points)}


def relay_role(
    *,
    source_row: Mapping[str, Any],
    destinations: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    ids = np.asarray(source_row["point_ids"], dtype=np.int64)
    xyz = np.asarray(source_row["object_points"], dtype=np.float64)
    source_points = np.asarray(source_row["image_points"], dtype=np.float64)
    if len(ids) != len(xyz) or len(ids) != len(source_points):
        raise ValueError("source XYZ arrays disagree")
    index = {int(point_id): position for position, point_id in enumerate(ids)}
    result = {
        "source": {
            "point_ids": ids,
            "object_points": xyz,
            "image_points": source_points,
        }
    }
    for view, destination in destinations.items():
        selected = sorted(set(index) & set(destination))
        positions = np.asarray([index[point_id] for point_id in selected], dtype=np.int64)
        result[str(view)] = {
            "point_ids": np.asarray(selected, dtype=np.int64),
            "object_points": xyz[positions],
            "image_points": np.asarray([destination[point_id] for point_id in selected], dtype=np.float64).reshape(-1, 2),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--component", type=ANCHOR.parse_tokens, required=True)
    parser.add_argument("--source-view", required=True)
    parser.add_argument("--frames", type=ANCHOR.parse_tokens, required=True)
    parser.add_argument("--source-xyz-cache", type=Path, required=True)
    parser.add_argument("--romav2-root", type=Path, required=True)
    parser.add_argument("--romav2-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-stride", type=int, default=8)
    parser.add_argument("--minimum-overlap", type=float, default=0.2)
    parser.add_argument("--maximum-cycle-error-px", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime = args.runtime_scene.resolve()
    component = [str(view) for view in args.component]
    source = str(args.source_view)
    frames = [int(frame) for frame in args.frames]
    validate_relay_component(component, source)
    targets = [view for view in component if view != source]
    with args.source_xyz_cache.open("rb") as handle:
        source_payload = pickle.load(handle)
    source_observations = source_payload["observations"]
    first_image = runtime / "images" / source / f"{frames[0]:06d}.png"
    with Image.open(first_image) as image:
        width, height = image.size
    queries = ANCHOR.query_grid(height=height, width=width, stride=args.query_stride)

    started = time.perf_counter()
    model, _device = ANCHOR._load_roma_model(
        source_root=args.romav2_root, weights=args.romav2_weights,
        device=str(args.device), setting="fast", force_bidirectional=True,
    )
    model_load_seconds = time.perf_counter() - started
    observations = {}
    frame_rows = []
    inference_seconds = 0.0
    for frame in frames:
        matches = {}
        for target in targets:
            match = ANCHOR.match_queries(
                model=model,
                static_image=runtime / "images" / source / f"{frame:06d}.png",
                target_image=runtime / "images" / target / f"{frame:06d}.png",
                queries=queries, height=height, width=width,
                minimum_overlap=float(args.minimum_overlap),
                maximum_cycle_error_px=float(args.maximum_cycle_error_px),
            )
            matches[target] = match
            inference_seconds += float(match["inference_seconds"])
        destinations = {view: destination_by_id(row) for view, row in matches.items()}
        role_counts = {}
        for role in ("fit", "confirm", "final"):
            relayed = relay_role(
                source_row=source_observations[(source, frame, role)],
                destinations=destinations,
            )
            source_row = relayed.pop("source")
            observations[(source, frame, role)] = source_row
            for view, row in relayed.items():
                observations[(view, frame, role)] = row
            role_counts[role] = {
                source: int(len(source_row["point_ids"])),
                **{view: int(len(relayed[view]["point_ids"])) for view in targets},
            }
        frame_rows.append({
            "frame": frame,
            "raw_match_count_by_target": {view: int(len(matches[view]["point_ids"])) for view in targets},
            "relayed_XYZ_count_by_role_and_view": role_counts,
        })
    payload = {
        "protocol": PROTOCOL, "status": "complete_public_same_time_PTZ_XYZ_relay",
        "component": component, "source_view": source, "target_views": targets,
        "frames": frames, "observations": observations,
        "config": {
            "query_origin": source, "query_stride": int(args.query_stride),
            "minimum_overlap": float(args.minimum_overlap),
            "maximum_cycle_error_px": float(args.maximum_cycle_error_px),
            "XYZ_is_frozen_from_two_public_static_cameras": True,
            "XYZ_is_frame_local": True,
        },
        "frames_audit": frame_rows,
        "runtime": {
            "model_load_seconds": float(model_load_seconds),
            "inference_seconds": float(inference_seconds),
            "inference_seconds_per_frame": float(inference_seconds / len(frames)),
        },
        "lineage": {
            "source_XYZ_cache": {"path": str(args.source_xyz_cache.resolve()), "sha256": ANCHOR.sha256(args.source_xyz_cache.resolve())},
            "roma_weights": {"path": str(args.romav2_weights.resolve()), "sha256": ANCHOR.sha256(args.romav2_weights.resolve())},
        },
        "attestation": {
            "GT_used": False, "NVS_used": False, "test_RGB_or_cameras_used": False,
            "reference_pose_used": False, "same_query_ID_used_across_band": True,
        },
    }
    write_pickle(args.output, payload)
    write_json(args.output.with_suffix(".json"), {key: value for key, value in payload.items() if key != "observations"})
    print(json.dumps({
        "component": component, "source_view": source,
        "runtime": payload["runtime"],
        "fit_relay_count": {
            view: ANCHOR.summary(np.asarray([row["relayed_XYZ_count_by_role_and_view"]["fit"][view] for row in frame_rows]))
            for view in component
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
