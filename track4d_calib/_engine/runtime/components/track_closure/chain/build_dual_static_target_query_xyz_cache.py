#!/usr/bin/env python3
"""Triangulate frame-local XYZ from shared queries in calibrated static cameras."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "geometry"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader
from track4d_calib.same_time_multi_ptz_closure import split_frame_local_track_ids
from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    triangulate_same_time_multiview_points,
)


ANCHOR_BUILDER = Path(__file__).resolve().with_name(
    "build_second_roma_component_anchor_cache.py"
)
PROTOCOL = "dual-public-static-target-query-frame-local-XYZ-v1"


def _load_anchor_builder():
    spec = importlib.util.spec_from_file_location("anchor_builder", ANCHOR_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {ANCHOR_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANCHOR = _load_anchor_builder()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def matched_destination_by_id(match: Mapping[str, Any]) -> dict[int, np.ndarray]:
    ids = np.asarray(match["point_ids"], dtype=np.int64)
    points = np.asarray(match["target_points"], dtype=np.float64)
    if len(ids) != len(points) or len(np.unique(ids)) != len(ids):
        raise ValueError("RoMa query IDs and destination points must be one-to-one")
    return {int(point_id): point for point_id, point in zip(ids, points)}


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid target image dimensions: {path}")
    return int(height), int(width)


def supported_query_ids(destinations: Mapping[str, Any], minimum: int) -> set[int]:
    """Keep query identities observed by enough distinct static cameras."""
    counts = Counter(point for points in destinations.values() for point in points)
    return {int(point) for point, count in counts.items() if count >= minimum}


def triangulate_roles(
    *,
    frame: int,
    queries: np.ndarray,
    matches: Mapping[str, Mapping[str, Any]],
    support_cameras: Mapping[str, Any],
    config: SameTimePnPConfig,
) -> dict[str, dict[str, Any]]:
    destination = {
        str(view): matched_destination_by_id(match)
        for view, match in matches.items()
    }
    shared = supported_query_ids(destination, config.min_support_views)
    split = split_frame_local_track_ids(
        frame=int(frame), point_ids=np.arange(len(queries), dtype=np.int64)
    )
    result = {}
    for role, role_ids in split.items():
        selected = sorted(shared & set(np.asarray(role_ids, dtype=np.int64)))
        support_observations = {
            point_id: {
                view: destination[view][point_id] for view in sorted(destination)
                if point_id in destination[view]
            }
            for point_id in selected
        }
        target_observations = {
            point_id: np.asarray(queries[point_id], dtype=np.float64)
            for point_id in selected
        }
        triangulated = triangulate_same_time_multiview_points(
            support_cameras=support_cameras,
            support_observations=support_observations,
            target_observations=target_observations,
            config=config,
        )
        result[str(role)] = triangulated
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--target-view", required=True)
    parser.add_argument("--static-views", type=ANCHOR.parse_tokens, required=True)
    parser.add_argument("--frames", type=ANCHOR.parse_tokens, required=True)
    parser.add_argument("--romav2-root", type=Path, required=True)
    parser.add_argument("--romav2-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matching-quality", choices=("fast", "precise"), default="fast")
    parser.add_argument("--query-stride", type=int, default=8)
    parser.add_argument("--minimum-overlap", type=float, default=0.2)
    parser.add_argument("--maximum-cycle-error-px", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target = str(args.target_view)
    static_views = [str(view) for view in args.static_views]
    frames = [int(frame) for frame in args.frames]
    if len(static_views) < 2 or len(set(static_views)) != len(static_views):
        raise ValueError("at least two distinct public static views are required")
    runtime = args.runtime_scene.resolve()
    support_cameras = CameraLoader(runtime / "cameras/static").load_cameras(
        static_views
    )
    first_target_image = runtime / "images" / target / f"{frames[0]:06d}.png"
    height, width = image_dimensions(first_target_image)
    queries = ANCHOR.query_grid(height=height, width=width, stride=args.query_stride)
    config = SameTimePnPConfig()

    model_started = time.perf_counter()
    model, _device = ANCHOR._load_roma_model(
        source_root=args.romav2_root,
        weights=args.romav2_weights,
        device=str(args.device),
        setting=args.matching_quality,
        force_bidirectional=True,
    )
    model_load_seconds = time.perf_counter() - model_started
    observations = {}
    frame_rows = []
    inference_seconds = 0.0
    for frame in frames:
        matches = {}
        for static_view in static_views:
            match = ANCHOR.match_queries(
                model=model,
                static_image=runtime / "images" / target / f"{frame:06d}.png",
                target_image=runtime / "images" / static_view / f"{frame:06d}.png",
                queries=queries,
                height=height,
                width=width,
                minimum_overlap=float(args.minimum_overlap),
                maximum_cycle_error_px=float(args.maximum_cycle_error_px),
            )
            matches[static_view] = match
            inference_seconds += float(match["inference_seconds"])
        roles = triangulate_roles(
            frame=frame,
            queries=queries,
            matches=matches,
            support_cameras=support_cameras,
            config=config,
        )
        for role, row in roles.items():
            observations[(target, frame, role)] = row
        frame_rows.append(
            {
                "frame": frame,
                "accepted_count_by_static": {
                    view: int(len(matches[view]["point_ids"]))
                    for view in static_views
                },
                "shared_query_count": int(
                    len(
                        supported_query_ids(
                            {view: matches[view]["point_ids"] for view in static_views},
                            config.min_support_views,
                        )
                    )
                ),
                "triangulated_count_by_role": {
                    role: int(row["triangulated_count"])
                    for role, row in roles.items()
                },
            }
        )

    payload = {
        "protocol": PROTOCOL,
        "status": "complete_public_same_time_XYZ_cache",
        "target_view": target,
        "static_views": static_views,
        "frames": frames,
        "observations": observations,
        "config": {
            "matching_quality": args.matching_quality,
            "query_origin": "long_focal_target_image",
            "query_stride": int(args.query_stride),
            "query_count": int(len(queries)),
            "minimum_overlap": float(args.minimum_overlap),
            "maximum_cycle_error_px": float(args.maximum_cycle_error_px),
            "triangulation": vars(config),
            "foreground_XYZ_is_frame_local": True,
        },
        "frames_audit": frame_rows,
        "runtime": {
            "model_load_seconds": float(model_load_seconds),
            "inference_seconds": float(inference_seconds),
            "inference_seconds_per_frame": float(inference_seconds / len(frames)),
        },
        "lineage": {
            "entrypoint": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "runtime_manifest": {"path": str(runtime / "runtime_manifest.json"), "sha256": sha256(runtime / "runtime_manifest.json")},
            "roma_weights": {"path": str(args.romav2_weights.resolve()), "sha256": sha256(args.romav2_weights.resolve())},
        },
        "attestation": {
            "GT_used": False,
            "NVS_used": False,
            "test_RGB_or_cameras_used": False,
            "reference_pose_used": False,
            "public_static_cameras_fixed": True,
            "target_query_ID_shared_by_support_observations": True,
        },
    }
    write_pickle(args.output, payload)
    write_json(
        args.output.with_suffix(".json"),
        {key: value for key, value in payload.items() if key != "observations"},
    )
    print(
        json.dumps(
            {
                "target_view": target,
                "static_views": static_views,
                "triangulated_per_frame": {
                    role: ANCHOR.summary(
                        np.asarray([row["triangulated_count_by_role"][role] for row in frame_rows])
                    )
                    for role in ("fit", "confirm", "final")
                },
                "runtime": payload["runtime"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
