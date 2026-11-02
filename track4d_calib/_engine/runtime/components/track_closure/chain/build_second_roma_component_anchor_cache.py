#!/usr/bin/env python3
"""Build public-static RoMa anchors for one same-time PTZ component."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "geometry"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader, CameraParams
from track4d_calib.roma_query_relay import relay_bidirectional_query_points
from track4d_calib.same_time_multi_ptz_closure import split_frame_local_track_ids
from scripts.geometry.export_same_time_dynamic_track_closure import (
    load_static_pnp_confidence,
)
from scripts.geometry.export_same_time_multi_ptz_closure import STATIC_VIEWS
from scripts.geometry.export_static_pnp_inlier_seeded_track_closure import (
    _load_roma_model,
)


PROTOCOL = "roma-fast-bidirectional-component-anchor-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def parse_tokens(value: str) -> list[str]:
    return [token for token in value.replace(",", " ").split() if token]


def query_grid(*, height: int, width: int, stride: int) -> np.ndarray:
    step = int(stride)
    if step <= 0:
        raise ValueError("query stride must be positive")
    offset = step / 2.0
    x = np.arange(offset, float(width), step, dtype=np.float64)
    y = np.arange(offset, float(height), step, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def optical_axis(camera: CameraParams) -> np.ndarray:
    return np.asarray(camera.R, dtype=np.float64).T @ np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float64
    )


def choose_anchor(
    *,
    component: Sequence[str],
    frames: Sequence[int],
    confidence: Mapping[tuple[int, str], float],
) -> str:
    views = [str(view) for view in component]
    if not views:
        raise ValueError("component must be non-empty")
    scores = {
        view: float(
            np.median([float(confidence[(int(frame), view)]) for frame in frames])
        )
        for view in views
    }
    return min(views, key=lambda view: (-scores[view], view))


def nearest_static_views(
    *,
    anchor_cameras: Mapping[int, CameraParams],
    static_cameras: Mapping[str, CameraParams],
    count: int,
) -> list[str]:
    anchor_axes = [optical_axis(camera) for camera in anchor_cameras.values()]
    rows = []
    for view, camera in static_cameras.items():
        axis = optical_axis(camera)
        angles = [
            math.acos(float(np.clip(np.dot(anchor, axis), -1.0, 1.0)))
            for anchor in anchor_axes
        ]
        rows.append((float(np.median(angles)), str(view)))
    rows.sort()
    return [view for _angle, view in rows[: int(count)]]


def arrays_from_prediction(prediction: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for direction in ("AB", "BA"):
        for field in ("warp", "overlap"):
            value = prediction.get(f"{field}_{direction}")
            if value is None:
                raise RuntimeError(
                    f"RoMa prediction lacks {field}_{direction}; bidirectional mode is required"
                )
            array = value[0].detach().to(device="cpu").numpy()
            if field == "overlap" and array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            arrays[f"{field}_{direction}"] = np.asarray(array, dtype=np.float32)
    return arrays


def match_queries(
    *,
    model: Any,
    static_image: Path,
    target_image: Path,
    queries: np.ndarray,
    height: int,
    width: int,
    minimum_overlap: float,
    maximum_cycle_error_px: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    prediction = model.match(str(static_image), str(target_image))
    inference_seconds = time.perf_counter() - started
    arrays = arrays_from_prediction(prediction)
    relay = relay_bidirectional_query_points(
        source_points_xy=queries,
        warp_source_to_target=arrays["warp_AB"],
        warp_target_to_source=arrays["warp_BA"],
        overlap_source_to_target=arrays["overlap_AB"],
        overlap_target_to_source=arrays["overlap_BA"],
        source_image_size=(height, width),
        target_image_size=(height, width),
        minimum_overlap=float(minimum_overlap),
        maximum_cycle_error_px=float(maximum_cycle_error_px),
    )
    valid = np.asarray(relay["valid"], dtype=bool)
    point_ids = np.flatnonzero(valid).astype(np.int64)
    return {
        "point_ids": point_ids,
        "static_points": np.asarray(queries[valid], dtype=np.float64),
        "target_points": np.asarray(relay["target_points"][valid], dtype=np.float64),
        "forward_overlap": np.asarray(relay["forward_overlap"][valid], dtype=np.float64),
        "backward_overlap": np.asarray(relay["backward_overlap"][valid], dtype=np.float64),
        "cycle_error_px": np.asarray(relay["cycle_error_px"][valid], dtype=np.float64),
        "inference_seconds": float(inference_seconds),
        "diagnostics": relay["diagnostics"],
    }


def summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def role_rows(
    *, frame: int, match: Mapping[str, Any], query_count: int
) -> dict[str, dict[str, np.ndarray]]:
    split = split_frame_local_track_ids(
        frame=int(frame), point_ids=np.arange(int(query_count), dtype=np.int64)
    )
    accepted = np.asarray(match["point_ids"], dtype=np.int64)
    result = {}
    for role, role_ids in split.items():
        keep = np.isin(accepted, np.asarray(role_ids, dtype=np.int64))
        result[str(role)] = {
            "point_ids": accepted[keep],
            "static_points": np.asarray(match["static_points"])[keep],
            "target_points": np.asarray(match["target_points"])[keep],
            "forward_overlap": np.asarray(match["forward_overlap"])[keep],
            "backward_overlap": np.asarray(match["backward_overlap"])[keep],
            "cycle_error_px": np.asarray(match["cycle_error_px"])[keep],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--base-pose-root", type=Path, required=True)
    parser.add_argument("--static-pnp-diagnostics", type=Path, required=True)
    parser.add_argument("--component", type=parse_tokens, required=True)
    parser.add_argument("--frames", type=parse_tokens, required=True)
    parser.add_argument("--romav2-root", type=Path, required=True)
    parser.add_argument("--romav2-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-stride", type=int, default=8)
    parser.add_argument("--static-candidates", type=int, default=4)
    parser.add_argument("--exclude-static-views", type=parse_tokens, default=[])
    parser.add_argument("--selection-frame", type=int, default=0)
    parser.add_argument("--minimum-overlap", type=float, default=0.2)
    parser.add_argument("--maximum-cycle-error-px", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = [int(frame) for frame in args.frames]
    component = [str(view) for view in args.component]
    if int(args.selection_frame) not in frames:
        raise ValueError("selection frame must be in frames")
    runtime = args.runtime_scene.resolve()
    static_cameras = CameraLoader(runtime / "cameras/static").load_cameras(
        STATIC_VIEWS
    )
    base = {
        frame: CameraLoader(
            args.base_pose_root / f"f{frame:06d}" / "cameras_pred"
        ).load_cameras(component)
        for frame in frames
    }
    confidence, confidence_audit = load_static_pnp_confidence(
        args.static_pnp_diagnostics,
        frames=frames,
        target_views=component,
    )
    anchor = choose_anchor(
        component=component, frames=frames, confidence=confidence
    )
    excluded_static = {str(view) for view in args.exclude_static_views}
    if excluded_static - set(STATIC_VIEWS):
        raise ValueError("excluded static views must belong to the public static roster")
    candidates = nearest_static_views(
        anchor_cameras={frame: base[frame][anchor] for frame in frames},
        static_cameras=static_cameras,
        count=int(args.static_candidates) + len(excluded_static),
    )
    candidates = [view for view in candidates if view not in excluded_static][
        : int(args.static_candidates)
    ]
    if len(candidates) != int(args.static_candidates):
        raise RuntimeError("not enough non-excluded public static candidates")
    height = int(base[frames[0]][anchor].H)
    width = int(base[frames[0]][anchor].W)
    queries = query_grid(
        height=height, width=width, stride=int(args.query_stride)
    )

    model_started = time.perf_counter()
    model, _device = _load_roma_model(
        source_root=args.romav2_root,
        weights=args.romav2_weights,
        device=str(args.device),
        setting="fast",
        force_bidirectional=True,
    )
    model_load_seconds = time.perf_counter() - model_started
    selection_rows = []
    selection_matches: dict[str, dict[str, Any]] = {}
    frame = int(args.selection_frame)
    for static_view in candidates:
        matched = match_queries(
            model=model,
            static_image=runtime / "images" / static_view / f"{frame:06d}.png",
            target_image=runtime / "images" / anchor / f"{frame:06d}.png",
            queries=queries,
            height=height,
            width=width,
            minimum_overlap=float(args.minimum_overlap),
            maximum_cycle_error_px=float(args.maximum_cycle_error_px),
        )
        selection_matches[static_view] = matched
        selection_rows.append(
            {
                "static_view": static_view,
                "accepted_count": int(len(matched["point_ids"])),
                "overlap": summary(
                    np.minimum(
                        matched["forward_overlap"], matched["backward_overlap"]
                    )
                ),
                "cycle_error_px": summary(matched["cycle_error_px"]),
                "inference_seconds": matched["inference_seconds"],
            }
        )
    eligible = [row for row in selection_rows if row["accepted_count"] >= 12]
    if not eligible:
        raise RuntimeError("no public-static candidate has at least 12 cycle matches")
    chosen = min(
        eligible,
        key=lambda row: (
            -float(row["overlap"]["median"]),
            -int(row["accepted_count"]),
            str(row["static_view"]),
        ),
    )["static_view"]

    observations: dict[tuple[str, int, str], dict[str, Any]] = {}
    frame_rows = []
    inference_seconds = 0.0
    for frame in frames:
        matched = (
            selection_matches[chosen]
            if frame == int(args.selection_frame)
            else match_queries(
                model=model,
                static_image=runtime / "images" / chosen / f"{frame:06d}.png",
                target_image=runtime / "images" / anchor / f"{frame:06d}.png",
                queries=queries,
                height=height,
                width=width,
                minimum_overlap=float(args.minimum_overlap),
                maximum_cycle_error_px=float(args.maximum_cycle_error_px),
            )
        )
        inference_seconds += float(matched["inference_seconds"])
        roles = role_rows(frame=frame, match=matched, query_count=len(queries))
        for role, row in roles.items():
            observations[(anchor, frame, role)] = {
                "static_view": str(chosen),
                **row,
            }
        frame_rows.append(
            {
                "frame": frame,
                "accepted_count": int(len(matched["point_ids"])),
                "role_counts": {
                    role: int(len(row["point_ids"])) for role, row in roles.items()
                },
                "overlap": summary(
                    np.minimum(
                        matched["forward_overlap"], matched["backward_overlap"]
                    )
                ),
                "cycle_error_px": summary(matched["cycle_error_px"]),
                "inference_seconds": matched["inference_seconds"],
            }
        )

    inputs = {
        "entrypoint": Path(__file__).resolve(),
        "runtime_manifest": runtime / "runtime_manifest.json",
        "static_pnp_diagnostics": args.static_pnp_diagnostics.resolve(),
        "romav2_weights": args.romav2_weights.resolve(),
    }
    payload = {
        "protocol": PROTOCOL,
        "status": "complete_public_match_cache",
        "component": component,
        "anchor_target": anchor,
        "selected_static": str(chosen),
        "frames": frames,
        "observations": observations,
        "config": {
            "setting": "fast",
            "force_bidirectional": True,
            "query_stride": int(args.query_stride),
            "query_count": int(len(queries)),
            "static_candidate_count": int(args.static_candidates),
            "excluded_static_views": sorted(excluded_static),
            "selection_frame": int(args.selection_frame),
            "minimum_overlap": float(args.minimum_overlap),
            "maximum_cycle_error_px": float(args.maximum_cycle_error_px),
        },
        "lineage": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "selection": selection_rows,
        "confidence_audit": confidence_audit,
        "frames_audit": frame_rows,
        "runtime": {
            "model_load_seconds": float(model_load_seconds),
            "inference_seconds": float(inference_seconds),
            "inference_seconds_per_frame": float(inference_seconds / len(frames)),
        },
        "attestation": {
            "GT_used": False,
            "NVS_used": False,
            "test_RGB_or_cameras_used": False,
            "reference_pose_used": False,
            "public_static_cameras_fixed": True,
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
                "component": component,
                "anchor_target": anchor,
                "selected_static": chosen,
                "runtime": payload["runtime"],
                "accepted_count": summary(
                    np.asarray([row["accepted_count"] for row in frame_rows])
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
