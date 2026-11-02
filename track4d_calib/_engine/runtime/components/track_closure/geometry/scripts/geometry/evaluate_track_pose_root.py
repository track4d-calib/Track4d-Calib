#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rotation_error_deg(predicted: np.ndarray, reference: np.ndarray) -> float:
    delta = np.asarray(predicted, dtype=np.float64) @ np.asarray(reference, dtype=np.float64).T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _static_scene_scale(runtime_scene: Path) -> float:
    runtime_manifest = json.loads(
        (runtime_scene / "runtime_manifest.json").read_text(encoding="utf-8")
    ) if (runtime_scene / "runtime_manifest.json").is_file() else None
    if runtime_manifest is not None:
        static_views = [str(view) for view in runtime_manifest["observed_short_views"]]
    else:
        storage = CameraLoader(runtime_scene / "cameras/static")
        raise ValueError(
            "runtime_manifest.json is required to define the static scene scale"
        )
    cameras = CameraLoader(runtime_scene / "cameras/static").load_cameras(static_views)
    distances = [
        float(np.linalg.norm(cameras[first].camera_center - cameras[second].camera_center))
        for first, second in combinations(static_views, 2)
    ]
    positive = [value for value in distances if value > 1e-12]
    if not positive:
        raise ValueError("static cameras do not define a nonzero scene scale")
    return float(np.median(positive))


def _camera_dir(pose_root: Path, frame: int) -> Path:
    nested = pose_root / f"f{frame:06d}" / "cameras_pred"
    if (nested / "intri.yml").is_file() and (nested / "extri.yml").is_file():
        return nested
    return pose_root / f"f{frame:06d}"


def evaluate_pose_root(
    *,
    runtime_scene: Path,
    pose_root: Path,
    sealed_reference: Path,
    frames: Sequence[int],
    target_views: Sequence[str],
) -> dict[str, Any]:
    runtime = Path(runtime_scene)
    root = Path(pose_root)
    frame_ids = [int(frame) for frame in frames]
    targets = [str(view) for view in target_views]
    scene_scale = _static_scene_scale(runtime)
    with np.load(sealed_reference, allow_pickle=True) as reference:
        reference_views = [str(view) for view in reference["long_views"]]
        reference_index = {view: index for index, view in enumerate(reference_views)}
        reference_r = np.asarray(reference["R"], dtype=np.float64)
        reference_t = np.asarray(reference["T"], dtype=np.float64)
    missing_reference = set(targets) - set(reference_index)
    if missing_reference:
        raise ValueError(f"reference is missing target views: {sorted(missing_reference)}")

    rows: list[dict[str, Any]] = []
    for frame in frame_ids:
        directory = _camera_dir(root, frame)
        loader = CameraLoader(directory) if (directory / "extri.yml").is_file() else None
        for view in targets:
            if loader is None:
                rows.append(
                    {
                        "frame": frame,
                        "view": view,
                        "status": "missing_prediction",
                        "rotation_error_deg": None,
                        "center_error": None,
                        "center_error_rel": None,
                    }
                )
                continue
            try:
                predicted = loader.load_camera(view)
            except (ValueError, FileNotFoundError):
                rows.append(
                    {
                        "frame": frame,
                        "view": view,
                        "status": "missing_prediction",
                        "rotation_error_deg": None,
                        "center_error": None,
                        "center_error_rel": None,
                    }
                )
                continue
            view_index = reference_index[view]
            gt_r = reference_r[frame, view_index]
            gt_t = reference_t[frame, view_index]
            gt_center = -gt_r.T @ gt_t
            center_error = float(np.linalg.norm(predicted.camera_center - gt_center))
            rows.append(
                {
                    "frame": frame,
                    "view": view,
                    "status": "success",
                    "rotation_error_deg": _rotation_error_deg(predicted.R, gt_r),
                    "center_error": center_error,
                    "center_error_rel": float(center_error / scene_scale),
                }
            )
    successful = [row for row in rows if row["status"] == "success"]
    by_view = {}
    for view in targets:
        view_rows = [row for row in successful if row["view"] == view]
        by_view[view] = {
            "success_rate": float(len(view_rows) / max(len(frame_ids), 1)),
            "rotation_error_deg": _stats([row["rotation_error_deg"] for row in view_rows]),
            "center_error": _stats([row["center_error"] for row in view_rows]),
            "center_error_rel": _stats([row["center_error_rel"] for row in view_rows]),
        }
    return {
        "protocol": "hidden-parent-metric-pose-evaluation-v1",
        "runtime_scene": str(runtime.resolve()),
        "pose_root": str(root.resolve()),
        "sealed_reference": str(Path(sealed_reference).resolve()),
        "sealed_reference_sha256": _sha256(Path(sealed_reference)),
        "frames": frame_ids,
        "target_views": targets,
        "static_scene_scale": scene_scale,
        "success_rate": float(len(successful) / max(len(rows), 1)),
        "rotation_error_deg": _stats([row["rotation_error_deg"] for row in successful]),
        "center_error": _stats([row["center_error"] for row in successful]),
        "center_error_rel": _stats([row["center_error_rel"] for row in successful]),
        "by_view": by_view,
        "rows": rows,
    }


def _tokens(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--sealed-reference", type=Path, required=True)
    parser.add_argument("--frames", type=_tokens, required=True)
    parser.add_argument("--target-views", type=_tokens, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_pose_root(
        runtime_scene=args.runtime_scene,
        pose_root=args.pose_root,
        sealed_reference=args.sealed_reference,
        frames=[int(frame) for frame in args.frames],
        target_views=args.target_views,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
