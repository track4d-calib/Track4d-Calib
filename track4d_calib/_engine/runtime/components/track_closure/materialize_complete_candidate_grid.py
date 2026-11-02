#!/usr/bin/env python3
"""Complete a partial public candidate grid with exact Static+PnP fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parent
GEOMETRY = VERSION_ROOT / "geometry"
if str(GEOMETRY) not in sys.path:
    sys.path.insert(0, str(GEOMETRY))

from track4d_calib.cameras import CameraLoader, CameraParams, save_cameras


STATIC_VIEWS = tuple(
    "00 02 03 05 06 08 09 11 12 14 15 17 18 20 21 23 "
    "24 26 27 29 30 32 33 35 36 38 39 41 42 44 45 47".split()
)
TARGET_VIEWS = tuple("81 82 83 84 86 87 88 89 91 92 93 94".split())
if os.environ.get("TRACK4D_CALIB_RIG_CONTRACT"):
    from relay_contract import STATIC_VIEWS, TARGET_VIEWS
ALL_VIEWS = STATIC_VIEWS + TARGET_VIEWS
FRAMES = tuple(range(30))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path) -> dict[str, Any]:
    base = Path(root).resolve()
    files = sorted(path for path in base.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256(path).encode("ascii") + b"\n")
    return {"file_count": len(files), "tree_digest": digest.hexdigest()}


def camera_equal(left: CameraParams, right: CameraParams) -> bool:
    return bool(
        left.view_id == right.view_id
        and left.H == right.H
        and left.W == right.W
        and np.array_equal(left.K, right.K)
        and np.array_equal(left.R, right.R)
        and np.array_equal(left.T, right.T)
        and np.array_equal(left.D, right.D)
    )


def try_load(root: Path, view: str) -> CameraParams | None:
    try:
        return CameraLoader(root).load_camera(view)
    except ValueError:
        return None


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def materialize(base_root: Path, candidate_root: Path, output_root: Path) -> dict[str, Any]:
    base_root = base_root.resolve()
    candidate_root = candidate_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    missing_nodes: list[dict[str, Any]] = []
    candidate_nodes = 0
    static_nodes = 0
    for frame in FRAMES:
        base_dir = base_root / f"f{frame:06d}/cameras_pred"
        candidate_dir = candidate_root / f"f{frame:06d}/cameras_pred"
        base = CameraLoader(base_dir).load_cameras(ALL_VIEWS)
        selected: dict[str, CameraParams] = {}
        for view in ALL_VIEWS:
            candidate = try_load(candidate_dir, view)
            if view in STATIC_VIEWS:
                if candidate is None:
                    raise ValueError(f"candidate static camera is missing at {frame}:{view}")
                if not camera_equal(candidate, base[view]):
                    raise ValueError(f"candidate changed a public static camera at {frame}:{view}")
                selected[view] = candidate
                static_nodes += 1
                continue
            if candidate is None:
                selected[view] = base[view]
                missing_nodes.append(
                    {
                        "frame": int(frame),
                        "view": str(view),
                        "policy": "exact_same_frame_StaticPnP_initialization_fallback",
                    }
                )
            else:
                selected[view] = candidate
                candidate_nodes += 1
        output_dir = output_root / f"f{frame:06d}/cameras_pred"
        save_cameras(output_dir, selected)
        reloaded = CameraLoader(output_dir).load_cameras(ALL_VIEWS)
        for view in ALL_VIEWS:
            if not camera_equal(reloaded[view], selected[view]):
                raise RuntimeError(f"saved candidate camera changed at {frame}:{view}")

    audit = {
        "protocol": "track_closure-complete-candidate-grid-with-exact-base-fallback-v1",
        "status": "complete_public_candidate_initialization_grid",
        "frames": list(FRAMES),
        "static_views": list(STATIC_VIEWS),
        "target_views": list(TARGET_VIEWS),
        "expected_camera_nodes": len(FRAMES) * len(ALL_VIEWS),
        "static_camera_nodes": static_nodes,
        "dynamic_candidate_nodes": candidate_nodes,
        "dynamic_exact_base_fallback_nodes": len(missing_nodes),
        "missing_candidate_nodes": missing_nodes,
        "candidate_parameter_values_changed": False,
        "fallback_source": "same frozen complete Static+PnP grid",
        "fallback_is_initialization_only": True,
        "GT_used": False,
        "NVS_used": False,
        "test_RGB_or_cameras_used": False,
        "reference_pose_used": False,
        "inputs": {
            "base_root": {"path": str(base_root), **tree_digest(base_root)},
            "partial_candidate_root": {
                "path": str(candidate_root),
                **tree_digest(candidate_root),
            },
        },
        "output": {"path": str(output_root), **tree_digest(output_root)},
    }
    write_json(output_root.parent / "COMPLETE_CANDIDATE_GRID_AUDIT.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            materialize(args.base_root, args.candidate_root, args.output_root),
            indent=2,
            sort_keys=True,
        )
    )
