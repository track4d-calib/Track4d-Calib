#!/usr/bin/env python3
"""Create a pose-free 30-frame runtime for cross-dataset calibration tests.

Only RGB images and intrinsics are exposed.  Static and dynamic extrinsics in
the source dataset, if present, are deliberately not copied or linked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalized(values: Sequence[str], label: str) -> list[str]:
    result = [str(value) for value in values]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be nonempty and unique")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--static-views", nargs="+", required=True)
    parser.add_argument("--target-views", nargs="+", required=True)
    parser.add_argument("--test-views", nargs="*", default=())
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--resolution", nargs=2, type=int)
    args = parser.parse_args()

    source = args.source.resolve()
    runtime = args.runtime.resolve()
    statics = normalized(args.static_views, "static views")
    targets = normalized(args.target_views, "target views")
    tests = [str(value) for value in args.test_views]
    if set(statics) & set(targets):
        raise ValueError("static and target rosters overlap")
    if len(tests) != len(set(tests)) or set(tests) - set(targets):
        raise ValueError("test views must be a unique subset of target views")
    if args.num_frames <= 0:
        raise ValueError("num-frames must be positive")
    if runtime.exists():
        raise FileExistsError(runtime)
    intrinsics = source / "cameras/intri.yml"
    if not intrinsics.is_file():
        raise FileNotFoundError(intrinsics)
    missing = [
        str(source / f"images/{view}/{frame:06d}.png")
        for view in [*statics, *targets]
        for frame in range(args.num_frames)
        if not (source / f"images/{view}/{frame:06d}.png").is_file()
    ]
    if missing:
        raise FileNotFoundError(missing[:10])

    runtime.mkdir(parents=True)
    (runtime / "images").symlink_to(source / "images", target_is_directory=True)
    (runtime / "cameras").mkdir()
    shutil.copy2(intrinsics, runtime / "cameras/intri.yml")
    manifest = {
        "protocol": "SCENE_IO-crossdataset-pose-free-runtime-v1",
        "scene": str(args.scene),
        "frame_count": int(args.num_frames),
        "local_frames": [0, int(args.num_frames) - 1],
        "global_frames": [0, int(args.num_frames) - 1],
        "resolution": list(args.resolution) if args.resolution else None,
        "observed_short_views": statics,
        "public_static_views": statics,
        "long_views": targets,
        "dynamic_train_views": targets,
        "heldout_test_views": tests,
        "runtime_views": [*statics, *targets],
        "source_lineage": {
            "source": str(source),
            "images": str((source / "images").resolve()),
            "intrinsics": str(intrinsics.resolve()),
            "intrinsics_sha256": sha256(intrinsics),
            "source_camera_extrinsics_opened": False,
        },
        "privacy": {
            "GT_long_pose_available": False,
            "dynamic_extrinsics_available": False,
            "heldout_images_available": False,
        },
        "claim_boundary": {
            "static_pose_available": False,
            "dynamic_pose_available": False,
            "old_track_or_match_dump_available": False,
            "fallback_available": False,
            "GT_available": False,
            "NVS_available": False,
        },
    }
    write_json(runtime / "runtime_manifest.json", manifest)
    digest = hashlib.sha256(
        (sha256(runtime / "runtime_manifest.json") + sha256(runtime / "cameras/intri.yml")).encode()
    ).hexdigest()
    write_json(runtime / "checksums.sha256.json", {
        "protocol": manifest["protocol"],
        "tree_digest": digest,
        "pose_files": [],
    })
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
