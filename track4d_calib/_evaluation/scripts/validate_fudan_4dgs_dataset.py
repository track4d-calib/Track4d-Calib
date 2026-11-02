#!/usr/bin/env python3
"""Validate that a dataset root is readable by the local Fudan 4DGS loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class ValidationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Failed to read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"Expected JSON object in {path}")
    return payload


def _resolve_image_path(source_path: Path, file_path: str, extension: str) -> Path:
    token = str(file_path)
    if not token.endswith(extension):
        token = token + extension
    image_path = Path(token)
    if image_path.is_absolute():
        return image_path
    return source_path / image_path


def _validate_transform_file(source_path: Path, name: str, extension: str) -> int:
    transform_path = source_path / name
    if not transform_path.is_file():
        if (source_path / "train_meta.json").is_file() or (source_path / "test_meta.json").is_file():
            raise ValidationError(
                f"Missing {name}; this looks like an intermediate panoptic dataset. "
                "Convert it with scripts/build_fudan_4dgs_time_init_pointcloud.py before Fudan 4DGS training."
            )
        raise ValidationError(f"Missing required Fudan transform file: {transform_path}")
    payload = _read_json(transform_path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValidationError(f"{transform_path} must contain a non-empty frames list")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValidationError(f"{transform_path} frame {index} is not an object")
        if "file_path" not in frame:
            raise ValidationError(f"{transform_path} frame {index} is missing file_path")
        if "transform_matrix" not in frame:
            raise ValidationError(f"{transform_path} frame {index} is missing transform_matrix")
        image_path = _resolve_image_path(source_path, str(frame["file_path"]), extension)
        if not image_path.is_file():
            raise ValidationError(f"Missing image for {name} frame {index}: {image_path}")
    return len(frames)


def _validate_points3d(source_path: Path) -> int:
    ply_path = source_path / "points3d.ply"
    if not ply_path.is_file():
        raise ValidationError(f"Missing required Fudan point cloud: {ply_path}")
    vertex_count = -1
    has_time = False
    try:
        with ply_path.open("r", encoding="ascii", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("element vertex "):
                    vertex_count = int(stripped.split()[-1])
                if stripped == "property float time":
                    has_time = True
                if stripped == "end_header":
                    break
    except Exception as exc:
        raise ValidationError(f"Failed to inspect {ply_path}: {exc}") from exc
    if vertex_count <= 0:
        raise ValidationError(f"{ply_path} must declare a positive vertex count")
    if not has_time:
        raise ValidationError(f"{ply_path} must include property float time for 4DGS initialization")
    return vertex_count


def validate_dataset(source_path: Path, extension: str = ".png") -> dict[str, int | str]:
    source_path = Path(source_path).resolve()
    if not source_path.is_dir():
        raise ValidationError(f"source_path is not a directory: {source_path}")
    train_frames = _validate_transform_file(source_path, "transforms_train.json", extension)
    test_frames = _validate_transform_file(source_path, "transforms_test.json", extension)
    points = _validate_points3d(source_path)
    return {
        "source_path": str(source_path),
        "train_frames": int(train_frames),
        "test_frames": int(test_frames),
        "points3d_vertices": int(points),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=Path, required=True)
    parser.add_argument("--extension", type=str, default=".png")
    args = parser.parse_args()
    try:
        summary = validate_dataset(args.source_path, extension=str(args.extension))
    except ValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
