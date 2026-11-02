#!/usr/bin/env python3
"""Export the points of a binary COLMAP reconstruction to PLY with a conversion summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

try:
    import read_write_model as colmap_io
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    colmap_io = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reconstruction(model: Path) -> tuple[int, int, dict[int, Any], str]:
    """Load a COLMAP model without requiring a compiled pycolmap wheel."""

    if colmap_io is not None:
        loaded = colmap_io.read_model(str(model), ext=".bin")
        if loaded is None:
            raise RuntimeError(f"failed to read COLMAP model: {model}")
        cameras, images, points = loaded
        return len(cameras), len(images), points, "read_write_model"

    # Retain compatibility for callers that have pycolmap but do not expose
    # COLMAP's reference Python reader on PYTHONPATH.  Import lazily so an
    # unusable binary wheel cannot block the preferred portable backend.
    import pycolmap  # type: ignore

    reconstruction = pycolmap.Reconstruction(model)
    return (
        reconstruction.num_cameras(),
        reconstruction.num_reg_images(),
        dict(reconstruction.points3D),
        "pycolmap",
    )


def write_binary_ply(path: Path, points: dict[int, Any]) -> None:
    """Write XYZ/RGB vertices in deterministic COLMAP point-id order."""

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for point_id in sorted(points):
            point = points[point_id]
            xyz = tuple(float(value) for value in point.xyz)
            rgb = tuple(int(value) for value in point.rgb)
            handle.write(struct.pack("<fffBBB", *xyz, *rgb))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    output = args.output.resolve()
    summary = args.summary.resolve()
    if output.exists() or summary.exists():
        raise FileExistsError(output if output.exists() else summary)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)

    num_cameras, num_images, points, backend = load_reconstruction(model)
    if num_images <= 0 or len(points) <= 0:
        raise RuntimeError("source reconstruction is empty")
    output.parent.mkdir(parents=True, exist_ok=False)
    write_binary_ply(output, points)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("PLY export is missing or empty")

    audit = {
        "status": "pass",
        "source_model": str(model),
        "registered_images": num_images,
        "cameras": num_cameras,
        "points3D": len(points),
        "backend": backend,
        "output": str(output),
        "sha256": sha256(output),
    }
    summary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
