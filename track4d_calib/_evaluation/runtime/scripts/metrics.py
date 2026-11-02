from __future__ import annotations

import hashlib, json, os, statistics, subprocess, time

from pathlib import Path

from typing import Any, Sequence

def metric_distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "std": statistics.pstdev(values),
        "min": min(values), "max": max(values),
    }

def source_digest(source: Path) -> dict[str, Any]:
    """Hash the actual consumed images as well as geometry, times and points."""
    result: dict[str, Any] = {"root": str(source), "files": {}, "splits": {}}
    for name in ("transforms_train.json", "transforms_test.json", "points3d.ply"):
        result["files"][name] = sha256(source / name)
    for split in ("train", "test"):
        payload = read_json(source / f"transforms_{split}.json")
        digest = hashlib.sha256()
        for row in payload["frames"]:
            path = source / row["file_path"]
            if not path.suffix:
                path = path.with_suffix(".png")
            digest.update(row["file_path"].encode() + b"\0" + sha256(path).encode())
        result["splits"][split] = {
            "images": len(payload["frames"]), "images_sha256": digest.hexdigest(),
        }
    return result

from run_evaluation import read_json, sha256
