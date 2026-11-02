"""Validate relay manifests and resolve their shared geometry inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from track4d_calib.same_time_multiview_pnp import SameTimePnPConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_relay_manifests(
    manifest_paths: Sequence[Path],
) -> dict[str, Any]:
    manifests = []
    for path in manifest_paths:
        resolved = Path(path).resolve()
        manifests.append((resolved, json.loads(resolved.read_text(encoding="utf-8"))))
    if not manifests:
        raise ValueError("at least one relay manifest is required")

    shared_keys = ("runtime_scene", "per_frame_matcher_root", "static_views", "config")
    reference = manifests[0][1]
    for path, manifest in manifests[1:]:
        for key in shared_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"relay manifests differ at {key}: {path}")

    pose_root_by_view: dict[str, Path] = {}
    for path, manifest in manifests:
        pose_root = Path(str(manifest["baseline_pose_root"])).resolve()
        for raw_view in manifest["target_views"]:
            view = str(raw_view)
            if view in pose_root_by_view:
                raise ValueError(f"target view appears in multiple relay manifests: {view}")
            pose_root_by_view[view] = pose_root
        fairness = manifest.get("fairness", {})
        if fairness.get("GT_used") or fairness.get("NVS_used"):
            raise ValueError(f"relay manifest contains forbidden readout: {path}")
    return {
        "runtime_scene": Path(str(reference["runtime_scene"])).resolve(),
        "base_matcher_root": Path(str(reference["per_frame_matcher_root"])).resolve(),
        "static_views": [str(view) for view in reference["static_views"]],
        "config": SameTimePnPConfig(**reference["config"]),
        "pose_root_by_view": pose_root_by_view,
        "manifest_hashes": {str(path): _sha256(path) for path, _manifest in manifests},
    }
