#!/usr/bin/env python3
"""Validate a committed pose tree and adapt its metadata for reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def pose_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def initialization_pose_tree_sha256(root: Path) -> str:
    """Reproduce the tree digest stored in the initialization record."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _first_boundary_value(boundary: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in boundary:
            return boundary[name]
    return None


def adapt_pose_seal(
    path: Path,
    *,
    pose_root_override: Path | None = None,
    expected_pose_tree_sha256: str | None = None,
) -> dict[str, Any]:
    seal = read_json(path)
    boundary = seal.get("claim_boundary", {})
    forbidden = {
        "old_pose": _first_boundary_value(
            boundary, "old_pose_used", "old_dynamic_pose_used"
        ),
        "old_tracks": _first_boundary_value(
            boundary,
            "old_track_or_match_dump_used",
            "old_track_or_match_used",
            "old_match_track_or_relay_used",
        ),
        "fallback": _first_boundary_value(
            boundary,
            "fallback_to_existing_result_used",
            "fallback_used",
            "old_layer1_or_fallback_used",
        ),
        "GT": boundary.get("GT_used"),
        "NVS": boundary.get("NVS_used"),
    }
    pose_root = seal.get("pose_root")
    status = seal.get("status")
    is_initialization_public_seal = status == "sealed_fresh_public_pose_before_GT_or_NVS"
    atomic = seal.get("atomic_commit")
    if atomic is None:
        # Initialization records imply an atomic commit of the pose tree.
        atomic = is_initialization_public_seal or status == "geometry_gate_pass"
    if (
        status not in {"geometry_gate_pass", "sealed_fresh_public_pose_before_GT_or_NVS"}
        or atomic is not True
        or not pose_root
        or any(value is not False for value in forbidden.values())
    ):
        raise ValueError(
            {"reason": "pose seal is not a fresh geometry-only atomic commit", "checks": forbidden}
        )
    source_pose = Path(str(pose_root))
    resolved_pose = (
        pose_root_override.resolve() if pose_root_override is not None else source_pose.resolve()
    )
    if not resolved_pose.is_dir():
        raise FileNotFoundError(resolved_pose)
    if is_initialization_public_seal:
        sealed_digest = seal.get("pose_tree", {}).get("tree_sha256")
        if not sealed_digest or initialization_pose_tree_sha256(source_pose.resolve()) != sealed_digest:
            raise ValueError("INITIALIZATION pose tree changed after its public seal")
    tree_digest = pose_tree_sha256(resolved_pose)
    relocated = pose_root_override is not None
    if relocated and (
        not expected_pose_tree_sha256 or tree_digest != expected_pose_tree_sha256
    ):
        raise ValueError("relocated pose tree digest does not match the source digest")
    return {
        "status": "pass_fullgraph_joint_BA_before_GT_NVS",
        "atomic_commit": True,
        "pose_root": str(resolved_pose),
        "gates": {"upstream_fresh_geometry_pose_seal": True},
        "claim_boundary": {
            "old_dynamic_pose_used": False,
            "old_track_or_match_dump_used": False,
            "fallback_used": False,
            "GT_used": False,
            "NVS_used": False,
        },
        "schema_adapter": {
            "source": {"path": str(path), "sha256": sha256(path)},
            "source_protocol": seal.get("protocol"),
            "camera_pose_changed": False,
            "selection_performed": False,
            "relocated_for_compute_host": relocated,
            "source_pose_root": str(source_pose),
            "pose_tree_sha256": tree_digest,
            "adapter": {"path": str(HERE), "sha256": sha256(HERE)},
        },
    }
