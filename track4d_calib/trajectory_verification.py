"""Recheck RGB trajectory decisions against the exported cameras."""

import json
from pathlib import Path

import numpy as np

from ._engine.runtime.sparse.center_trajectory import reprojection_errors
from ._engine.runtime.sparse.motion_verification import verify_camera_motion
from ._engine.runtime.sparse.track_partition import identity_roles
from .verification import _read_cameras, digest


def verify_rgb_motion(calibration, config):
    """Verify geometry, motion bounds and whole-trajectory publication."""
    root = Path(calibration)
    evidence = root / "center_motion"
    protocol = json.loads((evidence / "PROTOCOL.json").read_text())
    result = json.loads((evidence / "RESULT.json").read_text())
    collection = json.loads((evidence / "rgb/RESULT.json").read_text())
    motion = config["center_motion"]
    settings = {"center_model": "bounded", "radius_fraction": motion["radius_fraction"],
                "rotation_model": motion["rotation_model"]}
    if protocol["config"] != config or protocol["settings"] != settings:
        raise ValueError("RGB motion protocol differs from the calibration config")
    frames, views = range(config["frames"]), config["dynamic_views"]
    rows = {row["view"]: row for row in result["rows"]}
    if (len(rows) != len(result["rows"]) or set(rows) != set(views)
            or result["total"] != len(views)
            or result["audit"]["registered"] != len(views) * len(frames)):
        raise ValueError("Incomplete RGB trajectory decisions")
    expected = {f"f{frame:06d}_v{view}.npz" for frame in frames for view in views}
    if set(collection["observations_sha256"]) != expected:
        raise ValueError("Incomplete RGB verification observations")
    expected_poses = {f"f{frame:06d}/cameras_pred/{name}.yml"
                      for frame in frames for name in ("intri", "extri")}
    if set(protocol["input_poses_sha256"]) != expected_poses or any(
        digest(root / "pre_motion_poses" / name) != sha
        for name, sha in protocol["input_poses_sha256"].items()
    ):
        raise ValueError("RGB motion input poses changed")
    files = [evidence / name for name in ("PROTOCOL.json", "RESULT.json", "rgb/RESULT.json")]
    for name, sha in collection["observations_sha256"].items():
        path = evidence / "rgb/observations" / name
        if digest(path) != sha:
            raise ValueError("RGB verification observations changed")
        files.append(path)
    statics = _read_cameras(root / "runtime_scene/cameras/static", config["static_views"])
    centers = np.stack([-cam["R"].T @ cam["T"].reshape(3) for cam in statics.values()])
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    scale = float(np.median(distances[np.triu_indices(len(centers), 1)]))
    if not np.isfinite(scale) or scale <= 0 or not np.isclose(scale, protocol["scale"], rtol=0, atol=1e-10):
        raise ValueError("RGB motion rig scale changed")
    cameras = {name: [] for name in ("raw_poses", "candidate_poses", "poses")}
    roster = config["static_views"] + views
    pose_hashes = {}
    for frame in frames:
        suffix = Path(f"f{frame:06d}/cameras_pred")
        original = _read_cameras(root / "pre_motion_poses" / suffix, roster)
        published = _read_cameras(root / "poses" / suffix, roster)
        for name in cameras:
            folder = evidence / name / suffix
            cameras[name].append(_read_cameras(folder, roster))
            for filename in ("intri.yml", "extri.yml"):
                path = folder / filename
                pose_hashes[str(path.relative_to(evidence))] = digest(path)
                files.append(path)
        for view in roster:
            for field in ("K", "D", "R", "T"):
                raw = cameras["raw_poses"][-1][view][field]
                candidate = cameras["candidate_poses"][-1][view][field]
                committed = cameras["poses"][-1][view][field]
                changed = view in rows and rows[view]["accepted"] is True
                expected_camera = candidate if changed and field in ("R", "T") else raw
                pairs = [(raw, original[view][field]), (committed, expected_camera),
                         (committed, published[view][field])]
                if view not in rows or field in ("K", "D"):
                    pairs.append((candidate, raw))
                if any(a.shape != b.shape or not np.allclose(a, b, atol=1e-10, rtol=0) for a, b in pairs):
                    raise ValueError("RGB refinement changed an uncommitted camera or intrinsics")
    if pose_hashes != result["audit"]["pose_sha256"]:
        raise ValueError("RGB trajectory files changed after export")
    accepted = 0
    for view, row in rows.items():
        observations = {role: [] for role in ("fit", "confirm", "final")}
        for frame in frames:
            with np.load(evidence / f"rgb/observations/f{frame:06d}_v{view}.npz", allow_pickle=False) as data:
                for number, role in enumerate(observations):
                    ids, xyz, uv = (data[role + "_" + key] for key in ("ids", "xyz", "uv"))
                    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids < 0).any()
                            or len(np.unique(ids)) != len(ids) or not np.all(identity_roles(ids) == number)
                            or xyz.shape != (len(ids), 3) or uv.shape != (len(ids), 2)
                            or not np.isfinite(xyz).all() or not np.isfinite(uv).all()):
                        raise ValueError("Invalid RGB observations or overlapping track roles")
                    observations[role].append({"xyz": xyz, "uv": uv})
        scores, trajectories = [], []
        for name in ("raw_poses", "candidate_poses"):
            sequence = [frame[view] for frame in cameras[name]]
            r = np.stack([cam["R"] for cam in sequence])
            c = np.stack([-cam["R"].T @ cam["T"].reshape(3) for cam in sequence])
            k, d = [np.stack([cam[field] for cam in sequence]) for field in ("K", "D")]
            trajectories.append(c)
            scores.append({role: reprojection_errors(r, c, k, d, observations[role])
                           for role in ("confirm", "final")})
        check = verify_camera_motion(*scores, *trajectories, list(frames), scale)
        decision = bool(row["solver"]["success"] and check["accepted"])
        if type(row["accepted"]) is not bool or decision != row["accepted"]:
            raise ValueError("Published RGB trajectory fails independent verification")
        path = evidence / f"candidates/v{view}.npz"
        files.append(path)
        if decision:
            with np.load(path, allow_pickle=False) as data:
                if np.max(np.linalg.norm(trajectories[1] - data["common_center"], axis=1)) > motion["radius_fraction"] * scale + 1e-9:
                    raise ValueError("RGB trajectory exceeds its center motion bound")
        accepted += decision
    if accepted != result["accepted"]:
        raise ValueError("RGB accepted trajectory count differs")
    return {"status": "verified", "accepted_cameras": accepted, "retained_cameras": len(views) - accepted,
            "camera_nodes": len(views) * len(frames),
            "artifact_sha256": {str(p.relative_to(root)): digest(p) for p in sorted(files)}}
