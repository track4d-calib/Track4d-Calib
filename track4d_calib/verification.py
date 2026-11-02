"""Validate published sparse calibration artifacts before reconstruction."""
import json
import hashlib
import math
from pathlib import Path


def digest(path):
    """Hash artifacts without importing the calibration runtime."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()

def verify_improvement(before, after):
    """Require an improvement on observations excluded from pose fitting."""
    if (after["inliers"] < max(12, before["inliers"])
            or after["median_px"] is None or before["median_px"] is None
            or not math.isfinite(after["median_px"])
            or not math.isfinite(before["median_px"])
            or after["median_px"] >= before["median_px"] - 1e-6):
        raise ValueError("Repair failed independent geometry verification")


def verify_track_partition(calibration, config):
    """Check the actual PnP inputs and later RGB refinement observations."""
    if not config.get("track_validation"):
        return {"status": "disabled"}
    import numpy as np
    from ._engine.runtime.sparse.track_partition import frame_roles, exclude_reserved_queries

    root = Path(calibration)
    protocol = json.loads((root / "track_partition/PROTOCOL.json").read_text())
    for name, expected in protocol["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Track partition artifact changed: {name}")
    seen, references = set(), []
    for path in sorted((root / "pnp").glob("shard_*/predictions_pre_gt.json")):
        record = json.loads(path.read_text())
        references.append(path)
        for folder, hashes in record["track_files_sha256"].items():
            directory = Path(folder)
            if directory.parent.resolve() != (root / "fitting_tracks").resolve():
                raise ValueError("PnP did not use the reserved-track partition")
            seen.add(directory.name)
            for name, expected in hashes.items():
                if digest(directory / name) != expected:
                    raise ValueError("PnP track input changed after fitting")
    if seen != {f"f{frame:06d}" for frame in range(config["frames"])}:
        raise ValueError("PnP partition audit is incomplete")
    closure = json.loads((root / "CLOSURE.json").read_text())
    if Path(closure["track_source"]).resolve() != (root / "fitting_tracks").resolve():
        raise ValueError("Closure used unpartitioned tracks")
    for frame in range(config["frames"]):
        source = root / f"cotracker_cached_tracks/f{frame:06d}"
        target = root / f"fitting_tracks/f{frame:06d}"
        roles = frame_roles(root, frame)
        visibility = np.load(source / "visibility.npy")
        visibility[..., roles != 0] = False
        if not np.array_equal(visibility, np.load(target / "visibility.npy")):
            raise ValueError("Reserved tracks are visible to fitting")
        for name in ("tracks.npy", "track_quality.npy", "views.txt", "meta.json"):
            if digest(source / name) != digest(target / name):
                raise ValueError("Partition changed track observations or identities")
    count = 0
    for name in ("crop_matches", "nonanchor_matches"):
        result_path = root / name / "RESULT.json"
        if not result_path.is_file():
            continue
        references.append(result_path)
        for row in json.loads(result_path.read_text())["rows"]:
            frame, view = row["frame"], row["view"]
            path = root / name / f"f{frame:06d}_v{view}.npz"
            with np.load(path, allow_pickle=False) as data:
                for role in range(3):
                    uv = data[f"uv_{role}"]
                    if not exclude_reserved_queries(root, frame, view, uv).all():
                        raise ValueError("RGB refinement reused a reserved dynamic observation")
                    count += len(uv)
    return {"status": "verified", "frames": len(seen), "reserved_pnp_observations": 0,
            "rgb_refinement_observations_checked": count, "scope": protocol["scope"],
            "partition_sha256": digest(root / "track_partition/PROTOCOL.json"),
            "input_attestation_sha256": {str(p.relative_to(root)): digest(p) for p in references}}


def _read_cameras(folder, views):
    import cv2
    import numpy as np

    cameras = {view: {} for view in views}
    for filename, fields in (
        ("intri.yml", {"K": "K_", "D": "D_"}),
        ("extri.yml", {"R": "Rot_", "T": "T_"}),
    ):
        path = folder / filename
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not storage.isOpened():
                raise ValueError(f"Missing camera file: {path}")
            for view in views:
                for field, prefix in fields.items():
                    value = storage.getNode(prefix + view).mat()
                    if value is None or not np.isfinite(value).all():
                        raise ValueError(f"Missing or nonfinite camera {field}: {path}, view={view}")
                    cameras[view][field] = value
        finally:
            storage.release()
    for view, cam in cameras.items():
        if (cam["K"].shape != (3, 3) or cam["R"].shape != (3, 3)
                or cam["T"].size != 3 or cam["D"].size not in (4, 5, 8, 12, 14)
                or not np.allclose(cam["R"] @ cam["R"].T, np.eye(3), atol=1e-5)
                or abs(np.linalg.det(cam["R"]) - 1) > 1e-5):
            raise ValueError(f"Invalid camera matrices: {folder}, view={view}")
    return cameras


def _reprojection_metrics(camera, xyz, uv):
    import cv2
    import numpy as np

    if len(xyz) < 12:
        return {"count": len(xyz), "inliers": 0, "median_px": None}
    projected, _ = cv2.projectPoints(
        xyz, cv2.Rodrigues(camera["R"])[0], camera["T"], camera["K"], camera["D"]
    )
    error = np.linalg.norm(projected[:, 0] - uv, axis=1)
    depth = (xyz @ camera["R"].T + camera["T"].reshape(1, 3))[:, 2]
    if not np.isfinite(error).all() or not np.isfinite(depth).all():
        raise ValueError("Nonfinite crop reprojection")
    return {"count": len(xyz), "inliers": int(((error <= 4) & (depth > 0)).sum()),
            "median_px": float(np.median(error))}


def _verify_metrics(actual, recorded):
    if not isinstance(recorded, list) or len(recorded) != 2:
        raise ValueError("Expected two crop verification scores")
    for measured, saved in zip(actual, recorded):
        median, reference = measured["median_px"], saved.get("median_px")
        same_median = (median is None and reference is None) or (
            median is not None and type(reference) in (int, float)
            and math.isfinite(reference) and abs(median - reference) <= 1e-6
        )
        if (measured["count"] != saved.get("count")
                or measured["inliers"] != saved.get("inliers") or not same_median):
            raise ValueError("Saved crop scores differ from camera reprojection")


def verify_crop_refinement(calibration, config):
    """Reproject saved holdout tracks and check the complete crop output grid."""
    import numpy as np

    if not config.get("dynamic_overlap_crop", True):
        return {"status": "disabled"}
    root = Path(calibration)
    evidence = root / "crop_matches"
    protocol_path, result_path = evidence / "PROTOCOL.json", evidence / "RESULT.json"
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(result_path.read_text())
    anchors = np.rint(np.linspace(
        0, config["frames"] - 1, config.get("temporal_anchor_count", 5)
    )).astype(int).tolist()
    if (protocol.get("anchor_crop") is not True or protocol.get("anchors") != anchors
            or protocol.get("frames") != anchors):
        raise ValueError("Crop verification does not cover the configured anchors")
    expected = {(frame, view) for frame in anchors for view in config["dynamic_views"]}
    rows = result["rows"]
    indexed = {(row["frame"], row["view"]): row for row in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("Incomplete or duplicate crop verification rows")
    files = [protocol_path, result_path]
    committed = 0
    views = config["static_views"] + config["dynamic_views"]
    for frame in range(config["frames"]):
        suffix = Path(f"f{frame:06d}/cameras_pred")
        output = "pre_motion_poses" if config.get("center_motion") and config["center_radius"] != 0 else "poses"
        before_folder, after_folder = root / "pre_crop_poses" / suffix, root / output / suffix
        before_cameras, after_cameras = _read_cameras(before_folder, views), _read_cameras(after_folder, views)
        files.extend(folder / name for folder in (before_folder, after_folder)
                     for name in ("intri.yml", "extri.yml"))
        for view in views:
            before_cam, after_cam = before_cameras[view], after_cameras[view]
            row = indexed.get((frame, view))
            if row is not None and type(row.get("committed")) is not bool:
                raise ValueError("Missing crop commit decision")
            changed = row is not None and row["committed"]
            fields = ("K", "D") if changed else ("R", "T", "K", "D")
            if any(before_cam[key].shape != after_cam[key].shape
                   or not np.allclose(before_cam[key], after_cam[key], rtol=0, atol=1e-10)
                   for key in fields):
                raise ValueError(f"Crop changed an uncommitted pose or intrinsics: frame={frame}, view={view}")
            if row is None:
                continue
            path = evidence / f"f{frame:06d}_v{view}.npz"
            files.append(path)
            roles, identities = [], set()
            with np.load(path, allow_pickle=False) as observations:
                for role in range(3):
                    ids, xyz, uv = (observations[f"{key}_{role}"] for key in ("ids", "xyz", "uv"))
                    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids < 0).any()
                            or xyz.shape != (len(ids), 3) or uv.shape != (len(ids), 2)
                            or not np.isfinite(xyz).all() or not np.isfinite(uv).all()):
                        raise ValueError(f"Invalid crop observations: {path}, role={role}")
                    unique = set(ids.tolist())
                    if len(unique) != len(ids) or identities.intersection(unique):
                        raise ValueError(f"Crop fitting and verification point IDs overlap: {path}")
                    identities.update(unique)
                    roles.append((xyz, uv))
            if row.get("role_counts") != [len(xyz) for xyz, _ in roles]:
                raise ValueError(f"Crop observation counts differ: {path}")
            before = [_reprojection_metrics(before_cam, *role) for role in roles[1:]]
            _verify_metrics(before, row.get("before"))
            if changed:
                after = [_reprojection_metrics(after_cam, *role) for role in roles[1:]]
                _verify_metrics(after, row.get("after"))
                single_pass = row.get("single_pass")
                if not isinstance(single_pass, list) or len(single_pass) != 2:
                    raise ValueError("Missing single-pass crop comparison")
                for old, new, single in zip(before, after, single_pass):
                    verify_improvement(old, new)
                    verify_improvement(single, new)
                committed += 1
    if committed != result.get("committed"):
        raise ValueError("Crop commit count differs from verified updates")
    return {"status": "verified", "crop_nodes": len(rows), "committed_nodes": committed,
            "camera_nodes": config["frames"] * len(views),
            "artifact_sha256": {str(path.relative_to(root)): digest(path) for path in sorted(files)}}


def verify_center_motion(calibration, config):
    """Recheck complete trajectories and the tracks excluded from their fit."""
    if not config.get("center_motion") or config["center_radius"] == 0:
        return {"status": "fixed_center" if config["center_radius"] == 0 else "legacy"}
    if config["center_motion"].get("model") == "rgb_window":
        from .trajectory_verification import verify_rgb_motion

        return verify_rgb_motion(calibration, config)
    raise ValueError("Unsupported center-motion model")


def verify_calibration(root):
    root = Path(root)
    result = json.loads((root / "result.json").read_text())
    if result.get("status") != "complete" or not all(result.get(k) is True for k in (
            "input_audit_passed", "pose_grid_complete", "repair_updates_verified")):
        raise ValueError("Sparse calibration did not pass publication checks")
    protocol = json.loads((root / "protocol.json").read_text())
    config = protocol["config"]
    expected = config["frames"] * len(config["dynamic_views"])
    if result["final_center_contract"]["checked_nodes"] != expected:
        raise ValueError("Wrong dynamic pose count")
    if json.loads((root / "input_audit.json").read_text())["passed"] is not True:
        raise ValueError("Native input audit failed")
    for relative, sha in protocol["code_sha256"].items():
        if digest(root / "code" / relative) != sha:
            raise ValueError("Executed calibration source changed: " + relative)
    if len(result["pose_sha256"]) != config["frames"] * 2:
        raise ValueError("Incomplete camera files")
    for relative, sha in result["pose_sha256"].items():
        if digest(root / "poses" / relative) != sha:
            raise ValueError("Committed pose changed: " + relative)
    engine = json.loads((root / "calibration/RESULT.json").read_text())
    if engine["registered"] != expected or engine["pose_hashes"] != result["pose_sha256"]:
        raise ValueError("Engine result differs from published cameras")
    crop = verify_crop_refinement(root / "calibration", config)
    if "crop_refinement_verification" in result and result["crop_refinement_verification"] != crop:
        raise ValueError("Crop evidence changed after publication")
    motion = verify_center_motion(root / "calibration", config)
    if "center_motion" in config and result.get("center_motion_verification") != motion:
        raise ValueError("Center-motion evidence changed after publication")
    closure = json.loads((root / "calibration/CLOSURE.json").read_text())
    if len(closure["rows"]) != expected:
        raise ValueError("Incomplete geometric verification")
    for row in closure["rows"]:
        if row["committed"]:
            for role in ("confirm", "final"):
                before, after = row["base_" + role], row["candidate_" + role]
                verify_improvement(before, after)
    if config.get("nonanchor_static_supports", 0):
        extra = json.loads((root / "calibration/nonanchor_matches/RESULT.json").read_text())
        for row in extra["rows"]:
            if row["committed"]:
                if len(row["before"]) != 2 or len(row["after"]) != 2:
                    raise ValueError("Missing non-anchor verification observations")
                for before, after in zip(row["before"], row["after"]):
                    verify_improvement(before, after)
    if config["center_radius"] is not None:
        import cv2
        import numpy as np

        centers = json.loads((root / "calibration/motion_centers.json").read_text())
        if set(centers) != set(config["dynamic_views"]):
            raise ValueError("Incomplete camera center references")
        for frame in range(config["frames"]):
            path = root / f"poses/f{frame:06d}/cameras_pred/extri.yml"
            cameras = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            try:
                for view, reference in centers.items():
                    rotation = cameras.getNode("Rot_" + view).mat()
                    translation = cameras.getNode("T_" + view).mat()
                    center = (-rotation.T @ translation.reshape(3, 1)).reshape(3)
                    distance = np.linalg.norm(center - np.asarray(reference))
                    if not np.isfinite(distance) or distance > config["center_radius"] + 1e-6:
                        raise ValueError("Camera center exceeds its motion bound")
            finally:
                cameras.release()
    runtime = root / "calibration/runtime_scene"
    manifest = json.loads((runtime / "runtime_manifest.json").read_text())
    for name in ("intri", "extri"):
        if digest(runtime / ("cameras/static/" + name + ".yml")) != manifest["fresh_static_rig"][name + "_sha256"]:
            raise ValueError("Committed static rig changed")
    return result
