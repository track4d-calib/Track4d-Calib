"""Refine and commit complete camera trajectories using independent RGB evidence."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from camera_motion import fit_camera_motion
from center_trajectory import reprojection_errors
from motion_verification import verify_camera_motion
from rgb_observations import ROLES, acquire, digest, read, spatial_sample, write
from track_partition import identity_roles


def audit_roles(rows):
    identities = []
    for number, role in enumerate(ROLES):
        values = np.concatenate([row[role + "_ids"] for row in rows])
        if not np.all(identity_roles(values) == number):
            raise ValueError("RGB observations changed their persistent role")
        identities.append(set(values.tolist()))
    if any(identities[a] & identities[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Fitting and verification identities overlap across time")


def repair(task):
    root, output, view, settings, scale = task
    from track4d_calib.cameras import CameraLoader

    cv2.setNumThreads(1)
    config = read(root / "config.json")
    cameras, rows = [], []
    for frame in range(config["frames"]):
        cameras.append(CameraLoader(root / f"poses/f{frame:06d}/cameras_pred").load_cameras([view])[view])
        with np.load(output / f"rgb/observations/f{frame:06d}_v{view}.npz") as data:
            rows.append({key: data[key] for key in data.files})
    audit_roles(rows)
    observations = {role: [{"xyz": row[role + "_xyz"], "uv": row[role + "_uv"]} for row in rows] for role in ROLES}
    selections = []
    for row, original in zip(observations["fit"], rows):
        selected = spatial_sample(row["uv"], original["fit_ids"], 150)
        selections.append(original["fit_ids"][selected].tolist())
        row["xyz"], row["uv"] = row["xyz"][selected], row["uv"][selected]
    r, c, k, d = (np.stack(values) for values in ([cam.R for cam in cameras],
        [cam.camera_center.reshape(3) for cam in cameras], [cam.K for cam in cameras], [cam.D for cam in cameras]))
    fit = {"rotations": r, "centers": c, "solver": {"success": False, "reason": "insufficient_fit_coverage"}}
    if min(len(row["uv"]) for row in observations["fit"]) >= 12:
        fit = fit_camera_motion(r, c, k, d, observations["fit"], np.arange(config["frames"]),
            scale=scale, center_radius_fraction=settings["radius_fraction"],
            knot_spacing=10., max_evaluations=100)
    before = {role: reprojection_errors(r, c, k, d, observations[role]) for role in ROLES[1:]}
    after = {role: reprojection_errors(fit["rotations"], fit["centers"], k, d, observations[role]) for role in ROLES[1:]}
    check = verify_camera_motion(before, after, c, fit["centers"], np.arange(config["frames"]), scale)
    accepted = bool(fit["solver"]["success"] and check["accepted"])
    arrays = {"R": fit["rotations"], "C": fit["centers"], "raw_R": r, "raw_C": c,
              "common_center": fit.get("common_center", np.median(c, axis=0))}
    np.savez_compressed(output / f"candidates/v{view}.npz", **arrays)
    row = {"view": view, "solver": fit["solver"], "accepted": accepted, "verification": check,
           "fit_ids": selections, "center_span_percent_before": center_span(c, scale),
           "center_span_percent_candidate": center_span(fit["centers"], scale)}
    write(output / f"candidates/v{view}.json", row)
    return row


def center_span(centers, scale):
    return float(100 * np.linalg.norm(centers[:, None] - centers[None], axis=-1).max() / scale)


def export_and_audit(root, output, rows, scale):
    from track4d_calib.cameras import CameraLoader, save_cameras

    config = read(root / "config.json")
    candidates = {}
    for row in rows:
        with np.load(output / f"candidates/v{row['view']}.npz") as data:
            candidates[row["view"]] = {key: data[key] for key in data.files}
    accepted = {row["view"] for row in rows if row["accepted"]}
    hashes = {}
    for frame in range(config["frames"]):
        source = CameraLoader(root / f"poses/f{frame:06d}/cameras_pred").load_cameras(config["static_views"] + config["dynamic_views"])
        for name in ("raw_poses", "candidate_poses", "poses"):
            cameras = dict(source)
            if name != "raw_poses":
                for view, data in candidates.items():
                    if name == "candidate_poses" or view in accepted:
                        rotation, center = data["R"][frame], data["C"][frame]
                        cameras[view] = replace(cameras[view], R=rotation, T=-rotation @ center)
            folder = output / name / f"f{frame:06d}/cameras_pred"
            save_cameras(folder, cameras)
            actual = CameraLoader(folder).load_cameras(list(cameras))
            for view, expected in cameras.items():
                for field in ("K", "D", "R", "T"):
                    if not np.allclose(getattr(actual[view], field), getattr(expected, field), atol=1e-10, rtol=0):
                        raise ValueError("Export changed a camera matrix")
            for path in folder.glob("*.yml"):
                hashes[str(path.relative_to(output))] = digest(path)
    # Recompute acceptance from the exported cameras and stored observations.
    for row in rows:
        view = row["view"]
        raw, fitted, evidence = [], [], []
        for frame in range(config["frames"]):
            raw.append(CameraLoader(output / f"raw_poses/f{frame:06d}/cameras_pred").load_cameras([view])[view])
            fitted.append(CameraLoader(output / f"candidate_poses/f{frame:06d}/cameras_pred").load_cameras([view])[view])
            with np.load(output / f"rgb/observations/f{frame:06d}_v{view}.npz") as data:
                evidence.append({key: data[key] for key in data.files})
        audit_roles(evidence)
        scores = []
        for cameras in (raw, fitted):
            r, c, k, d = (np.stack(values) for values in ([cam.R for cam in cameras],
                [cam.camera_center.reshape(3) for cam in cameras], [cam.K for cam in cameras], [cam.D for cam in cameras]))
            scores.append({role: reprojection_errors(r, c, k, d,
                [{"xyz": item[role + "_xyz"], "uv": item[role + "_uv"]} for item in evidence]) for role in ROLES[1:]})
        check = verify_camera_motion(*scores, np.stack([cam.camera_center.reshape(3) for cam in raw]),
            np.stack([cam.camera_center.reshape(3) for cam in fitted]), np.arange(config["frames"]), scale)
        if bool(row["solver"]["success"] and check["accepted"]) != row["accepted"]:
            raise ValueError("Exported trajectory acceptance differs")
    return {"pose_sha256": hashes, "recomputed_acceptance": True,
            "uncommitted_cameras_unchanged": True, "intrinsics_and_static_poses_unchanged": True,
            "commit_scope": "whole camera trajectory", "registered": len(rows) * config["frames"]}


def refine(root, output, settings, workers=4):
    """Fit, verify and export a trajectory stage without changing input poses."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists")
    config = read(root / "config.json")
    if not config.get("track_validation"):
        raise ValueError("Reserve track identities before PnP with track_validation")
    if (config.get("center_motion") or {}).get("model") != "rgb_window" or config["center_radius"] is not None:
        raise ValueError("RGB motion refinement requires rgb_window and center_radius=null")
    if settings["center_model"] != "bounded" or settings["rotation_model"] != "framewise":
        raise ValueError("RGB refinement uses bounded centers and framewise rotations")
    if not np.isfinite(settings["radius_fraction"]) or settings["radius_fraction"] < 0 or workers < 1:
        raise ValueError("Invalid motion radius or worker count")
    from track4d_calib.cameras import CameraLoader

    output.mkdir(parents=True)
    (output / "candidates").mkdir()
    statics = CameraLoader(root / "runtime_scene/cameras/static").load_cameras(config["static_views"])
    centers = np.stack([cam.camera_center.reshape(3) for cam in statics.values()])
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    scale = float(np.median(distances[np.triu_indices(len(centers), 1)]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate static rig")
    sources = {str(p): digest(p) for p in Path(__file__).parent.glob("*.py")}
    input_poses = {str(p.relative_to(root / "poses")): digest(p) for f in range(config["frames"])
                   for p in (root / f"poses/f{f:06d}/cameras_pred").glob("*.yml")}
    write(output / "PROTOCOL.json", {"settings": settings, "scale": scale, "source_sha256": sources,
        "input_poses_sha256": input_poses, "config": config, "fit_points_per_frame": 150,
        "max_evaluations": 100, "knot_spacing": 10., "loss": "Huber 2 px", "verification": "Independent RGB role verification with non-increasing high-frequency center motion",
        "commit": "Accept complete camera trajectories only; retain rejected trajectories unchanged"})
    start = time.monotonic()
    acquire(root, output / "rgb")
    collected = time.monotonic()
    with ProcessPoolExecutor(min(workers, len(config["dynamic_views"]))) as pool:
        rows = list(pool.map(repair, [(root, output, view, settings, scale) for view in config["dynamic_views"]]))
    fitted = time.monotonic()
    audit = export_and_audit(root, output, rows, scale)
    collection = read(output / "rgb/RESULT.json")
    if any(digest(output / "rgb/observations" / name) != value
           for name, value in collection["observations_sha256"].items()):
        raise ValueError("RGB observations changed after collection")
    if (any(digest(Path(path)) != value for path, value in sources.items())
            or any(digest(root / "poses" / path) != value for path, value in input_poses.items())):
        raise ValueError("Source or input poses changed during repair")
    result = {"accepted": sum(row["accepted"] for row in rows), "total": len(rows), "rows": rows,
        "seconds": {"collection": collected - start, "fit_and_verify": fitted - collected,
                    "export_and_audit": time.monotonic() - fitted, "total": time.monotonic() - start},
        "audit": audit, "source_and_input_unchanged": True}
    write(output / "RESULT.json", result)
    print("RESULT", {key: result[key] for key in ("accepted", "total", "seconds")}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    root = args.calibration.resolve()
    motion = read(root / "config.json")["center_motion"]
    if motion["model"] != "rgb_window":
        raise ValueError("This stage requires center_motion.model=rgb_window")
    refine(root, root / "center_motion", {
        "center_model": "bounded", "radius_fraction": motion["radius_fraction"],
        "rotation_model": motion["rotation_model"],
    })
    (root / "motion_poses").symlink_to("center_motion/poses", target_is_directory=True)


if __name__ == "__main__":
    engine = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(engine / "components/track_closure/geometry"), str(engine / "scripts")]
    main()
