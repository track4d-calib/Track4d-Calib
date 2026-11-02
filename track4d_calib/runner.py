"""Portable launcher. Each run owns its code snapshot and all computed geometry."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .dataset import validate_scene
from .verification import verify_crop_refinement, verify_center_motion, verify_track_partition

ENGINE = Path(__file__).parent / "_engine"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def materialize_engine(destination: Path, dependencies: dict[str, Path]) -> None:
    shutil.copytree(ENGINE, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    vendor = destination / "third_party"
    vendor.mkdir()
    for name, source in (
        ("hloc", dependencies["hloc"]),
        ("co-tracker", dependencies["cotracker"]),
        ("RoMaV2", dependencies["romav2_src"].parent),
    ):
        (vendor / name).symlink_to(source, target_is_directory=True)


def command(
    config: dict,
    *,
    source: Path,
    output: Path,
    code: Path,
    dependencies: dict[str, Path],
    gpu: int,
    frames: int,
    pace: float,
) -> list[str]:
    """Build the same sparse calibration command for every declared scene."""
    if pace:
        raise ValueError("Sparse calibration does not support artificial matching pacing")
    if config.get("relay_source_quality", "fast") != "fast":
        raise ValueError("Sparse anchor matching currently supports fast quality only")
    if config.get("closure_loss", "huber") != "huber":
        raise ValueError("The geometric repair solver requires huber loss")
    return [
        sys.executable, str(code / "runtime/sparse/run_pipeline.py"),
        "--source", str(source), "--root", str(output),
        "--config-json", json.dumps(dict(config, frames=frames)),
        "--dependencies-json", json.dumps({key: str(path) for key, path in dependencies.items()}),
        "--stage-code", str(code), "--gpu", str(gpu),
    ]


def load_access_auditor(code: Path):
    scripts = code / "runtime/scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "input_audit.py"
        spec = importlib.util.spec_from_file_location("track4d_calib_native_audit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.audit_external_results
    finally:
        sys.path.remove(str(scripts))


def _publish_poses(output: Path, calibration: Path, frames: int) -> tuple[dict, dict]:
    """Publish finite camera poses and independently verified repair updates."""
    import cv2
    import numpy as np

    result = json.loads((calibration / "RESULT.json").read_text())
    config = json.loads((calibration / "config.json").read_text())
    closure = json.loads((calibration / "CLOSURE.json").read_text())
    radius = config["center_radius"]
    centers = (json.loads((calibration / "motion_centers.json").read_text())
               if radius is not None else {})
    expected = frames * len(config["dynamic_views"])
    if result["registered"] != expected or len(closure["rows"]) != expected:
        raise RuntimeError("incomplete dynamic camera grid")
    for row in closure["rows"]:
        if not row["committed"]:
            continue
        for role in ("confirm", "final"):
            before, after = row["base_" + role], row["candidate_" + role]
            if (after["inliers"] < max(12, before["inliers"])
                    or after["median_px"] >= before["median_px"] - 1e-6):
                raise RuntimeError("repair update failed independent verification")
    pose = calibration / "poses"
    for frame in range(frames):
        path = pose / f"f{frame:06d}/cameras_pred/extri.yml"
        cameras = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            for view in config["static_views"] + config["dynamic_views"]:
                rotation = cameras.getNode("Rot_" + view).mat()
                translation = cameras.getNode("T_" + view).mat()
                if (rotation is None or translation is None or rotation.shape != (3, 3)
                        or translation.size != 3 or not np.isfinite(rotation).all()
                        or not np.isfinite(translation).all()
                        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
                        or abs(np.linalg.det(rotation) - 1) > 1e-5):
                    raise RuntimeError(f"invalid camera pose: frame={frame}, view={view}")
                if view in centers:
                    center = (-rotation.T @ translation.reshape(3, 1)).reshape(3)
                    if np.linalg.norm(center - np.asarray(centers[view])) > radius + 1e-6:
                        raise RuntimeError(f"camera center exceeds its motion bound: frame={frame}, view={view}")
        finally:
            cameras.release()
    pose_hashes = {str(p.relative_to(pose)): digest(p)
                   for p in sorted(pose.glob("f*/cameras_pred/*.yml"))}
    if len(pose_hashes) != frames * 2 or pose_hashes != result["pose_hashes"]:
        raise RuntimeError("incomplete or modified output camera files")
    crop_verification = verify_crop_refinement(calibration, config)
    motion_verification = verify_center_motion(calibration, config)
    partition_verification = verify_track_partition(calibration, config)
    final_checks = [row.get("candidate_confirm") if row["committed"] else row["base_confirm"]
                    for row in closure["rows"]]
    unresolved = sum(check["median_px"] is None or check["median_px"] > 4
                     or check["inliers"] < 12 for check in final_checks)
    seal = {
        "status": "complete",
        "repair_updates_verified": True,
        "crop_refinement_verification": crop_verification,
        "center_motion_verification": motion_verification,
        "track_partition_verification": partition_verification,
        "unresolved_risk_count": unresolved,
        "risk_scope": "Track-closure confirmation observations, before optional non-anchor refinement",
        "final_center_contract": {"radius": radius, "finite_pose_grid": True, "checked_nodes": expected},
        "speed": {"engine_seconds": result["wall_seconds"], "stages": result["stages"]},
    }
    write_json(calibration / "POSE_AUDIT.json", seal)
    (output / "poses").symlink_to(pose.relative_to(output), target_is_directory=True)
    return seal, pose_hashes


def run(
    config: dict,
    *,
    source: Path,
    output: Path,
    dependencies: dict[str, Path],
    gpu: int,
    frames: int,
    pace: float,
    preflight_only: bool = False,
) -> dict:
    """Validate inputs, snapshot the engine, execute, and publish verified poses."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"choose a new output directory: {output}")
    output = output.resolve()
    tracer = shutil.which("strace")
    if not preflight_only and tracer is None:
        raise ValueError("Linux strace is required for the native input-boundary audit")
    output.mkdir(parents=True)
    started = time.perf_counter()
    code, calibration = output / "code", output / "calibration"
    try:
        write_json(output / "status.json", {"status": "validating_inputs"})
        inputs = validate_scene(source, config)
        write_json(output / "input_validation.json", inputs)
        materialize_engine(code, dependencies)
        args = command(
            config,
            source=source,
            output=calibration,
            code=code,
            dependencies=dependencies,
            gpu=gpu,
            frames=frames,
            pace=pace,
        )
        hashes = {
            str(p.relative_to(code)): digest(p)
            for p in sorted(code.rglob("*.py"))
            if "third_party" not in p.relative_to(code).parts
        }
        freeze = {
            "config": config,
            "command": args,
            "frames": list(range(frames)),
            "seed": 0,
            "source": str(source),
            "code_sha256": hashes,
            "dependencies": {k: str(v) for k, v in dependencies.items()},
            "weights_sha256": {k: digest(v) for k, v in dependencies.items() if v.is_file()},
            "status": "frozen_before_calibration",
            "NVS_used_for_calibration": False,
            "old_pose_tracks_or_matches_as_inputs": False,
            "matching_pace": pace,
        }
        write_json(output / "protocol.json", freeze)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": os.pathsep.join([str(code), str(dependencies["hloc"])]),
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        if preflight_only:
            completed = subprocess.run([*args, "--preflight-only"], cwd=code, env=env, check=False)
            if completed.returncode:
                raise RuntimeError("engine preflight failed")
            result = {"status": "preflight_pass", "calibration_performed": False}
        else:
            trace = output / "input_access.log"
            env["TRACK4D_CALIB_EXTERNAL_OPEN_AUDIT"] = str(trace)
            write_json(
                output / "status.json",
                {"status": "calibrating", "log": str(output / "calibration.log")},
            )
            with (output / "calibration.log").open("xb") as log:
                completed = subprocess.run(
                    [
                        tracer,
                        "-f",
                        "-yy",
                        "-e",
                        "trace=openat",
                        "-s",
                        "1024",
                        "-o",
                        str(trace),
                        *args,
                    ],
                    cwd=code,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            trusted = (
                code,
                Path(sys.prefix).resolve(),
                Path("/usr"),
                Path("/lib"),
                Path("/lib64"),
                *dependencies.values(),
            )
            audit = load_access_auditor(code)(trace, output, [source], trusted_roots=trusted)
            write_json(output / "input_audit.json", audit)
            if completed.returncode or not audit["passed"]:
                raise RuntimeError(
                    f"calibration or input audit failed (exit {completed.returncode}); see calibration.log and input_audit.json"
                )
            seal, pose_hashes = _publish_poses(output, calibration, frames)
            elapsed = time.perf_counter() - started
            result = {
                "status": "complete",
                "input_audit_passed": True,
                "pose_grid_complete": True,
                "repair_updates_verified": seal["repair_updates_verified"],
                "crop_refinement_verification": seal["crop_refinement_verification"],
                "center_motion_verification": seal["center_motion_verification"],
                "unresolved_risk_count": seal["unresolved_risk_count"],
                "risk_scope": seal["risk_scope"],
                "frames": frames,
                "pose_root": "poses",
                "pose_sha256": pose_hashes,
                "complete_wall_seconds": elapsed,
                "complete_seconds_per_frame": elapsed / frames,
                "paced": pace == 0.15,
                "NVS_performed": False,
                "final_center_contract": seal["final_center_contract"],
                "speed": seal["speed"],
            }
        write_json(output / "result.json", result)
        write_json(output / "status.json", {"status": result["status"]})
        return result
    except BaseException as error:
        write_json(
            output / "status.json",
            {
                "status": "failed",
                "error": str(error),
                "attempt_seconds": time.perf_counter() - started,
            },
        )
        raise
