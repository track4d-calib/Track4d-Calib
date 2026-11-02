#!/usr/bin/env python3
"""Measure independent per-frame HLoc calibration using the scene data contract."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track4d_calib.config import load_config, load_dependencies, resolve_spsg_settings, spsg_arguments
from track4d_calib.dataset import validate_scene
from track4d_calib.runner import write_json


def frame_command(python, code, source, output, config, frame):
    return [
        str(python), str(code / "scripts/run_hloc_baseline_sfm.py"),
        "--data_root", str(source / "images"),
        "--views", *config["static_views"], *config["dynamic_views"],
        "--frame_idx", str(frame), "--ref_view", config["static_views"][0],
        "--output_dir", str(output / "frontend"),
        "--cameras_out", str(output / "cameras"),
        "--intrinsics_from_cameras", str(source / "cameras"),
        "--intrinsics_mode", "per_view", "--feature_conf", "superpoint_aachen",
        "--matcher_conf", "superglue", "--match_pair_batch_size", "2",
        "--superglue_num_workers", "0", "--superglue_writer_threads", "1",
        "--camera_mode", "PER_IMAGE", "--pairs_mode", "exhaustive", "--mask_mode", "all",
        "--mapper_num_threads", "1", "--sfm_random_seed", "0",
        "--no-normalize_ref", "--allow_partial", "--timing_json", str(output / "timing.json"),
        *spsg_arguments(resolve_spsg_settings(config.get("spsg", {}))),
    ]


def camera_coverage(directory, config):
    extrinsics = directory / "extri.yml"
    cameras = cv2.FileStorage(str(extrinsics), cv2.FILE_STORAGE_READ) if extrinsics.is_file() else None
    missing = {}
    try:
        for kind in ("static", "dynamic"):
            absent = []
            for view in config[kind + "_views"]:
                rotation = cameras.getNode("Rot_" + view).mat() if cameras else None
                translation = cameras.getNode("T_" + view).mat() if cameras else None
                if (rotation is None or translation is None or rotation.shape != (3, 3)
                        or translation.size != 3 or not np.isfinite(rotation).all()
                        or not np.isfinite(translation).all()):
                    absent.append(view)
            missing["missing_" + kind] = absent
    finally:
        if cameras:
            cameras.release()
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "config", "dependencies", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device ordinal")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    import torch
    device = torch.cuda.get_device_properties(args.gpu)
    subprocess.run([sys.executable, str(ROOT / "tools/checks/verify_source.py")], check=True)
    config = load_config(args.config.resolve())
    dependencies = load_dependencies(args.dependencies.resolve())
    source, output = args.source.resolve(), args.output.absolute()
    validation = validate_scene(source, config)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    frame = None
    try:
        write_json(output / "status.json", {"status": "preparing"})
        code = output / "code"
        shutil.copytree(ROOT / "track4d_calib/_engine", code,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (code / "third_party").mkdir(exist_ok=True)
        (code / "third_party/hloc").symlink_to(dependencies["hloc"], target_is_directory=True)
        write_json(output / "input_validation.json", validation)
        write_json(output / "protocol.json", {
            "config": config, "gpu": args.gpu, "python": sys.version,
            "gpu_name": device.name, "gpu_uuid": str(device.uuid),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source_manifest_sha256": hashlib.sha256((ROOT / "track4d_calib/_sources.json").read_bytes()).hexdigest(),
            "method": "Independent HLoc SuperPoint/SuperGlue exhaustive SfM; fixed intrinsics",
            "timing": "Subprocess wall time including startup and export; frames run sequentially",
            "partial_registration": "Export available cameras; count missing cameras separately",
        })
        env = dict(os.environ, PYTHONPATH=str(code), CUDA_VISIBLE_DEVICES=str(args.gpu),
                   PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for frame in range(config["frames"]):
            write_json(output / "status.json", {
                "status": "running", "frame": frame, "attempted_frames": len(rows),
            })
            destination = output / f"f{frame:06d}"
            destination.mkdir()
            command = frame_command(sys.executable, code, source, destination, config, frame)
            start = time.perf_counter()
            with (destination / "run.log").open("x") as log:
                result = subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - start
            coverage = camera_coverage(destination / "cameras", config)
            rows.append({"frame": frame, "wall_seconds": elapsed,
                         "returncode": result.returncode, "command": command, **coverage})
            write_json(output / "timing.json", rows)
        complete = sum(not row["returncode"] and not row["missing_static"]
                       and not row["missing_dynamic"] for row in rows)
        summary = {
            "status": "complete" if complete == config["frames"] else "partial",
            "attempted_frames": len(rows), "complete_frames": complete,
            "wall_seconds": sum(row["wall_seconds"] for row in rows),
            "missing_dynamic_poses": sum(len(row["missing_dynamic"]) for row in rows),
        }
        write_json(output / "result.json", summary)
        write_json(output / "status.json", summary)
        return int(complete != config["frames"])
    except (Exception, KeyboardInterrupt) as error:
        failure = {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "frame": frame, "attempted_frames": len(rows),
            "error": f"{type(error).__name__}: {error}",
        }
        print(json.dumps(failure), file=sys.stderr)
        try:
            write_json(output / "status.json", failure)
        except OSError as write_error:
            print(f"Cannot save benchmark status: {write_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
