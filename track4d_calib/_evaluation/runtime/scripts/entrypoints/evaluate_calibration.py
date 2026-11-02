#!/usr/bin/env python3
"""Evaluate committed calibration using the configured cameras and image sequence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_evaluation import nvs_dynamic_roster, read_json, sha256, validate_seal, window_frames, write_json
from metrics import metric_distribution, source_digest


EVALUATOR_SHA256 = "bc32b3f46a10203424b8b22a75d351fe7dae734e49cc6fbf81fc1df4ecb086be"


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "calibration-root", "root", "settings", "project-root",
        "backend-dir", "calib-python", "train-python", "fudan-runtime",
        "evaluator", "prebuilt-dgr",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--prebuilt-dgr-sha256", required=True)
    parser.add_argument("--canonical-scene", type=Path, help="Required only for external evaluation cameras")
    args = parser.parse_args()
    frames = window_frames(args.frame_start)
    calibration = args.calibration_root.resolve()
    seal, pose_root = validate_seal(calibration)
    protocol = read_json(calibration / "config.json")
    statics, dynamics = protocol["static_views"], protocol["dynamic_views"]
    settings = read_json(args.settings)
    if settings["calibration"] != {key: protocol[key] for key in ("scene", "static_views", "dynamic_views")}:
        raise ValueError("evaluation settings differ from the committed calibration roster")
    if settings["camera_source"] not in ("external", "calibrated"):
        raise ValueError("unknown evaluation camera source")
    tests = list(settings["test_views"])
    external = settings["camera_source"] == "external"
    if set(tests) & set(statics) or set(tests).intersection(dynamics) != set(protocol.get("calibration_test_views", [])):
        raise ValueError("evaluation overlap must match calibration_test_views")
    if not external and not set(tests) <= set(dynamics):
        raise ValueError("calibrated test cameras must be in the dynamic roster")
    train_dynamics = nvs_dynamic_roster(dynamics, tests, settings["dynamic_train_views"])
    if external and args.canonical_scene is None:
        raise ValueError("external evaluation cameras require --canonical-scene")
    if sha256(args.evaluator) != EVALUATOR_SHA256 or sha256(args.prebuilt_dgr) != args.prebuilt_dgr_sha256:
        raise ValueError("frozen evaluator or renderer changed")
    pose_records = {
        str(path): sha256(path)
        for frame in frames
        for name in ("intri.yml", "extri.yml")
        for path in [pose_root / f"f{frame:06d}/cameras_pred" / name]
    }
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(root / "STATUS.json", {"status": "preparing_evaluation_only_cameras"})
    test_root, heldout = None, None
    evaluation_inputs = {}
    if external:
        # Reference cameras are evaluation inputs only, after the pose seal.
        sys.path.insert(0, str(args.project_root.resolve()))
        import numpy as np
        from track4d_calib.cameras import CameraIntrinsicsLoader, CameraLoader, CameraParams, save_cameras

        dataset = args.canonical_scene.resolve() / "dataset"
        canonical_static = dataset / "public" / protocol["scene"] / "cameras/static"
        references = CameraLoader(canonical_static).load_cameras(statics)
        heldout = dataset / "private/evaluation" / protocol["scene"] / "heldout"
        intrinsics = CameraIntrinsicsLoader(heldout / "cameras").load_camera_templates(tests)
        gt_file = dataset / "private/gt_long_poses.npz"
        test_root = root / "evaluation_only_cameras"
        with np.load(gt_file, allow_pickle=False) as data:
            views = [str(view) for view in data["long_views"].tolist()]
            for local, frame in enumerate(frames):
                cameras = dict(references)
                for view in tests:
                    i, template = views.index(view), intrinsics[view]
                    cameras[view] = CameraParams(
                        view_id=view, K=template.K.copy(), D=template.D.copy(),
                        R=data["R"][frame, i].copy(), T=data["T"][frame, i].copy(),
                        H=template.H, W=template.W,
                    )
                save_cameras(test_root / f"f{local:06d}/cameras_pred", cameras)
        evaluation_inputs = {str(p): record(p) for p in (
            gt_file, canonical_static / "intri.yml", canonical_static / "extri.yml",
            heldout / "cameras/intri.yml",
        )}

    probe = Path(__file__).resolve().parents[1] / "run_evaluation.py"
    command = [
        str(args.calib_python), str(probe), "--calibration-root", str(calibration),
        "--frame-start", str(args.frame_start),
        "--root", str(root / "readout"), "--backend-dir", str(args.backend_dir),
        "--project-root", str(args.project_root), "--test-views", *tests,
        "--dynamic-train-views", *train_dynamics,
        "--time-denominator", str(settings["time_denominator"]), "--resolution", *map(str, settings["resolution"]),
        "--source-resolution", *map(str, settings.get("source_resolution", settings["resolution"])),
        "--render-resolution-factor", str(settings.get("render_resolution_factor", 1)),
        "--time-duration-end", str(settings.get("time_duration_end", 1.0)),
        "--init-points", str(settings["init_points"]),
        "--roma-matches-per-pair", str(settings.get("roma_matches_per_pair", 2048)),
        "--init-mode", settings["init_mode"], "--roma-ref-views", *settings["roma_ref_views"],
        "--gpu", args.gpu, "--calib-python", str(args.calib_python),
        "--train-python", str(args.train_python), "--fudan-runtime", str(args.fudan_runtime),
        "--evaluator", str(args.evaluator), "--colmap-compat-dir",
        str(args.project_root / "third_party/hloc/hloc/utils"),
        "--prebuilt-dgr", str(args.prebuilt_dgr), "--prebuilt-dgr-sha256", args.prebuilt_dgr_sha256,
        "--init-temporal-sigma", str(settings["temporal_sigma"]),
    ]
    if settings.get("roma_target_views"):
        command.extend(["--roma-target-views", *settings["roma_target_views"]])
    if external:
        command.extend(["--extra-test-rgb-root", str(heldout), "--test-pose-root", str(test_root)])
    environment = os.environ.copy()
    environment.pop("TRACK4D_CALIB_FIXED_CAMERA_EXTENT", None)
    environment.update({
        "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    frozen = {
        "status": "frozen_before_initialization_training_and_metrics",
        "calibration": str(calibration), "pose_seal": record(calibration.parent / "result.json"),
        "calibration_policy": "Complete pose grid with independently verified repair updates", "input_pose_hashes": pose_records,
        "settings": record(args.settings),
        "evaluation_only_inputs": evaluation_inputs,
        "evaluation_gauge": ("Reference test cameras aligned to fresh training gauge using shared static cameras"
                             if external else "Calibrated test cameras in the current-run training gauge"),
        "GT_used_for_calibration_or_train_pose_or_initializer": False,
        "old_pose_tracks_points_or_checkpoint_used_for_training": False,

        "fairness": {
            "frame_ids": frames, "train_views": statics + train_dynamics,
            "test_views": tests, "resolution": settings["resolution"], "iters": settings["iters"],
            "seed": 0, "batch_size": 4, "checkpoint": "none",
            "init": settings["init_mode"], "roma_ref_views": settings["roma_ref_views"],
            "init_points": settings["init_points"],
            "source_resolution": settings.get("source_resolution", settings["resolution"]),
            "render_resolution_factor": settings.get("render_resolution_factor", 1),
            "time_duration": [0.0, settings.get("time_duration_end", 1.0)],
            "roma_target_views": settings.get("roma_target_views", statics),
            "roma_matches_per_pair": settings.get("roma_matches_per_pair", 2048),
            "preprocess": "canonical RGB; builder undistort/resize; lossless PNG",
            "time_denominator": settings["time_denominator"], "temporal_sigma": settings["temporal_sigma"], "camera_extent": "automatic",
        },
        "command": command,
        "code": {str(p): record(p) for p in (Path(__file__), probe, args.evaluator, args.prebuilt_dgr)},
    }
    write_json(root / "PROTOCOL_FROZEN.json", frozen)
    write_json(root / "STATUS.json", {"status": "running_fresh_initialization_training_evaluation"})
    with (root / "readout.log").open("x") as handle:
        completed = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        write_json(root / "STATUS.json", {"status": "failed", "exit_code": completed.returncode,
            "wall_seconds": time.perf_counter() - started, "log": str(root / "readout.log")})
        return completed.returncode
    validate_seal(calibration)
    if any(sha256(Path(path)) != expected for path, expected in pose_records.items()):
        raise ValueError("calibration pose changed during NVS")
    nvs = root / "readout/nvs"
    result = read_json(nvs / "eval_seed0_iter10000/summary.json")
    expected = {(view, frame) for view in tests for frame in range(settings["frames"])}
    rows = result["rows"]
    if len(rows) != len(expected) or {(str(row["view"]), int(row["frame"])) for row in rows} != expected:
        raise ValueError("test evaluation rows differ from the frozen protocol")
    write_json(root / "SOURCE_AUDIT.json", source_digest(nvs / "source"))
    write_json(root / "SUMMARY.json", {
        "status": "completed", "overall": result["overall"], "by_view": result["by_view"],
        "rows": [{**row, "local_frame": int(row["frame"]),
                  "frame": frames[int(row["frame"])]} for row in rows],
        "frame_ids": frames,
        "psnr_distribution": metric_distribution([float(row["psnr"]) for row in rows]),
        "pose_unchanged": True, "wall_seconds": time.perf_counter() - started,
        "protocol": record(root / "PROTOCOL_FROZEN.json"),
    })
    write_json(root / "STATUS.json", {"status": "completed", "exit_code": 0})
    print(result["overall"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
