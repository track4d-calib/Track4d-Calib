"""Run sparse anchor calibration and verified camera-motion refinement."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--dependencies-json", required=True)
    parser.add_argument("--stage-code", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    started = time.perf_counter()
    root = args.root
    root.mkdir(exist_ok=False)
    config = json.loads(args.config_json)
    frame_count = config["frames"]
    dependencies = json.loads(args.dependencies_json)
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[:8])
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for name in [
        "TRACK4D_CALIB_PIPELINE_CONFIG",
        "TRACK4D_CALIB_RIG_CONTRACT",
        "TRACK4D_CALIB_NUM_FRAMES",
    ]:
        env.pop(name, None)
    (root / "inputs").symlink_to(args.source.resolve(), target_is_directory=True)
    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "dependencies.json").write_text(json.dumps(dependencies, indent=2))
    code = args.stage_code.resolve()
    (root / "code").symlink_to(code, target_is_directory=True)
    vendor = code / "third_party"
    method = Path(__file__).resolve().parent
    if args.preflight_only:
        return
    scripts = code / "runtime/scripts"
    runtime = root / "runtime_scene"
    python = sys.executable
    env["PYTHONPATH"] = os.pathsep.join(
        [str(code), str(vendor / "hloc"), str(vendor / "co-tracker")]
    )
    protocol = {
        "status": "frozen_before_calibration",
        "config": config,
        "source": str(args.source),
        "anchor_count": config.get("temporal_anchor_count", 5),
        "anchor_supports": config.get("static_support_count", 2),
        "anchor_points_per_band": config.get("anchor_points_per_band", 600),
        "query_batch": 512,
        "pnp_max_points": config.get("pnp_max_points", 150),
        "gpu": args.gpu,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "ordinary_frame_cross_view_matching": bool(
            config.get("nonanchor_static_supports", 0)
            or (config.get("center_motion") or {}).get("model") == "rgb_window"
        ),
        "static_refinement": config.get("static_refinement", "off"),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in method.glob("*.py")
        },
        "reuse_previous_calibration": False,
    }
    (root / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
    rows = []


    def run(stage, cmd):
        t = time.perf_counter()
        with (root / (stage + ".log")).open("x") as log:
            result = subprocess.run(
                cmd, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        row = {
            "stage": stage,
            "command": cmd,
            "seconds": time.perf_counter() - t,
            "exit_code": result.returncode,
        }
        rows.append(row)
        (root / (stage + "_RUN.json")).write_text(json.dumps(row, indent=2))
        print(stage, row["seconds"], result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(stage)


    run(
        "prepare",
        [
            python,
            str(scripts / "prepare_scene.py"),
            "--source",
            str(args.source),
            "--runtime",
            str(runtime),
            "--scene",
            config["scene"],
            "--static-views",
            *config["static_views"],
            "--target-views",
            *config["dynamic_views"],
            "--test-views",
            "--num-frames",
            str(frame_count),
            "--resolution",
            *map(str, config["resolution"]),
        ],
    )
    run(
        "static_sfm",
        [
            python,
            str(scripts / "build_static_rig.py"),
            "--runtime",
            str(runtime),
            "--root",
            str(root / "static_sfm"),
            "--stage-code",
            str(code),
            "--python",
            python,
            "--spsg-json",
            json.dumps(config.get("spsg", {})),
            "--gpu",
            str(args.gpu),
        ],
    )
    if config.get("static_refinement", "off") != "off":
        run(
            "static_refinement",
            [
                python,
                str(scripts / "refine_static_rig.py"),
                "--runtime",
                str(runtime),
                "--root",
                str(root / "static_refinement"),
                "--num-frames",
                str(frame_count),
                "--gpu",
                str(args.gpu),
                "--romav2-root",
                dependencies["romav2_src"],
                "--romav2-weights",
                dependencies["romav2_weights"],
                "--matching-quality",
                config["static_refinement"],
                "--fit-frames",
                *map(str, config.get("static_refinement_fit_frames", [0])),
            ],
        )
    run("anchor_matching", [python, str(method / "match_anchors.py"), str(root)])
    run("tracking", [python, str(method / "track_anchors.py"), str(root)])
    if config.get("track_validation"):
        run("track_partition", [python, str(method / "track_partition.py"), str(root)])
    cmd = [
        python,
        str(scripts / "run_seeded_track_closure.py"),
        "--runtime-scene",
        str(runtime),
        "--per-frame-track-root",
        str(root / "cotracker_cached_tracks"),
        "--output-root",
        str(root / "pnp"),
        "--target-views",
        " ".join(config["dynamic_views"]),
        "--frames",
        " ".join(map(str, range(frame_count))),
        "--max-points",
        str(config.get("pnp_max_points", 150)),
        "--support-reprojection-px",
        "2.5",
        "--min-triangulation-angle-deg",
        "0.5",
        "--pnp-reprojection-px",
        "4.0",
        "--min-pnp-inliers",
        "12",
        "--seed",
        "0",
    ]
    (root / "PNP_COMMAND.json").write_text(json.dumps(cmd, indent=2))
    run("geometry", [python, str(method / "solve.py"), str(root)])
    poses = root / "initial_poses"
    poses.mkdir()
    for shard in sorted((root / "pnp").glob("shard_*")):
        for frame in (shard / "pose_track_same_time_pnp").glob("f*"):
            (poses / frame.name).symlink_to(frame.resolve(), target_is_directory=True)
    summary = json.loads((root / "PNP_SUMMARY.json").read_text())
    complete = summary["success"] == len(config["dynamic_views"]) * frame_count
    if complete:
        run("closure", [python, str(method / "close_tracks.py"), str(root)])
        poses = root / "poses"
        if config.get("nonanchor_static_supports", 0):
            run("nonanchor_matching_refinement", [python, str(method / "match_nonanchors.py"), str(root)])
            (root / "poses").rename(root / "track_poses")
            (root / "nonanchor_poses").rename(root / "poses")
        if config.get("dynamic_overlap_crop", True):
            run("anchor_crop_refinement", [python, str(method / "match_nonanchors.py"), str(root), "--anchor-crop"])
            (root / "poses").rename(root / "pre_crop_poses")
            (root / "crop_poses").rename(root / "poses")
        if config["center_radius"] != 0 and config.get("center_motion"):
            run("center_motion", [python, str(method / "refine_rgb_motion.py"), "--calibration", str(root)])
            (root / "poses").rename(root / "pre_motion_poses")
            (root / "motion_poses").rename(root / "poses")
    result = {
        "status": "complete_initial_calibration_pending_independent_quality_evaluation"
        if complete
        else "incomplete_initial_calibration",
        "wall_seconds": time.perf_counter() - started,
        "registered": summary["success"],
        "expected": summary["expected"],
        "stages": rows,
        "pose_hashes": {
            str(p.relative_to(poses)): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in poses.iterdir()
            for p in (d / "cameras_pred").glob("*.yml")
        },
    }
    (root / "RESULT.json").write_text(json.dumps(result, indent=2))
    print(
        {k: v for k, v in result.items() if k not in ["stages", "pose_hashes"]}, flush=True
    )


if __name__ == "__main__":
    main()
