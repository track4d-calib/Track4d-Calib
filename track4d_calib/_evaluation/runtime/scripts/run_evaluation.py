#!/usr/bin/env python3
"""Build and evaluate a reconstruction from committed camera poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_time(row: dict[str, Any], denominator: int) -> float:
    match = re.search(r"_f(\d+)", Path(row["file_path"]).name)
    if match is None or not 0 <= int(match[1]) < 30:
        raise ValueError(f"not a local 30-frame image: {row['file_path']}")
    if denominator not in (29, 30):
        raise ValueError("the frozen protocol must use denominator 29 or 30")
    return int(match[1]) / denominator


def window_frames(start: int) -> list[int]:
    if start < 0:
        raise ValueError("frame start must be nonnegative")
    return list(range(start, start + 30))


def link_window_images(source: Path, destination: Path, frames: list[int]) -> None:
    """Expose source frames under the reconstruction window's local indices."""
    paths = [source / f"{frame:06d}.png" for frame in frames]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.mkdir()
    for local, path in enumerate(paths):
        (destination / f"{local:06d}.png").symlink_to(path.resolve())


def with_temporal_sigma(environment: dict[str, str], sigma: float) -> dict[str, str]:
    """Make the 4DGS time initialization explicit without mutating the caller.

    Zero selects the runtime's default; positive values are standard
    deviations in normalized time. This parameter is outside the YAML config.
    """
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("init temporal sigma must be finite and nonnegative")
    return {**environment, "FUDAN_INIT_TEMPORAL_SIGMA": format(sigma, ".17g")}


def nvs_dynamic_roster(dynamics: list[str], tests: list[str], requested: list[str] | None) -> list[str]:
    """Apply a predeclared evaluation split without changing calibration."""
    selected = list(requested) if requested is not None else [v for v in dynamics if v not in tests]
    if (not selected or len(set(selected)) != len(selected)
            or not set(selected) <= set(dynamics) or set(selected) & set(tests)):
        raise ValueError("NVS dynamic train views must be a unique nonempty calibrated, non-test subset")
    return selected


def validate_seal(root: Path) -> tuple[dict[str, Any], Path]:
    from calibration_verification import verify_calibration

    publication = root.parent
    result = verify_calibration(publication)
    return result, publication / "poses"



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--backend-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--extra-test-rgb-root", type=Path)
    parser.add_argument("--test-pose-root", type=Path)
    parser.add_argument("--test-views", nargs="+", required=True)
    parser.add_argument("--dynamic-train-views", nargs="+", help="Frozen NVS-only subset; calibration is unchanged.")
    parser.add_argument("--time-denominator", type=int, choices=(29, 30), required=True)
    parser.add_argument("--resolution", nargs=2, type=int, required=True)
    parser.add_argument("--source-resolution", nargs=2, type=int)
    parser.add_argument("--render-resolution-factor", type=int, default=1)
    parser.add_argument("--time-duration-end", type=float, default=1.0)
    parser.add_argument("--init-points", type=int, default=300000)
    parser.add_argument("--roma-target-views", nargs="+")
    parser.add_argument("--roma-matches-per-pair", type=int, default=2048)
    parser.add_argument("--init-mode", choices=("sparse_sfm", "roma_ref_pairs"), required=True)
    parser.add_argument("--roma-ref-views", nargs="*", default=[])
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--calib-python", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--fudan-runtime", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--colmap-compat-dir", type=Path, required=True)
    parser.add_argument("--prebuilt-dgr", type=Path, required=True)
    parser.add_argument("--prebuilt-dgr-sha256", required=True)
    parser.add_argument(
        "--init-temporal-sigma", type=float, required=True,
        help="Explicit 4DGS time stddev; 0 selects the legacy runtime default.",
    )
    args = parser.parse_args()
    frames = window_frames(args.frame_start)
    if sha256(args.prebuilt_dgr) != args.prebuilt_dgr_sha256:
        raise ValueError("renderer changed after evaluation setup")
    with_temporal_sigma({}, args.init_temporal_sigma)
    source_resolution = args.source_resolution or args.resolution
    if source_resolution != [v * args.render_resolution_factor for v in args.resolution]:
        raise ValueError("source resolution and runtime resize differ from declared NVS resolution")
    root = args.root.resolve()
    calibration = args.calibration_root.resolve()
    seal, original_pose = validate_seal(calibration)
    protocol = read_json(calibration / "config.json")
    static_summary = read_json(calibration / "static_sfm/STATIC_SFM_SUMMARY.json")
    statics, dynamics = protocol["static_views"], protocol["dynamic_views"]
    tests = args.test_views
    train_dynamics = nvs_dynamic_roster(dynamics, tests, args.dynamic_train_views)
    if args.test_pose_root is None and not set(tests) <= set(dynamics):
        raise ValueError("unestimated test views require an explicit evaluation-only camera tree")
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.project_root))
    from track4d_calib.cameras import CameraLoader, save_cameras
    import numpy as np

    static_root = Path(static_summary["static_rig_for_downstream"]).resolve()
    static_loader = CameraLoader(static_root)
    merged = root / "current_run_all_train_cameras"
    pose_hashes: dict[str, str] = {}
    for local, frame in enumerate(frames):
        source = original_pose / f"f{frame:06d}/cameras_pred"
        dynamic_loader = CameraLoader(source)
        cameras = static_loader.load_cameras(statics)
        cameras.update(dynamic_loader.load_cameras(dynamics))
        output = merged / f"f{local:06d}/cameras_pred"
        save_cameras(output, cameras)
        reloaded = CameraLoader(output)
        for view, camera in cameras.items():
            exported = reloaded.load_camera(view)
            for key in ("K", "D", "R", "T"):
                if not np.array_equal(getattr(camera, key), getattr(exported, key)):
                    raise ValueError(f"camera changed during export: {frame} {view} {key}")
        for name in ("intri.yml", "extri.yml"):
            pose_hashes[str(source / name)] = sha256(source / name)
    render_root = root / "render_scene"
    (render_root / "cameras").mkdir(parents=True)
    (render_root / "cameras/static").symlink_to(static_root, target_is_directory=True)
    (render_root / "images").mkdir()
    all_views = list(statics) + [v for v in tests if v not in dynamics] + [
        v for v in dynamics if v in set(train_dynamics + tests)
    ]
    for view in all_views:
        inputs = calibration / "runtime_scene/images"
        if view not in set(statics + dynamics):
            if args.extra_test_rgb_root is None:
                raise ValueError("missing held-out RGB root")
            inputs = args.extra_test_rgb_root / "images"
        aliases = (view, str(int(view)), f"{int(view):02d}", f"{int(view):03d}")
        directory = next((inputs / v for v in aliases if (inputs / v).is_dir()), None)
        if directory is None:
            raise FileNotFoundError(inputs / view)
        link_window_images(directory, render_root / "images" / view, frames)

    sys.path.insert(0, str(args.backend_dir))
    import run_reconstruction as backend
    backend_records = {
        str(path): sha256(path) for path in (
            Path(__file__).resolve(), args.evaluator.resolve(),
            args.backend_dir / "run_reconstruction.py",
            args.backend_dir / "training_config.py",
        )
    }
    lineage = {
        "status": "frozen_before_source_build_or_metric",
        "calibration_root": str(calibration),
        "frame_ids": frames,
        "pose_root": str(original_pose), "pose_files_sha256": pose_hashes,
        "all_NVS_training_cameras_from_this_calibration": True,
        "dynamic_camera_values_changed": False,
        "static_camera_source": str(static_root),
        "old_train_pose_or_points_or_checkpoint_used": False,
        "test_camera_source": str(args.test_pose_root) if args.test_pose_root else str(original_pose),
        "test_camera_read_after_calibration_only": True,
        "camera_and_point_time_denominator": args.time_denominator,
        "NVS_resolution": args.resolution,
        "source_resolution": source_resolution,
        "render_resolution_factor": args.render_resolution_factor,
        "time_duration": [0.0, args.time_duration_end],
        "init_points": args.init_points,
        "roma_target_views": args.roma_target_views or statics,
        "roma_matches_per_pair": args.roma_matches_per_pair,
        "evaluator": str(args.evaluator), "code_sha256": backend_records,
        "NVS_init_temporal_sigma": args.init_temporal_sigma,
        "NVS_dynamic_train_views": train_dynamics,
        "calibrated_dynamic_views": dynamics,

    }
    write_json(root / "TRACK4D_CALIB_NVS_INPUT_AUDIT.json", lineage)

    def adapt_current_seal(path: Path, **kwargs: Any) -> dict[str, Any]:
        if path.resolve() != calibration.parent / "result.json":
            raise ValueError("unexpected pose input")
        validate_seal(calibration)
        return {"pose_root": str(merged), "source": str(path), "camera_pose_changed": False}

    backend.adapt_pose_seal = adapt_current_seal
    original_run = backend.base.run

    def audited_run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        if Path(command[1]).name in ("train.py", "eval_fudan_checkpoint_dataset.py"):
            kwargs["env"] = with_temporal_sigma(kwargs["env"], args.init_temporal_sigma)
        if command[1] == "train.py":
            source = root / "nvs/source"
            before = {"points3d.ply": sha256(source / "points3d.ply")}
            for name in ("transforms_train.json", "transforms_test.json"):
                path = source / name
                before[name] = sha256(path)
                payload = read_json(path)
                for row in payload["frames"]:
                    row["time"] = frame_time(row, args.time_denominator)
                write_json(path, payload)
            if args.time_denominator == 29:
                from plyfile import PlyData
                path = source / "points3d.ply"
                points = PlyData.read(str(path))
                old = points["vertex"]["time"]
                frames = np.rint(old * 30).astype(int)
                if set(frames) != set(range(30)) or np.max(np.abs(old - frames / 30)) > 1e-6:
                    raise ValueError("initializer is not the expected frame/30 time grid")
                points["vertex"]["time"][:] = frames / 29
                temporary = path.with_suffix(".retimed.ply")
                points.write(str(temporary))
                temporary.replace(path)
            write_json(root / "NVS_PRETRAIN_TIME_AUDIT.json", {
                "denominator": args.time_denominator, "before_sha256": before,
                "after_sha256": {name: sha256(source / name) for name in before},
                "init_temporal_sigma": args.init_temporal_sigma,
                "camera_extent_override": kwargs["env"].get("TRACK4D_CALIB_FIXED_CAMERA_EXTENT"),
            })
        if Path(command[1]).name == "eval_fudan_checkpoint_dataset.py":
            command = list(command)
            command[1] = str(args.evaluator.resolve())
        return original_run(command, **kwargs)

    backend.base.run = audited_run
    command = [
        "run_reconstruction.py", "--scene", protocol["scene"],
        "--render-scene-root", str(render_root), "--pose-seal",
        str(calibration.parent / "result.json"), "--static-sfm-summary",
        str(calibration / "static_sfm/STATIC_SFM_SUMMARY.json"),
        "--output-root", str(root / "nvs"), "--all-views", *all_views,
        "--test-views", *tests, "--static-gauge-views", *statics,
        "--dynamic-train-views", *train_dynamics,
        "--resolution", *map(str, source_resolution), "--gpu", args.gpu,
        "--render-resolution-factor", str(args.render_resolution_factor),
        "--time-duration-end", str(args.time_duration_end), "--init-points", str(args.init_points),
        "--roma-target-views", *(args.roma_target_views or statics),
        "--roma-matches-per-pair", str(args.roma_matches_per_pair),
        "--calib-python", str(args.calib_python), "--train-python", str(args.train_python),
        "--project-root", str(args.project_root), "--point-exporter",
        str(args.backend_dir / "export_colmap_points3d_ply.py"),
        "--fudan-runtime", str(args.fudan_runtime), "--colmap-compat-dir",
        str(args.colmap_compat_dir), "--prebuilt-dgr", str(args.prebuilt_dgr),
        "--prebuilt-dgr-sha256", args.prebuilt_dgr_sha256, "--init-mode", args.init_mode,
    ]
    if args.roma_ref_views:
        command.extend(["--roma-ref-views", *args.roma_ref_views])
    if args.test_pose_root:
        command.extend(["--test-pose-root", str(args.test_pose_root)])
    write_json(root / "NVS_BACKEND_COMMAND.json", command)
    sys.argv = command
    return backend.main()


if __name__ == "__main__":
    raise SystemExit(main())
