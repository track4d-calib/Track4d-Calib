#!/usr/bin/env python3
"""Prepare a reconstruction dataset, train 4DGS and evaluate held-out views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

try:
    import training_config as base
    from pose_adapter import adapt_pose_seal
except ModuleNotFoundError:  # pragma: no cover
    from scripts import training_config as base
    from scripts.pose_adapter import adapt_pose_seal


def unique(values: list[str], name: str) -> tuple[str, ...]:
    result = tuple(map(str, values))
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_render_static_gauge(*, render_scene_root: Path, static_summary: dict[str, Any]) -> dict[str, Any]:
    """Reject a pose/NVS run when static and dynamic world gauges differ.

    A fresh calibration pose is expressed in the fresh-SfM static gauge.  The
    renderer obtains static cameras from ``render_scene_root``; silently using
    a different reconstruction therefore combines two valid but incompatible
    gauges (the common symptom is a large PSNR collapse).  Require byte-identical
    static camera files and record the check in the protocol lineage.
    """

    expected_root = Path(str(static_summary.get("static_rig_for_downstream", ""))).resolve()
    render_root = (render_scene_root / "cameras" / "static").resolve()
    if not expected_root.is_dir():
        raise FileNotFoundError(
            "fresh static-SfM downstream rig is missing: " + str(expected_root)
        )
    if not render_root.is_dir():
        raise FileNotFoundError("render static rig is missing: " + str(render_root))
    files: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    for name in ("extri.yml", "intri.yml"):
        expected = expected_root / name
        rendered = render_root / name
        if not expected.is_file() or not rendered.is_file():
            mismatches.append(name)
            continue
        expected_hash = _sha256_file(expected)
        rendered_hash = _sha256_file(rendered)
        files[name] = {
            "fresh_static": str(expected),
            "render_static": str(rendered),
            "fresh_sha256": expected_hash,
            "render_sha256": rendered_hash,
            "identical": expected_hash == rendered_hash,
        }
        if expected_hash != rendered_hash:
            mismatches.append(name)
    if mismatches:
        raise ValueError(
            "render_scene_root static rig is not the current fresh-SfM gauge "
            f"(mismatched files: {mismatches}). Use a render root whose "
            "cameras/static is copied or linked from static_sfm_summary." 
            f" fresh={expected_root} render={render_root}"
        )
    return {
        "fresh_static_root": str(expected_root),
        "render_static_root": str(render_root),
        "files": files,
        "identical": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--render-scene-root", type=Path, required=True)
    parser.add_argument("--pose-seal", type=Path, required=True)
    parser.add_argument("--pose-root-override", type=Path)
    parser.add_argument("--expected-pose-tree-sha256")
    parser.add_argument("--static-sfm-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--all-views", nargs="+", required=True)
    parser.add_argument("--test-views", nargs="+", required=True)
    parser.add_argument("--static-gauge-views", nargs="+", required=True)
    parser.add_argument("--dynamic-train-views", nargs="*", default=[])
    parser.add_argument("--test-pose-root", type=Path)
    parser.add_argument("--resolution", nargs=2, type=int, required=True)
    parser.add_argument("--render-resolution-factor", type=int, default=1)
    parser.add_argument("--time-duration-end", type=float, default=1.0)
    parser.add_argument("--init-points", type=int, default=300000)
    parser.add_argument("--roma-target-views", nargs="+")
    parser.add_argument("--roma-matches-per-pair", type=int, default=2048)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--calib-python", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--point-exporter", type=Path, required=True)
    parser.add_argument("--fudan-runtime", type=Path, required=True)
    parser.add_argument("--colmap-compat-dir", type=Path, required=True)
    parser.add_argument("--prebuilt-dgr", type=Path, required=True)
    parser.add_argument("--prebuilt-dgr-sha256", required=True)
    parser.add_argument(
        "--init-mode",
        choices=("sparse_sfm", "roma_ref_pairs"),
        default="sparse_sfm",
        help=(
            "Fresh NVS initializer. roma_ref_pairs reproduces the paper "
            "Dyn-DNA protocol from current-run training RGB and poses."
        ),
    )
    parser.add_argument(
        "--roma-ref-views",
        nargs="*",
        default=[],
        help="Static training views used as references by roma_ref_pairs.",
    )
    args = parser.parse_args()
    if (min(*args.resolution, args.render_resolution_factor, args.init_points,
            args.roma_matches_per_pair) <= 0
            or any(v % args.render_resolution_factor for v in args.resolution)
            or not math.isfinite(args.time_duration_end) or args.time_duration_end <= 0):
        raise ValueError("invalid source/render resolution, initialization budget or time duration")

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    all_views = unique(args.all_views, "all-views")
    tests = unique(args.test_views, "test-views")
    statics = unique(args.static_gauge_views, "static-gauge-views")
    dynamic = tuple(map(str, args.dynamic_train_views))
    if set(tests) - set(all_views) or set(statics) - set(all_views):
        raise ValueError("test/static views must be included in all-views")
    train_views = tuple(view for view in all_views if view not in set(tests))
    if set(dynamic) - set(train_views):
        raise ValueError("dynamic train views must be training views")
    roma_refs = tuple(map(str, args.roma_ref_views))
    roma_targets = unique(args.roma_target_views or list(statics), "roma-target-views")
    if not set(roma_targets) <= set(train_views):
        raise ValueError("roma target views must be training views")
    if args.init_mode == "roma_ref_pairs":
        if not roma_refs or len(roma_refs) != len(set(roma_refs)):
            raise ValueError("roma_ref_pairs requires unique --roma-ref-views")
        if set(roma_refs) - set(statics):
            raise ValueError("roma reference views must be static training views")
    frames = tuple(range(30))

    pose_seal_path = args.pose_seal.resolve()
    pose_adapter = adapt_pose_seal(
        pose_seal_path,
        pose_root_override=args.pose_root_override,
        expected_pose_tree_sha256=args.expected_pose_tree_sha256,
    )
    pose_root = Path(str(pose_adapter["pose_root"])).resolve()
    static_path = args.static_sfm_summary.resolve()
    static = base.json.loads(static_path.read_text(encoding="utf-8"))
    if (
        static.get("status") != "complete_fresh_static_SfM"
        or static.get("claim_boundary", {}).get("packaged_static_pose_used") is not False
        or Path(static.get("raw_static_rig", static.get("static_rig_for_downstream", ""))).name
        != "static_rig_raw"
    ):
        raise ValueError("static rig is not strict fresh-SfM output")
    static_gauge_check = _check_render_static_gauge(
        render_scene_root=args.render_scene_root.resolve(),
        static_summary=static,
    )
    for frame in frames:
        camera_dir = pose_root / f"f{frame:06d}/cameras_pred"
        if not all((camera_dir / name).is_file() for name in ("intri.yml", "extri.yml")):
            raise FileNotFoundError(camera_dir)
    test_pose_root = args.test_pose_root.resolve() if args.test_pose_root else None
    if test_pose_root is not None:
        for frame in frames:
            camera_dir = test_pose_root / f"f{frame:06d}/cameras_pred"
            if not all((camera_dir / name).is_file() for name in ("intri.yml", "extri.yml")):
                raise FileNotFoundError(camera_dir)

    source = output / "source"
    model = output / "model_seed0"
    evaluation = output / "eval_seed0_iter10000"
    panoptic = output / "panoptic"
    config_path = output / "full4dgs_config_seed0.yaml"
    cfg = base.config(source, model)
    cfg["time_duration"] = [0.0, args.time_duration_end]
    cfg["ModelParams"]["resolution"] = args.render_resolution_factor
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    freeze = {
        "status": "frozen_before_fresh_NVS_source_or_metric",
        "protocol": "SCENE_IO-generic-fresh-pose-30f-10k-NVS-v1",
        "scene": args.scene,
        "frames": list(frames),
        "train_views": list(train_views),
        "test_views": list(tests),
        "resolution": [v // args.render_resolution_factor for v in args.resolution],
        "source_resolution": list(args.resolution),
        "render_resolution_factor": args.render_resolution_factor,
        "time_duration": cfg["time_duration"],
        "iterations": 10000,
        "batch_size": 4,
        "seed": 0,
        "init_points": args.init_points,
        "pose_root": str(pose_root),
        "test_pose_source": "sealed_evaluation_only_root" if test_pose_root else "same_as_train",
        "inputs": {
            "pose_seal": {"path": str(pose_seal_path), "sha256": base.sha256(pose_seal_path)},
            "static_sfm_summary": {"path": str(static_path), "sha256": base.sha256(static_path)},
            "controller": {"path": str(Path(__file__).resolve()), "sha256": base.sha256(Path(__file__).resolve())},
            "point_exporter": {
                "path": str(args.point_exporter.resolve()),
                "sha256": base.sha256(args.point_exporter.resolve()),
            },
            "builder": {
                "path": str(args.project_root.resolve() / "scripts/apps/build_given_pose_fudan4dgs_source.py"),
                "sha256": base.sha256(args.project_root.resolve() / "scripts/apps/build_given_pose_fudan4dgs_source.py"),
            },
            "panoptic_builder": {
                "path": str(args.project_root.resolve() / "scripts/build_4dgs_panoptic_from_calib.py"),
                "sha256": base.sha256(args.project_root.resolve() / "scripts/build_4dgs_panoptic_from_calib.py"),
            },
            "dense_point_builder": {
                "path": str(args.project_root.resolve() / "scripts/build_fudan_4dgs_time_init_pointcloud.py"),
                "sha256": base.sha256(args.project_root.resolve() / "scripts/build_fudan_4dgs_time_init_pointcloud.py"),
            },
            "evaluator": {
                "path": str(args.project_root.resolve() / "scripts/apps/eval_fudan_checkpoint_dataset.py"),
                "sha256": base.sha256(args.project_root.resolve() / "scripts/apps/eval_fudan_checkpoint_dataset.py"),
            },
            "train": {
                "path": str(args.fudan_runtime.resolve() / "train.py"),
                "sha256": base.sha256(args.fudan_runtime.resolve() / "train.py"),
            },
            "prebuilt_renderer": {
                "path": str(args.prebuilt_dgr.resolve()),
                "sha256": base.sha256(args.prebuilt_dgr.resolve()),
            },
        },
        "fresh_initialization": {
            "method": (
                "roma_ref_pairs: current-run training RGB and poses"
                if args.init_mode == "roma_ref_pairs"
                else "current-run frame0 static-SfM points, sampled over 30 timestamps"
            ),
            "mode": args.init_mode,
            "roma_ref_views": list(roma_refs),
            "roma_target_views": list(roma_targets) if args.init_mode == "roma_ref_pairs" else [],
            "roma_matches_per_pair": args.roma_matches_per_pair,
            "old_point_cloud_or_source_used": False,
            "test_RGB_used_for_initialization": False,
        },
        "static_gauge_check": static_gauge_check,
        "fairness": {
            "same_iterations_resolution_split_frames_seed_preprocess_init_across_compared_pose_arms": True,
            "only_allowed_difference": "fresh committed training/test calibration pose",
        },
        "claim_boundary": {
            "prior_pose_used_for_calibration": False,
            "old_pose_track_or_match_dump_used": False,
            "fallback_used": False,
            "old_NVS_source_point_cloud_or_checkpoint_used": False,
            "GT_or_NVS_used_for_pose_selection": False,
            "evaluation_test_pose_read_only_after_pose_commit": bool(test_pose_root),
        },
    }
    base.write_json(output / "NVS_PROTOCOL_FROZEN.json", freeze)

    project = args.project_root.resolve()
    builder = project / "scripts/apps/build_given_pose_fudan4dgs_source.py"
    evaluator = project / "scripts/apps/eval_fudan_checkpoint_dataset.py"
    runtime_4dgs = args.fudan_runtime.resolve()
    colmap_reader = args.colmap_compat_dir.resolve() / "read_write_model.py"
    if not colmap_reader.is_file():
        raise FileNotFoundError(colmap_reader)
    # HLoc's utils directory also contains io.py, which shadows Python's io.
    colmap_compat = output / "colmap_compat"
    colmap_compat.mkdir()
    shutil.copyfile(colmap_reader, colmap_compat / colmap_reader.name)
    base.write_json(colmap_compat / "source.json", {
        "path": str(colmap_reader), "sha256": base.sha256(colmap_reader),
    })
    prebuilt_dgr = args.prebuilt_dgr.resolve()
    if base.sha256(prebuilt_dgr) != args.prebuilt_dgr_sha256:
        raise ValueError("prebuilt renderer hash mismatch")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": (
                f"{colmap_compat}:{runtime_4dgs / 'scene'}:{project}:"
                f"{project / 'third_party/RoMaV2'}:{environment.get('PYTHONPATH', '')}"
            ),
        }
    )
    runs: dict[str, Any] = {}
    fresh_points = output / "fresh_static_points/points3D.ply"
    if args.init_mode == "roma_ref_pairs":
        dense_panoptic = output / "fresh_dense_init/panoptic"
        dense_source = output / "fresh_dense_init/source"
        runs["dense_init_panoptic"] = base.run(
            [
                str(args.calib_python),
                str(project / "scripts/build_4dgs_panoptic_from_calib.py"),
                "--render_scene_root",
                str(args.render_scene_root.resolve()),
                "--calib_run_root",
                str(pose_root),
                "--output_dir",
                str(dense_panoptic),
                "--frame_start",
                "0",
                "--frame_end",
                "29",
                "--source_frame_index_offset",
                "0",
                "--views",
                *dict.fromkeys((*statics, *roma_targets)),
                "--test_views",
                "--auto_test_view_strategy",
                "none",
                "--test_stride",
                "1",
                "--test_first_frame",
                "0",
                "--no-drop_test_frames_from_train_except_first",
                "--resize_wh",
                str(args.resolution[0]),
                str(args.resolution[1]),
                "--point_cloud_mode",
                "synthetic",
                "--num_init_points",
                "1000",
                "--seed",
                "0",
            ],
            cwd=project,
            env=environment,
            log=output / "dense_init_panoptic.log",
        )
        runs["dense_init_points"] = base.run(
            [
                str(args.calib_python),
                str(project / "scripts/build_fudan_4dgs_time_init_pointcloud.py"),
                "--panoptic_root",
                str(dense_panoptic),
                "--per_frame_root",
                str(pose_root),
                "--output_dir",
                str(dense_source),
                "--mode",
                "roma_ref_pairs",
                "--start_frame",
                "0",
                "--end_frame",
                "29",
                "--total_points",
                str(args.init_points),
                "--seed",
                "0",
                "--device",
                "cuda",
                "--roma_setting",
                "fast",
                "--roma_ref_view",
                roma_refs[0],
                "--roma_ref_views",
                *roma_refs,
                "--roma_target_views",
                *roma_targets,
                "--roma_pairs_per_frame",
                "4",
                "--roma_pairs_per_ref",
                "4",
                "--roma_matches_per_pair",
                str(args.roma_matches_per_pair),
                "--roma_candidate_multiplier",
                "4",
                "--roma_foreground_mode",
                "ref",
                "--roma_foreground_min_ratio",
                "0.6",
                "--roma_foreground_dilate_px",
                "5",
                "--roma_reproj_px",
                "2",
                "--roma_max_depth",
                "0",
                "--roma_target_strategy",
                "spread",
                "--timestamp_mode",
                "normalized",
                "--fps",
                "30",
                "--align_frame_gauge",
                "--gauge_anchor_frame",
                "0",
                "--gauge_align_views",
                *statics,
                "--gauge_min_common",
                "6",
                "--no-init_point_filter_outliers",
                "--force",
            ],
            cwd=project,
            env=environment,
            log=output / "dense_init_points.log",
        )
        fresh_points = dense_source / "points3d.ply"
        if not fresh_points.is_file():
            raise FileNotFoundError(fresh_points)
    else:
        runs["point_export"] = base.run(
            [
                str(args.calib_python),
                str(args.point_exporter.resolve()),
                "--model",
                str(static_path.parent / "frontend/hloc_sfm/sfm"),
                "--output",
                str(fresh_points),
                "--summary",
                str(output / "fresh_static_points/EXPORT_SUMMARY.json"),
            ],
            cwd=project,
            env=environment,
            log=output / "point_export.log",
        )
    builder_command = [
        str(args.calib_python),
        str(builder),
        "--render_scene_root",
        str(args.render_scene_root.resolve()),
        "--per_frame_root",
        str(pose_root),
        "--panoptic_root",
        str(panoptic),
        "--fudan_source",
        str(source),
        "--pose_source",
        "per_frame_calib",
        "--pose_source_label",
        "fresh_geometry_committed_pose",
        "--frame_start",
        "0",
        "--frame_end",
        "29",
        "--views",
        *all_views,
        "--test_views",
        *tests,
        "--test_stride",
        "1",
        "--resize_wh",
        str(args.resolution[0]),
        str(args.resolution[1]),
        "--point_cloud_mode",
        "ply" if args.init_mode == "sparse_sfm" else "synthetic",
        "--num_init_points",
        "10000" if args.init_mode == "sparse_sfm" else "1000",
        "--fudan_mode",
        "panoptic_init_npz",
        "--fudan_total_points",
        str(args.init_points),
        "--device",
        "cuda",
        "--timestamp_mode",
        "normalized",
        "--seed",
        "0",
        "--python",
        str(args.calib_python),
        "--summary_json",
        str(output / "SOURCE_BUILD_SUMMARY.json"),
    ]
    if args.init_mode == "sparse_sfm":
        builder_command.extend(["--point_cloud_ply", str(fresh_points)])
    else:
        builder_command.extend(
            [
                "--stable_fudan_points3d_ply",
                str(fresh_points),
                "--stable_fudan_points3d_source",
                str(fresh_points.parent),
                "--no-align_stable_points3d_to_target_gauge",
            ]
        )
    if dynamic:
        builder_command.extend(
            ["--require_pose_motion_views", *dynamic, "--pose_motion_audit_mode", "record_only"]
        )
    if test_pose_root is not None:
        builder_command.extend(
            [
                "--test_pose_source",
                "existing_per_frame",
                "--test_per_frame_root",
                str(test_pose_root),
                "--test_static_gauge_views",
                *statics,
            ]
        )
    runs["source"] = base.run(
        builder_command,
        cwd=project,
        env=environment,
        log=output / "source_build.log",
    )
    expected_train = len(train_views) * len(frames)
    expected_test = len(tests) * len(frames)
    train_rows = json.loads((source / "transforms_train.json").read_text())["frames"]
    test_rows = json.loads((source / "transforms_test.json").read_text())["frames"]
    if len(train_rows) != expected_train or len(test_rows) != expected_test:
        raise ValueError(
            {"expected_train": expected_train, "actual_train": len(train_rows), "expected_test": expected_test, "actual_test": len(test_rows)}
        )

    train_env = os.environ.copy()
    train_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": f"{project / 'scripts/geometry/nvs_4dgs/runtime_compat'}:{project}",
            "TRACK4D_CALIB_PREBUILT_DGR_SO": str(prebuilt_dgr),
            "TRACK4D_CALIB_PREBUILT_DGR_SHA256": args.prebuilt_dgr_sha256,
            "TORCH_EXTENSIONS_DIR": str(output / "runtime/torch_extensions"),
        }
    )
    Path(train_env["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True)
    started = time.perf_counter()
    runs["train"] = base.run(
        [
            str(args.train_python),
            "train.py",
            "--config",
            str(config_path),
            "--seed",
            "0",
            "--save_iterations",
            "10000",
            "--test_iterations",
            "10001",
            "--quiet",
        ],
        cwd=runtime_4dgs,
        env=train_env,
        log=output / "train10000.log",
    )
    checkpoint = model / "chkpnt10000.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    runs["eval"] = base.run(
        [
            str(args.train_python),
            str(evaluator),
            "--fudan_root",
            str(runtime_4dgs),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--source_path",
            str(source),
            "--out_dir",
            str(evaluation),
            "--max_frames_per_view",
            "30",
            "--time_duration",
            "0",
            str(args.time_duration_end),
            "--skip_side_images",
            "--skip_videos",
        ],
        cwd=runtime_4dgs,
        env=train_env,
        log=output / "eval.log",
    )
    result = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    if int(result["overall"]["frames"]) != expected_test or set(result["by_view"]) != set(tests):
        raise ValueError("evaluation is not the frozen test set")
    final = {
        "status": "pass",
        "protocol": freeze["protocol"],
        "scene": args.scene,
        "fairness": {
            key: freeze[key]
            for key in (
                "frames",
                "train_views",
                "test_views",
                "resolution",
                "iterations",
                "batch_size",
                "seed",
                "init_points",
            )
        },
        "claim_boundary": freeze["claim_boundary"],
        "new": result["overall"],
        "new_by_view": result["by_view"],
        "checkpoint": {"path": str(checkpoint), "sha256": base.sha256(checkpoint)},
        "runs": runs,
        "train_eval_wall_seconds": time.perf_counter() - started,
    }
    base.write_json(output / "FINAL_SUMMARY.json", final)
    print(json.dumps({"status": "pass", "new": final["new"], "new_by_view": final["new_by_view"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
