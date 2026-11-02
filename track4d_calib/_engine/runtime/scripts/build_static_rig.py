#!/usr/bin/env python3
"""Reconstruct and install a fresh static rig from frame-0 RGB only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
from spsg_settings import resolve_spsg_settings, spsg_arguments


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_runtime_checksums(runtime: Path, protocol: str) -> None:
    """Record the installed rig and its current runtime manifest."""
    paths = [runtime / "runtime_manifest.json", runtime / "cameras/intri.yml",
             runtime / "cameras/static/intri.yml", runtime / "cameras/static/extri.yml"]
    digest_inputs = [sha256(path) for path in paths]
    write_json(runtime / "checksums.sha256.json", {
        "protocol": protocol,
        "tree_digest": hashlib.sha256("\n".join(digest_inputs).encode()).hexdigest(),
        "digest_inputs": digest_inputs,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage-code", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--instrument-gpu", action="store_true")
    parser.add_argument("--spsg-json", default="{}")
    args = parser.parse_args()
    spsg = resolve_spsg_settings(json.loads(args.spsg_json))

    runtime = args.runtime.resolve()
    root = args.root.resolve()
    stage = args.stage_code.resolve()
    runner = stage / "scripts/run_hloc_baseline_sfm.py"
    manifest_path = runtime / "runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    statics = [str(view) for view in manifest["public_static_views"]]
    if len(statics) < 2 or set(statics) & set(map(str, manifest["long_views"])):
        raise ValueError("invalid static roster")
    if root.exists():
        raise FileExistsError(root)
    installed = runtime / "cameras/static"
    if installed.exists():
        raise FileExistsError(installed)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    root.mkdir(parents=True)
    command = [
        str(args.python), str(runner),
        "--data_root", str(runtime / "images"),
        "--views", *statics,
        "--frame_idx", "0",
        "--ref_view", statics[0],
        "--output_dir", str(root / "frontend"),
        "--cameras_out", str(root / "static_rig_raw"),
        "--intrinsics_from_cameras", str(runtime / "cameras"),
        "--intrinsics_mode", "per_view",
        "--feature_conf", "superpoint_aachen",
        "--matcher_conf", "superglue",
        "--match_pair_batch_size", "2",
        "--superglue_num_workers", "0",
        "--superglue_writer_threads", "1",
        "--camera_mode", "PER_IMAGE",
        "--pairs_mode", "exhaustive",
        "--mask_mode", "all",
        # Thread count alone does not seed C++ verification/mapper RANSAC.
        # Fix both before any outcomes are known, for every declared rig.
        "--mapper_num_threads", "1",
        "--sfm_random_seed", "0",
        "--no-normalize_ref",
        "--timing_json", str(root / "timing.json"),
        "--overwrite",
    ]
    command.extend(spsg_arguments(spsg))
    if args.instrument_gpu:
        command = [command[0], str(Path(__file__).with_name("run_static_sfm.py")),
                   "--sfm-runner", command[1], "--hloc-root", str(stage / "third_party/hloc"),
                   *command[2:]]
    freeze = {
        "status": "frozen_before_fresh_static_SfM",
        "protocol": "SCENE_IO-crossdataset-fresh-static-frame0-v1",
        "runtime": str(runtime),
        "frame": 0,
        "static_views": statics,
        "pair_count": len(statics) * (len(statics) - 1) // 2,
        "method": {
            "feature": f"SuperPoint Aachen max{spsg['max_keypoints']}",
            "matcher": "SuperGlue exhaustive",
            "intrinsics": "source per-view intrinsics",
            "solver": "COLMAP incremental SfM plus global BA",
            "gauge": "arbitrary current-run SfM gauge",
            "determinism": {
                "seed": 0,
                "colmap_verification_ransac_seed": 0,
                "colmap_pipeline_mapper_triangulation_seeds": [0, 0, 0],
                "mapper_num_threads": 1,
                "python_hash_seed": 0,
                "cublas_workspace_config": ":4096:8",
            },
        },
        "command": command,
        "claim_boundary": {
            "packaged_static_pose_used": False,
            "dynamic_RGB_used": False,
            "old_feature_or_match_dump_used": False,
            "GT_used": False,
            "NVS_used": False,
        },
    }
    write_json(root / "PROTOCOL_FROZEN.json", freeze)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": os.pathsep.join(
            [str(stage), str(stage / "third_party/hloc"), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
    })
    started = time.perf_counter()
    log = root / "static_sfm.log"
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=stage, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    wall = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"static SfM failed ({completed.returncode}); see {log}")

    # The runtime is still upstream of both compared methods.  Install exactly
    # the current run's arbitrary-gauge rig, never a source/package pose.
    shutil.copytree(root / "static_rig_raw", installed)
    manifest["fresh_static_rig"] = {
        "source": str((root / "static_rig_raw").resolve()),
        "intri_sha256": sha256(installed / "intri.yml"),
        "extri_sha256": sha256(installed / "extri.yml"),
        "packaged_static_pose_used": False,
        "gauge": "arbitrary current-run frame0 static SfM gauge",
    }
    manifest["claim_boundary"]["static_pose_available"] = True
    manifest["claim_boundary"]["static_pose_source"] = "current_run_fresh_SfM"
    write_json(manifest_path, manifest)
    write_runtime_checksums(runtime, manifest["protocol"])
    timing = json.loads((root / "timing.json").read_text(encoding="utf-8"))
    summary = {
        "status": "complete_fresh_static_SfM",
        "protocol": freeze["protocol"],
        "registered_static_views": len(statics),
        "expected_static_views": len(statics),
        "static_pairs": freeze["pair_count"],
        "wall_seconds": wall,
        "timing": timing,
        "static_rig_for_downstream": str(installed.resolve()),
        "raw_static_rig": str((root / "static_rig_raw").resolve()),
        "claim_boundary": freeze["claim_boundary"],
    }
    write_json(root / "STATIC_SFM_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
