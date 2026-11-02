#!/usr/bin/env python3
"""Build the track_closure conditioned relay cache for one scene."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any

import numpy as np

from relay_contract import BANDS, RIG_CONTRACT, RUNTIME_TARGETS


VERSION_ROOT = Path(__file__).resolve().parent
CODE = VERSION_ROOT / "geometry"
SELECTOR = VERSION_ROOT / "chain/select_band_static_pair_nullsafe.py"
SOURCE_BUILDER = VERSION_ROOT / "chain/build_dual_static_target_query_xyz_cache.py"
RELAY_BUILDER = VERSION_ROOT / "chain/build_dual_static_xyz_ptz_relay_cache.py"
FRAMES = tuple(range(30))
PAIR_SECONDS = 0.15
STATIC_CANDIDATE_COUNT = 8
STATIC_SUPPORT_COUNT = (RIG_CONTRACT or {}).get("static_support_count", 2)
SOURCE_MATCHING_QUALITY = os.environ.get("TRACK4D_CALIB_RELAY_SOURCE_QUALITY", "fast")
if SOURCE_MATCHING_QUALITY not in ("fast", "precise"):
    raise ValueError("relay source matching quality must be fast or precise")


def relay_pair_budget(
    bands: tuple[tuple[str, ...], ...] = BANDS,
    frames: tuple[int, ...] = FRAMES,
) -> dict[str, Any]:
    """Count every image pair evaluated by the conditioned relay.

    Each band evaluates eight static candidates once, matches its selected
    anchor to its configured static supports at every frame, and relays the anchor query
    to every other PTZ in that band.  Bidirectional cycle checking remains one
    image-pair request under the project's fixed 0.15 s/pair accounting.
    """

    if not bands or not frames or any(len(band) < 2 for band in bands):
        raise ValueError("relay pair accounting requires nonempty multi-view bands")
    selection = len(bands) * STATIC_CANDIDATE_COUNT
    source = len(bands) * STATIC_SUPPORT_COUNT * len(frames)
    ptz_relay = sum((len(band) - 1) * len(frames) for band in bands)
    total = selection + source + ptz_relay
    return {
        "selection_pairs": int(selection),
        "source_static_pairs": int(source),
        "ptz_relay_pairs": int(ptz_relay),
        "total_pairs": int(total),
        "pair_seconds": PAIR_SECONDS,
        "equivalent_seconds_per_frame": float(total * PAIR_SECONDS / len(frames)),
    }


def runtime_shard_metadata(
    band_index: int, band: tuple[str, ...]
) -> dict[str, Any]:
    """Emit the shard identity consumed by the closure runtime."""

    component = tuple(str(view) for view in band)
    if band_index < 0 or band_index >= len(BANDS) or component != BANDS[band_index]:
        raise ValueError(f"unexpected relay band identity: {band_index}:{component}")
    target = RUNTIME_TARGETS[band_index]
    if target not in component:
        raise ValueError(f"runtime target {target} is outside component {component}")
    return {"component": list(component), "target_view": target}


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


def write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def scene_paths(args: argparse.Namespace) -> dict[str, Path]:
    work = args.scene_root / "calibration/pose_method"
    return {
        "runtime": args.scene_root / "dataset/public" / args.scene,
        "base": work / "static_pnp_complete/pose_static_pnp_temporal_complete",
        "diagnostics": work / "stage_compat/base_same_frame_pnp/predictions_pre_gt.json",
        "base_freeze": work / "static_pnp_complete/PUBLIC_OUTPUT_FREEZE.json",
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    paths = scene_paths(args)
    required = {
        "runner": Path(__file__).resolve(),
        "selector": SELECTOR,
        "source_builder": SOURCE_BUILDER,
        "relay_builder": RELAY_BUILDER,
        "runtime_manifest": paths["runtime"] / "runtime_manifest.json",
        "base_pose_freeze": paths["base_freeze"],
        "static_pnp_diagnostics": paths["diagnostics"],
        "roma_weights": args.romav2_weights,
    }
    if missing := [str(path) for path in required.values() if not path.is_file()]:
        raise FileNotFoundError(missing)
    payload = {
        "protocol": "track_closure-conditioned-dual-static-band-relay-v1",
        "status": "frozen_before_new_public_matching_private_GT_or_NVS",
        "scene": args.scene,
        "method_version": "track_closure",
        "bands": [list(band) for band in BANDS],
        "config": {
            "frames": list(FRAMES),
            "query_stride": 8,
            "minimum_overlap": 0.2,
            "maximum_cycle_error_px": 1.5,
            "static_candidate_count": STATIC_CANDIDATE_COUNT,
            "static_support_count": STATIC_SUPPORT_COUNT,
            "source_matching_quality": SOURCE_MATCHING_QUALITY,
            "different_static_physical_azimuth_required": True,
            "gpus": args.gpus,
            "python": str(args.python),
            "romav2_root": str(args.romav2_root),
        },
        "pair_budget": relay_pair_budget(),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in required.items()
        },
        "fairness": {
            "GT_used": False,
            "NVS_used": False,
            "test_RGB_or_cameras_used": False,
            "reference_pose_used": False,
        },
    }
    path = args.root / "METHOD_FREEZE.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("existing conditioned-relay freeze differs")
    if not path.exists():
        write_json(path, payload)
    return payload


def run_logged(
    command: list[str], *, environment: dict[str, str], stdout: Path, stderr: Path
) -> None:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        completed = subprocess.run(command, cwd=CODE, env=environment, stdout=out, stderr=err)
    if completed.returncode:
        raise RuntimeError({"command": command, "exit_code": completed.returncode})


def run_band(
    args: argparse.Namespace, index: int, band: tuple[str, ...], gpu: int
) -> dict[str, Any]:
    paths = scene_paths(args)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": os.pathsep.join([str(CODE), str(VERSION_ROOT), str(args.romav2_root)]),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
    )
    shard_root = args.root / "shards" / f"band_{index}"
    if shard_root.exists():
        raise FileExistsError(shard_root)
    shard_root.mkdir(parents=True)
    selection = shard_root / "STATIC_PAIR_SELECTION.json"
    selection_command = [
        str(args.python), str(SELECTOR),
        "--runtime-scene", str(paths["runtime"]),
        "--base-pose-root", str(paths["base"]),
        "--static-pnp-diagnostics", str(paths["diagnostics"]),
        "--component", " ".join(band),
        "--frames", " ".join(map(str, FRAMES)),
        "--romav2-root", str(args.romav2_root),
        "--romav2-weights", str(args.romav2_weights),
        "--device", "cuda",
        "--selection-frame", "0",
        "--output", str(selection),
    ]
    started = time.time()
    run_logged(
        selection_command,
        environment=environment,
        stdout=args.root / "logs" / f"band_{index}_selection.stdout.log",
        stderr=args.root / "logs" / f"band_{index}_selection.stderr.log",
    )
    selected = json.loads(selection.read_text())
    anchor = str(selected["anchor_target"])
    static_views = [str(view) for view in selected["selected_static_views"]]
    if STATIC_SUPPORT_COUNT > 2:
        remaining = sorted(
            (row for row in selected["candidate_rows"] if str(row["static_view"]) not in static_views),
            key=lambda row: (-int(row["accepted_count"]), str(row["static_view"])),
        )
        static_views.extend(str(row["static_view"]) for row in remaining[:STATIC_SUPPORT_COUNT - 2])
    if len(static_views) != STATIC_SUPPORT_COUNT:
        raise ValueError("not enough static candidates for the configured support count")
    source_cache = shard_root / "SOURCE_DUAL_STATIC_XYZ.pkl"
    source_command = [
        str(args.python), str(SOURCE_BUILDER),
        "--runtime-scene", str(paths["runtime"]),
        "--target-view", anchor,
        "--matching-quality", SOURCE_MATCHING_QUALITY,
        "--static-views", " ".join(static_views),
        "--frames", " ".join(map(str, FRAMES)),
        "--romav2-root", str(args.romav2_root),
        "--romav2-weights", str(args.romav2_weights),
        "--device", "cuda",
        "--query-stride", "8",
        "--minimum-overlap", "0.2",
        "--maximum-cycle-error-px", "1.5",
        "--output", str(source_cache),
    ]
    run_logged(
        source_command,
        environment=environment,
        stdout=args.root / "logs" / f"band_{index}_source.stdout.log",
        stderr=args.root / "logs" / f"band_{index}_source.stderr.log",
    )
    relay_cache = shard_root / "RELAY_DUAL_STATIC_XYZ.pkl"
    relay_command = [
        str(args.python), str(RELAY_BUILDER),
        "--runtime-scene", str(paths["runtime"]),
        "--component", " ".join(band),
        "--source-view", anchor,
        "--frames", " ".join(map(str, FRAMES)),
        "--source-xyz-cache", str(source_cache),
        "--romav2-root", str(args.romav2_root),
        "--romav2-weights", str(args.romav2_weights),
        "--device", "cuda",
        "--query-stride", "8",
        "--minimum-overlap", "0.2",
        "--maximum-cycle-error-px", "1.5",
        "--output", str(relay_cache),
    ]
    run_logged(
        relay_command,
        environment=environment,
        stdout=args.root / "logs" / f"band_{index}_relay.stdout.log",
        stderr=args.root / "logs" / f"band_{index}_relay.stderr.log",
    )
    with source_cache.open("rb") as handle:
        source_payload = pickle.load(handle)
    with relay_cache.open("rb") as handle:
        relay_payload = pickle.load(handle)
    result = {
        "band_index": index,
        "band": list(band),
        **runtime_shard_metadata(index, band),
        "gpu": gpu,
        "anchor": anchor,
        "static_views": static_views,
        "selection": {"path": str(selection), "sha256": sha256(selection)},
        "source": {"path": str(source_cache), "sha256": sha256(source_cache)},
        "relay": {"path": str(relay_cache), "sha256": sha256(relay_cache)},
        "runtime": {
            "selection_inference_seconds": selected["runtime"]["inference_seconds"],
            "source_inference_seconds": source_payload["runtime"]["inference_seconds"],
            "relay_inference_seconds": relay_payload["runtime"]["inference_seconds"],
            "wall_seconds": time.time() - started,
        },
    }
    write_json(shard_root / "STATUS.json", result)
    return result


def run_lane(
    args: argparse.Namespace, gpu: int, jobs: list[tuple[int, tuple[str, ...]]]
) -> list[dict[str, Any]]:
    return [run_band(args, index, band, gpu) for index, band in jobs]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.root.mkdir(parents=True, exist_ok=True)
    freeze(args)
    lanes = {gpu: [] for gpu in args.gpus}
    for index, band in enumerate(BANDS):
        lanes[args.gpus[index % len(args.gpus)]].append((index, band))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [executor.submit(run_lane, args, gpu, jobs) for gpu, jobs in lanes.items()]
        shards = [row for future in futures for row in future.result()]
    shards.sort(key=lambda row: row["band_index"])
    observations = {}
    inference = 0.0
    for shard in shards:
        with Path(shard["relay"]["path"]).open("rb") as handle:
            payload = pickle.load(handle)
        expected = {
            (view, frame, role)
            for view in shard["band"]
            for frame in FRAMES
            for role in ("fit", "confirm", "final")
        }
        if set(payload["observations"]) != expected or set(observations) & expected:
            raise RuntimeError(f"relay coverage mismatch for band {shard['band_index']}")
        observations.update(payload["observations"])
        inference += sum(
            float(shard["runtime"][key])
            for key in (
                "selection_inference_seconds",
                "source_inference_seconds",
                "relay_inference_seconds",
            )
        )
    counts = [
        len(np.asarray(row["point_ids"]).reshape(-1)) for row in observations.values()
    ]
    payload = {
        "protocol": "track_closure-conditioned-dual-static-band-relay-merged-v1",
        "status": "complete_public_same_time_XYZ_cache",
        "observations": observations,
        "shards": shards,
        "coverage": {
            "node_role_count": len(observations),
            "minimum_points": int(min(counts)),
            "median_points": float(np.median(counts)),
            "node_roles_at_least_12": int(np.count_nonzero(np.asarray(counts) >= 12)),
        },
        "runtime": {
            "summed_inference_seconds": inference,
            "single_model_equivalent_seconds_per_frame": inference / 30.0,
            "single_model_equivalent_inference_seconds_per_frame": inference / 30.0,
            "model_load_excluded": True,
            "pair_budget": relay_pair_budget(),
        },
        "attestation": {
            "GT_used": False,
            "NVS_used": False,
            "test_RGB_or_cameras_used": False,
            "reference_pose_used": False,
            "same_frame_local_XYZ_relayed_within_each_band": True,
        },
    }
    output = args.root / "CONDITIONED_BAND_RELAY_XYZ_CACHE.pkl"
    write_pickle(output, payload)
    write_json(
        args.root / "CONDITIONED_BAND_RELAY_XYZ_CACHE.json",
        {key: value for key, value in payload.items() if key != "observations"},
    )
    result = {
        "status": "complete",
        "cache": str(output),
        "cache_sha256": sha256(output),
        "coverage": payload["coverage"],
        "runtime": payload["runtime"],
        "shards": shards,
    }
    write_json(args.root / "RUN_STATUS.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--romav2-root", type=Path, required=True)
    parser.add_argument("--romav2-weights", type=Path, required=True)
    args = parser.parse_args()
    for field in ("scene_root", "root", "python", "romav2_root", "romav2_weights"):
        setattr(args, field, getattr(args, field).expanduser().absolute())
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
