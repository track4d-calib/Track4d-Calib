#!/usr/bin/env python3
"""Overlap independent relay-band CPU work on one serialized GPU lane.

Selection -> source triangulation -> PTZ relay remains ordered within every
band. The cache builder still assembles and verifies the complete cache.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
METHOD = HERE.parents[1] / "components/track_closure"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_frame_count(module, frame_count: int) -> None:
    """Adapt sequence scope without editing the relay geometry source."""

    if frame_count not in (30, 120):
        raise ValueError("relay sequence must contain 30 or 120 frames")
    module.FRAMES = tuple(range(frame_count))
    module.relay_pair_budget = partial(
        module.relay_pair_budget, bands=module.BANDS, frames=module.FRAMES
    )
    if frame_count == 30:
        return
    original_write_pickle = module.write_pickle

    def write_pickle(path, payload):


        runtime = payload["runtime"]
        per_frame = float(runtime["summed_inference_seconds"]) / frame_count
        runtime["single_model_equivalent_seconds_per_frame"] = per_frame
        runtime["single_model_equivalent_inference_seconds_per_frame"] = per_frame
        runtime["frame_count"] = frame_count
        original_write_pickle(path, payload)

    module.write_pickle = write_pickle


def validate_stage(path: Path) -> Path:
    stage = path.resolve()
    allowed = {METHOD / "chain" / name for name in (
        "select_band_static_pair_nullsafe.py", "build_dual_static_target_query_xyz_cache.py",
        "build_dual_static_xyz_ptz_relay_cache.py")}
    if stage not in allowed:
        raise ValueError("unknown relay stage")
    return stage


def install_roma_cache(factory, service, *, retain):
    """Cache inference models, never match results; keep every pair on the lane."""
    original = factory._load_roma_model
    cache, stats = {}, {"constructed": 0, "reused": 0}

    def timed_model(*, source_root, weights, device, setting=None, force_bidirectional=False):
        key = (str(source_root.resolve()), str(weights.resolve()), str(device), setting, force_bidirectional)
        if retain and key in cache:
            stats["reused"] += 1
            return cache[key]
        model, resolved_device = service.call(
            "model_load", 0, original, source_root=source_root, weights=weights,
            device=device, setting=setting, force_bidirectional=force_bidirectional)
        match = model.match

        def timed_match(*args, **kwargs):
            return service.call("match", 1, match, *args, **kwargs)

        model.match = timed_match
        stats["constructed"] += 1
        if retain:
            cache[key] = model, resolved_device
        return model, resolved_device

    factory._load_roma_model = timed_model
    return stats


def run_resident_band(args, index, band, gpu, frame_count):
    """Run support selection, XYZ construction and relay assembly in one process."""
    from resident_frontend import redirected_logs
    from pipeline import write_json
    from gpu_timing import recorder

    audit = os.environ.get("TRACK4D_CALIB_EXTERNAL_OPEN_AUDIT")
    if not audit or not Path(audit).is_file():
        raise ValueError("resident relay requires the outer native process-tree audit")
    if not os.path.samefile(args.python, sys.executable):
        raise ValueError("resident relay must use the declared Python")
    # No CUDA/BLAS imports precede the owning GPU/thread environment.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                      MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    sys.path.insert(0, str(METHOD))
    sys.path.insert(0, str(METHOD / "geometry"))
    from scripts.geometry import export_static_pnp_inlier_seeded_track_closure as factory
    service = recorder()
    if service is None:
        raise ValueError("resident relay requires a GPU service ledger")
    stats = install_roma_cache(factory, service, retain=True)
    module = load(METHOD / "build_conditioned_relay.py", "calibration_resident_relay_builder")
    configure_frame_count(module, frame_count)

    def run_logged(command, *, environment, stdout, stderr):
        stage = validate_stage(Path(command[1]))
        stdout.parent.mkdir(parents=True, exist_ok=True)
        previous = sys.argv
        try:
            os.environ.update(environment)
            with redirected_logs(stdout, stderr):
                sys.argv = [str(stage), *command[2:]]
                child = load(stage, "calibration_resident_relay_stage")
                if code := int(child.main()):
                    raise RuntimeError({"command": command, "exit_code": code})
        finally:
            sys.argv = previous

    module.run_logged = run_logged
    result = module.run_band(args, index, band, gpu)
    write_json(args.root / "shards" / f"band_{index}" / "RESIDENT_EXECUTION.json", {
        "pid": os.getpid(), "model_cache": stats,
        "scope": "current band process only; no image/match/track/pose result reuse",
        "native_open_audit": str(audit),
    })
    return result


def run_isolated_band(args, index, band, gpu, frame_count):
    """Give each band its own spawned process and model lifetime."""
    with ProcessPoolExecutor(max_workers=1,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(run_resident_band, args, index, band, gpu, frame_count).result()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage-script", type=Path)
    parser.add_argument("--pipeline-workers", type=int)
    parser.add_argument("--num-frames", type=int, choices=(30, 120), default=30)
    options, remaining = parser.parse_known_args()
    if options.pipeline_workers is None:
        quality = os.environ.get("TRACK4D_CALIB_RELAY_SOURCE_QUALITY", "fast")
        # Precise source models need substantially more resident GPU memory.
        options.pipeline_workers = 1 if quality == "precise" else 4
    if options.pipeline_workers < 1:
        raise ValueError("pipeline workers must be positive")
    sys.path.insert(0, str(METHOD))
    sys.path.insert(0, str(METHOD / "geometry"))
    if options.stage_script is not None:
        stage = validate_stage(options.stage_script)
        from gpu_timing import recorder
        service = recorder()
        if service is None:
            raise ValueError("relay pipeline requires the declared GPU service ledger")
        from scripts.geometry import export_static_pnp_inlier_seeded_track_closure as factory
        install_roma_cache(factory, service, retain=False)
        module = load(stage, "calibration_relay_stage")
        sys.argv = [str(stage), *remaining]
        return int(module.main())

    module = load(METHOD / "build_conditioned_relay.py", "calibration_frozen_relay")
    configure_frame_count(module, options.num_frames)
    original_run_logged = module.run_logged

    def run_logged(command, **kwargs):
        return original_run_logged(
            [command[0], str(HERE), "--stage-script", command[1], *command[2:]], **kwargs)

    def run_lane(args, gpu, jobs):
        if os.environ.get("TRACK4D_CALIB_EXTERNAL_OPEN_AUDIT"):
            # One job per worker prevents adapters/models from crossing bands.
            # Independent bands overlap CPU work; actual GPU calls still flock.
            with ThreadPoolExecutor(max_workers=min(options.pipeline_workers, len(jobs))) as pool:
                futures = [pool.submit(run_isolated_band, args, index, band, gpu, options.num_frames)
                           for index, band in jobs]
                return [future.result() for future in futures]
        with ThreadPoolExecutor(max_workers=min(options.pipeline_workers, len(jobs))) as pool:
            futures = [pool.submit(module.run_band, args, index, band, gpu) for index, band in jobs]
            return [future.result() for future in futures]

    module.run_logged = run_logged
    module.run_lane = run_lane
    sys.argv = [str(module.__file__), *remaining]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
