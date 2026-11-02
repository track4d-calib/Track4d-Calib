"""Build window-local relay tracks and validate their source cameras."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace

try:
    from pipeline import sha256, write_json
except ModuleNotFoundError:
    from scripts.pipeline import sha256, write_json


POLICY = "single_fresh_source_refresh_per_low_support_window_v1"
CONFIG_ENV = "TRACK4D_CALIB_WINDOW_RELAY_CONFIG"
HERE = Path(__file__).resolve()


def plan_windows(payload, *, bands, frames, minimum=12, window_size=5, static_support_count=2):
    """Schedule once by support counts, never by held-out residuals or NVS."""
    frames = tuple(frames)
    if not frames or frames != tuple(range(len(frames))) or len(frames) % window_size:
        raise ValueError("repair requires the complete ordered frame grid")
    jobs = []
    observations = payload["observations"]
    if type(static_support_count) is not int or not 2 <= static_support_count <= 8:
        raise ValueError("static support count must be an integer from 2 to 8")
    for index, band in enumerate(bands):
        for offset in range(0, len(frames), window_size):
            window = frames[offset:offset + window_size]
            if all(len(observations[v, f, r]["point_ids"]) >= minimum
                   for v in band for f in window for r in ("fit", "confirm", "final")):
                continue
            selected = min(window, key=lambda f: (
                sum(len(observations[v, f, "fit"]["point_ids"]) for v in band), f))
            jobs.append({"band_index": index, "band": list(band), "frames": list(window),
                         "selection_frame": selected,
                         "additional_pairs": 8 + len(window) * (static_support_count + len(band)-1)})
    return jobs


def add_repair_budget(budget, additional_pairs, frame_count):
    if additional_pairs < 0 or int(additional_pairs) != additional_pairs or frame_count <= 0:
        raise ValueError("invalid repair matching budget")
    if budget.get("repair_additional_pairs", 0):
        raise ValueError("repair matching already accounted")
    result = dict(budget)
    result["repair_additional_pairs"] = int(additional_pairs)
    result["total_pairs"] += int(additional_pairs)
    result["normalized_matching_seconds_per_frame"] = result["total_pairs"] * .15 / frame_count
    return result


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_window(config, job, root):
    """Fresh RGB matching through the existing serialized, paced GPU lane."""
    if not os.environ.get("TRACK4D_CALIB_PIPELINE_CONFIG") or not os.environ.get("TRACK4D_CALIB_EXTERNAL_OPEN_AUDIT"):
        raise ValueError("relay repair requires GPU timing and native process-tree input audit")
    root.mkdir(parents=True, exist_ok=False)
    method = HERE.parents[1] / "components/track_closure"
    sys.path[:0] = [str(method), str(method / "geometry")]
    builder = load(method / "build_conditioned_relay.py", "window_relay_builder")
    builder.FRAMES = tuple(job["frames"])
    builder.scene_paths = lambda unused: {
        "runtime": Path(config["runtime"]), "base": Path(config["base"]),
        "diagnostics": Path(config["diagnostics"])}
    original_run = builder.run_logged

    def run_logged(command, **kwargs):
        command = list(command)
        if "--selection-frame" in command:
            command[command.index("--selection-frame") + 1] = str(job["selection_frame"])
        # This launcher counts every actual match call; model loads are separate
        # timed GPU services, not hidden inside the 0.15s/pair quota.
        command = [command[0], str(HERE.with_name("run_relay.py")),
                   "--stage-script", command[1], *command[2:]]
        return original_run(command, **kwargs)

    builder.run_logged = run_logged
    args = SimpleNamespace(root=root, scene_root=Path(config["run_root"]), scene="declared_rig",
                           python=Path(sys.executable), romav2_root=Path(config["romav2_root"]),
                           romav2_weights=Path(config["romav2_weights"]))
    result = builder.run_band(args, job["band_index"], tuple(job["band"]), config["gpu"])
    result.update(selection_frame=job["selection_frame"], frames=job["frames"],
                  additional_pairs=job["additional_pairs"])
    write_json(root / "RESULT.json", result)
    return result


def refresh_low_support(config_path, relay_path, output, *, bands, frames, validator):
    """Refresh scheduled windows before fit-only shared-center estimation."""
    started = time.perf_counter()
    config = json.loads(Path(config_path).read_text())
    root = Path(config["run_root"]).resolve()
    for key in ("base", "diagnostics", "relay"):
        if not Path(config[key]).resolve().is_relative_to(root):
            raise ValueError(f"repair {key} is outside the current run")
    if Path(config["relay"]).resolve() != relay_path.resolve() or sha256(relay_path) != config["relay_sha256"]:
        raise ValueError("original current-run relay changed")
    if not output.resolve().is_relative_to(root):
        raise ValueError("repair output escapes current run")
    output.mkdir(parents=True, exist_ok=False)
    with relay_path.open("rb") as stream:
        payload = pickle.load(stream)
    roles = ("fit", "confirm", "final")
    validator(payload, bands=bands, frames=frames, roles=roles)
    from rig_contract import ENVIRONMENT_KEY, load_rig_contract
    rig = load_rig_contract(Path(os.environ[ENVIRONMENT_KEY]))
    jobs = plan_windows(payload, bands=bands, frames=frames,
                        static_support_count=rig.get("static_support_count", 2))
    plan = {"policy": POLICY, "jobs": jobs, "maximum_attempts_per_window": 1,
            "selection_uses_fit_counts_only": True, "trigger_uses_role_availability_not_residuals": True,
            "original_relay": str(relay_path), "original_relay_sha256": sha256(relay_path),
            "centers_frozen_before_refresh": False, "additional_pairs": sum(j["additional_pairs"] for j in jobs)}
    write_json(output / "PLAN.json", plan)
    records = []
    payload["calibration_window_sources"] = []
    for index, job in enumerate(jobs):
        job_path = output / f"job_{index}.json"
        write_json(job_path, job)
        window_root = output / f"window_{index}"
        command = [sys.executable, str(HERE), "--config", str(config_path), "--job", str(job_path),
                   "--root", str(window_root)]
        with (output / f"window_{index}.log").open("xb") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        record = json.loads((window_root / "RESULT.json").read_text())
        with Path(record["relay"]["path"]).open("rb") as stream:
            fresh = pickle.load(stream)
        validator(fresh, bands=[job["band"]], frames=job["frames"], roles=roles)
        payload["observations"].update(fresh["observations"])
        payload["calibration_window_sources"].append({"band": job["band"], "frames": job["frames"],
                                                "anchor": record["anchor"]})
        records.append(record)
    audit = validate_windowed_relay(payload, original=validator, bands=bands, frames=frames, roles=roles)
    result_path = relay_path
    if jobs:
        runtime = payload["runtime"]
        inference = sum(row["runtime"][k] for row in records for k in (
            "selection_inference_seconds", "source_inference_seconds", "relay_inference_seconds"))
        runtime["summed_inference_seconds"] += inference
        for key in ("single_model_equivalent_seconds_per_frame", "single_model_equivalent_inference_seconds_per_frame"):
            runtime[key] = runtime["summed_inference_seconds"] / len(frames)
        runtime["window_repair_additional_pairs"] = plan["additional_pairs"]
        if "pair_budget" in runtime:
            budget = dict(runtime["pair_budget"])
            budget["window_repair_additional_pairs"] = plan["additional_pairs"]
            budget["total_pairs"] += plan["additional_pairs"]
            budget["equivalent_seconds_per_frame"] = budget["total_pairs"] * .15 / len(frames)
            runtime["pair_budget"] = budget
        result_path = output / "CURRENT_RUN_REPAIRED_RELAY.pkl"
        with result_path.open("xb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    result = {**plan, "relay": str(result_path), "relay_sha256": sha256(result_path),
              "records": records, "validation": audit, "wall_seconds": time.perf_counter() - started,
              "full_geometric_verify_still_required": True}
    write_json(output / "RESULT.json", result)
    return result


def validate_windowed_relay(payload, *, original, bands, frames, roles, **kwargs):
    windows = payload.get("calibration_window_sources", [])
    if not windows:
        return original(payload, bands=bands, frames=frames, roles=roles, **kwargs)
    bands = tuple(tuple(map(str, band)) for band in bands)
    frames = tuple(map(int, frames))
    observations = payload["observations"]
    expected = {(v, f, r) for band in bands for v in band for f in frames for r in roles}
    if set(observations) != expected:
        raise ValueError("relay cache coverage mismatch")
    sources = {}
    for shard in payload["shards"]:
        band = tuple(map(str, shard["component"]))
        if band not in bands or band in sources:
            raise ValueError("original relay source coverage mismatch")
        sources[band] = str(shard["anchor"])
    if set(sources) != set(bands):
        raise ValueError("original relay source coverage mismatch")
    replaced = set()
    partitions = []
    for window in windows:
        band = tuple(map(str, window["band"]))
        selected = tuple(map(int, window["frames"]))
        if band not in bands or not selected or len(set(selected)) != len(selected):
            raise ValueError("invalid relay repair window")
        if not set(selected) <= set(frames):
            raise ValueError("relay repair escapes declared frames")
        keys = {(band, f) for f in selected}
        if replaced & keys:
            raise ValueError("overlapping relay repair windows")
        replaced.update(keys)
        partitions.append((band, selected, str(window["anchor"])))
    for band in bands:
        remaining = tuple(f for f in frames if (band, f) not in replaced)
        if remaining:
            partitions.append((band, remaining, sources[band]))
    checks = []
    for band, selected, source in partitions:
        local = {"component": band, "source_view": source, "observations": {
            (v, f, r): observations[v, f, r] for v in band for f in selected for r in roles}}
        checks.append(original(local, bands=[band], frames=selected, roles=roles, **kwargs))
    weak = {tuple(key) for check in checks for key in check["below_minimum_keys"]}
    blocked = []
    for band in bands:
        for start in range(0, len(frames), 5):
            window = frames[start:start + 5]
            keys = sorted(k for k in weak if k[0] in band and k[1] in window)
            if keys:
                blocked.append({"component": list(band), "window": list(window),
                                "blocked_node_roles": list(map(list, keys))})
    report = dict(checks[0])
    report.update(bands=list(map(list, bands)), node_role_count=len(expected),
                  minimum_points_per_node_role=min(c["minimum_points_per_node_role"] for c in checks),
                  node_roles_at_least_minimum=len(expected) - len(weak),
                  node_roles_below_minimum=len(weak), below_minimum_keys=list(map(list, sorted(weak))),
                  blocked_component_windows=blocked, blocked_component_window_count=len(blocked),
                  all_node_roles_have_minimum_support=not weak,
                  support_status="pass_all_supported" if not weak else "pass_with_recoverable_component_fallback",
                  source_by_component=None, relay_source_identity_declared=True,
                  source_identity_scope="per_window_partition",
                  source_partitions=[{"band": list(b), "frames": list(f), "anchor": s}
                                     for b, f, s in partitions],
                  source_partition_node_roles=sum(r["node_role_count"] for r in checks))
    if report["source_partition_node_roles"] != report["node_role_count"]:
        raise ValueError("relay source partitions do not cover the whole grid")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run_window(json.loads(args.config.read_text()), json.loads(args.job.read_text()), args.root)
