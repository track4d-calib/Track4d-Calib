#!/usr/bin/env python3
"""Calibrate a sequence and evaluate its reconstruction with one configuration."""

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track4d_calib.config import LEVELS, evaluation_settings, load_config
from track4d_calib.evaluation import verify_calibration
from track4d_calib.runner import digest, write_json

# Use independent 30-frame windows to support parallel reconstruction jobs.
WINDOW_SIZE = 30
METRICS = ("psnr", "ssim", "lpips_alex")


def aggregate(results: list[dict], frames: int, test_views: list[str]) -> dict:
    """Require one measurement per test camera and source frame."""
    rows = [row for result in results for row in result["rows"]]
    expected = {(view, frame) for view in test_views for frame in range(frames)}
    observed = [(str(row["view"]), int(row["frame"])) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("evaluation has missing, duplicate or unexpected camera/frame rows")
    if any(not math.isfinite(float(row[key])) for row in rows for key in METRICS):
        raise ValueError("evaluation contains nonfinite metrics")

    def summarize(group: list[dict]) -> dict:
        summary = {"frames": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            summary.update({f"mean_{metric}": statistics.mean(values),
                            f"median_{metric}": statistics.median(values),
                            f"std_{metric}": statistics.pstdev(values)})
        return summary

    return {"status": "complete", "overall": summarize(rows),
            "by_view": {view: summarize([row for row in rows if str(row["view"]) == view])
                        for view in test_views},
            "rows": sorted(rows, key=lambda row: (int(row["frame"]), str(row["view"]))) }


def remove_checkpoints(window: Path) -> None:
    """Remove only this evaluation's checkpoints after its metrics are verified."""
    model = window / "nvs/readout/nvs/model_seed0"
    records = []
    for path in sorted(model.glob("chkpnt*.pth")):
        if path.is_symlink() or not path.resolve().is_relative_to(window.resolve()):
            raise ValueError(f"checkpoint is outside this evaluation: {path}")
        records.append({"path": str(path.relative_to(window)),
                        "bytes": path.stat().st_size, "sha256": digest(path)})
    write_json(window / "checkpoint_retention.json", {"policy": "remove_after_evaluation", "files": records})
    for record in records:
        (window / record["path"]).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--level", choices=LEVELS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--calibration-run", type=Path, help="Use an existing completed calibration")
    parser.add_argument("--frames", type=int, choices=(30, 120), default=120)
    parser.add_argument("--canonical-scene", type=Path)
    parser.add_argument("--fudan-runtime", type=Path, default=Path("third_party/4dgs"))
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--dinov3-src", type=Path, default=Path("third_party/dinov3"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print commands without creating outputs")
    args = parser.parse_args()
    if args.source is None and args.calibration_run is None:
        parser.error("--source is required when starting a calibration")
    config = load_config(args.config.resolve(), level=args.level, frames=args.frames)
    settings = evaluation_settings(config)
    if settings["camera_source"] == "external" and args.canonical_scene is None:
        parser.error("external evaluation cameras require --canonical-scene")
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    output = args.output.expanduser().absolute()
    calibration = args.calibration_run.resolve() if args.calibration_run else output / "calibration"
    config_path = output / "settings.json"
    steps = []
    if not args.calibration_run:
        steps.append({"name": "calibration", "command": [
            sys.executable, "-m", "track4d_calib", "run", "--config", str(config_path),
            "--source", str(args.source.resolve()), "--output", str(calibration),
            "--dependencies", str(args.dependencies.resolve()), "--frames", str(args.frames),
            "--gpu", str(args.gpu),
        ]})
    for start in range(0, args.frames, WINDOW_SIZE):
        window = output / f"evaluation/f{start:06d}"
        command = [sys.executable, "-m", "track4d_calib.evaluation", "--run", str(calibration),
                   "--output", str(window), "--config", str(config_path),
                   "--dependencies", str(args.dependencies.resolve()),
                   "--frame-start", str(start), "--gpu", str(args.gpu)]
        for name in ("fudan_runtime", "train_python", "renderer", "dinov3_src", "canonical_scene"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value.expanduser().absolute())])
        steps.append({"name": f"evaluation_{start:06d}", "frame_ids": list(range(start, start + WINDOW_SIZE)),
                      "output": str(window), "command": command})
    plan = {"frames": args.frames, "window_size": WINDOW_SIZE, "settings": settings,
            "calibration_settings": config,
            "input_config_sha256": digest(args.config.resolve()), "steps": steps,
            "runner_sha256": digest(Path(__file__)),
            "checkpoint_retention": "keep" if args.keep_checkpoints else "remove_after_evaluation"}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if args.calibration_run:
        result = verify_calibration(calibration)
        saved = json.loads((calibration / "protocol.json").read_text())["config"]
        if result["frames"] != args.frames or any(saved[key] != config[key] for key in ("scene", "static_views", "dynamic_views")):
            raise ValueError("existing calibration does not match the requested sequence")
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation").mkdir()
    write_json(config_path, config)
    write_json(output / "protocol.json", plan)
    started = time.perf_counter()
    results, timings = [], []
    try:
        for step in steps:
            write_json(output / "status.json", {"status": "running", "stage": step["name"]})
            stage_started = time.perf_counter()
            with (output / (step["name"] + ".log")).open("x") as log:
                subprocess.run(step["command"], check=True, stdout=log, stderr=subprocess.STDOUT)
            timings.append({"stage": step["name"], "wall_seconds": time.perf_counter() - stage_started})
            write_json(output / "timings.json", timings)
            if "output" in step:
                window = Path(step["output"])
                result = json.loads((window / "result.json").read_text())
                if result["status"] != "completed" or result["frame_ids"] != step["frame_ids"]:
                    raise ValueError("window evaluation did not complete the requested frames")
                local = [{**row, "frame": int(row["frame"]) - step["frame_ids"][0]}
                         for row in result["rows"]]
                aggregate([{"rows": local}], WINDOW_SIZE, settings["test_views"])
                results.append(result)
                if not args.keep_checkpoints:
                    remove_checkpoints(window)
        summary = aggregate(results, args.frames, settings["test_views"])
        summary.update({"calibration": str(calibration), "windows": len(results),
                        "wall_seconds": time.perf_counter() - started, "timings": timings})
        write_json(output / "result.json", summary)
        write_json(output / "status.json", {"status": "complete"})
        print(json.dumps(summary["overall"], indent=2))
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
