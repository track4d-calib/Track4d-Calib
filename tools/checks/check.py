#!/usr/bin/env python3
"""Check Python syntax, dataset settings, entry points and source integrity."""

import ast
import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
import json
import re
from urllib.parse import unquote, urlsplit
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track4d_calib.config import LEVELS, LEVEL_SETTINGS, evaluation_settings, load_config, load_dependencies
from track4d_calib.runner import command


def isolated_band_probe(args, index, band, gpu, frames):
    if index < 0:
        raise RuntimeError("worker failure probe")
    global _probe_calls
    _probe_calls = globals().get("_probe_calls", 0) + 1
    return index, band, gpu, frames, _probe_calls, os.getpid()


def check_process_isolation():
    path = ROOT / "track4d_calib/_engine/runtime/scripts/run_relay.py"
    spec = importlib.util.spec_from_file_location("relay_check", path)
    relay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relay)
    relay.run_resident_band = isolated_band_probe
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(relay.run_isolated_band, None, i, "band", "cpu", 30)
                   for i in range(4)]
        rows = [future.result() for future in futures]
    if any(row[:5] != (i, "band", "cpu", 30, 1) for i, row in enumerate(rows)):
        raise SystemExit("Relay worker state or arguments crossed band boundaries")
    if len({row[5] for row in rows}) != len(rows):
        raise SystemExit("Relay bands reused a worker process")
    try:
        relay.run_isolated_band(None, -1, "band", "cpu", 30)
    except RuntimeError as error:
        if str(error) != "worker failure probe":
            raise
    else:
        raise SystemExit("Relay worker failure was not propagated")


def main() -> None:
    check_process_isolation()
    subprocess.run([sys.executable, str(ROOT / "tools/checks/check_crop_verification.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/checks/check_track_validation.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/checks/check_rgb_motion.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/checks/check_static_triangulation.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/checks/check_motion_config.py")], check=True)
    sources = sorted((ROOT / "track4d_calib").rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    third_party = json.loads((ROOT / "dependencies/third-party.json").read_text())
    for name, source in third_party["sources"].items():
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise SystemExit(f"Invalid dependency revision: {name}")
        if not (ROOT / "dependencies" / source["source_manifest"]).is_file():
            raise SystemExit(f"Missing dependency source manifest: {name}")
    dependencies = load_dependencies(ROOT / "configs/dependencies.example.json")
    presets = [
        path
        for path in sorted((ROOT / "configs").glob("*.json"))
        if not path.name.startswith("dependencies.") and not path.name.endswith(".local.json")
    ]
    for path in presets:
        config = load_config(path)
        if "levels" in json.loads(path.read_text()):
            derived = {"level", "motion_model", "track_validation"}
            invariant = set(config) - LEVEL_SETTINGS - derived
            for level in LEVELS:
                resolved = load_config(path, level=level)
                if any(resolved[key] != config[key] for key in invariant):
                    raise SystemExit(f"Calibration level changed scene or evaluation: {path.name}/{level}")
        if "evaluation" in config:
            evaluation_settings(config)
        argv = command(
            config,
            source=Path("scene"),
            output=Path("output"),
            code=ROOT / "track4d_calib/_engine",
            dependencies=dependencies,
            gpu=0,
            frames=config["frames"],
            pace=0.0,
        )
        if not Path(argv[1]).is_file():
            raise SystemExit(f"Missing calibration entry point: {argv[1]}")
    documents = [*sorted(ROOT.glob("*.md")), *sorted((ROOT / "docs").glob("*.md")),
                 *sorted((ROOT / "tools").rglob("*.md")),
                 ROOT / "third_party/README.md"]
    for path in documents:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            link = urlsplit(target)
            if link.scheme or not link.path:
                continue
            if not (path.parent / unquote(link.path)).exists():
                raise SystemExit(f"Broken link in {path.relative_to(ROOT)}: {target}")
    for module in ("track4d_calib", "track4d_calib.evaluation"):
        subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run([sys.executable, str(ROOT / "tools/checks/verify_source.py")], check=True)
    print(
        f"Checked {len(sources)} Python files, {len(presets)} presets, {len(documents)} documents, both command-line entry points and relay process isolation."
    )


if __name__ == "__main__":
    main()
