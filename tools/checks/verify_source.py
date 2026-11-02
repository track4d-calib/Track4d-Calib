#!/usr/bin/env python3
"""Check source hashes and the track-closure component manifest."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "track4d_calib/_sources.json"


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "track4d_calib").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"} and path != MANIFEST
    }


def closure_identity(engine: Path) -> dict:
    component = engine / "runtime/components/track_closure"
    sys.path.insert(0, str(component))
    try:
        spec = importlib.util.spec_from_file_location("closure_source_check", component / "closure_component.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        digest, count = module.algorithm_source_digest()
        return {"tree_digest": digest, "file_count": count}
    finally:
        sys.path.remove(str(component))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Record the current source tree after review")
    args = parser.parse_args()
    engine = ROOT / "track4d_calib/_engine"
    path = engine / "runtime/components/track_closure/VERSION_MANIFEST.json"
    with tempfile.TemporaryDirectory(prefix="track4d_calib-source-") as directory:
        snapshot = Path(directory) / "engine"
        shutil.copytree(engine, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        identity = closure_identity(snapshot)
    component = json.loads(path.read_text())
    if args.update:
        component["algorithm_source"] = identity
        path.write_text(json.dumps(component, indent=2) + "\n")
        MANIFEST.write_text(json.dumps({"schema": 1, "files": source_hashes()}, indent=2) + "\n")
    if component["algorithm_source"] != identity:
        raise SystemExit("Track-closure source does not match VERSION_MANIFEST.json")
    expected = json.loads(MANIFEST.read_text())["files"]
    actual = source_hashes()
    differences = [key for key in sorted(expected.keys() | actual.keys()) if expected.get(key) != actual.get(key)]
    if differences:
        raise SystemExit("Source manifest differs: " + ", ".join(differences))
    print(f"Verified {len(actual)} source files and the {identity['file_count']}-file closure component.")


if __name__ == "__main__":
    main()
