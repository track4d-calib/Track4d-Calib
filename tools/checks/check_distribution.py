#!/usr/bin/env python3
"""Verify a wheel's contents and installed commands outside the source checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("track4d_calib/_sources.json"))["files"]
        for name, digest in manifest.items():
            if name not in names or hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise SystemExit(f"Missing or changed wheel source: {name}")
        for name in names:
            if name.endswith((".pyc", ".local.json")) or "/__pycache__/" in name:
                raise SystemExit(f"Unexpected wheel content: {name}")
        expected = {name for name in names if name.startswith("track4d_calib/") and name.endswith((".py", ".json"))}
        if expected != set(manifest) | {"track4d_calib/_sources.json"}:
            raise SystemExit("Wheel contains untracked package sources")
    with tempfile.TemporaryDirectory(prefix="track4d_calib-wheel-") as directory:
        root = Path(directory)
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root / "env")
        python = root / "env/bin/python"
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                        "--force-reinstall", str(wheel)], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run([str(python), "-c",
                        "import track4d_calib,sys; from pathlib import Path; "
                        "assert Path(track4d_calib.__file__).is_relative_to(sys.prefix); "
                        "from importlib.metadata import version; "
                        "assert track4d_calib.__version__ == version('track4d-calib')"],
                       cwd=root, check=True)
        for command in ("track4d-calib", "track4d-calib-evaluate"):
            subprocess.run([str(root / "env/bin" / command), "--help"],
                           cwd=root, check=True, stdout=subprocess.DEVNULL)
        config = root / "scene.json"
        config.write_text(json.dumps({
            "scene": "example", "static_views": [f"{i:02d}" for i in range(16)],
            "dynamic_views": ["80", "81", "82"], "resolution": [1280, 720],
            "frames": 30, "center_radius": None,
        }))
        dependencies = root / "dependencies.json"
        dependencies.write_text(json.dumps({key: str(root / key) for key in
            ("hloc", "cotracker", "romav2_src", "cotracker_checkpoint", "romav2_weights")}))
        result = subprocess.run([str(root / "env/bin/track4d-calib"), "plan",
            "--config", str(config), "--dependencies", str(dependencies),
            "--source", str(root / "scene"), "--output", str(root / "output")],
            cwd=root, check=True, capture_output=True, text=True)
        plan = json.loads(result.stdout)
        if plan["config"]["frames"] != 30 or (root / "output").exists():
            raise SystemExit("Installed plan command violated its contract")
        payload = json.loads(config.read_text())
        payload["levels"] = {
            "fast": {"temporal_anchor_spacing": 29},
            "balanced": {"temporal_anchor_spacing": 15},
            "quality": {"temporal_anchor_spacing": 8},
        }
        config.write_text(json.dumps(payload))
        for level, count in (("fast", 2), ("balanced", 3), ("quality", 5)):
            result = subprocess.run([str(root / "env/bin/track4d-calib"), "plan",
                "--config", str(config), "--level", level,
                "--dependencies", str(dependencies), "--source", str(root / "scene"),
                "--output", str(root / "output")],
                cwd=root, check=True, capture_output=True, text=True)
            resolved = json.loads(result.stdout)["config"]
            if resolved["level"] != level or resolved["temporal_anchor_count"] != count:
                raise SystemExit("Installed calibration level resolved incorrectly")
    print(f"Verified {len(manifest)} wheel sources, installed entry points and configuration plan.")


if __name__ == "__main__":
    main()
