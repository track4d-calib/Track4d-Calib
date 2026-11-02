#!/usr/bin/env python3
"""Decode a camera-video package into the calibration image layout."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track4d_calib.config import load_config


def decode_view(package, destination, view, record, config, ffmpeg):
    video = package / "videos" / f"{view}.mkv"
    directory = destination / "images" / view
    directory.mkdir(parents=True)
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-i", str(video),
         "-map", "0:v:0", "-pix_fmt", "rgb24", "-vsync", "0", "-threads", "1",
         "-start_number", "0", str(directory / "%06d.png"),
         "-map", "0:v:0", "-pix_fmt", "rgb24", "-vsync", "0", "-threads", "1",
         "-f", "hash", "-hash", "sha256", "-"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip() != "SHA256=" + record["rgb_sha256"]:
        raise ValueError(f"Decoded RGB checksum differs for camera {view}")
    images = sorted(directory.glob("*.png"))
    if [path.name for path in images] != [f"{frame:06d}.png" for frame in range(config["frames"])]:
        raise ValueError(f"Frame sequence differs for camera {view}")
    for path in images:
        with path.open("rb") as image:
            header = image.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or list(struct.unpack(">II", header[16:24])) != config["resolution"]:
            raise ValueError(f"Unexpected image dimensions: {path}")
    return view


def extract(package, output, ffmpeg, workers):
    config = load_config(package / "config.json")
    manifest = json.loads((package / "manifest.json").read_text())
    views = config["static_views"] + config["dynamic_views"]
    if manifest.get("format") != "h264-rgb-lossless" or manifest.get("frames") != config["frames"]:
        raise ValueError("Video manifest does not match the scene configuration")
    if set(manifest["videos"]) != set(views) or manifest.get("resolution") != config["resolution"]:
        raise ValueError("Video manifest does not match the camera roster or resolution")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".decode-", dir=output.parent) as temporary:
        scene = Path(temporary) / "scene"
        scene.mkdir()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(decode_view, package, scene, view,
                                   manifest["videos"][view], config, ffmpeg) for view in views]
            for future in futures:
                print(f"Decoded camera {future.result()}", flush=True)
        (scene / "cameras").mkdir()
        shutil.copy2(package / "cameras/intri.yml", scene / "cameras/intri.yml")
        shutil.copy2(package / "config.json", scene / "config.json")
        scene.rename(output)
    print(f"Ready: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        extract(args.package.resolve(), args.output.absolute(), args.ffmpeg, args.workers)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
