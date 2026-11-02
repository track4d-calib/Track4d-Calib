#!/usr/bin/env python3
"""Export committed cameras and synchronized RGB for an external renderer."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track4d_calib._engine.track4d_calib.cameras import CameraLoader
from track4d_calib.evaluation import verify_calibration
from track4d_calib.runner import digest, write_json


def export(run: Path, source: Path, output: Path, test_views: list[str],
           time_denominator: float | None = None, copy_images: bool = False) -> dict:
    """Write Blender camera-to-world transforms with per-image intrinsics."""
    run, source = run.resolve(), source.resolve()
    result = verify_calibration(run)
    config = json.loads((run / "protocol.json").read_text())["config"]
    frames = result["frames"]
    statics, dynamics = config["static_views"], config["dynamic_views"]
    views = statics + dynamics
    tests = set(test_views)
    if len(tests) != len(test_views) or not tests <= set(views) or tests == set(views):
        raise ValueError("test views must be unique calibrated cameras and leave training views")
    denominator = frames - 1 if time_denominator is None else time_denominator
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("time denominator must be finite and positive")
    output = output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    rows = {"train": [], "test": []}
    static = CameraLoader(run / "calibration/runtime_scene/cameras/static").load_cameras(statics)
    for frame in range(frames):
        cameras = dict(static)
        cameras.update(CameraLoader(run / "poses" / f"f{frame:06d}/cameras_pred").load_cameras(dynamics))
        for view in views:
            camera = cameras[view]
            if (not np.isfinite(camera.K).all() or np.count_nonzero(camera.D)
                    or not np.allclose(camera.K[2], [0, 0, 1])
                    or camera.K[0, 1] != 0 or camera.K[1, 0] != 0
                    or camera.K[0, 0] <= 0 or camera.K[1, 1] <= 0):
                raise ValueError(f"camera {view}: export requires undistorted pinhole intrinsics without skew")
            c2w = np.linalg.inv(camera.extrinsic)
            if not np.isfinite(c2w).all():
                raise ValueError(f"camera {view}, frame {frame}: nonfinite pose")
            c2w[:3, 1:3] *= -1
            image = source / "images" / view / f"{frame:06d}.png"
            with Image.open(image) as rgb:
                if rgb.mode != "RGB" or rgb.format != "PNG" or rgb.size != (camera.W, camera.H):
                    raise ValueError(f"image format or dimensions differ from camera: {image}")
            split = "test" if view in tests else "train"
            relative = f"images/{split}_f{frame:06d}_v{view}"
            target = output / (relative + ".png")
            if copy_images:
                shutil.copyfile(image, target)
            else:
                target.symlink_to(image)
            rows[split].append({
                "file_path": relative, "frame_id": frame, "camera_id": view,
                "time": frame / denominator, "transform_matrix": c2w.tolist(),
                "fl_x": float(camera.K[0, 0]), "fl_y": float(camera.K[1, 1]),
                "cx": float(camera.K[0, 2]), "cy": float(camera.K[1, 2]),
                "w": camera.W, "h": camera.H,
            })
    verify_calibration(run)
    for split, entries in rows.items():
        write_json(output / f"transforms_{split}.json", {"frames": entries})
    manifest = {
        "status": "complete", "calibration": str(run), "source": str(source),
        "calibration_result_sha256": digest(run / "result.json"),
        "convention": "OpenGL camera-to-world: x right, y up, z backward",
        "time_denominator": denominator, "frames": frames,
        "train_views": [view for view in views if view not in tests],
        "test_views": test_views, "images_copied": copy_images,
        "transforms_sha256": {split: digest(output / f"transforms_{split}.json") for split in rows},
    }
    write_json(output / "export.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="Prepared calibration scene; RGB must match output intrinsics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-views", nargs="*", default=[])
    parser.add_argument("--time-denominator", type=float,
                        help="Timestamp is frame ID divided by this value; default: frames minus one")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.source, args.output, args.test_views,
                            args.time_denominator, args.copy_images), indent=2))


if __name__ == "__main__":
    main()
