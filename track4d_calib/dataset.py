"""Validate the RGB and pinhole-camera layout consumed by calibration."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _validate_intrinsics(intrinsics: Path, views: list[str], width: int, height: int) -> None:
    """Require calibrated pinhole intrinsics for each declared image stream."""
    try:
        cameras = cv2.FileStorage(str(intrinsics), cv2.FILE_STORAGE_READ)
    except (cv2.error, SystemError) as error:
        raise ValueError(f"cannot read OpenCV camera file: {intrinsics}") from error
    try:
        if not cameras.isOpened():
            raise ValueError(f"cannot read OpenCV camera file: {intrinsics}")
        for view in views:
            matrix = cameras.getNode(f"K_{view}").mat()
            if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError(f"K_{view} must be a finite 3 x 3 matrix")
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
                raise ValueError(f"K_{view} must have positive focal lengths")
            if not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-8):
                raise ValueError(f"K_{view} must have last row [0, 0, 1]")

            distortion = cameras.getNode(f"D_{view}").mat()
            if distortion is None or distortion.size != 5 or not np.isfinite(distortion).all():
                raise ValueError(f"D_{view} must contain five finite distortion coefficients")
            if np.count_nonzero(distortion):
                raise ValueError(
                    f"camera {view}: supply undistorted images, their updated K, and zero D"
                )
            for name, expected in (("W", width), ("H", height)):
                node = cameras.getNode(f"{name}_{view}")
                if node.empty() or node.real() != expected:
                    raise ValueError(f"{name}_{view} must equal the configured value {expected}")
    except cv2.error as error:
        raise ValueError(f"invalid OpenCV camera entries in {intrinsics}: {error}") from error
    finally:
        cameras.release()


def _validate_images(
    source: Path, views: list[str], frame_count: int, width: int, height: int
) -> None:
    """Check synchronized RGB frames and their configured pixel dimensions."""
    for view in views:
        for frame in range(frame_count):
            path = source / "images" / view / f"{frame:06d}.png"
            if not path.is_file():
                raise FileNotFoundError(f"missing synchronized image: {path}")
            try:
                with Image.open(path) as image:
                    if image.format != "PNG" or image.mode != "RGB":
                        raise ValueError(f"{path}: expected an 8-bit RGB PNG image")
                    if image.size != (width, height):
                        raise ValueError(
                            f"{path}: image size {image.size} does not match {(width, height)}"
                        )
                    image.verify()
            except (OSError, SyntaxError) as error:
                raise ValueError(f"cannot read RGB image {path}: {error}") from error


def validate_scene(source: Path, config: dict) -> dict:
    """Check every declared camera and frame without loading models or poses."""
    source = source.resolve()
    intrinsics = source / "cameras/intri.yml"
    if not intrinsics.is_file():
        raise FileNotFoundError(f"missing camera intrinsics: {intrinsics}")
    width, height = config["resolution"]
    views = config["static_views"] + config["dynamic_views"]
    frame_count = config["frames"]
    _validate_intrinsics(intrinsics, views, width, height)
    _validate_images(source, views, frame_count, width, height)
    return {
        "source": str(source),
        "intrinsics": str(intrinsics),
        "views": views,
        "frames": frame_count,
        "images": len(views) * frame_count,
        "resolution": [width, height],
        "camera_model": "undistorted_pinhole",
        "passed": True,
    }
