from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraParams:
    """Pinhole camera parameters for a single view (OpenCV convention)."""

    view_id: str
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) world -> camera
    T: np.ndarray  # (3,) world -> camera translation
    D: np.ndarray  # (5,) distortion coeffs (optional; often zeros)
    H: int
    W: int

    @property
    def P(self) -> np.ndarray:
        """Projection matrix P = K [R | T]. Shape: (3, 4)."""
        rt = np.hstack([self.R, self.T.reshape(3, 1)])
        return self.K @ rt

    @property
    def extrinsic(self) -> np.ndarray:
        """Extrinsic matrix [R | T; 0 0 0 1]. Shape: (4, 4)."""
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = self.R
        ext[:3, 3] = self.T
        return ext

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world coords: C = -R^T T. Shape: (3,)."""
        return -self.R.T @ self.T


class CameraLoader:
    """Load camera intrinsics/extrinsics from OpenCV YAML files (extri.yml/intri.yml)."""

    def __init__(self, cameras_path: Path):
        self.cameras_path = Path(cameras_path)
        self.extri_path = self.cameras_path / "extri.yml"
        self.intri_path = self.cameras_path / "intri.yml"

        if not self.extri_path.exists():
            raise FileNotFoundError(f"Extrinsics file not found: {self.extri_path}")
        if not self.intri_path.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {self.intri_path}")

    def load_camera(self, view_id: str) -> CameraParams:
        fs_extri = cv2.FileStorage(str(self.extri_path), cv2.FILE_STORAGE_READ)
        R = fs_extri.getNode(f"Rot_{view_id}").mat()
        T = fs_extri.getNode(f"T_{view_id}").mat()
        fs_extri.release()

        if R is None or T is None:
            raise ValueError(f"Camera {view_id} not found in extrinsics file")

        fs_intri = cv2.FileStorage(str(self.intri_path), cv2.FILE_STORAGE_READ)
        K = fs_intri.getNode(f"K_{view_id}").mat()
        D = fs_intri.getNode(f"D_{view_id}").mat()
        H = fs_intri.getNode(f"H_{view_id}").real()
        W = fs_intri.getNode(f"W_{view_id}").real()
        fs_intri.release()

        if K is None:
            raise ValueError(f"Camera {view_id} not found in intrinsics file")

        return CameraParams(
            view_id=str(view_id),
            K=np.asarray(K, dtype=np.float64),
            R=np.asarray(R, dtype=np.float64),
            T=np.asarray(T, dtype=np.float64).reshape(3),
            D=np.asarray(D, dtype=np.float64).reshape(-1) if D is not None else np.zeros(5, dtype=np.float64),
            H=int(H),
            W=int(W),
        )

    def load_cameras(self, view_ids: Iterable[str]) -> Dict[str, CameraParams]:
        cams: Dict[str, CameraParams] = {}
        for vid in view_ids:
            cams[str(vid)] = self.load_camera(str(vid))
        return cams


class CameraIntrinsicsLoader:
    """Load camera templates from ``intri.yml`` without reading extrinsics."""

    def __init__(self, cameras_path: Path):
        self.cameras_path = Path(cameras_path)
        self.intri_path = self.cameras_path / "intri.yml"
        if not self.intri_path.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {self.intri_path}")

    def load_camera_template(self, view_id: str) -> CameraParams:
        fs_intri = cv2.FileStorage(str(self.intri_path), cv2.FILE_STORAGE_READ)
        try:
            K = fs_intri.getNode(f"K_{view_id}").mat()
            D = fs_intri.getNode(f"D_{view_id}").mat()
            H = fs_intri.getNode(f"H_{view_id}").real()
            W = fs_intri.getNode(f"W_{view_id}").real()
        finally:
            fs_intri.release()
        if K is None:
            raise ValueError(f"Camera {view_id} not found in intrinsics file")
        return CameraParams(
            view_id=str(view_id),
            K=np.asarray(K, dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            T=np.zeros(3, dtype=np.float64),
            D=(
                np.asarray(D, dtype=np.float64).reshape(-1)
                if D is not None
                else np.zeros(5, dtype=np.float64)
            ),
            H=int(H),
            W=int(W),
        )

    def load_camera_templates(
        self,
        view_ids: Iterable[str],
    ) -> Dict[str, CameraParams]:
        return {
            str(view_id): self.load_camera_template(str(view_id))
            for view_id in view_ids
        }


def save_cameras(cameras_path: Path, cameras: Dict[str, CameraParams]) -> None:
    """
    Save camera intrinsics/extrinsics in OpenCV YAML format (extri.yml/intri.yml).

    This matches the format expected by :class:`CameraLoader`.
    """
    cameras_path = Path(cameras_path)
    cameras_path.mkdir(parents=True, exist_ok=True)

    extri_path = cameras_path / "extri.yml"
    intri_path = cameras_path / "intri.yml"

    fs_extri = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_WRITE)
    try:
        for vid, cam in cameras.items():
            fs_extri.write(f"Rot_{vid}", np.asarray(cam.R, dtype=np.float64))
            fs_extri.write(f"T_{vid}", np.asarray(cam.T, dtype=np.float64).reshape(3, 1))
    finally:
        fs_extri.release()

    fs_intri = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_WRITE)
    try:
        for vid, cam in cameras.items():
            fs_intri.write(f"K_{vid}", np.asarray(cam.K, dtype=np.float64))
            d = np.asarray(cam.D, dtype=np.float64).reshape(-1)
            if d.size < 5:
                d5 = np.zeros((5,), dtype=np.float64)
                d5[: d.size] = d
                d = d5
            fs_intri.write(f"D_{vid}", d[:5].reshape(5, 1))
            fs_intri.write(f"H_{vid}", float(cam.H))
            fs_intri.write(f"W_{vid}", float(cam.W))
    finally:
        fs_intri.release()
