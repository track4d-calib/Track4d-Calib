from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_DATASET_CONFIG = "track4d_calib_dataset.json"
DATASET_TYPES = ("standard", "custom", "diffuman4d", "panoptic")


@dataclass(frozen=True)
class DatasetLayout:
    dataset_root: Path
    images_root: Path
    masks_root: Optional[Path]
    cameras_root: Optional[Path]
    name: str
    dataset_type: str = "standard"
    dataset_config: Optional[Path] = None

    def as_meta(self) -> Dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "images_root": str(self.images_root),
            "masks_root": str(self.masks_root) if self.masks_root else "",
            "cameras_root": str(self.cameras_root) if self.cameras_root else "",
            "dataset_type": str(self.dataset_type),
            "dataset_config": str(self.dataset_config) if self.dataset_config else "",
            "dataset_name": str(self.name),
        }


def _resolve_path(value: Optional[str | Path], *, dataset_root: Path) -> Optional[Path]:
    if value is None or str(value) == "":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    return path


def _load_dataset_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset config must be a JSON object: {path}")
    return data


def infer_dataset_root_from_meta(meta: Dict[str, Any], dataset_root_arg: str | None) -> Optional[Path]:
    if dataset_root_arg:
        return Path(dataset_root_arg)
    if isinstance(meta, dict):
        root = meta.get("dataset_root")
        if root:
            return Path(root)
        data_root = meta.get("data_root")
        if data_root:
            return Path(data_root).parent
    return None


def infer_dataset_type_from_meta(meta: Dict[str, Any], dataset_type_arg: str | None) -> str:
    if dataset_type_arg:
        return str(dataset_type_arg)
    if isinstance(meta, dict) and meta.get("dataset_type"):
        return str(meta.get("dataset_type"))
    return "standard"


def infer_dataset_config_from_meta(meta: Dict[str, Any], dataset_config_arg: str | None) -> Optional[Path]:
    if dataset_config_arg:
        return Path(dataset_config_arg)
    if isinstance(meta, dict) and meta.get("dataset_config"):
        return Path(str(meta.get("dataset_config")))
    return None


def resolve_dataset_layout(
    *,
    dataset_root: Optional[Path],
    images_root: Optional[Path],
    masks_root: Optional[Path] = None,
    cameras_root: Optional[Path] = None,
    dataset_type: str = "standard",
    dataset_config: Optional[Path] = None,
    require_images: bool = True,
) -> DatasetLayout:
    dtype = str(dataset_type or "standard").lower().strip()
    if dtype not in DATASET_TYPES:
        raise ValueError(f"Unknown dataset_type={dataset_type}. Choices: {DATASET_TYPES}")

    if dtype in ("standard",):
        if images_root is None:
            if dataset_root is None:
                raise ValueError("Need dataset_root or images_root to resolve dataset layout.")
            images_root = dataset_root / "images"
            if not images_root.is_dir():
                alt = dataset_root / "image"
                if alt.is_dir():
                    images_root = alt
        if dataset_root is None:
            dataset_root = Path(images_root).parent
        if masks_root is None:
            cand = dataset_root / "fmasks"
            masks_root = cand if cand.is_dir() else None
        if cameras_root is None:
            cand = dataset_root / "cameras"
            cameras_root = cand if cand.is_dir() else None
        name = dataset_root.name
        if require_images and images_root is not None and not images_root.is_dir():
            raise FileNotFoundError(f"images_root not found: {images_root}")
        return DatasetLayout(
            dataset_root=dataset_root,
            images_root=images_root,
            masks_root=masks_root,
            cameras_root=cameras_root,
            name=name,
            dataset_type=dtype,
            dataset_config=None,
        )

    # Non-standard layouts rely on a JSON config.
    if dataset_root is None and images_root is not None:
        dataset_root = Path(images_root).parent
    if dataset_root is None:
        raise ValueError("Need dataset_root or images_root to resolve dataset layout.")

    cfg_path = Path(dataset_config) if dataset_config else (dataset_root / DEFAULT_DATASET_CONFIG)
    cfg = _load_dataset_config(cfg_path)

    if images_root is None:
        images_root = _resolve_path(cfg.get("images_root"), dataset_root=dataset_root)
    if images_root is None:
        raise ValueError(f"images_root missing in dataset config: {cfg_path}")

    if masks_root is None:
        masks_root = _resolve_path(cfg.get("masks_root"), dataset_root=dataset_root)
    if cameras_root is None:
        cameras_root = _resolve_path(cfg.get("cameras_root"), dataset_root=dataset_root)

    name = str(cfg.get("name") or dataset_root.name)
    if require_images and not images_root.is_dir():
        raise FileNotFoundError(f"images_root not found: {images_root}")

    return DatasetLayout(
        dataset_root=dataset_root,
        images_root=images_root,
        masks_root=masks_root,
        cameras_root=cameras_root,
        name=name,
        dataset_type=dtype,
        dataset_config=cfg_path,
    )
