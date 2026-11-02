from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


def resolve_frame_image(
    images_dir: Path,
    *,
    frame: int,
    exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> Path:
    """Resolve an indexed image by its exact zero-padded frame stem."""
    images_dir = Path(images_dir)
    allowed = {str(ext).lower() for ext in exts}
    stem = f"{int(frame):06d}"
    matches = sorted(
        path
        for path in images_dir.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in allowed
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one image for frame={int(frame)} under {images_dir}, found {matches}"
        )
    return matches[0]


def list_image_paths(
    images_dir: Path,
    *,
    start: int = 0,
    count: Optional[int] = None,
    exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> List[Path]:
    """
    List image files under a directory, sorted lexicographically, with slicing.

    This works for datasets that use zero-padded frame indices like `000000.webp`.
    """
    images_dir = Path(images_dir)
    paths: List[Path] = []
    for ext in exts:
        paths.extend(images_dir.glob(f"*{ext}"))
        paths.extend(images_dir.glob(f"*{ext.upper()}"))
    paths = sorted(paths)
    if start < 0:
        raise ValueError("start must be >= 0")
    if count is None:
        return paths[start:]
    if count < 0:
        raise ValueError("count must be >= 0")
    return paths[start : start + count]


def mask_to_bool(mask: np.ndarray) -> np.ndarray:
    """Convert a uint8/float mask to boolean foreground mask (mask > 0)."""
    mask = np.asarray(mask)
    return mask > 0


def select_mask_region(mask: np.ndarray, *, mode: str) -> np.ndarray:
    """
    Select a region from a (possibly soft) mask and return a uint8 {0,1} mask.

    Args:
        mask: 2D mask (uint8/float) where foreground is typically >0.
        mode:
          - "foreground": select mask>0
          - "background": select mask==0
          - "all": select everything
    Returns:
        uint8 mask with the same shape, where selected pixels are 1.
    """
    mode = str(mode).lower().strip()
    m = mask_to_bool(mask)
    if mode == "foreground":
        out = m
    elif mode == "background":
        out = ~m
    elif mode == "all":
        out = np.ones_like(m, dtype=bool)
    else:
        raise ValueError(f"Unknown mask mode: {mode} (expected foreground/background/all)")
    return out.astype(np.uint8)


def infer_common_num_frames(
    images_root: Path,
    *,
    views: Sequence[str],
    start: int = 0,
    exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> int:
    """
    Infer the common number of frames across multiple views.

    This is useful for "process all frames" mode: we take the minimum available
    length across the requested views so indexing stays aligned.
    """
    if start < 0:
        raise ValueError("start must be >= 0")
    if not views:
        raise ValueError("views must be non-empty")

    lengths: List[int] = []
    for vid in [str(v) for v in views]:
        n = len(list_image_paths(Path(images_root) / vid, start=int(start), count=None, exts=exts))
        if n <= 0:
            raise ValueError(f"No frames found for view {vid} under {Path(images_root) / vid}")
        lengths.append(int(n))
    return int(min(lengths))
