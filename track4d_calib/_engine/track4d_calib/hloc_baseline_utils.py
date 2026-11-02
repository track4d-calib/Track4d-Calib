from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _try_int(x: str) -> Optional[int]:
    try:
        return int(str(x))
    except ValueError:
        return None


def find_frame_mask_path(masks_root: Union[str, Path], view_id: str, src_image: Union[str, Path]) -> Optional[Path]:
    """
    Resolve a per-frame foreground mask for an image.

    Some datasets store RGB frames as .webp while masks are .png.  Prefer an
    exact filename match, then fall back to common mask extensions with the same
    frame stem.
    """
    masks_root = Path(masks_root)
    src_image = Path(src_image)
    view_dir = masks_root / str(view_id)
    exact = view_dir / src_image.name
    if exact.is_file():
        return exact
    stem = src_image.stem
    for ext in _MASK_EXTENSIONS:
        cand = view_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    matches = sorted(p for p in view_dir.glob(f"{stem}.*") if p.is_file())
    return matches[0] if matches else None


def pairs_ring_knn(
    views: Iterable[str],
    *,
    k: int,
    num_total_views: int,
) -> List[Tuple[str, str]]:
    """
    Build a sparse pairing graph for a circular multi-camera rig.

    Each view is connected to its K nearest other views by circular index distance.
    This reduces spurious wide-baseline matches compared to exhaustive pairing.

    Args:
        views: iterable of view ids (typically zero-padded integers like "00").
        k: number of nearest neighbors per view.
        num_total_views: total number of cameras in the rig (for wrap-around distance).
    Returns:
        Sorted list of unique view-id pairs (a,b) with a<b by numeric order when possible.
    """
    views = [str(v) for v in views]
    if len(views) < 2:
        return []
    k = int(k)
    if k <= 0:
        return []
    num_total_views = int(num_total_views)
    if num_total_views <= 0:
        raise ValueError("num_total_views must be positive")

    view_num: Dict[str, Optional[int]] = {v: _try_int(v) for v in views}
    view_idx: Dict[str, int] = {v: i for i, v in enumerate(views)}

    def _dist(a: str, b: str) -> int:
        na = view_num[a]
        nb = view_num[b]
        if na is None or nb is None:
            return abs(view_idx[a] - view_idx[b])
        d = abs(na - nb)
        return int(min(d, num_total_views - d))

    def _sort_key(v: str):
        n = view_num[v]
        return (n if n is not None else 10**9, v)

    pairs = set()
    for v in views:
        cand = []
        for u in views:
            if u == v:
                continue
            cand.append((_dist(v, u), _sort_key(u), u))
        cand.sort(key=lambda x: (x[0], x[1]))
        for _, __, u in cand[: min(k, len(cand))]:
            a, b = sorted([v, u], key=_sort_key)
            pairs.add((a, b))

    return sorted(pairs, key=lambda p: (_sort_key(p[0]), _sort_key(p[1])))


def filter_hloc_features_by_masks(
    features_h5: Union[str, Path],
    image_masks: Dict[str, np.ndarray],
    *,
    mask_thresh: int = 0,
    mode: str = "background",
) -> Dict[str, Dict[str, int]]:
    """
    Filter HLOC (SuperPoint) features by a per-image foreground mask.

    This is useful for multi-view human capture where foreground (people) is dynamic
    and background is static for calibration.

    Args:
        features_h5: path to HLOC features .h5
        image_masks: mapping image_name -> mask (H,W) uint8; masked pixels are > mask_thresh.
        mask_thresh: background keeps keypoints where mask[y, x] <= mask_thresh;
            foreground keeps keypoints where mask[y, x] > mask_thresh.
        mode: "background" removes masked foreground; "foreground" keeps only
            masked foreground.
    Returns:
        Per-image stats: {name: {"before": N, "kept": M}} for processed images.
    """
    import h5py

    features_h5 = Path(features_h5)
    stats: Dict[str, Dict[str, int]] = {}

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"background", "foreground"}:
        raise ValueError(f"mode must be 'background' or 'foreground', got {mode!r}")

    with h5py.File(str(features_h5), "r+") as f:
        for name, mask in image_masks.items():
            if name not in f:
                continue
            g = f[name]
            if "keypoints" not in g:
                continue

            kps = np.asarray(g["keypoints"])
            before = int(kps.shape[0])
            if before == 0:
                stats[name] = {"before": 0, "kept": 0}
                continue

            mask = np.asarray(mask)
            if mask.ndim != 2:
                raise ValueError(f"mask for {name} must be 2D, got shape={mask.shape}")

            xs = np.rint(kps[:, 0]).astype(np.int64)
            ys = np.rint(kps[:, 1]).astype(np.int64)
            xs = np.clip(xs, 0, mask.shape[1] - 1)
            ys = np.clip(ys, 0, mask.shape[0] - 1)
            if mode_norm == "background":
                keep = mask[ys, xs] <= int(mask_thresh)
            else:
                keep = mask[ys, xs] > int(mask_thresh)
            kept = int(np.count_nonzero(keep))
            stats[name] = {"before": before, "kept": kept}

            if kept == before:
                continue

            # Filter arrays.
            kp_new = kps[keep]
            sc_new = np.asarray(g["scores"])[keep] if "scores" in g else None
            desc = np.asarray(g["descriptors"]) if "descriptors" in g else None
            desc_new = None
            if desc is not None:
                if desc.ndim == 2 and desc.shape[1] == before:
                    desc_new = desc[:, keep]
                elif desc.ndim == 2 and desc.shape[0] == before:
                    desc_new = desc[keep]
                else:
                    raise ValueError(f"Unexpected descriptors shape for {name}: {desc.shape}, N={before}")

            # Replace datasets (HDF5 doesn't support shrinking fixed-size datasets in-place).
            del g["keypoints"]
            g.create_dataset("keypoints", data=kp_new)
            if sc_new is not None:
                del g["scores"]
                g.create_dataset("scores", data=sc_new)
            if desc_new is not None:
                del g["descriptors"]
                g.create_dataset("descriptors", data=desc_new)

    return stats
