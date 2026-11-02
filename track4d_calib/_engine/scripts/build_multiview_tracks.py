#!/usr/bin/env python3
"""
Build multiview tracks from seed-frame RoMaV2 matches and temporal tracking.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F


def _add_third_party_to_syspath(project_root: Path) -> None:
    roma_src = project_root / "third_party" / "RoMaV2" / "src"
    cotracker_root = project_root / "third_party" / "co-tracker"
    cowtracker_root = project_root / "third_party" / "cowtracker"
    hloc_root = project_root / "third_party" / "hloc"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(roma_src))
    sys.path.insert(0, str(cotracker_root))
    if cowtracker_root.is_dir():
        sys.path.insert(0, str(cowtracker_root))
    if (hloc_root / "hloc").is_dir():
        sys.path.insert(0, str(hloc_root))


def _load_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_mask(mask_path: Path, *, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    h, w = shape_hw
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def _load_focal_px_by_view(cameras_root: Path, views: List[str]) -> Dict[str, float]:
    intri_path = Path(cameras_root) / "intri.yml"
    if not intri_path.is_file():
        return {}
    fs = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return {}
    out: Dict[str, float] = {}
    try:
        for view in views:
            key = str(view)
            K = fs.getNode(f"K_{key}").mat()
            if K is None:
                continue
            K_np = np.asarray(K, dtype=np.float64)
            if K_np.shape != (3, 3):
                continue
            fx = float(K_np[0, 0])
            fy = float(K_np[1, 1])
            f = 0.5 * (fx + fy)
            if np.isfinite(f) and f > 1e-6:
                out[key] = float(f)
    finally:
        fs.release()
    return out


def _resize_rgb_by_scale(
    image: np.ndarray,
    *,
    scale: float,
    min_side: int,
) -> Tuple[np.ndarray, float]:
    scale_f = float(scale)
    if (not np.isfinite(scale_f)) or scale_f <= 0.0:
        return image, 1.0
    h0, w0 = int(image.shape[0]), int(image.shape[1])
    if h0 <= 1 or w0 <= 1:
        return image, 1.0

    h1 = max(8, int(round(float(h0) * scale_f)))
    w1 = max(8, int(round(float(w0) * scale_f)))
    min_side_i = int(max(8, int(min_side)))
    cur_min = min(h1, w1)
    if cur_min < min_side_i:
        boost = float(min_side_i) / float(max(cur_min, 1))
        h1 = max(8, int(round(float(h1) * boost)))
        w1 = max(8, int(round(float(w1) * boost)))

    if h1 == h0 and w1 == w0:
        return image, 1.0

    resized = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_LINEAR)
    scale_eff = float(w1) / float(max(w0, 1))
    return resized, scale_eff


def _normalized_grid(h: int, w: int) -> np.ndarray:
    """Normalized (x, y) grid in [-1, 1], aligned with RoMa warp sampling."""
    ys = np.linspace(-1.0 + 1.0 / float(h), 1.0 - 1.0 / float(h), int(h), dtype=np.float32)
    xs = np.linspace(-1.0 + 1.0 / float(w), 1.0 - 1.0 / float(w), int(w), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def _bilinear_sample_warp_norm(
    warp_map: np.ndarray,
    query_xy_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample normalized warp map at normalized query coords via bilinear interpolation."""
    if warp_map.ndim != 3 or warp_map.shape[-1] != 2:
        raise ValueError(f"warp_map must have shape (h, w, 2), got {warp_map.shape}")
    if query_xy_norm.ndim != 3 or query_xy_norm.shape[-1] != 2:
        raise ValueError(f"query_xy_norm must have shape (h, w, 2), got {query_xy_norm.shape}")

    h_map, w_map = warp_map.shape[:2]
    x = (query_xy_norm[..., 0] + 1.0) * 0.5 * float(w_map) - 0.5
    y = (query_xy_norm[..., 1] + 1.0) * 0.5 * float(h_map) - 0.5

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= float(w_map - 1))
        & (y >= 0.0)
        & (y <= float(h_map - 1))
    )

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, w_map - 1)
    x1c = np.clip(x1, 0, w_map - 1)
    y0c = np.clip(y0, 0, h_map - 1)
    y1c = np.clip(y1, 0, h_map - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    Ia = warp_map[y0c, x0c]
    Ib = warp_map[y1c, x0c]
    Ic = warp_map[y0c, x1c]
    Id = warp_map[y1c, x1c]

    sampled = (
        Ia * wa[..., None]
        + Ib * wb[..., None]
        + Ic * wc[..., None]
        + Id * wd[..., None]
    ).astype(np.float32)
    sampled[~valid] = np.nan
    return sampled, valid


def _compute_cycle_consistency_error_map_ref_px(
    *,
    warp_ref_to_view: np.ndarray,
    warp_view_to_ref: np.ndarray,
    ref_shape_hw: Tuple[int, int],
) -> np.ndarray:
    """RoMa cycle-consistency error on ref grid, expressed in ref-image pixels."""
    wab = np.asarray(warp_ref_to_view, dtype=np.float32)
    wba = np.asarray(warp_view_to_ref, dtype=np.float32)
    if wab.ndim != 3 or wab.shape[-1] != 2:
        raise ValueError(f"warp_ref_to_view must have shape (h, w, 2), got {wab.shape}")
    if wba.ndim != 3 or wba.shape[-1] != 2:
        raise ValueError(f"warp_view_to_ref must have shape (h, w, 2), got {wba.shape}")

    h_ref, w_ref = int(ref_shape_hw[0]), int(ref_shape_hw[1])
    if h_ref <= 0 or w_ref <= 0:
        raise ValueError(f"Invalid ref shape: {ref_shape_hw}")

    h_ab, w_ab = wab.shape[:2]
    ref_grid = _normalized_grid(h_ab, w_ab)
    back_warp, back_valid = _bilinear_sample_warp_norm(wba, wab)

    ab_valid = np.isfinite(wab).all(axis=-1) & (np.abs(wab).max(axis=-1) <= 1.0)
    ba_valid = np.isfinite(back_warp).all(axis=-1)
    valid = ab_valid & back_valid & ba_valid

    delta = back_warp - ref_grid
    dx_px = delta[..., 0] * (0.5 * float(w_ref))
    dy_px = delta[..., 1] * (0.5 * float(h_ref))
    err_px = np.sqrt(dx_px * dx_px + dy_px * dy_px).astype(np.float32)
    err_px[~valid] = np.inf
    return err_px


def _patch_torch_hub_dinov3_offline_if_possible() -> None:
    """
    ReTracker uses torch.hub to load DINOv3 model code (hubconf) from GitHub.
    In restricted environments without reliable network, torch.hub may fail even
    when a cached checkout exists locally.

    This best-effort patch rewrites torch.hub.load("facebookresearch/dinov3", ...)
    to load from the local cached repo under ~/.cache/torch/hub/.
    """

    try:
        hub_dir = Path(torch.hub.get_dir())
    except Exception:
        hub_dir = Path.home() / ".cache" / "torch" / "hub"

    if not hub_dir.is_dir():
        return

    # Prefer the newest cached checkout (matches torch.hub naming).
    candidates = sorted(hub_dir.glob("facebookresearch_dinov3_*"))
    local_repo = None
    for cand in reversed(candidates):
        if (cand / "hubconf.py").is_file():
            local_repo = cand
            break
    if local_repo is None:
        return

    orig_load = torch.hub.load
    if getattr(orig_load, "_track4d_calib_dinov3_offline_patch", False):
        return

    def _wrapped_load(repo_or_dir, model, *args, **kwargs):
        try:
            source = kwargs.get("source", "github")
            if (
                isinstance(repo_or_dir, str)
                and repo_or_dir.startswith("facebookresearch/dinov3")
                and str(source) == "github"
            ):
                kwargs = dict(kwargs)
                kwargs["source"] = "local"
                repo_or_dir = str(local_repo)
        except Exception:
            # Fall back to original behavior.
            pass
        return orig_load(repo_or_dir, model, *args, **kwargs)

    setattr(_wrapped_load, "_track4d_calib_dinov3_offline_patch", True)
    torch.hub.load = _wrapped_load  # type: ignore[assignment]
    print(f"[INFO] torch.hub: using cached DINOv3 repo for offline runs: {local_repo}")


def _patch_torch_amp_custom_fwd_bwd_compat() -> None:
    """
    DINOv3 code imports `custom_fwd/custom_bwd` from `torch.amp`, while some
    Torch builds (e.g. 2.2.x) only expose them under `torch.cuda.amp`.
    """
    try:
        amp_mod = getattr(torch, "amp", None)
        cuda_amp_mod = getattr(getattr(torch, "cuda", None), "amp", None)
        if amp_mod is None or cuda_amp_mod is None:
            return
        if (not hasattr(amp_mod, "custom_fwd")) and hasattr(cuda_amp_mod, "custom_fwd"):
            cuda_custom_fwd = getattr(cuda_amp_mod, "custom_fwd")

            def _custom_fwd_compat(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs.pop("device_type", None)
                cast_inputs = kwargs.get("cast_inputs", None)
                if args:
                    return cuda_custom_fwd(*args, cast_inputs=cast_inputs)

                def _decorator(fn):
                    return cuda_custom_fwd(fn, cast_inputs=cast_inputs)

                return _decorator

            setattr(amp_mod, "custom_fwd", _custom_fwd_compat)

        if (not hasattr(amp_mod, "custom_bwd")) and hasattr(cuda_amp_mod, "custom_bwd"):
            cuda_custom_bwd = getattr(cuda_amp_mod, "custom_bwd")

            def _custom_bwd_compat(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs.pop("device_type", None)
                if args:
                    return cuda_custom_bwd(*args)

                def _decorator(fn):
                    return cuda_custom_bwd(fn)

                return _decorator

            setattr(amp_mod, "custom_bwd", _custom_bwd_compat)
    except Exception:
        return


def _is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and ("cuda" in msg or "cublas" in msg or "cudnn" in msg)


def _alltracker_candidate_image_sizes(initial: int, min_size: int, step: int) -> List[int]:
    hi = max(8, int(initial))
    lo = min(hi, max(8, int(min_size)))
    st = max(8, int(step))
    sizes = [hi]
    cur = hi
    while cur > lo:
        cur = max(lo, cur - st)
        if sizes[-1] != cur:
            sizes.append(cur)
    return sizes


def _alltracker_resize_hw_from_max_size(src_hw: Tuple[int, int], max_size: int) -> Tuple[int, int]:
    h0, w0 = int(src_hw[0]), int(src_hw[1])
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"Invalid source shape: {src_hw}")
    max_dim = max(h0, w0)
    target_dim = min(max_dim, max(8, int(max_size)))
    scale = float(target_dim) / float(max_dim)
    h = max(8, int(round(h0 * scale)))
    w = max(8, int(round(w0 * scale)))
    h = max(8, (h // 8) * 8)
    w = max(8, (w // 8) * 8)
    return int(h), int(w)


def _cowtracker_resize_hw_from_max_size(src_hw: Tuple[int, int], max_size: int) -> Tuple[int, int]:
    h0, w0 = int(src_hw[0]), int(src_hw[1])
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"Invalid source shape: {src_hw}")
    stride = 112
    max_dim = max(h0, w0)
    if int(max_size) > 0:
        target_dim = min(max_dim, max(stride, int(max_size)))
    else:
        target_dim = max_dim
    scale = float(target_dim) / float(max_dim)
    h = max(stride, int(round(h0 * scale)))
    w = max(stride, int(round(w0 * scale)))
    # CoWTracker combines a /14 ViT branch and convolutional side features; the
    # official demo size (336x560) is aligned to lcm(14,16)=112.
    h = max(stride, (h // stride) * stride)
    w = max(stride, (w // stride) * stride)
    return int(h), int(w)


def _alltracker_grid_xy(h: int, w: int, *, dtype: torch.dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(int(h), dtype=dtype),
        torch.arange(int(w), dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0)


def _alltracker_forward_segment(
    *,
    alltracker,
    video_seg: torch.Tensor,
    inference_iters: int,
    window_len: int,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run AllTracker on one temporal segment.

    Returns:
      flows: (B,S,2,H,W) on CPU, float32.
      visconfs: (B,S,2,H,W) on CPU, float32.
    """
    seg_len = int(video_seg.shape[1])
    b = int(video_seg.shape[0])
    h = int(video_seg.shape[-2])
    w = int(video_seg.shape[-1])
    if b != 1:
        raise ValueError(f"AllTracker segment expects B=1, got B={b}")

    if seg_len <= 1:
        flows = torch.zeros((b, 1, 2, h, w), dtype=torch.float32, device="cpu")
        visconfs = torch.ones((b, 1, 2, h, w), dtype=torch.float32, device="cpu")
        return flows, visconfs

    run_stride = None if int(stride) <= 0 else int(stride)
    flows, visconfs, _, _ = alltracker.forward_sliding(
        video_seg,
        iters=int(inference_iters),
        sw=None,
        is_training=False,
        window_len=int(window_len),
        stride=run_stride,
    )
    flows = flows.detach().to(device="cpu", dtype=torch.float32)
    visconfs = visconfs.detach().to(device="cpu", dtype=torch.float32)

    # Pair mode may return (B,2,H,W). Promote to temporal form and prepend anchor identity.
    if flows.ndim == 4:
        flows = flows[:, None]
        visconfs = visconfs[:, None]

    if flows.ndim != 5 or visconfs.ndim != 5:
        raise RuntimeError(f"Unexpected AllTracker output shapes: flows={tuple(flows.shape)} vis={tuple(visconfs.shape)}")

    if int(flows.shape[1]) == seg_len - 1:
        zf = torch.zeros((b, 1, 2, h, w), dtype=flows.dtype, device=flows.device)
        zv = torch.ones((b, 1, 2, h, w), dtype=visconfs.dtype, device=visconfs.device)
        flows = torch.cat([zf, flows], dim=1)
        visconfs = torch.cat([zv, visconfs], dim=1)

    if int(flows.shape[1]) != seg_len:
        raise RuntimeError(
            f"AllTracker segment length mismatch: got {int(flows.shape[1])}, expected {seg_len}"
        )
    return flows, visconfs


def _alltracker_dense_maps_for_query_frame(
    *,
    alltracker,
    video: torch.Tensor,
    query_frame: int,
    inference_iters: int,
    window_len: int,
    stride: int,
    confidence_channel: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dense trajectory and confidence maps for one query frame.

    Returns:
      traj_maps: (T,2,H,W) on CPU.
      conf_map: (T,H,W) on CPU.
    """
    b, t, _, h, w = video.shape
    if int(b) != 1:
        raise ValueError(f"AllTracker expects B=1, got B={int(b)}")
    q = int(query_frame)
    if q < 0 or q >= int(t):
        raise ValueError(f"query_frame out of range: {q} not in [0,{int(t)-1}]")

    grid = _alltracker_grid_xy(int(h), int(w), dtype=torch.float32)[None, None]  # (1,1,2,H,W)

    flows_f, vis_f = _alltracker_forward_segment(
        alltracker=alltracker,
        video_seg=video[:, q:],
        inference_iters=int(inference_iters),
        window_len=int(window_len),
        stride=int(stride),
    )
    traj_f = flows_f + grid

    if q > 0:
        flows_b, vis_b = _alltracker_forward_segment(
            alltracker=alltracker,
            video_seg=video[:, : q + 1].flip([1]),
            inference_iters=int(inference_iters),
            window_len=int(window_len),
            stride=int(stride),
        )
        traj_b = flows_b + grid
        traj_b = traj_b.flip([1])[:, 1:]
        vis_b = vis_b.flip([1])[:, 1:]
        traj = torch.cat([traj_b, traj_f], dim=1)
        vis = torch.cat([vis_b, vis_f], dim=1)
    else:
        traj = traj_f
        vis = vis_f

    if int(traj.shape[1]) != int(t):
        raise RuntimeError(
            f"AllTracker dense-map T mismatch: got {int(traj.shape[1])}, expected {int(t)}"
        )

    ch = int(confidence_channel)
    if ch < 0 or ch >= int(vis.shape[2]):
        raise ValueError(f"Invalid AllTracker confidence channel: {ch}, shape={tuple(vis.shape)}")
    return traj[0], vis[0, :, ch]


def _alltracker_sample_dense_maps(
    *,
    traj_maps: torch.Tensor,
    conf_map: torch.Tensor,
    query_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample dense maps at query points.

    Args:
      traj_maps: (T,2,H,W)
      conf_map: (T,H,W)
      query_xy: (N,2) in pixel coords of dense map resolution.

    Returns:
      tracks: (T,N,2)
      conf: (T,N)
    """
    t, c, h, w = traj_maps.shape
    if c != 2:
        raise ValueError(f"traj_maps must have channel=2, got shape={tuple(traj_maps.shape)}")
    pts = np.asarray(query_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"query_xy must have shape (N,2), got {pts.shape}")
    n = int(pts.shape[0])
    if n == 0:
        return (
            np.zeros((int(t), 0, 2), dtype=np.float32),
            np.zeros((int(t), 0), dtype=np.float32),
        )

    x = torch.from_numpy(pts[:, 0])
    y = torch.from_numpy(pts[:, 1])
    gx = 2.0 * x / max(float(w - 1), 1.0) - 1.0
    gy = 2.0 * y / max(float(h - 1), 1.0) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, n, 1, 2).expand(int(t), -1, -1, -1)

    sampled_traj = F.grid_sample(
        traj_maps.to(dtype=torch.float32),
        grid.to(dtype=torch.float32),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled_traj = sampled_traj.squeeze(-1).permute(0, 2, 1).contiguous().numpy().astype(np.float32)

    sampled_conf = F.grid_sample(
        conf_map[:, None].to(dtype=torch.float32),
        grid.to(dtype=torch.float32),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled_conf = sampled_conf.squeeze(1).squeeze(-1).contiguous().numpy().astype(np.float32)
    return sampled_traj, sampled_conf


def _sample_flow_bilinear_np(flow_xy: np.ndarray, points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a dense flow map at sparse pixel coordinates.

    Args:
      flow_xy: (H,W,2), displacement in pixel units.
      points_xy: (N,2), pixel coordinates in the same resolution.

    Returns:
      sampled: (N,2), NaN where the point is outside the image.
      valid: (N,) bool.
    """
    flow = np.asarray(flow_xy, dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"flow_xy must have shape (H,W,2), got {flow.shape}")

    h, w = int(flow.shape[0]), int(flow.shape[1])
    x = pts[:, 0]
    y = pts[:, 1]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= float(w - 1))
        & (y >= 0.0)
        & (y <= float(h - 1))
    )

    x0 = np.floor(np.nan_to_num(x, nan=0.0)).astype(np.int64)
    y0 = np.floor(np.nan_to_num(y, nan=0.0)).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, max(w - 1, 0))
    x1c = np.clip(x1, 0, max(w - 1, 0))
    y0c = np.clip(y0, 0, max(h - 1, 0))
    y1c = np.clip(y1, 0, max(h - 1, 0))

    wa = (x1.astype(np.float32) - x) * (y1.astype(np.float32) - y)
    wb = (x1.astype(np.float32) - x) * (y - y0.astype(np.float32))
    wc = (x - x0.astype(np.float32)) * (y1.astype(np.float32) - y)
    wd = (x - x0.astype(np.float32)) * (y - y0.astype(np.float32))

    sampled = (
        flow[y0c, x0c] * wa[:, None]
        + flow[y1c, x0c] * wb[:, None]
        + flow[y0c, x1c] * wc[:, None]
        + flow[y1c, x1c] * wd[:, None]
    ).astype(np.float32)
    sampled[~valid] = np.nan
    return sampled, valid.astype(bool)


def _points_inside_image_np(points_xy: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    h, w = int(image_hw[0]), int(image_hw[1])
    return (
        np.isfinite(pts[:, 0])
        & np.isfinite(pts[:, 1])
        & (pts[:, 0] >= 0.0)
        & (pts[:, 0] <= float(w - 1))
        & (pts[:, 1] >= 0.0)
        & (pts[:, 1] <= float(h - 1))
    )


def _propagate_sparse_tracks_with_pair_flows(
    *,
    forward_flows: List[np.ndarray],
    backward_flows: List[np.ndarray],
    query_xy: np.ndarray,
    query_t: np.ndarray,
    image_hw: Tuple[int, int],
    fb_consistency_threshold_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pairwise optical flow into sparse all-frame tracks.

    ``forward_flows[t]`` maps frame t -> t+1. ``backward_flows[t]`` maps
    frame t+1 -> t. Visibility confidence is 1/(1+FB error); inconsistent
    edges terminate propagation in that direction.
    """
    if len(forward_flows) != len(backward_flows):
        raise ValueError("forward_flows and backward_flows must have the same length")
    num_frames = int(len(forward_flows) + 1)
    pts = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    qts = np.asarray(query_t, dtype=np.int64).reshape(-1)
    if pts.shape[0] != qts.shape[0]:
        raise ValueError(f"query_xy/query_t length mismatch: {pts.shape[0]} vs {qts.shape[0]}")

    n = int(pts.shape[0])
    tracks = np.full((num_frames, n, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((num_frames, n), dtype=np.float32)
    threshold = float(fb_consistency_threshold_px)
    use_fb_gate = threshold > 0.0

    if n == 0:
        return tracks, confidence

    seed_inside = _points_inside_image_np(pts, image_hw)
    for point_id in range(n):
        q = int(qts[point_id])
        if q < 0 or q >= num_frames or not bool(seed_inside[point_id]):
            continue

        seed_xy = pts[point_id].astype(np.float32)
        tracks[q, point_id] = seed_xy
        confidence[q, point_id] = 1.0

        cur = seed_xy.copy()
        for edge_t in range(q, num_frames - 1):
            flow, valid = _sample_flow_bilinear_np(forward_flows[edge_t], cur.reshape(1, 2))
            if not bool(valid[0]):
                break
            nxt = cur + flow[0]
            if not bool(_points_inside_image_np(nxt.reshape(1, 2), image_hw)[0]):
                break
            rev, rev_valid = _sample_flow_bilinear_np(backward_flows[edge_t], nxt.reshape(1, 2))
            if not bool(rev_valid[0]):
                break
            fb_err = float(np.linalg.norm(flow[0] + rev[0]))
            if use_fb_gate and fb_err > threshold:
                break
            tracks[edge_t + 1, point_id] = nxt.astype(np.float32)
            confidence[edge_t + 1, point_id] = float(1.0 / (1.0 + max(fb_err, 0.0)))
            cur = nxt.astype(np.float32)

        cur = seed_xy.copy()
        for edge_t in range(q - 1, -1, -1):
            flow, valid = _sample_flow_bilinear_np(backward_flows[edge_t], cur.reshape(1, 2))
            if not bool(valid[0]):
                break
            prev = cur + flow[0]
            if not bool(_points_inside_image_np(prev.reshape(1, 2), image_hw)[0]):
                break
            rev, rev_valid = _sample_flow_bilinear_np(forward_flows[edge_t], prev.reshape(1, 2))
            if not bool(rev_valid[0]):
                break
            fb_err = float(np.linalg.norm(flow[0] + rev[0]))
            if use_fb_gate and fb_err > threshold:
                break
            tracks[edge_t, point_id] = prev.astype(np.float32)
            confidence[edge_t, point_id] = float(1.0 / (1.0 + max(fb_err, 0.0)))
            cur = prev.astype(np.float32)

    return tracks, confidence


def _raft_resize_hw_from_max_size(src_hw: Tuple[int, int], max_size: int) -> Tuple[int, int]:
    h0, w0 = int(src_hw[0]), int(src_hw[1])
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"Invalid source shape: {src_hw}")
    max_dim = max(h0, w0)
    if int(max_size) > 0:
        target_dim = min(max_dim, max(8, int(max_size)))
    else:
        target_dim = max_dim
    scale = float(target_dim) / float(max_dim)
    h = max(8, int(round(h0 * scale)))
    w = max(8, int(round(w0 * scale)))
    # Torchvision RAFT requires dimensions divisible by 8.
    h = max(8, (h // 8) * 8)
    w = max(8, (w // 8) * 8)
    return int(h), int(w)


def _raft_pair_flow(
    *,
    raft,
    raft_transform,
    frame0: np.ndarray,
    frame1: np.ndarray,
    device: torch.device,
    num_flow_updates: int,
    use_amp: bool,
) -> np.ndarray:
    img0 = torch.from_numpy(np.asarray(frame0, dtype=np.uint8)).permute(2, 0, 1)[None].to(device)
    img1 = torch.from_numpy(np.asarray(frame1, dtype=np.uint8)).permute(2, 0, 1)[None].to(device)
    if raft_transform is not None:
        img0, img1 = raft_transform(img0, img1)
    else:
        img0 = img0.float().div(127.5).sub(1.0).contiguous()
        img1 = img1.float().div(127.5).sub(1.0).contiguous()

    amp_enabled = bool(use_amp) and str(device).startswith("cuda")
    with torch.no_grad():
        autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
            preds = raft(img0, img1, num_flow_updates=int(num_flow_updates))
    flow = preds[-1][0].detach().to(device="cpu", dtype=torch.float32).permute(1, 2, 0).contiguous().numpy()
    return flow.astype(np.float32)


def _track_with_raft_pairwise(
    *,
    raft,
    raft_transform,
    device: torch.device,
    frames_track: List[np.ndarray],
    query_xy: np.ndarray,
    query_t: np.ndarray,
    image_size: int,
    num_flow_updates: int,
    fb_consistency_threshold_px: float,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if len(frames_track) <= 0:
        raise ValueError("frames_track is empty")

    h0, w0 = int(frames_track[0].shape[0]), int(frames_track[0].shape[1])
    h1, w1 = _raft_resize_hw_from_max_size((h0, w0), int(image_size))
    sx = max(float(w1 - 1), 1.0) / max(float(w0 - 1), 1.0)
    sy = max(float(h1 - 1), 1.0) / max(float(h0 - 1), 1.0)

    frames_small = [
        cv2.resize(fr, (w1, h1), interpolation=cv2.INTER_LINEAR)
        if (int(fr.shape[0]) != h1 or int(fr.shape[1]) != w1)
        else fr
        for fr in frames_track
    ]

    forward_flows: List[np.ndarray] = []
    backward_flows: List[np.ndarray] = []
    for t in range(int(len(frames_small)) - 1):
        forward_flows.append(
            _raft_pair_flow(
                raft=raft,
                raft_transform=raft_transform,
                frame0=frames_small[t],
                frame1=frames_small[t + 1],
                device=device,
                num_flow_updates=int(num_flow_updates),
                use_amp=bool(use_amp),
            )
        )
        backward_flows.append(
            _raft_pair_flow(
                raft=raft,
                raft_transform=raft_transform,
                frame0=frames_small[t + 1],
                frame1=frames_small[t],
                device=device,
                num_flow_updates=int(num_flow_updates),
                use_amp=bool(use_amp),
            )
        )

    qxy = np.asarray(query_xy, dtype=np.float32).copy()
    qxy[:, 0] *= sx
    qxy[:, 1] *= sy
    tracks, conf = _propagate_sparse_tracks_with_pair_flows(
        forward_flows=forward_flows,
        backward_flows=backward_flows,
        query_xy=qxy,
        query_t=np.asarray(query_t, dtype=np.int64),
        image_hw=(h1, w1),
        fb_consistency_threshold_px=float(fb_consistency_threshold_px),
    )
    info = {
        "interp_h": int(h1),
        "interp_w": int(w1),
        "image_size": int(max(h1, w1)),
    }
    return tracks.astype(np.float32), conf.astype(np.float32), info


def _tapir_resize_hw_from_max_size(src_hw: Tuple[int, int], max_size: int) -> Tuple[int, int]:
    h0, w0 = int(src_hw[0]), int(src_hw[1])
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"Invalid source shape: {src_hw}")
    max_dim = max(h0, w0)
    if int(max_size) > 0:
        target_dim = min(max_dim, max(8, int(max_size)))
    else:
        target_dim = max_dim
    scale = float(target_dim) / float(max_dim)
    h = max(8, int(round(h0 * scale)))
    w = max(8, int(round(w0 * scale)))
    return int(h), int(w)


def _track_with_tapir_offline(
    *,
    tapir,
    device: torch.device,
    frames_track: List[np.ndarray],
    query_xy: np.ndarray,
    query_t: np.ndarray,
    image_size: int,
    query_chunk_size: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Track sparse points with the official PyTorch TAPIR model.

    TAPIR expects query points in (t, y, x), while this script keeps sparse
    points in (x, y). The returned tracks stay in the resized TAPIR raster.
    """
    if len(frames_track) <= 0:
        raise ValueError("frames_track is empty")

    t = int(len(frames_track))
    pts = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    qts = np.asarray(query_t, dtype=np.int64).reshape(-1)
    if pts.shape[0] != qts.shape[0]:
        raise ValueError(f"query_xy/query_t length mismatch: {pts.shape[0]} vs {qts.shape[0]}")
    if np.any(qts < 0) or np.any(qts >= t):
        raise ValueError(f"Invalid query_t range, expected within [0,{t-1}]")

    n = int(pts.shape[0])
    h0, w0 = int(frames_track[0].shape[0]), int(frames_track[0].shape[1])
    h1, w1 = _tapir_resize_hw_from_max_size((h0, w0), int(image_size))
    sx = max(float(w1 - 1), 1.0) / max(float(w0 - 1), 1.0)
    sy = max(float(h1 - 1), 1.0) / max(float(h0 - 1), 1.0)

    frames_small = [
        cv2.resize(fr, (w1, h1), interpolation=cv2.INTER_LINEAR)
        if (int(fr.shape[0]) != h1 or int(fr.shape[1]) != w1)
        else fr
        for fr in frames_track
    ]
    qxy = pts.copy()
    qxy[:, 0] *= sx
    qxy[:, 1] *= sy

    query_tyx = np.stack([qts.astype(np.float32), qxy[:, 1], qxy[:, 0]], axis=-1)
    video = torch.from_numpy(np.stack(frames_small, axis=0)).to(device=device, dtype=torch.float32)
    video = video.div(127.5).sub(1.0)[None]
    queries = torch.from_numpy(query_tyx).to(device=device, dtype=torch.float32)[None]

    try:
        amp_enabled = bool(use_amp) and str(device).startswith("cuda")
        autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.no_grad():
            with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
                outputs = tapir(
                    video,
                    queries,
                    query_chunk_size=max(1, int(query_chunk_size)),
                )
        tracks_n_t = outputs["tracks"][0].detach().to(device="cpu", dtype=torch.float32).numpy()
        occlusion = outputs["occlusion"][0].detach().to(device="cpu", dtype=torch.float32)
        expected_dist = outputs["expected_dist"][0].detach().to(device="cpu", dtype=torch.float32)
        confidence_n_t = ((1.0 - torch.sigmoid(occlusion)) * (1.0 - torch.sigmoid(expected_dist))).numpy()
    finally:
        del video
        del queries
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if tracks_n_t.shape != (n, t, 2):
        raise RuntimeError(f"TAPIR returned tracks with shape {tracks_n_t.shape}, expected {(n, t, 2)}")
    if confidence_n_t.shape != (n, t):
        raise RuntimeError(f"TAPIR returned confidence with shape {confidence_n_t.shape}, expected {(n, t)}")

    tracks = np.transpose(tracks_n_t, (1, 0, 2)).astype(np.float32)
    confidence = np.transpose(confidence_n_t, (1, 0)).astype(np.float32)
    info = {
        "interp_h": int(h1),
        "interp_w": int(w1),
        "image_size": int(max(h1, w1)),
        "query_chunk_size": int(query_chunk_size),
    }
    return tracks, confidence, info


def _track_with_alltracker_auto_resize(
    *,
    alltracker,
    device: torch.device,
    frames_track: List[np.ndarray],
    query_xy: np.ndarray,
    query_t: np.ndarray,
    image_size: int,
    min_image_size: int,
    downscale_step: int,
    auto_downscale_on_oom: bool,
    inference_iters: int,
    window_len: int,
    stride: int,
    confidence_channel: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Track sparse points with AllTracker, retrying smaller resolutions on CUDA OOM."""
    if len(frames_track) <= 0:
        raise ValueError("frames_track is empty")

    t = int(len(frames_track))
    n = int(query_xy.shape[0])
    h0, w0 = frames_track[0].shape[:2]
    max_dim0 = max(int(h0), int(w0))
    init_size = min(max_dim0, max(8, int(image_size)))
    floor_size = min(init_size, max(8, int(min_image_size)))

    if bool(auto_downscale_on_oom):
        candidates = _alltracker_candidate_image_sizes(
            initial=int(init_size),
            min_size=int(floor_size),
            step=int(downscale_step),
        )
    else:
        candidates = [int(init_size)]

    query_t = np.asarray(query_t, dtype=np.int64)
    if np.any(query_t < 0) or np.any(query_t >= t):
        raise ValueError(f"Invalid query_t range, expected within [0,{t-1}]")

    last_err_msg = ""
    for size in candidates:
        h1, w1 = _alltracker_resize_hw_from_max_size((h0, w0), int(size))
        sx = max(float(w1 - 1), 1.0) / max(float(w0 - 1), 1.0)
        sy = max(float(h1 - 1), 1.0) / max(float(h0 - 1), 1.0)

        frames_small = [
            cv2.resize(fr, (w1, h1), interpolation=cv2.INTER_LINEAR)
            if (fr.shape[0] != h1 or fr.shape[1] != w1)
            else fr
            for fr in frames_track
        ]
        qxy = np.asarray(query_xy, dtype=np.float32).copy()
        qxy[:, 0] *= sx
        qxy[:, 1] *= sy

        video = None
        try:
            video = (
                torch.from_numpy(np.stack(frames_small, axis=0))
                .permute(0, 3, 1, 2)[None]
                .float()
                .to(device)
            )

            tracks = np.full((t, n, 2), np.nan, dtype=np.float32)
            conf = np.zeros((t, n), dtype=np.float32)
            unique_q = sorted({int(v) for v in query_t.tolist()})
            with torch.no_grad():
                for q in unique_q:
                    idx = np.flatnonzero(query_t == q)
                    if idx.size == 0:
                        continue
                    traj_maps, conf_map = _alltracker_dense_maps_for_query_frame(
                        alltracker=alltracker,
                        video=video,
                        query_frame=int(q),
                        inference_iters=int(inference_iters),
                        window_len=int(window_len),
                        stride=int(stride),
                        confidence_channel=int(confidence_channel),
                    )
                    tr_q, cf_q = _alltracker_sample_dense_maps(
                        traj_maps=traj_maps,
                        conf_map=conf_map,
                        query_xy=qxy[idx],
                    )
                    tracks[:, idx] = tr_q
                    conf[:, idx] = cf_q

            del video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            info = {
                "interp_h": int(h1),
                "interp_w": int(w1),
                "image_size": int(size),
            }
            return tracks, conf, info
        except RuntimeError as exc:
            err_msg = str(exc)
            last_err_msg = err_msg
            if video is not None:
                del video
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            has_next = int(size) != int(candidates[-1])
            if bool(auto_downscale_on_oom) and _is_cuda_oom_error(exc) and has_next:
                print(
                    f"[WARN] AllTracker OOM at image_size={int(size)}; retrying smaller size "
                    f"(next <= {int(candidates[candidates.index(size) + 1])})"
                )
                continue
            raise

    tail = f": {last_err_msg}" if last_err_msg else ""
    raise RuntimeError(f"AllTracker failed at all candidate image sizes{tail}")


def _make_colors(n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Bright-ish RGB colors
    colors = rng.uniform(0.2, 1.0, size=(n, 3))
    colors = (colors * 255.0).astype(np.uint8)
    return colors


def _draw_tracks_rgb(
    frame_rgb: np.ndarray,
    tracks_xy: np.ndarray,  # (T, N, 2)
    vis: np.ndarray,  # (T, N) bool
    colors_rgb: np.ndarray,  # (N, 3) uint8
    *,
    t: int,
    point_radius: int = 2,
    draw_history: bool = True,
) -> np.ndarray:
    # Draw with OpenCV in BGR, then convert back.
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    n = tracks_xy.shape[1]
    for i in range(n):
        if not bool(vis[t, i]):
            continue
        color_rgb = colors_rgb[i].tolist()
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

        x, y = tracks_xy[t, i]
        xi, yi = int(round(float(x))), int(round(float(y)))
        cv2.circle(img, (xi, yi), point_radius, color_bgr, thickness=-1, lineType=cv2.LINE_AA)

        if draw_history and t > 0:
            # Only draw history for frames where the point is visible.
            pts = []
            for tt in range(0, t + 1):
                if bool(vis[tt, i]):
                    xh, yh = tracks_xy[tt, i]
                    pts.append([int(round(float(xh))), int(round(float(yh)))])
            if len(pts) >= 2:
                pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts_arr], isClosed=False, color=color_bgr, thickness=1, lineType=cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _overlay_mask_rgb(rgb: np.ndarray, mask: np.ndarray, *, color=(0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    """Overlay a (binary/soft) mask onto an RGB image for debugging."""
    rgb = np.asarray(rgb).copy()
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8)
    if m.shape[:2] != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = np.zeros_like(rgb, dtype=np.uint8)
    overlay[:, :] = np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
    idx = m.astype(bool)
    rgb[idx] = (alpha * overlay[idx] + (1.0 - alpha) * rgb[idx]).astype(np.uint8)
    return rgb


def _save_seed_debug_images(
    *,
    out_dir: Path,
    views: List[str],
    seed_frame: int,
    seed_rgb: Dict[str, np.ndarray],
    seed_mask: Dict[str, np.ndarray],
    points_seed_by_view: Dict[str, np.ndarray],
    valid_seed_by_view: Dict[str, np.ndarray],
) -> None:
    """Save per-view debug images of the seed frame: mask overlay + sampled seed points."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for vid in views:
        rgb = seed_rgb[vid]
        m = seed_mask[vid]
        img = _overlay_mask_rgb(rgb, m, color=(0, 255, 0), alpha=0.35)

        pts = np.asarray(points_seed_by_view[vid], dtype=np.float32)
        v = np.asarray(valid_seed_by_view[vid], dtype=bool).reshape(-1)
        for i, (x, y) in enumerate(pts):
            c = (0, 255, 0) if bool(v[i]) else (255, 0, 0)
            cv2.circle(img, (int(round(float(x))), int(round(float(y)))), 2, c, -1, lineType=cv2.LINE_AA)

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"seed_frame{int(seed_frame):06d}_view{vid}.png"), bgr)


def _parse_pairs_txt(pairs_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        a, b = s.split()
        pairs.append((a, b))
    return pairs


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(int(n), dtype=np.int64)
        self.rank = np.zeros((int(n),), dtype=np.uint8)

    def find(self, x: int) -> int:
        p = self.parent
        i = int(x)
        while int(p[i]) != i:
            p[i] = p[int(p[i])]
            i = int(p[i])
        return i

    def union(self, a: int, b: int) -> int:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return ra
        if int(self.rank[ra]) < int(self.rank[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        if int(self.rank[ra]) == int(self.rank[rb]):
            self.rank[ra] = np.uint8(int(self.rank[ra]) + 1)
        return ra


def _points_in_mask(mask: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    h, w = int(m.shape[0]), int(m.shape[1])
    x = np.clip(pts[:, 0].astype(np.int64), 0, max(w - 1, 0))
    y = np.clip(pts[:, 1].astype(np.int64), 0, max(h - 1, 0))
    return m[y, x] > 0


def _project_seed_geometry_point(X: np.ndarray, cam: Any) -> Optional[np.ndarray]:
    if hasattr(cam, "P"):
        P = np.asarray(cam.P, dtype=np.float64)
    elif isinstance(cam, dict):
        K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
        R = np.asarray(cam["R"], dtype=np.float64).reshape(3, 3)
        T = np.asarray(cam["T"], dtype=np.float64).reshape(3)
        P = K @ np.hstack([R, T.reshape(3, 1)])
    else:
        return None
    x = P @ np.array([float(X[0]), float(X[1]), float(X[2]), 1.0], dtype=np.float64)
    if not np.isfinite(x).all() or float(x[2]) <= 1e-12:
        return None
    return np.asarray(x[:2] / float(x[2]), dtype=np.float64)


def _triangulate_seed_geometry_robust(
    *,
    observations_xy_by_view: Dict[str, np.ndarray],
    cameras: Dict[str, Any],
    reproj_max_px: float,
    min_views: int,
) -> Tuple[Optional[np.ndarray], float, List[str]]:
    """Triangulate from the largest deterministic static-view consensus."""
    from track4d_calib.triangulation import triangulate_point_dlt, triangulate_point_multi_view

    views = sorted(str(v) for v in observations_xy_by_view if str(v) in cameras)
    threshold = float(reproj_max_px)
    required = int(max(2, min_views))
    if len(views) < required:
        return None, float("inf"), []

    def projection_matrix(view: str) -> Optional[np.ndarray]:
        cam = cameras[view]
        if hasattr(cam, "P"):
            return np.asarray(cam.P, dtype=np.float64)
        if isinstance(cam, dict):
            K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
            R = np.asarray(cam["R"], dtype=np.float64).reshape(3, 3)
            T = np.asarray(cam["T"], dtype=np.float64).reshape(3)
            return K @ np.hstack([R, T.reshape(3, 1)])
        return None

    best_key: Optional[Tuple[int, float, float, str, str]] = None
    best_inliers: List[str] = []
    best_X: Optional[np.ndarray] = None
    for i, view0 in enumerate(views):
        P0 = projection_matrix(view0)
        if P0 is None:
            continue
        for view1 in views[i + 1 :]:
            P1 = projection_matrix(view1)
            if P1 is None:
                continue
            X = triangulate_point_dlt(
                [P0, P1],
                [
                    np.asarray(observations_xy_by_view[view0], dtype=np.float64),
                    np.asarray(observations_xy_by_view[view1], dtype=np.float64),
                ],
            )
            if X is None:
                continue
            errors = {}
            for view in views:
                uv = _project_seed_geometry_point(np.asarray(X, dtype=np.float64), cameras[view])
                errors[view] = (
                    float(np.linalg.norm(uv - observations_xy_by_view[view]))
                    if uv is not None
                    else float("inf")
                )
            inliers = sorted(view for view, err in errors.items() if err <= threshold)
            if len(inliers) < required:
                continue
            inlier_errors = np.asarray([errors[view] for view in inliers], dtype=np.float64)
            key = (
                -int(len(inliers)),
                float(np.median(inlier_errors)),
                float(np.mean(inlier_errors)),
                str(view0),
                str(view1),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_inliers = inliers
                best_X = np.asarray(X, dtype=np.float64)

    if best_X is None or len(best_inliers) < required:
        return None, float("inf"), []

    refit_X, refit_error, _ = triangulate_point_multi_view(
        cameras={view: cameras[view] for view in best_inliers},
        points_2d={view: observations_xy_by_view[view] for view in best_inliers},
        visibility={view: True for view in best_inliers},
        reprojection_error_threshold=threshold,
        min_views_for_triangulation=required,
    )
    if refit_X is not None:
        return np.asarray(refit_X, dtype=np.float64), float(refit_error), best_inliers

    best_errors = []
    for view in best_inliers:
        uv = _project_seed_geometry_point(best_X, cameras[view])
        if uv is None:
            return None, float("inf"), []
        best_errors.append(float(np.linalg.norm(uv - observations_xy_by_view[view])))
    return best_X, float(np.mean(best_errors)), best_inliers


def _filter_spsg_seed_observations_by_geometry(
    observations_xy_by_view: Dict[str, np.ndarray],
    *,
    cameras: Dict[str, Any],
    ref_view: str,
    require_ref_view: bool = True,
    bridge_view_ids: set[str],
    bridge_observation_policy: str = "reproject",
    reproj_max_px: float,
    min_views: int,
    min_anchor_views: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Geometry-filter static observations without circularly gating repair evidence."""
    bridge_policy = str(bridge_observation_policy).strip().lower()
    if bridge_policy not in {"reproject", "retain"}:
        raise ValueError(
            "bridge_observation_policy must be one of {'reproject', 'retain'}, "
            f"got {bridge_observation_policy!r}"
        )
    obs = {
        str(v): np.asarray(xy, dtype=np.float64).reshape(2)
        for v, xy in observations_xy_by_view.items()
        if str(v) in cameras and np.isfinite(np.asarray(xy, dtype=np.float64)).all()
    }
    diag: Dict[str, Any] = {
        "accepted": False,
        "reason": "",
        "num_obs_before": int(len(obs)),
        "num_obs_after": 0,
        "removed_views": [],
        "anchor_views_used": 0,
        "triangulation_error_px": float("inf"),
        "require_ref_view": bool(require_ref_view),
        "bridge_observation_policy": bridge_policy,
        "retained_bridge_views": [],
    }

    ref_key = str(ref_view)
    if bool(require_ref_view) and ref_key not in obs:
        diag["reason"] = "missing_ref_view"
        return {}, diag

    threshold = float(reproj_max_px)
    if threshold <= 0.0:
        diag.update(
            {
                "accepted": int(len(obs)) >= int(min_views),
                "reason": "disabled" if int(len(obs)) >= int(min_views) else "too_few_views",
                "num_obs_after": int(len(obs)) if int(len(obs)) >= int(min_views) else 0,
            }
        )
        return (dict(obs) if int(len(obs)) >= int(min_views) else {}), diag

    bridge_set = {str(v) for v in bridge_view_ids}
    anchor_views = [v for v in obs.keys() if v not in bridge_set]
    if len(anchor_views) < int(min_anchor_views):
        diag["reason"] = "too_few_anchor_views"
        diag["anchor_views_used"] = int(len(anchor_views))
        return {}, diag

    X, tri_err, anchor_inliers = _triangulate_seed_geometry_robust(
        observations_xy_by_view={v: obs[v] for v in anchor_views},
        cameras={v: cameras[v] for v in anchor_views},
        reproj_max_px=float(threshold),
        min_views=int(min_anchor_views),
    )
    diag["anchor_views_used"] = int(len(anchor_inliers))
    diag["anchor_inlier_views"] = list(anchor_inliers)
    diag["triangulation_error_px"] = float(tri_err)
    if X is None:
        diag["reason"] = "triangulation_failed"
        return {}, diag

    keep: Dict[str, np.ndarray] = {}
    removed: List[str] = []
    retained_bridge_views: List[str] = []
    errors: Dict[str, float] = {}
    for view, xy in obs.items():
        uv = _project_seed_geometry_point(np.asarray(X, dtype=np.float64), cameras[view])
        if str(view) in bridge_set and bridge_policy == "retain":
            errors[view] = (
                float(np.linalg.norm(uv - xy)) if uv is not None else float("inf")
            )
            keep[view] = np.asarray(xy, dtype=np.float64)
            retained_bridge_views.append(str(view))
            continue
        if uv is None:
            errors[view] = float("inf")
            removed.append(str(view))
            continue
        err = float(np.linalg.norm(uv - xy))
        errors[view] = err
        if err <= threshold:
            keep[view] = np.asarray(xy, dtype=np.float64)
        else:
            removed.append(str(view))

    diag["errors_px"] = errors
    diag["removed_views"] = sorted(removed)
    diag["retained_bridge_views"] = sorted(retained_bridge_views)
    diag["num_obs_after"] = int(len(keep))
    if bool(require_ref_view) and ref_key not in keep:
        diag["reason"] = "ref_view_outlier"
        return {}, diag
    if len(keep) < int(min_views):
        diag["reason"] = "too_few_views_after_filter"
        return {}, diag

    diag["accepted"] = True
    diag["reason"] = "accepted"
    return keep, diag


def _select_spsg_landmarks_with_bridge_coverage(
    candidate_landmarks: List[Tuple[float, Dict[str, int]]],
    *,
    num_points: int,
    bridge_view_ids: Sequence[str],
    bridge_min_points_per_view: int,
    seed: int,
) -> Tuple[List[Tuple[float, Dict[str, int]]], Dict[str, Any]]:
    """Select SPSG landmarks while reserving coverage for bridge views."""
    candidates = sorted(
        [(float(score), dict(obs)) for score, obs in candidate_landmarks],
        key=lambda x: float(x[0]),
        reverse=True,
    )
    target_total = int(max(0, int(num_points)))
    bridge_views = [str(v) for v in bridge_view_ids]
    min_per_view = int(max(0, int(bridge_min_points_per_view)))
    diag: Dict[str, Any] = {
        "bridge_coverage_min_points_per_view": int(min_per_view),
        "bridge_coverage_requested_views": list(bridge_views),
        "bridge_coverage_counts": {str(v): 0 for v in bridge_views},
        "bridge_coverage_unique_selected": 0,
        "bridge_coverage_sampled_with_replacement": 0,
        "bridge_coverage_candidate_shortfall": 0,
    }
    if target_total <= 0 or len(candidates) <= 0:
        return [], diag

    unique_target = int(min(target_total, len(candidates)))
    selected_idx: List[int] = []
    selected_set: set[int] = set()

    def _counts(indices: Sequence[int]) -> Dict[str, int]:
        counts = {str(v): 0 for v in bridge_views}
        for idx in indices:
            obs = candidates[int(idx)][1]
            for view in bridge_views:
                if str(view) in obs:
                    counts[str(view)] += 1
        return counts

    if bridge_views and min_per_view > 0:
        for view in bridge_views:
            while len(selected_idx) < unique_target:
                counts_now = _counts(selected_idx)
                if int(counts_now.get(str(view), 0)) >= int(min_per_view):
                    break
                next_idx = None
                for cand_idx, (_score, obs) in enumerate(candidates):
                    if cand_idx in selected_set:
                        continue
                    if str(view) in obs:
                        next_idx = int(cand_idx)
                        break
                if next_idx is None:
                    break
                selected_idx.append(int(next_idx))
                selected_set.add(int(next_idx))

    for cand_idx in range(len(candidates)):
        if len(selected_idx) >= unique_target:
            break
        if cand_idx in selected_set:
            continue
        selected_idx.append(int(cand_idx))
        selected_set.add(int(cand_idx))

    selected_idx_full: List[int] = [int(i) for i in selected_idx]
    selected = [candidates[int(i)] for i in selected_idx_full]

    diag["bridge_coverage_unique_selected"] = int(len(selected_idx))
    diag["bridge_coverage_candidate_shortfall"] = int(max(0, target_total - len(selected)))
    diag["bridge_coverage_counts"] = _counts(selected_idx_full)
    return selected, diag


def _resolve_spsg_seed_geometry_camera_dir(cameras_root: Path, frame_i: int) -> Optional[Path]:
    root = Path(cameras_root)
    frame_tag = f"f{int(frame_i):06d}"
    candidates = [
        root / frame_tag / "cameras_pred",
        root / frame_tag / "cameras_gt",
        root / frame_tag / "cameras",
        root / frame_tag,
        root / "cameras_pred",
        root / "cameras_gt",
        root / "cameras",
        root,
    ]
    for cand in candidates:
        if (cand / "intri.yml").is_file() and (cand / "extri.yml").is_file():
            return cand
    return None


def _build_seed_points_spsg_global(
    *,
    sf_rgb: Dict[str, np.ndarray],
    sf_mask: Dict[str, np.ndarray],
    views: List[str],
    ref_view: str,
    num_points: int,
    min_visible_views: int,
    seed: int,
    spsg_f_ransac_px: float,
    seed_cameras: Optional[Dict[str, Any]] = None,
    bridge_view_ids: Optional[List[str]] = None,
    bridge_min_points_per_view: int = 0,
    geometry_reproj_px: float = 0.0,
    geometry_min_anchor_views: int = 3,
    geometry_bridge_observation_policy: str = "reproject",
    require_ref_view: bool = True,
) -> Dict[str, Any]:
    import h5py

    from hloc import extract_features, match_features, pairs_from_exhaustive
    from hloc.utils.parsers import names_to_pair

    if int(num_points) <= 0:
        raise ValueError("num_points must be > 0")
    if str(ref_view) not in views:
        raise ValueError(f"ref_view={ref_view} not found in views={views}")

    image_name_by_view = {str(v): f"view{v}.png" for v in views}
    view_by_image_name = {name: vid for vid, name in image_name_by_view.items()}
    image_names = [image_name_by_view[str(v)] for v in views]
    other_views = [str(v) for v in views if str(v) != str(ref_view)]

    with tempfile.TemporaryDirectory(prefix="track4d_calib_spsg_seed_") as td:
        tmp_root = Path(td)
        image_dir = tmp_root / "images"
        hloc_dir = tmp_root / "hloc"
        image_dir.mkdir(parents=True, exist_ok=True)
        hloc_dir.mkdir(parents=True, exist_ok=True)

        for vid in views:
            name = image_name_by_view[str(vid)]
            rgb = np.asarray(sf_rgb[str(vid)], dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(image_dir / name), bgr)
            if not ok:
                raise RuntimeError(f"Failed to write seed image for SPSG: {image_dir / name}")

        pairs_path = hloc_dir / "pairs.txt"
        pairs_from_exhaustive.main(pairs_path, image_list=image_names)
        pairs = _parse_pairs_txt(pairs_path)

        feat_conf = extract_features.confs["superpoint_aachen"]
        match_conf = match_features.confs["superglue"]

        features_path = extract_features.main(
            feat_conf,
            image_dir=image_dir,
            export_dir=hloc_dir,
            image_list=image_names,
            overwrite=True,
        )

        matches_out = hloc_dir / f"{features_path.stem}_{match_conf['output']}_{pairs_path.stem}.h5"
        matches_path = match_features.main(
            match_conf,
            pairs=pairs_path,
            features=features_path,
            matches=matches_out,
            overwrite=True,
        )

        keypoints_by_view: Dict[str, np.ndarray] = {}
        keypoint_count_by_view: Dict[str, int] = {}
        with h5py.File(features_path, "r") as f_feat:
            for vid in views:
                name = image_name_by_view[str(vid)]
                if name not in f_feat:
                    raise RuntimeError(f"Missing keypoints for {name} in {features_path}")
                kps = np.asarray(f_feat[name]["keypoints"], dtype=np.float32)
                keypoints_by_view[str(vid)] = kps
                keypoint_count_by_view[str(vid)] = int(kps.shape[0])

        node_offset: Dict[str, int] = {}
        total_nodes = 0
        for vid in views:
            node_offset[str(vid)] = int(total_nodes)
            total_nodes += int(keypoint_count_by_view[str(vid)])

        if int(total_nodes) <= 0:
            raise RuntimeError("SPSG seed builder: no keypoints extracted")

        node_view_idx = np.zeros((int(total_nodes),), dtype=np.int16)
        node_kp_idx = np.zeros((int(total_nodes),), dtype=np.int32)
        for view_idx, vid in enumerate(views):
            off = int(node_offset[str(vid)])
            n_kp = int(keypoint_count_by_view[str(vid)])
            if n_kp <= 0:
                continue
            node_view_idx[off : off + n_kp] = np.int16(int(view_idx))
            node_kp_idx[off : off + n_kp] = np.arange(n_kp, dtype=np.int32)

        uf = _UnionFind(int(total_nodes))
        node_score = np.zeros((int(total_nodes),), dtype=np.float32)
        node_active = np.zeros((int(total_nodes),), dtype=bool)

        pair_total = int(len(pairs))
        pair_used = 0
        match_inlier_total = 0
        f_method = int(getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC))

        with h5py.File(matches_path, "r") as f_match:
            for name0, name1 in pairs:
                vid0 = view_by_image_name.get(str(name0), None)
                vid1 = view_by_image_name.get(str(name1), None)
                if vid0 is None or vid1 is None:
                    continue

                pair_key = names_to_pair(name0, name1)
                grp = None
                if pair_key in f_match:
                    grp = f_match[pair_key]
                elif str(name0) in f_match and str(name1) in f_match[str(name0)]:
                    grp = f_match[str(name0)][str(name1)]
                if grp is None:
                    continue

                if "matches0" not in grp:
                    continue
                matches0 = np.asarray(grp["matches0"])
                if matches0.ndim != 1:
                    matches0 = matches0.reshape(-1)
                idx0 = np.flatnonzero(matches0 >= 0)
                if idx0.size < 8:
                    continue
                idx1 = matches0[idx0].astype(np.int64)

                kps0 = keypoints_by_view[str(vid0)]
                kps1 = keypoints_by_view[str(vid1)]
                valid_idx1 = (idx1 >= 0) & (idx1 < int(kps1.shape[0]))
                if not np.any(valid_idx1):
                    continue
                idx0 = idx0[valid_idx1]
                idx1 = idx1[valid_idx1]
                if idx0.size < 8:
                    continue

                pts0 = kps0[idx0]
                pts1 = kps1[idx1]

                keep_mask = _points_in_mask(sf_mask[str(vid0)], pts0) & _points_in_mask(sf_mask[str(vid1)], pts1)
                if not np.any(keep_mask):
                    continue
                idx0 = idx0[keep_mask]
                idx1 = idx1[keep_mask]
                pts0 = pts0[keep_mask]
                pts1 = pts1[keep_mask]
                if idx0.size < 8:
                    continue

                try:
                    _, inlier_mask = cv2.findFundamentalMat(
                        pts0,
                        pts1,
                        method=f_method,
                        ransacReprojThreshold=float(spsg_f_ransac_px),
                        confidence=0.999,
                        maxIters=2000,
                    )
                except Exception:
                    inlier_mask = None

                if inlier_mask is None:
                    continue

                inlier = np.asarray(inlier_mask).reshape(-1) > 0
                if not np.any(inlier):
                    continue

                idx0 = idx0[inlier]
                idx1 = idx1[inlier]
                if idx0.size < 8:
                    continue

                if "matching_scores0" in grp:
                    scores0 = np.asarray(grp["matching_scores0"]).reshape(-1)
                    if scores0.shape[0] >= int(matches0.shape[0]):
                        edge_score = np.asarray(scores0[idx0], dtype=np.float32)
                    else:
                        edge_score = np.ones((idx0.shape[0],), dtype=np.float32)
                else:
                    edge_score = np.ones((idx0.shape[0],), dtype=np.float32)

                pair_used += 1
                match_inlier_total += int(idx0.shape[0])

                off0 = int(node_offset[str(vid0)])
                off1 = int(node_offset[str(vid1)])
                for i in range(int(idx0.shape[0])):
                    n0 = int(off0 + int(idx0[i]))
                    n1 = int(off1 + int(idx1[i]))
                    uf.union(n0, n1)
                    s = float(max(edge_score[i], 1e-6))
                    node_score[n0] += s
                    node_score[n1] += s
                    node_active[n0] = True
                    node_active[n1] = True

    active_nodes = np.flatnonzero(node_active)
    if active_nodes.size <= 0:
        raise RuntimeError("SPSG seed builder: no inlier matches survived filtering")

    components: Dict[int, List[int]] = {}
    for node in active_nodes.tolist():
        root = int(uf.find(int(node)))
        components.setdefault(root, []).append(int(node))

    min_comp_views = max(2, int(min_visible_views))
    min_comp_views = min(int(min_comp_views), int(len(views)))
    conflict_components = 0
    candidate_landmarks: List[Tuple[float, Dict[str, int]]] = []
    geometry_enabled = seed_cameras is not None and float(geometry_reproj_px) > 0.0
    geometry_bridge_views = {str(v) for v in (bridge_view_ids or [])}
    geometry_rejected = 0
    geometry_trimmed = 0
    geometry_removed_observations = 0

    for nodes in components.values():
        best_node_by_view: Dict[str, int] = {}
        has_conflict = False
        for node in nodes:
            vid = str(views[int(node_view_idx[int(node)])])
            prev = best_node_by_view.get(vid, None)
            if prev is None:
                best_node_by_view[vid] = int(node)
            elif float(node_score[int(node)]) > float(node_score[int(prev)]):
                best_node_by_view[vid] = int(node)
                has_conflict = True
            else:
                has_conflict = True
        if has_conflict:
            conflict_components += 1

        if bool(require_ref_view) and str(ref_view) not in best_node_by_view:
            continue
        if int(len(best_node_by_view)) < int(min_comp_views):
            continue

        if geometry_enabled:
            obs_xy_by_view = {}
            for vid, node in best_node_by_view.items():
                kp_idx = int(node_kp_idx[int(node)])
                obs_xy_by_view[str(vid)] = np.asarray(keypoints_by_view[str(vid)][kp_idx], dtype=np.float64)
            filtered_obs, geom_diag = _filter_spsg_seed_observations_by_geometry(
                obs_xy_by_view,
                cameras=seed_cameras or {},
                ref_view=str(ref_view),
                require_ref_view=bool(require_ref_view),
                bridge_view_ids=geometry_bridge_views,
                bridge_observation_policy=str(geometry_bridge_observation_policy),
                reproj_max_px=float(geometry_reproj_px),
                min_views=int(min_comp_views),
                min_anchor_views=int(max(2, geometry_min_anchor_views)),
            )
            if not bool(geom_diag.get("accepted", False)):
                geometry_rejected += 1
                continue
            if len(filtered_obs) < len(best_node_by_view):
                geometry_trimmed += 1
                geometry_removed_observations += int(len(best_node_by_view) - len(filtered_obs))
                best_node_by_view = {vid: best_node_by_view[vid] for vid in filtered_obs.keys()}

        comp_score = float(sum(float(node_score[int(n)]) for n in best_node_by_view.values()))
        comp_score *= float(len(best_node_by_view))
        candidate_landmarks.append((float(comp_score), best_node_by_view))

    if len(candidate_landmarks) <= 0:
        raise RuntimeError(
            "SPSG seed builder: no valid global landmarks after DSU merge "
            f"(min_visible_views={min_comp_views}, require ref_view={ref_view})"
        )

    selected_scored, bridge_coverage_diag = _select_spsg_landmarks_with_bridge_coverage(
        candidate_landmarks,
        num_points=int(num_points),
        bridge_view_ids=[str(v) for v in (bridge_view_ids or [])],
        bridge_min_points_per_view=int(bridge_min_points_per_view),
        seed=int(seed),
    )
    selected_landmarks: List[Dict[str, int]] = [dict(obs) for _score, obs in selected_scored]
    unique_selected = int(bridge_coverage_diag.get("bridge_coverage_unique_selected", 0))

    selected_count = int(len(selected_landmarks))
    ref_xy = np.zeros((selected_count, 2), dtype=np.float32)
    ref_valid = np.zeros((selected_count,), dtype=bool)
    xy_by_view: Dict[str, np.ndarray] = {vid: np.zeros((selected_count, 2), dtype=np.float32) for vid in other_views}
    valid_by_view: Dict[str, np.ndarray] = {vid: np.zeros((selected_count,), dtype=bool) for vid in other_views}

    for i, obs_by_view in enumerate(selected_landmarks):
        ref_node_raw = obs_by_view.get(str(ref_view), None)
        if ref_node_raw is not None:
            ref_node = int(ref_node_raw)
            ref_kp_idx = int(node_kp_idx[ref_node])
            ref_xy[i] = np.asarray(keypoints_by_view[str(ref_view)][ref_kp_idx], dtype=np.float32)
            ref_valid[i] = True

        for vid in other_views:
            node = obs_by_view.get(str(vid), None)
            if node is None:
                continue
            kp_idx = int(node_kp_idx[int(node)])
            xy_by_view[str(vid)][i] = np.asarray(keypoints_by_view[str(vid)][kp_idx], dtype=np.float32)
            valid_by_view[str(vid)][i] = True

    diag = {
        "pair_total": int(pair_total),
        "pair_used": int(pair_used),
        "match_inlier_total": int(match_inlier_total),
        "component_total": int(len(components)),
        "component_conflict_total": int(conflict_components),
        "landmark_candidates": int(len(candidate_landmarks)),
        "landmark_selected_unique": int(unique_selected),
        "landmark_selected_total": int(len(selected_landmarks)),
        "min_component_views": int(min_comp_views),
        "f_ransac_px": float(spsg_f_ransac_px),
        "spsg_seed_geometry_filtered": {
            "enabled": bool(geometry_enabled),
            "reproj_px": float(geometry_reproj_px),
            "min_anchor_views": int(max(2, geometry_min_anchor_views)),
            "bridge_observation_policy": str(geometry_bridge_observation_policy),
            "require_ref_view": bool(require_ref_view),
            "bridge_view_count": int(len(geometry_bridge_views)),
            "rejected_components": int(geometry_rejected),
            "trimmed_components": int(geometry_trimmed),
            "removed_observations": int(geometry_removed_observations),
        },
        "spsg_bridge_coverage_selection": bridge_coverage_diag,
    }

    return {
        "ref_xy": ref_xy.astype(np.float32),
        "ref_valid": ref_valid.astype(bool),
        "xy_by_view": {k: v.astype(np.float32) for k, v in xy_by_view.items()},
        "valid_by_view": {k: v.astype(bool) for k, v in valid_by_view.items()},
        "diag": diag,
    }


def _seed_output_point_count(seed_out: Dict[str, Any], *, other_views: Sequence[str]) -> int:
    ref_xy = np.asarray(seed_out["ref_xy"])
    if ref_xy.ndim != 2 or int(ref_xy.shape[1]) != 2:
        raise ValueError(f"seed ref_xy must have shape (N,2), got {ref_xy.shape}")
    n = int(ref_xy.shape[0])
    xy_by_view = seed_out.get("xy_by_view", {})
    valid_by_view = seed_out.get("valid_by_view", {})
    for vid in other_views:
        xy = np.asarray(xy_by_view[str(vid)])
        valid = np.asarray(valid_by_view[str(vid)])
        if xy.shape != (n, 2):
            raise ValueError(f"seed xy_by_view[{vid}] must have shape ({n},2), got {xy.shape}")
        if valid.shape != (n,):
            raise ValueError(f"seed valid_by_view[{vid}] must have shape ({n},), got {valid.shape}")
    return n


def _expand_seed_query_offsets(query_offsets: Sequence[int], point_counts: Sequence[int]) -> np.ndarray:
    if len(query_offsets) != len(point_counts):
        raise ValueError("query_offsets and point_counts must have the same length")
    chunks: List[np.ndarray] = []
    for query_t, count in zip(query_offsets, point_counts):
        n = int(count)
        if n < 0:
            raise ValueError("point_counts must be non-negative")
        if n == 0:
            continue
        chunks.append(np.full((n,), int(query_t), dtype=np.int64))
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(chunks, axis=0)


def _build_seed_count_metadata(
    *,
    seed_frames: Sequence[int],
    point_counts: Sequence[int],
    requested_points_per_seed: int,
) -> Dict[str, Any]:
    frames = [int(frame) for frame in seed_frames]
    counts = [int(count) for count in point_counts]
    if len(frames) != len(counts):
        raise ValueError("seed_frames and point_counts must have the same length")
    if any(count < 0 for count in counts):
        raise ValueError("point_counts must be non-negative")
    return {
        "num_points": int(sum(counts)),
        "num_points_by_seed": counts,
        "num_points_per_seed": int(requested_points_per_seed),
        "num_seeds": int(len(frames)),
    }


def _open_video_writer(path: Path, *, fps: int, quality: int, codec: str):
    import imageio

    if str(codec).lower() != "auto":
        return imageio.get_writer(str(path), fps=int(fps), codec=str(codec), quality=int(quality))
    last_err: Exception | None = None
    for cand in ("libx264", "libopenh264", "mpeg4"):
        try:
            return imageio.get_writer(str(path), fps=int(fps), codec=cand, quality=int(quality))
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to open video writer for {path} with any known codec") from last_err


def _visibility_from_raw(
    vis_raw: np.ndarray,
    *,
    threshold: float = 0.0,
    hysteresis_low: float = 0.0,
    hysteresis_high: float = 0.0,
) -> np.ndarray:
    """Convert raw tracker visibility scores to a boolean mask.

    - A threshold <= 0 uses ``astype(bool)``.
    - If raw visibility is already bool, thresholds are ignored.
    - Optional hysteresis reduces flicker when score hovers around threshold.
    """
    arr = np.asarray(vis_raw)
    if arr.ndim != 2:
        raise ValueError(f"visibility must have shape (T,N), got {arr.shape}")

    if arr.dtype == np.bool_:
        return arr.copy()

    if float(threshold) > 0.0:
        vis = arr >= float(threshold)
    else:
        vis = arr.astype(bool)

    low = float(hysteresis_low)
    high = float(hysteresis_high)
    if not (high > 0.0 or low > 0.0):
        return vis.astype(bool)

    hi = max(high, low)
    lo = min(high, low)
    out = np.zeros_like(vis, dtype=bool)
    out[0] = arr[0] >= hi
    for t in range(1, arr.shape[0]):
        s = arr[t]
        turn_on = s >= hi
        turn_off = s <= lo
        out[t] = np.where(turn_on, True, np.where(turn_off, False, out[t - 1]))
    return out


def _apply_track_sanity_filters(
    *,
    tracks_xy: np.ndarray,
    vis: np.ndarray,
    image_hw: Tuple[int, int],
    border_margin_px: float = 0.0,
    max_step_px: float = 0.0,
    reentry_dist_px: float = 0.0,
    reentry_lost_frames: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Filter implausible track states with lightweight geometric checks.

    Args:
      tracks_xy: (T,N,2) in original image pixels.
      vis: (T,N) bool mask.
      image_hw: (H,W) of original view.
      border_margin_px: invalidate points too close to image border.
      max_step_px: invalidate per-frame jumps larger than this threshold.
      reentry_dist_px: after being lost for ``reentry_lost_frames`` frames,
          reject a reappearance if distance to last valid point is too large.
    """
    tr = np.asarray(tracks_xy, dtype=np.float32)
    vs = np.asarray(vis, dtype=bool).copy()
    if tr.ndim != 3 or tr.shape[2] != 2:
        raise ValueError(f"tracks_xy must have shape (T,N,2), got {tr.shape}")
    if vs.shape != tr.shape[:2]:
        raise ValueError(f"vis shape {vs.shape} mismatches tracks {tr.shape[:2]}")

    h, w = int(image_hw[0]), int(image_hw[1])
    stats: Dict[str, float] = {
        "visible_before": float(vs.sum()),
        "removed_nonfinite": 0.0,
        "removed_border": 0.0,
        "removed_step": 0.0,
        "removed_reentry": 0.0,
        "visible_after": 0.0,
    }

    finite = np.isfinite(tr[..., 0]) & np.isfinite(tr[..., 1])
    removed_nonfinite = vs & (~finite)
    stats["removed_nonfinite"] = float(removed_nonfinite.sum())
    vs &= finite

    margin = float(border_margin_px)
    if margin > 0.0:
        inside = (
            (tr[..., 0] >= margin)
            & (tr[..., 0] <= (w - 1 - margin))
            & (tr[..., 1] >= margin)
            & (tr[..., 1] <= (h - 1 - margin))
        )
        removed_border = vs & (~inside)
        stats["removed_border"] = float(removed_border.sum())
        vs &= inside

    step_thr = float(max_step_px)
    if step_thr > 0.0 and tr.shape[0] > 1:
        step = np.linalg.norm(tr[1:] - tr[:-1], axis=-1)
        removed_step = 0
        for t in range(1, tr.shape[0]):
            bad = vs[t] & vs[t - 1] & (step[t - 1] > step_thr)
            if np.any(bad):
                removed_step += int(bad.sum())
                vs[t, bad] = False
        stats["removed_step"] = float(removed_step)

    reentry_thr = float(reentry_dist_px)
    lost_need = int(max(1, reentry_lost_frames))
    if reentry_thr > 0.0 and tr.shape[0] > 1:
        removed_reentry = 0
        num_points = tr.shape[1]
        for point_id in range(num_points):
            last_xy = None
            lost_streak = 0
            for t in range(tr.shape[0]):
                if not bool(vs[t, point_id]):
                    lost_streak += 1
                    continue
                xy = tr[t, point_id]
                if last_xy is not None and lost_streak >= lost_need:
                    jump = float(np.linalg.norm(xy - last_xy))
                    if jump > reentry_thr:
                        vs[t, point_id] = False
                        removed_reentry += 1
                        lost_streak += 1
                        continue
                last_xy = xy
                lost_streak = 0
        stats["removed_reentry"] = float(removed_reentry)

    stats["visible_after"] = float(vs.sum())
    before = max(stats["visible_before"], 1.0)
    stats["visible_keep_ratio"] = float(stats["visible_after"] / before)
    return vs, stats


def _compute_track_quality_weights(
    *,
    tracks_xy: np.ndarray,
    vis: np.ndarray,
    vis_raw: np.ndarray | None,
    image_hw: Tuple[int, int],
    vis_weight: float = 0.7,
    border_margin_px: float = 0.0,
    step_px: float = 0.0,
    reentry_dist_px: float = 0.0,
    reentry_lost_frames: int = 3,
    min_weight: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute per-observation soft quality weights in [0,1].

    This is a training-free reliability score used by downstream localization.
    It preserves observations but downweights likely-bad ones (border/large jump/
    suspicious re-entry), unlike hard rejection.
    """
    tr = np.asarray(tracks_xy, dtype=np.float32)
    vs = np.asarray(vis, dtype=bool)
    if tr.ndim != 3 or tr.shape[2] != 2:
        raise ValueError(f"tracks_xy must have shape (T,N,2), got {tr.shape}")
    if vs.shape != tr.shape[:2]:
        raise ValueError(f"vis shape {vs.shape} mismatches tracks {tr.shape[:2]}")

    h, w = int(image_hw[0]), int(image_hw[1])
    ww = np.ones(vs.shape, dtype=np.float32)

    raw_factor_mean = 1.0
    if vis_raw is not None:
        vr = np.asarray(vis_raw)
        if vr.shape == vs.shape and vr.dtype != np.bool_:
            vr = np.nan_to_num(vr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            vr = np.clip(vr, 0.0, 1.0)
            blend = float(np.clip(vis_weight, 0.0, 1.0))
            raw_factor = (1.0 - blend) + blend * vr
            raw_factor_mean = float(raw_factor[vs].mean()) if np.any(vs) else 1.0
            ww *= raw_factor

    border_factor_mean = 1.0
    margin = float(border_margin_px)
    if margin > 0.0:
        dist_x = np.minimum(tr[..., 0], (w - 1) - tr[..., 0])
        dist_y = np.minimum(tr[..., 1], (h - 1) - tr[..., 1])
        dist_to_border = np.minimum(dist_x, dist_y)
        scale = np.clip(dist_to_border / max(margin, 1e-6), 0.0, 1.0)
        # Keep a floor > 0 so we downweight border observations instead of deleting.
        border_factor = 0.25 + 0.75 * scale
        border_factor_mean = float(border_factor[vs].mean()) if np.any(vs) else 1.0
        ww *= border_factor.astype(np.float32)

    step_factor_mean = 1.0
    step_thr = float(step_px)
    if step_thr > 0.0 and tr.shape[0] > 1:
        step = np.linalg.norm(tr[1:] - tr[:-1], axis=-1)
        step_factor = np.ones_like(step, dtype=np.float32)
        bad = step > step_thr
        if np.any(bad):
            step_factor[bad] = np.clip((step_thr / np.maximum(step[bad], 1e-6)).astype(np.float32), 0.1, 1.0)
        ww[1:] *= step_factor
        valid_step = vs[1:] & vs[:-1]
        step_factor_mean = float(step_factor[valid_step].mean()) if np.any(valid_step) else 1.0

    reentry_penalized = 0
    reentry_thr = float(reentry_dist_px)
    lost_need = int(max(1, reentry_lost_frames))
    if reentry_thr > 0.0 and tr.shape[0] > 1:
        num_points = tr.shape[1]
        for point_id in range(num_points):
            last_xy = None
            lost_streak = 0
            for t in range(tr.shape[0]):
                if not bool(vs[t, point_id]):
                    lost_streak += 1
                    continue
                xy = tr[t, point_id]
                if last_xy is not None and lost_streak >= lost_need:
                    jump = float(np.linalg.norm(xy - last_xy))
                    if jump > reentry_thr:
                        factor = float(np.clip(reentry_thr / max(jump, 1e-6), 0.1, 1.0))
                        ww[t, point_id] *= factor
                        reentry_penalized += 1
                last_xy = xy
                lost_streak = 0

    # Invalidate invisible observations; clamp visible weights.
    ww *= vs.astype(np.float32)
    min_w = float(np.clip(min_weight, 0.0, 1.0))
    ww[vs] = np.clip(ww[vs], min_w, 1.0)

    stats: Dict[str, float] = {
        "quality_mean_visible": float(ww[vs].mean()) if np.any(vs) else 0.0,
        "quality_median_visible": float(np.median(ww[vs])) if np.any(vs) else 0.0,
        "quality_min_visible": float(ww[vs].min()) if np.any(vs) else 0.0,
        "quality_max_visible": float(ww[vs].max()) if np.any(vs) else 0.0,
        "raw_factor_mean": float(raw_factor_mean),
        "border_factor_mean": float(border_factor_mean),
        "step_factor_mean": float(step_factor_mean),
        "reentry_penalized": float(reentry_penalized),
        "visible_count": float(vs.sum()),
    }
    return ww.astype(np.float32), stats


def main(*, track_view_sink=None) -> int:
    parser = argparse.ArgumentParser()
    default_data_root = "data/multiview_tracker/0012_09/images"
    parser.add_argument("--data_root", type=str, default=default_data_root)
    parser.add_argument("--dataset_root", type=str, default="", help="Optional dataset root (contains images/). If set, overrides default --data_root.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="standard",
        choices=["standard", "custom", "diffuman4d", "panoptic"],
        help="Dataset layout type. Non-standard types expect a dataset config JSON.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="",
        help="Optional dataset layout JSON (defaults to <dataset_root>/track4d_calib_dataset.json for non-standard layouts).",
    )
    parser.add_argument("--views", type=str, nargs="+", default=["14", "15", "16"])
    parser.add_argument("--ref_view", type=str, default="15")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--seed_frame",
        type=int,
        default=-1,
        help="Frame index used for RoMaV2 matching + point seeding. Default: start_frame.",
    )
    parser.add_argument(
        "--seed_frames",
        type=int,
        nargs="*",
        default=[],
        help="Additional seed frames for multi-seed matching (in addition to --seed_frame).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of frames to process. Use 0 to process all frames (uses the common min-length over views).",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=200,
        help=(
            "Requested seed landmarks per seed frame. For anchor_map, this is a "
            "per-view contribution cap before packing the union of COLMAP point3D IDs."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed_builder",
        type=str,
        default="roma_ref",
        choices=[
            "roma_ref",
            "spsg_global",
            "spsg_target_star",
            "spsg_target_closure",
            "spsg_target_verified_closure",
            "spsg_target_relative_closure",
            "spsg_target_relay",
            "anchor_map",
        ],
        help=(
            "Seed-frame correspondence builder: RoMa(ref->all), SuperPoint+SuperGlue "
            "global-ID DSU, direct target-to-support star IDs, target stars closed by "
            "target-target edges, direct target-centered relay IDs, or per-view observations "
            "from an existing COLMAP anchor map."
        ),
    )
    parser.add_argument(
        "--anchor_map_model_dir",
        type=str,
        default="",
        help="COLMAP model whose point3D IDs seed per-view temporal tracks for seed_builder=anchor_map.",
    )
    parser.add_argument(
        "--spsg_f_ransac_px",
        type=float,
        default=2.0,
        help="RANSAC reprojection threshold (px) for SPSG pair filtering before DSU merge.",
    )
    parser.add_argument(
        "--spsg_feature_conf",
        choices=["superpoint_aachen", "superpoint_max", "superpoint_native"],
        default="superpoint_aachen",
        help=(
            "HLoc SuperPoint extraction config for SPSG seed builders. "
            "superpoint_max uses the existing 1600 px localization-oriented config; "
            "superpoint_native preserves up-to-4096 px input coordinates for "
            "high-focal cross-view localization."
        ),
    )
    parser.add_argument(
        "--spsg_seed_geometry_cameras_root",
        type=str,
        default="",
        help="Optional seed-frame camera root for geometry-filtering SPSG global landmarks.",
    )
    parser.add_argument(
        "--spsg_seed_geometry_reproj_px",
        type=float,
        default=0.0,
        help="If >0 with --spsg_seed_geometry_cameras_root, drop SPSG global observations above this reprojection error.",
    )
    parser.add_argument(
        "--spsg_seed_geometry_min_anchor_views",
        type=int,
        default=3,
        help="Minimum non-bridge views used to triangulate each SPSG global seed landmark.",
    )
    parser.add_argument(
        "--spsg_seed_geometry_bridge_policy",
        type=str,
        default="reproject",
        choices=["reproject", "retain"],
        help=(
            "How seed geometry handles bridge-view observations. 'reproject' gates them "
            "with the supplied bridge camera; 'retain' validates the landmark only with "
            "non-bridge cameras and preserves matched bridge observations for later closure."
        ),
    )
    parser.add_argument(
        "--spsg_require_ref_view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every SPSG global seed landmark to be observed in --ref_view. "
            "Disable for wide dynamic-view coverage when landmarks can be triangulated "
            "from other static views."
        ),
    )
    parser.add_argument("--roma_setting", type=str, default="precise", choices=["precise", "fast", "base", "turbo"])
    parser.add_argument("--roma_compile", action="store_true", help="Enable torch.compile for RoMaV2 (slow to start).")
    parser.add_argument("--roma_min_overlap", type=float, default=0.2)
    parser.add_argument(
        "--roma_match_batch_size",
        type=int,
        default=0,
        help="Batch size for ref->views RoMa matching. <=0 uses automatic value by setting/device.",
    )
    parser.add_argument(
        "--roma_scale_align_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resize per-view image by focal ratio before RoMa matching to reduce cross-focal scale gap.",
    )
    parser.add_argument(
        "--roma_scale_align_cameras_root",
        type=str,
        default="",
        help="Optional cameras dir containing intri.yml for scale alignment (default: dataset layout cameras_root).",
    )
    parser.add_argument(
        "--roma_scale_align_min_factor",
        type=float,
        default=0.5,
        help="Lower clamp for per-view resize factor in RoMa scale alignment.",
    )
    parser.add_argument(
        "--roma_scale_align_max_factor",
        type=float,
        default=1.0,
        help="Upper clamp for per-view resize factor in RoMa scale alignment.",
    )
    parser.add_argument(
        "--roma_scale_align_min_side",
        type=int,
        default=320,
        help="Minimum image short side after RoMa scale alignment resize.",
    )
    parser.add_argument(
        "--min_visible_views",
        type=int,
        default=3,
        help="Minimum number of views (including reference) a point must be valid in at the seed frame to be sampled.",
    )
    parser.add_argument(
        "--seed_bridge_view_ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional designated views (subset of --views excluding ref) used for bridge-aware seed sampling.",
    )
    parser.add_argument(
        "--seed_bridge_min_visible",
        type=int,
        default=0,
        help="Require sampled points to be valid in at least this many bridge views (0 disables).",
    )
    parser.add_argument(
        "--seed_bridge_min_points_per_view",
        type=int,
        default=0,
        help="Reserve at least this many sampled points per bridge view before global fill (0 disables).",
    )
    parser.add_argument(
        "--seed_consistency_max_error_px",
        type=float,
        default=0.0,
        help="If >0, require ref->view->ref cycle error <= threshold (pixels) during seed sampling.",
    )
    parser.add_argument(
        "--seed_consistency_weight",
        type=float,
        default=0.0,
        help="If >0, downweight high cycle-error observations during global seed sampling.",
    )
    parser.add_argument(
        "--seed_bridge_consistency_max_error_px",
        type=float,
        default=0.0,
        help="Bridge-only cycle error threshold (pixels). <=0 means fallback to --seed_consistency_max_error_px.",
    )
    parser.add_argument(
        "--seed_bridge_consistency_weight",
        type=float,
        default=0.0,
        help="Bridge-only consistency downweight strength during quota ranking.",
    )
    parser.add_argument(
        "--masks_root",
        type=str,
        default="",
        help="Optional mask root (expects {masks_root}/{view}/{frame}.png). Empty -> auto-detect next to images/ as fmasks/.",
    )
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="foreground",
        choices=["foreground", "background", "all"],
        help="Which region to sample points from using the masks. "
        "For camera calibration, background is usually more stable than foreground.",
    )
    parser.add_argument(
        "--romav2_weights",
        type=str,
        default="",
        help="Optional path to romav2.pt. If empty, we look for checkpoints/romav2.pt.",
    )
    parser.add_argument(
        "--tracker_backend",
        type=str,
        default="cotracker3",
        choices=["cotracker3", "retracker", "alltracker", "raft", "tapir", "cowtracker"],
        help="Temporal tracker backend used per-view after seed-frame matching.",
    )
    parser.add_argument(
        "--track_direction",
        type=str,
        default="forward",
        choices=["forward", "backward"],
        help="Tracking order within [start_frame, start_frame+num_frames). "
        "Use backward to track from the last frame back to start_frame.",
    )
    parser.add_argument(
        "--cotracker_ckpt",
        type=str,
        default="checkpoints/scaled_offline.pth",
    )
    parser.add_argument(
        "--cotracker_mode",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="Use CoTracker3 offline (full sequence) or online (sliding window) mode. Online helps avoid OOM for long videos.",
    )
    parser.add_argument(
        "--cotracker_online_no_chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In CoTracker3 online mode, process the full clip in one pass after init (no temporal chunk loop).",
    )
    parser.add_argument(
        "--cotracker_vis_threshold",
        type=float,
        default=0.0,
        help="Visibility threshold for CoTracker scores. <=0 keeps legacy astype(bool) behavior.",
    )
    parser.add_argument(
        "--cotracker_vis_hysteresis_low",
        type=float,
        default=0.0,
        help="Low hysteresis threshold for visibility state (0 disables).",
    )
    parser.add_argument(
        "--cotracker_vis_hysteresis_high",
        type=float,
        default=0.0,
        help="High hysteresis threshold for visibility state (0 disables).",
    )
    parser.add_argument(
        "--track_filter_border_margin_px",
        type=float,
        default=0.0,
        help="Invalidate tracks within this border margin (px). 0 disables.",
    )
    parser.add_argument(
        "--track_filter_max_step_px",
        type=float,
        default=0.0,
        help="Invalidate per-frame displacements larger than this threshold (px). 0 disables.",
    )
    parser.add_argument(
        "--track_filter_reentry_dist_px",
        type=float,
        default=0.0,
        help="After temporary loss, reject far re-entry jumps larger than this threshold (px). 0 disables.",
    )
    parser.add_argument(
        "--track_filter_reentry_lost_frames",
        type=int,
        default=3,
        help="Minimum consecutive lost frames before re-entry distance gate applies.",
    )
    parser.add_argument(
        "--track_quality_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export per-observation soft quality weights (track_quality.npy).",
    )
    parser.add_argument(
        "--track_quality_vis_weight",
        type=float,
        default=0.7,
        help="Blend weight of raw visibility score in quality estimation.",
    )
    parser.add_argument(
        "--track_quality_border_margin_px",
        type=float,
        default=0.0,
        help="Downweight observations near border within this margin (px). 0 disables.",
    )
    parser.add_argument(
        "--track_quality_step_px",
        type=float,
        default=0.0,
        help="Downweight large per-frame jumps above this threshold (px). 0 disables.",
    )
    parser.add_argument(
        "--track_quality_reentry_dist_px",
        type=float,
        default=0.0,
        help="Downweight suspicious far re-entry jumps (px). 0 disables.",
    )
    parser.add_argument(
        "--track_quality_reentry_lost_frames",
        type=int,
        default=3,
        help="Minimum lost frames before re-entry quality penalty applies.",
    )
    parser.add_argument(
        "--track_quality_min_weight",
        type=float,
        default=0.05,
        help="Minimum non-zero weight for visible observations.",
    )
    parser.add_argument(
        "--retracker_root",
        type=str,
        default="../KTTR_project/re-tracker",
        help="Path to the ReTracker repo root (for tracker_backend=retracker).",
    )
    parser.add_argument(
        "--retracker_ckpt",
        type=str,
        default="",
        help="Optional ReTracker checkpoint path. If empty, we try to auto-locate under {retracker_root}/weights/.",
    )
    parser.add_argument(
        "--retracker_dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="ReTracker inference dtype (amp).",
    )
    parser.add_argument(
        "--retracker_query_batch_size",
        type=int,
        default=256,
        help="ReTracker query batch size (splits large N to avoid OOM).",
    )
    parser.add_argument(
        "--retracker_interp_shape",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="ReTracker internal interp_shape (H W). Also used as the per-frame resize for this demo for speed.",
    )
    parser.add_argument(
        "--alltracker_repo_root",
        type=str,
        default="third_party/alltracker",
        help="Path to AllTracker repo root (for tracker_backend=alltracker).",
    )
    parser.add_argument(
        "--alltracker_ckpt",
        type=str,
        default="",
        help="Optional AllTracker checkpoint path (.pth). Empty uses HuggingFace default weights.",
    )
    parser.add_argument(
        "--alltracker_tiny",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use AllTracker tiny model variant.",
    )
    parser.add_argument(
        "--alltracker_image_size",
        type=int,
        default=1024,
        help="Preferred AllTracker max frame size (long side); clamped to input size.",
    )
    parser.add_argument(
        "--alltracker_min_image_size",
        type=int,
        default=512,
        help="Lower bound for OOM auto-downscale retries.",
    )
    parser.add_argument(
        "--alltracker_downscale_step",
        type=int,
        default=128,
        help="AllTracker retry step for image_size when CUDA OOM occurs.",
    )
    parser.add_argument(
        "--alltracker_auto_downscale_on_oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically retry smaller AllTracker image_size on CUDA OOM.",
    )
    parser.add_argument(
        "--alltracker_inference_iters",
        type=int,
        default=4,
        help="AllTracker inference iterations.",
    )
    parser.add_argument(
        "--alltracker_window_len",
        type=int,
        default=16,
        help="AllTracker temporal window length (model seqlen).",
    )
    parser.add_argument(
        "--alltracker_stride",
        type=int,
        default=0,
        help="AllTracker sliding stride. <=0 uses model default (window_len//2).",
    )
    parser.add_argument(
        "--alltracker_confidence_channel",
        type=int,
        default=1,
        choices=[0, 1],
        help="AllTracker vis/conf channel used as visibility score.",
    )
    parser.add_argument(
        "--alltracker_conf_threshold",
        type=float,
        default=0.1,
        help="Visibility threshold for AllTracker confidence score.",
    )
    parser.add_argument(
        "--cowtracker_root",
        type=str,
        default="third_party/cowtracker",
        help="Path to CoWTracker repo root (for tracker_backend=cowtracker).",
    )
    parser.add_argument(
        "--cowtracker_ckpt",
        type=str,
        default="checkpoints/cowtracker_model.pth",
        help="CoWTracker checkpoint path. If missing/empty, CoWTracker may auto-download from HuggingFace.",
    )
    parser.add_argument(
        "--cowtracker_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="CoWTracker inference dtype.",
    )
    parser.add_argument(
        "--cowtracker_image_size",
        type=int,
        default=560,
        help="CoWTracker max frame long side before inference, rounded down to /112. <=0 keeps input size rounded to /112.",
    )
    parser.add_argument(
        "--cowtracker_visconf_threshold",
        type=float,
        default=0.1,
        help="Visibility threshold on CoWTracker vis*conf score.",
    )
    parser.add_argument(
        "--raft_model",
        type=str,
        default="large",
        choices=["large", "small"],
        help="Torchvision RAFT model size for tracker_backend=raft.",
    )
    parser.add_argument(
        "--raft_weights",
        type=str,
        default="default",
        help="RAFT weights source: 'default', 'none', or a local state_dict path.",
    )
    parser.add_argument(
        "--raft_image_size",
        type=int,
        default=640,
        help="RAFT max frame long side before pairwise flow inference. <=0 keeps input size, rounded to /8.",
    )
    parser.add_argument(
        "--raft_iters",
        type=int,
        default=12,
        help="Number of RAFT flow updates per adjacent-frame inference.",
    )
    parser.add_argument(
        "--raft_fb_consistency_threshold_px",
        type=float,
        default=3.0,
        help="Forward-backward consistency threshold in RAFT inference pixels. <=0 disables this gate.",
    )
    parser.add_argument(
        "--raft_conf_threshold",
        type=float,
        default=0.0,
        help="Visibility threshold for RAFT confidence 1/(1+forward-backward-error).",
    )
    parser.add_argument(
        "--raft_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA autocast for RAFT inference.",
    )
    parser.add_argument(
        "--tapir_repo_root",
        type=str,
        default="third_party/tapnet",
        help="Path to the official TAPNet repo root (for tracker_backend=tapir).",
    )
    parser.add_argument(
        "--tapir_checkpoint",
        type=str,
        default="checkpoints/tapir/tapir_checkpoint_panning.pt",
        help="Path to the official PyTorch TAPIR checkpoint.",
    )
    parser.add_argument(
        "--tapir_variant",
        type=str,
        default="tapir",
        choices=["tapir", "bootstapir"],
        help="TAPIR model variant. 'tapir' uses the standard TAPIR panning checkpoint config.",
    )
    parser.add_argument(
        "--tapir_image_size",
        type=int,
        default=256,
        help="TAPIR max frame long side before inference. <=0 keeps input size.",
    )
    parser.add_argument(
        "--tapir_query_chunk_size",
        type=int,
        default=64,
        help="TAPIR internal query chunk size.",
    )
    parser.add_argument(
        "--tapir_conf_threshold",
        type=float,
        default=0.5,
        help="Visibility threshold for TAPIR confidence from occlusion/expected-distance logits.",
    )
    parser.add_argument(
        "--tapir_amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA autocast for TAPIR inference.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/multiview_tracks")
    parser.add_argument("--save_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--video_codec", type=str, default="auto", help="ffmpeg codec name (e.g., libopenh264, mpeg4) or 'auto'.")
    parser.add_argument(
        "--save_seed_vis",
        action="store_true",
        help="Save debug images overlaying seed-frame masks + sampled seed points for each view.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    _add_third_party_to_syspath(project_root)
    _patch_torch_amp_custom_fwd_bwd_compat()

    # Local imports after sys.path setup (keeps unit tests lightweight).
    from track4d_calib.dataset_layout import resolve_dataset_layout
    from track4d_calib.anchor_map_track_seed import load_anchor_map_track_seed
    from track4d_calib.cameras import CameraLoader
    from track4d_calib.io import list_image_paths, resolve_frame_image
    from track4d_calib.io import infer_common_num_frames
    from track4d_calib.io import select_mask_region
    from track4d_calib.target_star_correspondence import (
        build_target_closure_seed_from_images,
        build_target_relative_closure_seed_from_images,
        build_target_relay_seed_from_images,
        build_target_star_seed_from_images,
        build_target_verified_closure_seed_from_images,
    )
    from track4d_calib.triplet_sampling import sample_multiview_points
    RoMaV2 = None
    if str(args.seed_builder) == "roma_ref":
        from romav2 import RoMaV2

    dataset_root_arg = Path(args.dataset_root) if args.dataset_root else None
    data_root_arg = Path(args.data_root) if args.data_root else None
    images_root_arg = None
    if dataset_root_arg is None:
        images_root_arg = data_root_arg
    else:
        # An explicit data root overrides the configured scene path.
        if args.data_root and args.data_root != default_data_root:
            images_root_arg = data_root_arg

    layout = resolve_dataset_layout(
        dataset_root=dataset_root_arg,
        images_root=images_root_arg,
        masks_root=Path(args.masks_root) if args.masks_root else None,
        dataset_type=str(args.dataset_type),
        dataset_config=Path(args.dataset_config) if args.dataset_config else None,
    )

    data_root = Path(layout.images_root)
    views = [str(v) for v in args.views]
    if len(views) < 2:
        raise SystemExit("--views must contain at least 2 view ids")
    if args.ref_view not in views:
        raise SystemExit(f"--ref_view {args.ref_view} must be in --views {views}")

    ref_view = args.ref_view
    other_views = [v for v in views if v != ref_view]
    if args.min_visible_views > len(views):
        raise SystemExit(f"--min_visible_views ({args.min_visible_views}) must be <= number of views ({len(views)})")
    bridge_view_ids = [str(v) for v in args.seed_bridge_view_ids if str(v) in other_views]
    invalid_bridge = sorted({str(v) for v in args.seed_bridge_view_ids if str(v) not in other_views})
    if invalid_bridge:
        raise SystemExit(
            f"--seed_bridge_view_ids contains invalid views not in --views (or equals ref): {invalid_bridge}"
        )
    if int(args.seed_bridge_min_visible) > len(bridge_view_ids):
        raise SystemExit(
            f"--seed_bridge_min_visible ({args.seed_bridge_min_visible}) must be <= number of valid bridge views ({len(bridge_view_ids)})"
        )
    if str(args.seed_builder) in {
        "spsg_target_star",
        "spsg_target_closure",
        "spsg_target_verified_closure",
        "spsg_target_relay",
    } and not bridge_view_ids:
        raise SystemExit(
            f"--seed_builder={args.seed_builder} requires --seed_bridge_view_ids"
        )
    if float(args.seed_consistency_max_error_px) < 0.0:
        raise SystemExit("--seed_consistency_max_error_px must be >= 0")
    if float(args.seed_consistency_weight) < 0.0:
        raise SystemExit("--seed_consistency_weight must be >= 0")
    if float(args.seed_bridge_consistency_max_error_px) < 0.0:
        raise SystemExit("--seed_bridge_consistency_max_error_px must be >= 0")
    if float(args.seed_bridge_consistency_weight) < 0.0:
        raise SystemExit("--seed_bridge_consistency_weight must be >= 0")
    if float(args.spsg_seed_geometry_reproj_px) < 0.0:
        raise SystemExit("--spsg_seed_geometry_reproj_px must be >= 0")
    if int(args.spsg_seed_geometry_min_anchor_views) < 2:
        raise SystemExit("--spsg_seed_geometry_min_anchor_views must be >= 2")
    spsg_seed_geometry_root = Path(args.spsg_seed_geometry_cameras_root) if str(args.spsg_seed_geometry_cameras_root).strip() else None
    if spsg_seed_geometry_root is not None and not spsg_seed_geometry_root.exists():
        raise SystemExit(f"--spsg_seed_geometry_cameras_root does not exist: {spsg_seed_geometry_root}")
    if float(args.spsg_f_ransac_px) <= 0.0:
        raise SystemExit("--spsg_f_ransac_px must be > 0")
    anchor_map_model_dir = Path(args.anchor_map_model_dir) if str(args.anchor_map_model_dir).strip() else None
    if str(args.seed_builder) == "anchor_map":
        if anchor_map_model_dir is None:
            raise SystemExit("--anchor_map_model_dir is required for --seed_builder=anchor_map")
        if not anchor_map_model_dir.is_dir():
            raise SystemExit(f"--anchor_map_model_dir does not exist: {anchor_map_model_dir}")

    if float(args.roma_scale_align_min_factor) <= 0.0:
        raise SystemExit("--roma_scale_align_min_factor must be > 0")
    if float(args.roma_scale_align_max_factor) <= 0.0:
        raise SystemExit("--roma_scale_align_max_factor must be > 0")
    if float(args.roma_scale_align_max_factor) < float(args.roma_scale_align_min_factor):
        raise SystemExit("--roma_scale_align_max_factor must be >= --roma_scale_align_min_factor")
    if int(args.roma_scale_align_min_side) < 8:
        raise SystemExit("--roma_scale_align_min_side must be >= 8")
    if int(args.raft_iters) <= 0:
        raise SystemExit("--raft_iters must be > 0")
    if float(args.raft_fb_consistency_threshold_px) < 0.0:
        raise SystemExit("--raft_fb_consistency_threshold_px must be >= 0")
    if float(args.raft_conf_threshold) < 0.0:
        raise SystemExit("--raft_conf_threshold must be >= 0")
    if int(args.tapir_query_chunk_size) <= 0:
        raise SystemExit("--tapir_query_chunk_size must be > 0")
    if float(args.tapir_conf_threshold) < 0.0:
        raise SystemExit("--tapir_conf_threshold must be >= 0")

    roma_scale_align_enable = bool(args.roma_scale_align_enable)
    roma_scale_align_cameras_root = ""
    roma_focal_px_by_view: Dict[str, float] = {}
    if roma_scale_align_enable:
        if str(args.roma_scale_align_cameras_root).strip():
            cameras_root_scale = Path(str(args.roma_scale_align_cameras_root).strip())
        elif layout.cameras_root is not None:
            cameras_root_scale = Path(layout.cameras_root)
        else:
            cameras_root_scale = data_root.parent / "cameras"
        roma_scale_align_cameras_root = str(cameras_root_scale)
        roma_focal_px_by_view = _load_focal_px_by_view(cameras_root_scale, views=views)
        if ref_view not in roma_focal_px_by_view:
            print(
                f"[WARN] roma_scale_align_enable requested but ref_view={ref_view} focal not found "
                f"under {cameras_root_scale}. Disable scale alignment."
            )
            roma_scale_align_enable = False
        else:
            num_loaded = int(len(roma_focal_px_by_view))
            print(
                f"[INFO] RoMa scale alignment enabled: focal loaded for "
                f"{num_loaded}/{len(views)} views from {cameras_root_scale}"
            )

    start_frame = int(args.start_frame)
    if int(args.num_frames) == 0:
        num_frames = infer_common_num_frames(data_root, views=views, start=start_frame)
    else:
        num_frames = int(args.num_frames)
    if num_frames <= 0:
        raise SystemExit(f"--num_frames must be > 0 (got {num_frames})")

    seed_frame = int(args.seed_frame) if int(args.seed_frame) >= 0 else start_frame
    if seed_frame < start_frame or seed_frame >= start_frame + num_frames:
        raise SystemExit(
            f"--seed_frame={seed_frame} must be within [start_frame={start_frame}, "
            f"start_frame+num_frames-1={start_frame + num_frames - 1}]"
        )

    extra_seed_frames = [int(s) for s in args.seed_frames]
    for sf in extra_seed_frames:
        if sf < start_frame or sf >= start_frame + num_frames:
            raise SystemExit(
                f"--seed_frames: {sf} must be within [start_frame={start_frame}, "
                f"start_frame+num_frames-1={start_frame + num_frames - 1}]"
            )
    all_seed_frames = [seed_frame] + [s for s in extra_seed_frames if s != seed_frame]
    if str(args.seed_builder) == "anchor_map" and len(all_seed_frames) != 1:
        raise SystemExit("--seed_builder=anchor_map currently requires exactly one seed frame")

    track_direction = str(args.track_direction).lower()
    if track_direction == "backward":
        expected_seed = start_frame + num_frames - 1
        if seed_frame != expected_seed:
            raise SystemExit(
                f"--track_direction=backward requires --seed_frame={expected_seed} "
                f"(got {seed_frame}) for the current window."
            )
    elif track_direction != "forward":
        raise SystemExit(f"Unknown --track_direction: {track_direction}")

    # Resolve mask dir: use explicit override, dataset layout, or standard fmasks fallback.
    if args.masks_root:
        mask_dir = Path(args.masks_root)
    elif layout.masks_root is not None:
        mask_dir = Path(layout.masks_root)
    else:
        mask_dir = (data_root.parent / "fmasks") if str(layout.dataset_type) == "standard" else None
    use_masks = bool(mask_dir and mask_dir.is_dir())
    if not use_masks and str(args.mask_mode).lower() != "all":
        print(f"[WARN] masks_root not found ({mask_dir}), falling back to --mask_mode=all")

    # Load seed frame + mask for all views
    seed_paths: Dict[str, Path] = {}
    seed_rgb: Dict[str, np.ndarray] = {}
    seed_mask: Dict[str, np.ndarray] = {}
    for vid in views:
        img_path = resolve_frame_image(data_root / vid, frame=seed_frame)
        seed_paths[vid] = img_path
        rgb = _load_rgb(img_path)
        seed_rgb[vid] = rgb
        if use_masks and str(args.mask_mode).lower() != "all":
            mpath = mask_dir / vid / (img_path.stem + ".png")
            m = _load_mask(mpath, shape_hw=rgb.shape[:2])
        else:
            m = np.ones(rgb.shape[:2], dtype=np.uint8)
        seed_mask[vid] = m

    seed_builder = str(args.seed_builder)
    # Seed-frame matching + point seeding setup.
    torch.set_float32_matmul_precision("highest")
    roma = None
    roma_device = None
    roma_match_batch_size_base = 0
    roma_match_oom_fallbacks = 0

    if seed_builder == "roma_ref":
        # Prefer local weights to avoid slow/blocked downloads.
        default_weights = project_root / "checkpoints" / "romav2.pt"
        if args.romav2_weights:
            wpath = Path(args.romav2_weights)
            if not wpath.is_file():
                raise RuntimeError(f"--romav2_weights not found: {wpath}")
            os.environ["ROMAV2_WEIGHTS"] = str(wpath)
        elif default_weights.is_file():
            os.environ["ROMAV2_WEIGHTS"] = str(default_weights)
        else:
            raise RuntimeError(
                "RoMaV2 weights not found.\n"
                f"Please download `romav2.pt` to: {default_weights}\n"
                "Or pass `--romav2_weights /path/to/romav2.pt`.\n"
                "Upstream URL: https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt"
            )
        if RoMaV2 is None:
            raise RuntimeError("RoMaV2 import failed while --seed_builder=roma_ref")
        roma_cfg = RoMaV2.Cfg(setting=args.roma_setting, compile=bool(args.roma_compile))
        roma = RoMaV2(roma_cfg)
        # Ensure RoMa follows --device instead of silently defaulting to cuda:0.
        roma_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        roma = roma.to(roma_device)
        if int(args.roma_match_batch_size) > 0:
            roma_match_batch_size_base = int(args.roma_match_batch_size)
        else:
            if (str(args.roma_setting) == "precise") and torch.cuda.is_available():
                roma_match_batch_size_base = 2
            elif torch.cuda.is_available():
                roma_match_batch_size_base = 4
            else:
                roma_match_batch_size_base = 1
        roma_match_batch_size_base = max(1, int(roma_match_batch_size_base))
        print(f"[INFO] Seed builder=roma_ref (RoMaV2, setting={args.roma_setting})")
    elif seed_builder == "spsg_global":
        print("[INFO] Seed builder=spsg_global (SuperPoint+SuperGlue + DSU global IDs)")
    elif seed_builder == "spsg_target_star":
        print("[INFO] Seed builder=spsg_target_star (direct target-to-support IDs)")
    elif seed_builder == "spsg_target_closure":
        print("[INFO] Seed builder=spsg_target_closure (cross-target shared IDs)")
    elif seed_builder == "spsg_target_verified_closure":
        print(
            "[INFO] Seed builder=spsg_target_verified_closure "
            "(cross-target IDs verified by common static observations)"
        )
    elif seed_builder == "spsg_target_relative_closure":
        print(
            "[INFO] Seed builder=spsg_target_relative_closure "
            "(target-relative IDs; absolute gauge deferred)"
        )
    elif seed_builder == "spsg_target_relay":
        print("[INFO] Seed builder=spsg_target_relay (direct target-centered relay IDs)")
    else:
        print(f"[INFO] Seed builder=anchor_map (COLMAP model={anchor_map_model_dir})")

    # Convert grayscale masks into {0,1} validity masks for sampling.
    mask_mode = str(args.mask_mode).lower()

    # --- Multi-seed matching: loop over all seed frames ---
    # SPSG may produce fewer than the requested points when unique landmarks are
    # exhausted, so point counts are tracked per seed.
    all_points_by_view: Dict[str, List[np.ndarray]] = {vid: [] for vid in views}
    all_valid_by_view: Dict[str, List[np.ndarray]] = {vid: [] for vid in views}
    per_seed_query_t: List[int] = []  # query_t (frame offset) for each point group
    per_seed_point_counts: List[int] = []
    roma_scale_align_diag_by_seed: Dict[str, Dict[str, Dict[str, float]]] = {}
    seed_builder_diag_by_seed: Dict[str, Dict[str, Any]] = {}
    anchor_map_point3d_ids_by_seed: List[np.ndarray] = []
    anchor_map_points3d_xyz_by_seed: List[np.ndarray] = []
    use_seed_consistency = any(
        [
            float(args.seed_consistency_max_error_px) > 0.0,
            float(args.seed_consistency_weight) > 0.0,
            float(args.seed_bridge_consistency_max_error_px) > 0.0,
            float(args.seed_bridge_consistency_weight) > 0.0,
        ]
    )

    for sf_idx, sf in enumerate(all_seed_frames):
        sf_offset = sf - start_frame

        # Load images/masks for this seed frame.
        sf_rgb: Dict[str, np.ndarray] = {}
        sf_mask: Dict[str, np.ndarray] = {}
        if sf == seed_frame:
            sf_rgb = seed_rgb
            sf_mask = dict(seed_mask)
        else:
            for vid in views:
                image_path = resolve_frame_image(data_root / vid, frame=sf)
                rgb = _load_rgb(image_path)
                sf_rgb[vid] = rgb
                if use_masks and mask_mode != "all":
                    mpath = mask_dir / vid / (image_path.stem + ".png")
                    m = _load_mask(mpath, shape_hw=rgb.shape[:2])
                else:
                    m = np.ones(rgb.shape[:2], dtype=np.uint8)
                sf_mask[vid] = m

        # Convert masks for this seed frame.
        if mask_mode != "all":
            sf_mask = {k: select_mask_region(v, mode=mask_mode) for k, v in sf_mask.items()}

        if seed_builder == "anchor_map":
            if anchor_map_model_dir is None:
                raise RuntimeError("anchor_map_model_dir unexpectedly missing")
            sf_seed_out = load_anchor_map_track_seed(
                anchor_map_model_dir,
                views=views,
                ref_view=ref_view,
                max_points_per_view=int(args.num_points),
                masks_by_view=sf_mask,
            )
            seed_n = _seed_output_point_count(sf_seed_out, other_views=other_views)
            all_points_by_view[ref_view].append(np.asarray(sf_seed_out["ref_xy"], dtype=np.float32))
            all_valid_by_view[ref_view].append(np.asarray(sf_seed_out["ref_valid"], dtype=bool))
            for vid in other_views:
                all_points_by_view[vid].append(np.asarray(sf_seed_out["xy_by_view"][vid], dtype=np.float32))
                all_valid_by_view[vid].append(np.asarray(sf_seed_out["valid_by_view"][vid], dtype=bool))

            point3d_ids = np.asarray(sf_seed_out["point3D_ids"], dtype=np.int64)
            points3d_xyz = np.asarray(sf_seed_out["points3D_xyz"], dtype=np.float64)
            if point3d_ids.shape != (seed_n,) or points3d_xyz.shape != (seed_n, 3):
                raise RuntimeError(
                    "Anchor-map seed metadata shape mismatch: "
                    f"ids={point3d_ids.shape}, xyz={points3d_xyz.shape}, seed_n={seed_n}"
                )
            anchor_map_point3d_ids_by_seed.append(point3d_ids)
            anchor_map_points3d_xyz_by_seed.append(points3d_xyz)
            per_seed_query_t.append(int(sf_offset))
            per_seed_point_counts.append(int(seed_n))
            diag = dict(sf_seed_out.get("diag", {}))
            seed_builder_diag_by_seed[str(int(sf))] = diag
            coverage = diag.get("anchor_observations_by_view", {})
            coverage_values = np.asarray(list(coverage.values()), dtype=np.int64)
            print(
                f"[INFO] Seed frame {sf}: anchor-map landmarks={seed_n}, "
                f"per-view support min/median/max="
                f"{int(coverage_values.min())}/{float(np.median(coverage_values)):.1f}/{int(coverage_values.max())}"
            )
            continue

        if seed_builder in {
            "spsg_target_star",
            "spsg_target_closure",
            "spsg_target_verified_closure",
            "spsg_target_relative_closure",
            "spsg_target_relay",
        }:
            target_seed_builder = {
                "spsg_target_star": build_target_star_seed_from_images,
                "spsg_target_closure": build_target_closure_seed_from_images,
                "spsg_target_verified_closure": (
                    build_target_verified_closure_seed_from_images
                ),
                "spsg_target_relative_closure": (
                    build_target_relative_closure_seed_from_images
                ),
                "spsg_target_relay": build_target_relay_seed_from_images,
            }[seed_builder]
            sf_seed_out = target_seed_builder(
                rgb_by_view=sf_rgb,
                mask_by_view=sf_mask,
                views=views,
                ref_view=ref_view,
                target_views=bridge_view_ids,
                num_points=int(args.num_points),
                min_support_views=max(2, int(args.min_visible_views) - 1),
                f_ransac_px=float(args.spsg_f_ransac_px),
                rng_seed=int(args.seed + sf_idx),
                feature_conf_name=str(args.spsg_feature_conf),
            )
            seed_n = _seed_output_point_count(sf_seed_out, other_views=other_views)
            all_points_by_view[ref_view].append(
                np.asarray(sf_seed_out["ref_xy"], dtype=np.float32)
            )
            all_valid_by_view[ref_view].append(
                np.asarray(sf_seed_out["ref_valid"], dtype=bool)
            )
            for vid in other_views:
                all_points_by_view[vid].append(
                    np.asarray(sf_seed_out["xy_by_view"][vid], dtype=np.float32)
                )
                all_valid_by_view[vid].append(
                    np.asarray(sf_seed_out["valid_by_view"][vid], dtype=bool)
                )
            per_seed_query_t.append(int(sf_offset))
            per_seed_point_counts.append(int(seed_n))
            diag = dict(sf_seed_out.get("diag", {}))
            seed_builder_diag_by_seed[str(int(sf))] = diag
            print(
                f"[INFO] Seed frame {sf}: {seed_builder} candidates="
                f"{diag.get('candidate_count_by_target', diag.get('candidate_total', 0))} selected="
                f"{diag.get('selected_count_by_target', {})}"
            )
            continue

        if seed_builder == "spsg_global":
            spsg_seed_cameras = None
            if spsg_seed_geometry_root is not None and float(args.spsg_seed_geometry_reproj_px) > 0.0:
                cam_dir = _resolve_spsg_seed_geometry_camera_dir(spsg_seed_geometry_root, int(sf))
                if cam_dir is None:
                    raise RuntimeError(
                        f"SPSG seed geometry camera root has no cameras for frame {sf}: "
                        f"{spsg_seed_geometry_root}"
                    )
                spsg_seed_cameras = CameraLoader(cam_dir).load_cameras(views)
                print(f"[INFO] Seed frame {sf}: SPSG geometry filter cameras={cam_dir}")
            sf_seed_out = _build_seed_points_spsg_global(
                sf_rgb=sf_rgb,
                sf_mask=sf_mask,
                views=views,
                ref_view=ref_view,
                num_points=int(args.num_points),
                min_visible_views=int(args.min_visible_views),
                seed=int(args.seed + sf_idx),
                spsg_f_ransac_px=float(args.spsg_f_ransac_px),
                seed_cameras=spsg_seed_cameras,
                bridge_view_ids=bridge_view_ids,
                bridge_min_points_per_view=int(args.seed_bridge_min_points_per_view),
                geometry_reproj_px=float(args.spsg_seed_geometry_reproj_px),
                geometry_min_anchor_views=int(args.spsg_seed_geometry_min_anchor_views),
                geometry_bridge_observation_policy=str(args.spsg_seed_geometry_bridge_policy),
                require_ref_view=bool(args.spsg_require_ref_view),
            )

            seed_n = _seed_output_point_count(sf_seed_out, other_views=other_views)
            all_points_by_view[ref_view].append(np.asarray(sf_seed_out["ref_xy"], dtype=np.float32))
            all_valid_by_view[ref_view].append(
                np.asarray(sf_seed_out.get("ref_valid", np.ones((seed_n,), dtype=bool)), dtype=bool)
            )
            for vid in other_views:
                all_points_by_view[vid].append(np.asarray(sf_seed_out["xy_by_view"][vid], dtype=np.float32))
                all_valid_by_view[vid].append(np.asarray(sf_seed_out["valid_by_view"][vid], dtype=bool))

            per_seed_query_t.append(int(sf_offset))
            per_seed_point_counts.append(int(seed_n))
            diag = dict(sf_seed_out.get("diag", {}))
            seed_builder_diag_by_seed[str(int(sf))] = diag
            print(
                f"[INFO] Seed frame {sf}: SPSG global landmarks={diag.get('landmark_candidates', 0)} "
                f"selected={diag.get('landmark_selected_total', seed_n)}"
            )
            continue

        # RoMaV2 matching: ref -> others at this seed frame.
        warps_sf: Dict[str, np.ndarray] = {}
        overlaps_sf: Dict[str, np.ndarray] = {}
        consistency_errors_sf: Dict[str, np.ndarray] = {}
        scale_align_diag_sf: Dict[str, Dict[str, float]] = {}
        pair_entries: List[Dict[str, object]] = []
        for vid in other_views:
            rgb_ref_match = sf_rgb[ref_view]
            rgb_vid_match = sf_rgb[vid]
            scale_factor_req = 1.0
            scale_factor_eff = 1.0
            if roma_scale_align_enable:
                f_ref = float(roma_focal_px_by_view.get(ref_view, float("nan")))
                f_vid = float(roma_focal_px_by_view.get(vid, float("nan")))
                if np.isfinite(f_ref) and np.isfinite(f_vid) and f_ref > 1e-6 and f_vid > 1e-6:
                    scale_factor_req = float(np.clip(
                        f_ref / f_vid,
                        float(args.roma_scale_align_min_factor),
                        float(args.roma_scale_align_max_factor),
                    ))
                    rgb_vid_match, scale_factor_eff = _resize_rgb_by_scale(
                        rgb_vid_match,
                        scale=scale_factor_req,
                        min_side=int(args.roma_scale_align_min_side),
                    )
                scale_align_diag_sf[str(vid)] = {
                    "f_ref": float(f_ref) if np.isfinite(f_ref) else float("nan"),
                    "f_view": float(f_vid) if np.isfinite(f_vid) else float("nan"),
                    "scale_factor_req": float(scale_factor_req),
                    "scale_factor_eff": float(scale_factor_eff),
                    "match_h": float(rgb_vid_match.shape[0]),
                    "match_w": float(rgb_vid_match.shape[1]),
                }

            pair_entries.append(
                {
                    "vid": str(vid),
                    "rgb_ref_match": rgb_ref_match,
                    "rgb_vid_match": rgb_vid_match,
                }
            )

        pair_groups: Dict[Tuple[int, int, int, int], List[Dict[str, object]]] = {}
        for entry in pair_entries:
            rgb_ref_match = np.asarray(entry["rgb_ref_match"])
            rgb_vid_match = np.asarray(entry["rgb_vid_match"])
            shape_key = (
                int(rgb_ref_match.shape[0]),
                int(rgb_ref_match.shape[1]),
                int(rgb_vid_match.shape[0]),
                int(rgb_vid_match.shape[1]),
            )
            pair_groups.setdefault(shape_key, []).append(entry)

        for _, entries in pair_groups.items():
            idx = 0
            while idx < len(entries):
                remaining = len(entries) - idx
                cur_bs = min(int(roma_match_batch_size_base), int(remaining))
                cur_bs = max(1, int(cur_bs))

                while True:
                    batch_entries = entries[idx : idx + cur_bs]
                    try:
                        ref_batch = np.stack([
                            np.asarray(e["rgb_ref_match"]) for e in batch_entries
                        ], axis=0)
                        vid_batch = np.stack([
                            np.asarray(e["rgb_vid_match"]) for e in batch_entries
                        ], axis=0)

                        ref_t = (
                            torch.from_numpy(ref_batch)
                            .permute(0, 3, 1, 2)
                            .to(roma_device)
                            .float()
                            / 255.0
                        )
                        vid_t = (
                            torch.from_numpy(vid_batch)
                            .permute(0, 3, 1, 2)
                            .to(roma_device)
                            .float()
                            / 255.0
                        )

                        preds = roma.match(ref_t, vid_t)
                        warp_ab_np = preds["warp_AB"].detach().cpu().numpy()
                        ov_ab_np = preds["overlap_AB"].detach().cpu().numpy()
                        warp_ba_t = preds.get("warp_BA", None)
                        warp_ba_np = warp_ba_t.detach().cpu().numpy() if warp_ba_t is not None else None

                        warp_ba_fallback_np = None
                        if use_seed_consistency and warp_ba_np is None:
                            preds_ba = roma.match(vid_t, ref_t)
                            warp_ba_fallback_np = preds_ba["warp_AB"].detach().cpu().numpy()

                        for bi, entry in enumerate(batch_entries):
                            vid = str(entry["vid"])
                            warp = warp_ab_np[bi]
                            ov = ov_ab_np[bi]
                            if ov.ndim == 3:
                                ov = ov[..., 0]
                            warps_sf[vid] = warp
                            overlaps_sf[vid] = ov

                            if use_seed_consistency:
                                if warp_ba_np is not None:
                                    warp_ba = warp_ba_np[bi]
                                elif warp_ba_fallback_np is not None:
                                    warp_ba = warp_ba_fallback_np[bi]
                                else:
                                    continue
                                try:
                                    consistency_errors_sf[vid] = _compute_cycle_consistency_error_map_ref_px(
                                        warp_ref_to_view=warp,
                                        warp_view_to_ref=warp_ba,
                                        ref_shape_hw=sf_mask[ref_view].shape[:2],
                                    )
                                except Exception as exc:
                                    print(f"[WARN] Failed to compute cycle consistency for view {vid}: {exc}")

                        idx += cur_bs
                        break
                    except Exception as exc:
                        if _is_cuda_oom_error(exc) and cur_bs > 1:
                            roma_match_oom_fallbacks += 1
                            cur_bs = max(1, int(cur_bs // 2))
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print(
                                f"[WARN] RoMa batch OOM at seed={sf}, retry smaller batch size={cur_bs}"
                            )
                            continue
                        raise

        if use_seed_consistency:
            print(
                f"[INFO] Seed frame {sf}: cycle-consistency maps for "
                f"{len(consistency_errors_sf)}/{len(other_views)} views"
            )
        if roma_scale_align_enable:
            eff_scales = [float(v.get("scale_factor_eff", 1.0)) for v in scale_align_diag_sf.values()]
            if eff_scales:
                print(
                    f"[INFO] Seed frame {sf}: RoMa scale-align factors "
                    f"min/median/max={min(eff_scales):.3f}/{np.median(np.asarray(eff_scales)):.3f}/{max(eff_scales):.3f}"
                )
            roma_scale_align_diag_by_seed[f"{int(sf)}"] = scale_align_diag_sf

        # Sample points at this seed frame.
        sf_seed_out = sample_multiview_points(
            ref_mask=sf_mask[ref_view],
            view_masks={vid: sf_mask[vid] for vid in other_views},
            warps_ref_to_view=warps_sf,
            overlaps_ref_to_view=overlaps_sf,
            num_points=args.num_points,
            min_overlap=args.roma_min_overlap,
            min_visible_views=int(args.min_visible_views),
            bridge_view_ids=bridge_view_ids,
            bridge_min_visible=int(args.seed_bridge_min_visible),
            bridge_min_points_per_view=int(args.seed_bridge_min_points_per_view),
            consistency_error_ref_to_view=consistency_errors_sf if consistency_errors_sf else None,
            consistency_max_error_px=float(args.seed_consistency_max_error_px),
            consistency_weight=float(args.seed_consistency_weight),
            bridge_consistency_max_error_px=float(args.seed_bridge_consistency_max_error_px),
            bridge_consistency_weight=float(args.seed_bridge_consistency_weight),
            seed=args.seed + sf_idx,
        )

        # Collect per-view points and validity.
        seed_n = _seed_output_point_count(sf_seed_out, other_views=other_views)
        all_points_by_view[ref_view].append(sf_seed_out["ref_xy"])
        all_valid_by_view[ref_view].append(np.ones((seed_n,), dtype=bool))
        for vid in other_views:
            all_points_by_view[vid].append(sf_seed_out["xy_by_view"][vid])
            all_valid_by_view[vid].append(sf_seed_out["valid_by_view"][vid])

        per_seed_query_t.append(int(sf_offset))
        per_seed_point_counts.append(int(seed_n))
        print(f"[INFO] Seed frame {sf}: matched {args.num_points} points (offset {sf_offset})")

    # Concatenate across all seed frames.
    num_seeds = len(all_seed_frames)
    total_num_points = int(sum(int(n) for n in per_seed_point_counts))
    points_seed_by_view: Dict[str, np.ndarray] = {
        vid: np.concatenate(all_points_by_view[vid], axis=0) for vid in views
    }
    valid_seed_by_view: Dict[str, np.ndarray] = {
        vid: np.concatenate(all_valid_by_view[vid], axis=0) for vid in views
    }
    # Build per-point query_t array: each seed group shares the same offset.
    per_point_query_t = _expand_seed_query_offsets(per_seed_query_t, per_seed_point_counts)
    if int(per_point_query_t.shape[0]) != int(total_num_points):
        raise RuntimeError(
            "Internal seed point count mismatch: "
            f"query_t={per_point_query_t.shape[0]} total_num_points={total_num_points}"
        )
    print(
        f"[INFO] Seed construction complete: seeds={num_seeds} "
        f"points={total_num_points} per_seed={per_seed_point_counts}",
        flush=True,
    )

    if bool(args.save_seed_vis):
        _save_seed_debug_images(
            out_dir=Path(args.output_dir),
            views=views,
            seed_frame=seed_frame,
            seed_rgb=seed_rgb,
            seed_mask=seed_mask,
            points_seed_by_view=points_seed_by_view,
            valid_seed_by_view=valid_seed_by_view,
        )


    if track_view_sink is not None:
        track_view_sink.configure(
            views=views, start_frame=start_frame, num_frames=num_frames,
            seed_frames=all_seed_frames, point_counts=per_seed_point_counts,
        )

    # CoTracker3 temporal tracking for each view.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tracker_backend = str(args.tracker_backend)

    cotracker = None
    cotracker_ckpt = None
    retracker = None
    retracker_ckpt = None
    alltracker = None
    alltracker_ckpt = ""
    cowtracker = None
    cowtracker_ckpt = ""
    cowtracker_dtype = torch.float16
    raft = None
    raft_transform = None
    raft_weights_source = ""
    tapir = None
    tapir_ckpt = ""

    if tracker_backend == "cotracker3":
        from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor

        cotracker_ckpt = Path(args.cotracker_ckpt)
        if args.cotracker_mode == "online" and cotracker_ckpt.name == "scaled_offline.pth":
            # Select the online checkpoint for the online tracker.
            cand = cotracker_ckpt.with_name("scaled_online.pth")
            if cand.is_file():
                print(f"[INFO] --cotracker_mode=online: switching ckpt {cotracker_ckpt} -> {cand}")
                cotracker_ckpt = cand
        if not cotracker_ckpt.exists():
            raise RuntimeError(f"CoTracker checkpoint not found: {cotracker_ckpt}")

        print(
            f"[INFO] Loading CoTracker3 mode={args.cotracker_mode} checkpoint={cotracker_ckpt}",
            flush=True,
        )
        if args.cotracker_mode == "online":
            cotracker = CoTrackerOnlinePredictor(checkpoint=str(cotracker_ckpt)).to(device)
        else:
            cotracker = CoTrackerPredictor(checkpoint=str(cotracker_ckpt), offline=True).to(device)
        print(f"[INFO] CoTracker3 ready on {device}", flush=True)
    elif tracker_backend == "retracker":
        if not torch.cuda.is_available():
            raise RuntimeError("--tracker_backend=retracker currently requires CUDA")

        _patch_torch_hub_dinov3_offline_if_possible()

        retracker_root = Path(args.retracker_root)
        if not retracker_root.is_dir():
            raise RuntimeError(f"--retracker_root not found: {retracker_root}")

        sys.path.insert(0, str(retracker_root))

        # Lazy import to avoid importing ReTracker deps when not needed.
        from retracker.apps.config import ModelConfig
        from retracker.apps.runtime.tracker import Tracker

        # Resolve ckpt path.
        ckpt_arg = str(args.retracker_ckpt).strip()
        ckpt = Path(ckpt_arg) if ckpt_arg else None
        if ckpt is not None and not ckpt.is_file():
            # Allow passing a path relative to retracker_root.
            ckpt2 = retracker_root / ckpt
            ckpt = ckpt2 if ckpt2.is_file() else ckpt
        if ckpt is None or not ckpt.is_file():
            cand1 = retracker_root / "weights" / "epoch_86_weights.ckpt"
            cand2 = retracker_root / "weights" / "model.ckpt"
            if cand1.is_file():
                ckpt = cand1
            elif cand2.is_file():
                ckpt = cand2
            else:
                raise RuntimeError(
                    "ReTracker checkpoint not found.\n"
                    f"Tried: {cand1}\n"
                    f"       {cand2}\n"
                    "Pass --retracker_ckpt /path/to/model.ckpt"
                )
        retracker_ckpt = ckpt

        retr_dtype = str(args.retracker_dtype)
        # ReTracker uses autocast('cuda', ...); disable amp for fp32.
        use_amp = bool(retr_dtype in ("fp16", "bf16"))

        interp_shape = tuple(int(x) for x in args.retracker_interp_shape)
        model_cfg = ModelConfig(
            ckpt_path=str(retracker_ckpt),
            interp_shape=interp_shape,
            dtype=retr_dtype,
            device=str(args.device),
            use_amp=use_amp,
            query_batch_size=int(args.retracker_query_batch_size),
        )
        retracker = Tracker(model_cfg)
    elif tracker_backend == "alltracker":
        if not torch.cuda.is_available():
            raise RuntimeError("--tracker_backend=alltracker currently requires CUDA")

        alltracker_root = Path(args.alltracker_repo_root)
        if not alltracker_root.is_dir():
            raise RuntimeError(f"--alltracker_repo_root not found: {alltracker_root}")

        sys.path.insert(0, str(alltracker_root))

        from nets.alltracker import Net as AllTrackerNet

        alltracker = AllTrackerNet(
            int(args.alltracker_window_len),
            use_basicencoder=bool(args.alltracker_tiny),
            no_split=bool(args.alltracker_tiny),
        ).to(device)

        ckpt_arg = str(args.alltracker_ckpt).strip()
        ckpt = Path(ckpt_arg) if ckpt_arg else None
        if ckpt is not None and not ckpt.is_file():
            ckpt2 = alltracker_root / ckpt
            ckpt = ckpt2 if ckpt2.is_file() else ckpt

        if ckpt is not None and ckpt.is_file():
            sd = torch.load(str(ckpt), map_location="cpu")
            model_sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            alltracker.load_state_dict(model_sd, strict=True)
            alltracker_ckpt = str(ckpt)
            print(f"[INFO] Loaded AllTracker checkpoint: {alltracker_ckpt}")
        else:
            if bool(args.alltracker_tiny):
                url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
            else:
                url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
            sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            model_sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            alltracker.load_state_dict(model_sd, strict=True)
            alltracker_ckpt = url
            print(f"[INFO] Loaded AllTracker checkpoint from: {url}")

        alltracker.eval()
        for _, p in alltracker.named_parameters():
            p.requires_grad = False
    elif tracker_backend == "cowtracker":
        if not torch.cuda.is_available():
            raise RuntimeError("--tracker_backend=cowtracker currently requires CUDA")

        cow_root = Path(args.cowtracker_root)
        if not cow_root.is_absolute():
            cow_root = project_root / cow_root
        if not cow_root.is_dir():
            raise RuntimeError(f"--cowtracker_root not found: {cow_root}")
        sys.path.insert(0, str(cow_root))

        from cowtracker import CoWTracker

        dtype_name = str(args.cowtracker_dtype)
        cowtracker_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]

        ckpt_arg = str(args.cowtracker_ckpt).strip()
        ckpt_path: str | None = None
        if ckpt_arg:
            ckpt = Path(ckpt_arg)
            if not ckpt.is_absolute():
                ckpt = project_root / ckpt
            if ckpt.is_file():
                ckpt_path = str(ckpt)
                cowtracker_ckpt = str(ckpt)
            else:
                raise RuntimeError(f"--cowtracker_ckpt not found: {ckpt}")

        cowtracker = CoWTracker.from_checkpoint(
            ckpt_path,
            device=str(device),
            dtype=cowtracker_dtype,
        )
    elif tracker_backend == "raft":
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        model_name = str(args.raft_model)
        weights_arg = str(args.raft_weights).strip()
        weights_lower = weights_arg.lower()
        if model_name == "small":
            model_fn = raft_small
            default_weights = Raft_Small_Weights.DEFAULT
        else:
            model_fn = raft_large
            default_weights = Raft_Large_Weights.DEFAULT

        weights = None
        if weights_lower in ("", "default"):
            weights = default_weights
            raft_weights_source = str(default_weights)
        elif weights_lower in ("none", "random"):
            weights = None
            raft_weights_source = "none"
        else:
            weights = None
            raft_weights_source = str(Path(weights_arg))

        raft = model_fn(weights=weights, progress=True).to(device)
        if weights is not None:
            raft_transform = weights.transforms()
        else:
            raft_transform = None

        if weights is None and weights_lower not in ("none", "random"):
            ckpt = Path(weights_arg)
            if not ckpt.is_file():
                raise RuntimeError(f"RAFT weights path not found: {ckpt}")
            sd = torch.load(str(ckpt), map_location="cpu")
            model_sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            raft.load_state_dict(model_sd, strict=True)
            print(f"[INFO] Loaded RAFT checkpoint: {ckpt}")

        raft.eval()
        for _, p in raft.named_parameters():
            p.requires_grad = False
    elif tracker_backend == "tapir":
        tapir_root = Path(args.tapir_repo_root)
        if not tapir_root.is_absolute():
            tapir_root = project_root / tapir_root
        if not tapir_root.is_dir():
            raise RuntimeError(f"--tapir_repo_root not found: {tapir_root}")

        sys.path.insert(0, str(tapir_root))
        from tapnet.torch import tapir_model as tapir_model_lib

        variant = str(args.tapir_variant)
        if variant == "tapir":
            tapir = tapir_model_lib.TAPIR(pyramid_level=0, extra_convs=False).to(device)
        elif variant == "bootstapir":
            tapir = tapir_model_lib.TAPIR(
                pyramid_level=1,
                extra_convs=True,
                softmax_temperature=10.0,
            ).to(device)
        else:
            raise RuntimeError(f"Unknown --tapir_variant: {variant}")

        ckpt = Path(str(args.tapir_checkpoint).strip())
        if not ckpt.is_absolute():
            ckpt = project_root / ckpt
        if not ckpt.is_file():
            ckpt2 = tapir_root / str(args.tapir_checkpoint).strip()
            ckpt = ckpt2 if ckpt2.is_file() else ckpt
        if not ckpt.is_file():
            raise RuntimeError(f"--tapir_checkpoint not found: {ckpt}")

        sd = torch.load(str(ckpt), map_location="cpu")
        model_sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        tapir.load_state_dict(model_sd, strict=True)
        tapir_ckpt = str(ckpt)
        print(f"[INFO] Loaded TAPIR checkpoint: {tapir_ckpt}")

        tapir.eval()
        for _, p in tapir.named_parameters():
            p.requires_grad = False
    else:
        raise RuntimeError(f"Unknown --tracker_backend: {tracker_backend}")

    # num_frames already resolved above (after dataset layout).
    seed_offset = seed_frame - start_frame
    if tracker_backend == "cowtracker":
        if str(track_direction) != "forward" or int(seed_offset) != 0 or int(num_seeds) != 1:
            raise SystemExit(
                "--tracker_backend=cowtracker currently supports only single-seed first-frame tracking "
                "(track_direction=forward, seed_frame=start_frame)."
            )
    if track_direction == "forward" and seed_offset != 0:
        if tracker_backend == "cotracker3" and str(args.cotracker_mode) == "online":
            if num_seeds == 1:
                raise SystemExit("--cotracker_mode=online requires --seed_frame == --start_frame (forward tracking)")
            else:
                print("[WARN] --cotracker_mode=online with multi-seed: only primary seed_frame constraint checked")

    frames_by_view: Dict[str, List[np.ndarray]] = {}  # only populated when save_video=True
    tracks_by_view: Dict[str, np.ndarray] = {}
    vis_by_view: Dict[str, np.ndarray] = {}
    track_filter_stats_by_view: Dict[str, Dict[str, float]] = {}
    quality_by_view: Dict[str, np.ndarray] = {}
    track_quality_stats_by_view: Dict[str, Dict[str, float]] = {}
    alltracker_runtime_by_view: Dict[str, Dict[str, int]] = {}
    cowtracker_runtime_by_view: Dict[str, Dict[str, int]] = {}
    raft_runtime_by_view: Dict[str, Dict[str, int]] = {}
    tapir_runtime_by_view: Dict[str, Dict[str, int]] = {}

    for vid in views:
        print(f"[INFO] Tracking view {vid}: loading {num_frames} frames", flush=True)
        img_paths = [
            resolve_frame_image(data_root / vid, frame=frame)
            for frame in range(start_frame, start_frame + int(num_frames))
        ]

        # Tracker internally interpolates to a fixed model resolution for CoTracker/ReTracker;
        # RAFT/AllTracker handle dynamic resize in their own branches below.
        target_interp_hw: Tuple[int, int] | None = None
        if tracker_backend == "cotracker3":
            target_interp_hw = (int(cotracker.interp_shape[0]), int(cotracker.interp_shape[1]))
        elif tracker_backend == "retracker":
            target_interp_hw = (int(args.retracker_interp_shape[0]), int(args.retracker_interp_shape[1]))

        frames_small: List[np.ndarray] = []
        if bool(args.save_video):
            frames_by_view[vid] = []

        for p in img_paths:
            rgb = _load_rgb(p)
            if bool(args.save_video):
                frames_by_view[vid].append(rgb)
            if target_interp_hw is not None:
                interp_h, interp_w = target_interp_hw
                if rgb.shape[0] != interp_h or rgb.shape[1] != interp_w:
                    rgb = cv2.resize(rgb, (interp_w, interp_h), interpolation=cv2.INTER_LINEAR)
            frames_small.append(rgb)

        if target_interp_hw is None:
            interp_h, interp_w = int(frames_small[0].shape[0]), int(frames_small[0].shape[1])
        else:
            interp_h, interp_w = int(target_interp_hw[0]), int(target_interp_hw[1])

        # Track only points that have a valid seed-frame match in this view.
        valid0 = valid_seed_by_view.get(vid, np.ones((total_num_points,), dtype=bool))
        valid_idx = np.flatnonzero(valid0.astype(bool))

        tracks_full = np.full((num_frames, total_num_points, 2), np.nan, dtype=np.float32)
        vis_full = np.zeros((num_frames, total_num_points), dtype=bool)
        quality_full = np.zeros((num_frames, total_num_points), dtype=np.float32)

        if valid_idx.size > 0:
            print(
                f"[INFO] Tracking view {vid}: valid_queries={valid_idx.size}/{total_num_points}",
                flush=True,
            )
            # Scale query points from original image coords -> model-resolution coords.
            h0, w0 = seed_rgb[vid].shape[:2]
            sx = (interp_w - 1) / max((w0 - 1), 1)
            sy = (interp_h - 1) / max((h0 - 1), 1)
            xy = points_seed_by_view[vid].astype(np.float32)[valid_idx]
            xy_small = xy.copy()
            xy_small[:, 0] *= sx
            xy_small[:, 1] *= sy

            n = int(xy_small.shape[0])
            queries = torch.zeros((1, n, 3), dtype=torch.float32, device=device)
            # Per-point query_t: each point gets the offset of its seed frame.
            valid_query_t = per_point_query_t[valid_idx]
            if track_direction == "backward":
                # Reverse: map each offset to reversed-clip time.
                valid_query_t = (num_frames - 1) - valid_query_t
            queries[0, :, 0] = torch.from_numpy(valid_query_t.astype(np.float32)).to(device)
            queries[0, :, 1] = torch.from_numpy(xy_small[:, 0]).to(device)
            queries[0, :, 2] = torch.from_numpy(xy_small[:, 1]).to(device)

            frames_track = frames_small
            if track_direction == "backward":
                frames_track = list(reversed(frames_small))

            tracks_coord_h, tracks_coord_w = int(interp_h), int(interp_w)

            if tracker_backend == "cotracker3":
                if args.cotracker_mode == "online":
                    # Initialize per-video online state.
                    init_video = (
                        torch.from_numpy(np.stack(frames_track[:1], axis=0))
                        .permute(0, 3, 1, 2)[None]
                        .float()
                        .to(device)
                    )
                    cotracker(
                        init_video,
                        is_first_step=True,
                        queries=queries,
                        grid_size=0,
                        add_support_grid=False,
                    )
                    if bool(args.cotracker_online_no_chunk):
                        video_online = (
                            torch.from_numpy(np.stack(frames_track, axis=0))
                            .permute(0, 3, 1, 2)[None]
                            .float()
                            .to(device)
                        )
                        pred_tracks, pred_vis = cotracker(video_online, grid_size=0, add_support_grid=False)
                    else:
                        step = int(getattr(cotracker, "step", 8))
                        pred_tracks = None
                        pred_vis = None
                        for start in range(0, int(num_frames) - step, step):
                            end = min(start + step * 2, int(num_frames))
                            chunk = (
                                torch.from_numpy(np.stack(frames_track[start:end], axis=0))
                                .permute(0, 3, 1, 2)[None]
                                .float()
                                .to(device)
                            )
                            pred_tracks, pred_vis = cotracker(chunk, grid_size=0, add_support_grid=False)
                        if pred_tracks is None or pred_vis is None:
                            chunk = (
                                torch.from_numpy(np.stack(frames_track, axis=0))
                                .permute(0, 3, 1, 2)[None]
                                .float()
                                .to(device)
                            )
                            pred_tracks, pred_vis = cotracker(chunk, grid_size=0, add_support_grid=False)
                else:
                    video = (
                        torch.from_numpy(np.stack(frames_track, axis=0))
                        .permute(0, 3, 1, 2)[None]
                        .float()
                        .to(device)
                    )
                    print(f"[INFO] Tracking view {vid}: CoTracker3 inference start", flush=True)
                    with torch.no_grad():
                        pred_tracks, pred_vis = cotracker(video, queries=queries, grid_size=0)
                    print(f"[INFO] Tracking view {vid}: CoTracker3 inference complete", flush=True)
                tracks_small = pred_tracks[0].detach().cpu().numpy().astype(np.float32)  # (T, n, 2)
                vis_raw = pred_vis[0].detach().cpu().numpy()
            elif tracker_backend == "retracker":
                # ReTracker expects uint8 video and (t,x,y) queries in input pixel coords.
                video = torch.from_numpy(np.stack(frames_track, axis=0)).permute(0, 3, 1, 2)[None].cpu()
                with torch.no_grad():
                    pred_tracks, pred_vis = retracker.track(video, queries, use_aug=False)  # (1, T, n, 2), (1, T, n)
                tracks_small = pred_tracks[0].detach().cpu().numpy().astype(np.float32)  # (T, n, 2)
                vis_raw = pred_vis[0].detach().cpu().numpy()
            elif tracker_backend == "raft":
                tracks_small, vis_raw, raft_info = _track_with_raft_pairwise(
                    raft=raft,
                    raft_transform=raft_transform,
                    device=device,
                    frames_track=frames_track,
                    query_xy=xy_small,
                    query_t=valid_query_t,
                    image_size=int(args.raft_image_size),
                    num_flow_updates=int(args.raft_iters),
                    fb_consistency_threshold_px=float(args.raft_fb_consistency_threshold_px),
                    use_amp=bool(args.raft_amp),
                )
                tracks_coord_h = int(raft_info["interp_h"])
                tracks_coord_w = int(raft_info["interp_w"])
                raft_runtime_by_view[vid] = {
                    "interp_h": int(tracks_coord_h),
                    "interp_w": int(tracks_coord_w),
                    "image_size": int(raft_info["image_size"]),
                }
            elif tracker_backend == "tapir":
                tracks_small, vis_raw, tapir_info = _track_with_tapir_offline(
                    tapir=tapir,
                    device=device,
                    frames_track=frames_track,
                    query_xy=xy_small,
                    query_t=valid_query_t,
                    image_size=int(args.tapir_image_size),
                    query_chunk_size=int(args.tapir_query_chunk_size),
                    use_amp=bool(args.tapir_amp),
                )
                tracks_coord_h = int(tapir_info["interp_h"])
                tracks_coord_w = int(tapir_info["interp_w"])
                tapir_runtime_by_view[vid] = {
                    "interp_h": int(tracks_coord_h),
                    "interp_w": int(tracks_coord_w),
                    "image_size": int(tapir_info["image_size"]),
                    "query_chunk_size": int(tapir_info["query_chunk_size"]),
                }
            elif tracker_backend == "alltracker":
                tracks_small, vis_raw, alltracker_info = _track_with_alltracker_auto_resize(
                    alltracker=alltracker,
                    device=device,
                    frames_track=frames_track,
                    query_xy=xy_small,
                    query_t=valid_query_t,
                    image_size=int(args.alltracker_image_size),
                    min_image_size=int(args.alltracker_min_image_size),
                    downscale_step=int(args.alltracker_downscale_step),
                    auto_downscale_on_oom=bool(args.alltracker_auto_downscale_on_oom),
                    inference_iters=int(args.alltracker_inference_iters),
                    window_len=int(args.alltracker_window_len),
                    stride=int(args.alltracker_stride),
                    confidence_channel=int(args.alltracker_confidence_channel),
                )
                tracks_coord_h = int(alltracker_info["interp_h"])
                tracks_coord_w = int(alltracker_info["interp_w"])
                alltracker_runtime_by_view[vid] = {
                    "interp_h": int(tracks_coord_h),
                    "interp_w": int(tracks_coord_w),
                    "image_size": int(alltracker_info["image_size"]),
                }
            elif tracker_backend == "cowtracker":
                if np.any(valid_query_t.astype(np.int64) != 0):
                    raise RuntimeError("CoWTracker backend supports only query_t=0 for this first-frame adapter.")
                cow_h, cow_w = _cowtracker_resize_hw_from_max_size(
                    (int(interp_h), int(interp_w)),
                    int(args.cowtracker_image_size),
                )
                cow_xy = xy_small.copy()
                cow_xy[:, 0] *= max(float(cow_w - 1), 1.0) / max(float(interp_w - 1), 1.0)
                cow_xy[:, 1] *= max(float(cow_h - 1), 1.0) / max(float(interp_h - 1), 1.0)
                frames_cow = [
                    cv2.resize(fr, (cow_w, cow_h), interpolation=cv2.INTER_LINEAR)
                    if (int(fr.shape[0]) != cow_h or int(fr.shape[1]) != cow_w)
                    else fr
                    for fr in frames_track
                ]
                video = (
                    torch.from_numpy(np.stack(frames_cow, axis=0))
                    .permute(0, 3, 1, 2)
                    .float()
                    .to(device)
                )
                amp_enabled = bool(str(device).startswith("cuda") and cowtracker_dtype != torch.float32)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=cowtracker_dtype, enabled=amp_enabled):
                        pred = cowtracker(video=video, queries=None)
                track_maps = pred["track"][0].detach().to(device="cpu", dtype=torch.float32)  # (T,H,W,2)
                vis_map = pred["vis"][0].detach().to(device="cpu", dtype=torch.float32)
                conf_map = pred["conf"][0].detach().to(device="cpu", dtype=torch.float32)
                tracks_small, vis_raw = _alltracker_sample_dense_maps(
                    traj_maps=track_maps.permute(0, 3, 1, 2).contiguous(),
                    conf_map=(vis_map * conf_map).contiguous(),
                    query_xy=cow_xy,
                )
                tracks_coord_h = int(cow_h)
                tracks_coord_w = int(cow_w)
                cowtracker_runtime_by_view[vid] = {
                    "interp_h": int(tracks_coord_h),
                    "interp_w": int(tracks_coord_w),
                    "image_size": int(max(cow_h, cow_w)),
                }
                del pred, track_maps, vis_map, conf_map, video
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                raise RuntimeError(f"Unknown tracker backend: {tracker_backend}")

            if vis_raw.ndim == 3 and vis_raw.shape[-1] == 1:
                vis_raw = vis_raw[..., 0]
            vis_threshold = float(args.cotracker_vis_threshold)
            if tracker_backend == "alltracker":
                vis_threshold = float(args.alltracker_conf_threshold)
            elif tracker_backend == "raft":
                vis_threshold = float(args.raft_conf_threshold)
            elif tracker_backend == "tapir":
                vis_threshold = float(args.tapir_conf_threshold)
            elif tracker_backend == "cowtracker":
                vis_threshold = float(args.cowtracker_visconf_threshold)
            vis_small = _visibility_from_raw(
                vis_raw,
                threshold=float(vis_threshold),
                hysteresis_low=float(args.cotracker_vis_hysteresis_low),
                hysteresis_high=float(args.cotracker_vis_hysteresis_high),
            )

            if tracks_small.shape[0] != int(num_frames) or vis_small.shape[0] != int(num_frames):
                raise RuntimeError(
                    f"Tracker {tracker_backend} returned unexpected T={tracks_small.shape[0]} (expected {num_frames})"
                )

            if track_direction == "backward":
                tracks_small = tracks_small[::-1].copy()
                vis_small = vis_small[::-1].copy()
                vis_raw = vis_raw[::-1].copy()

            # Scale tracks back to original image coords.
            inv_sx = max((w0 - 1), 1) / max((tracks_coord_w - 1), 1)
            inv_sy = max((h0 - 1), 1) / max((tracks_coord_h - 1), 1)
            tracks_small[..., 0] *= inv_sx
            tracks_small[..., 1] *= inv_sy

            vis_small, filter_stats = _apply_track_sanity_filters(
                tracks_xy=tracks_small,
                vis=vis_small,
                image_hw=(h0, w0),
                border_margin_px=float(args.track_filter_border_margin_px),
                max_step_px=float(args.track_filter_max_step_px),
                reentry_dist_px=float(args.track_filter_reentry_dist_px),
                reentry_lost_frames=int(args.track_filter_reentry_lost_frames),
            )
            tracks_small[~vis_small] = np.nan
            if filter_stats.get("removed_border", 0.0) > 0.0 or filter_stats.get("removed_step", 0.0) > 0.0 or filter_stats.get("removed_reentry", 0.0) > 0.0:
                print(
                    f"[INFO] view {vid}: track_filter keep={filter_stats.get('visible_keep_ratio', 1.0):.3f} "
                    f"(border={int(filter_stats.get('removed_border', 0.0))}, "
                    f"step={int(filter_stats.get('removed_step', 0.0))}, "
                    f"reentry={int(filter_stats.get('removed_reentry', 0.0))})"
                )
            track_filter_stats_by_view[vid] = {k: float(v) for k, v in filter_stats.items()}

            if bool(args.track_quality_enable):
                quality_small, quality_stats = _compute_track_quality_weights(
                    tracks_xy=tracks_small,
                    vis=vis_small,
                    vis_raw=vis_raw,
                    image_hw=(h0, w0),
                    vis_weight=float(args.track_quality_vis_weight),
                    border_margin_px=float(args.track_quality_border_margin_px),
                    step_px=float(args.track_quality_step_px),
                    reentry_dist_px=float(args.track_quality_reentry_dist_px),
                    reentry_lost_frames=int(args.track_quality_reentry_lost_frames),
                    min_weight=float(args.track_quality_min_weight),
                )
            else:
                quality_small = vis_small.astype(np.float32)
                quality_stats = {
                    "quality_mean_visible": float(quality_small[vis_small].mean()) if np.any(vis_small) else 0.0,
                    "quality_median_visible": float(np.median(quality_small[vis_small])) if np.any(vis_small) else 0.0,
                    "quality_min_visible": float(quality_small[vis_small].min()) if np.any(vis_small) else 0.0,
                    "quality_max_visible": float(quality_small[vis_small].max()) if np.any(vis_small) else 0.0,
                    "visible_count": float(vis_small.sum()),
                }
            track_quality_stats_by_view[vid] = {k: float(v) for k, v in quality_stats.items()}

            tracks_full[:, valid_idx] = tracks_small
            vis_full[:, valid_idx] = vis_small
            quality_full[:, valid_idx] = quality_small

        tracks_by_view[vid] = tracks_full
        vis_by_view[vid] = vis_full
        quality_by_view[vid] = quality_full
        if track_view_sink is not None:
            track_view_sink.submit(vid, tracks_full, vis_full, quality_full)
        print(f"[INFO] Tracking view {vid}: complete", flush=True)
        if vid not in track_filter_stats_by_view:
            track_filter_stats_by_view[vid] = {
                "visible_before": 0.0,
                "removed_nonfinite": 0.0,
                "removed_border": 0.0,
                "removed_step": 0.0,
                "removed_reentry": 0.0,
                "visible_after": 0.0,
                "visible_keep_ratio": 1.0,
            }
        if vid not in track_quality_stats_by_view:
            track_quality_stats_by_view[vid] = {
                "quality_mean_visible": 0.0,
                "quality_median_visible": 0.0,
                "quality_min_visible": 0.0,
                "quality_max_visible": 0.0,
                "visible_count": 0.0,
            }
        if tracker_backend == "alltracker" and vid not in alltracker_runtime_by_view:
            alltracker_runtime_by_view[vid] = {
                "interp_h": int(interp_h),
                "interp_w": int(interp_w),
                "image_size": int(max(interp_h, interp_w)),
            }
        if tracker_backend == "cowtracker" and vid not in cowtracker_runtime_by_view:
            cow_h, cow_w = _cowtracker_resize_hw_from_max_size(
                (int(interp_h), int(interp_w)),
                int(args.cowtracker_image_size),
            )
            cowtracker_runtime_by_view[vid] = {
                "interp_h": int(cow_h),
                "interp_w": int(cow_w),
                "image_size": int(max(cow_h, cow_w)),
            }
        if tracker_backend == "raft" and vid not in raft_runtime_by_view:
            raft_h, raft_w = _raft_resize_hw_from_max_size((int(interp_h), int(interp_w)), int(args.raft_image_size))
            raft_runtime_by_view[vid] = {
                "interp_h": int(raft_h),
                "interp_w": int(raft_w),
                "image_size": int(max(raft_h, raft_w)),
            }
        if tracker_backend == "tapir" and vid not in tapir_runtime_by_view:
            tapir_h, tapir_w = _tapir_resize_hw_from_max_size((int(interp_h), int(interp_w)), int(args.tapir_image_size))
            tapir_runtime_by_view[vid] = {
                "interp_h": int(tapir_h),
                "interp_w": int(tapir_w),
                "image_size": int(max(tapir_h, tapir_w)),
                "query_chunk_size": int(args.tapir_query_chunk_size),
            }

    # Save results.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    view_order = views
    tracks = np.stack([tracks_by_view[v] for v in view_order], axis=0)  # (V, T, N, 2)
    visibility = np.stack([vis_by_view[v] for v in view_order], axis=0)  # (V, T, N)
    track_quality = np.stack([quality_by_view[v] for v in view_order], axis=0)  # (V, T, N)

    np.save(out_dir / "tracks.npy", tracks)
    np.save(out_dir / "visibility.npy", visibility)
    np.save(out_dir / "track_quality.npy", track_quality)
    np.save(out_dir / "points_t0.npy", np.stack([points_seed_by_view[v] for v in view_order], axis=0))
    if anchor_map_point3d_ids_by_seed:
        np.save(out_dir / "anchor_point3D_ids.npy", np.concatenate(anchor_map_point3d_ids_by_seed, axis=0))
        np.save(out_dir / "anchor_points3D_xyz.npy", np.concatenate(anchor_map_points3d_xyz_by_seed, axis=0))
    (out_dir / "views.txt").write_text("\n".join(view_order) + "\n", encoding="utf-8")
    seed_count_metadata = _build_seed_count_metadata(
        seed_frames=all_seed_frames,
        point_counts=per_seed_point_counts,
        requested_points_per_seed=int(args.num_points),
    )
    if int(seed_count_metadata["num_points"]) != int(total_num_points):
        raise RuntimeError("Seed count metadata does not match generated track points")
    meta = {
        "data_root": str(data_root),
        "views": view_order,
        "ref_view": ref_view,
        "start_frame": int(start_frame),
        "seed_frame": int(seed_frame),
        "seed_frames": [int(s) for s in all_seed_frames],
        "track_direction": str(track_direction),
        "num_frames": int(num_frames),
        **seed_count_metadata,
        "seed_builder": str(seed_builder),
        "anchor_map_model_dir": str(anchor_map_model_dir) if anchor_map_model_dir is not None else "",
        "spsg_f_ransac_px": float(args.spsg_f_ransac_px),
        "spsg_feature_conf": str(args.spsg_feature_conf),
        "spsg_seed_geometry_cameras_root": str(args.spsg_seed_geometry_cameras_root),
        "spsg_seed_geometry_reproj_px": float(args.spsg_seed_geometry_reproj_px),
        "spsg_seed_geometry_min_anchor_views": int(args.spsg_seed_geometry_min_anchor_views),
        "spsg_seed_geometry_bridge_policy": str(args.spsg_seed_geometry_bridge_policy),
        "spsg_require_ref_view": bool(args.spsg_require_ref_view),
        "seed_builder_diag_by_seed": seed_builder_diag_by_seed,
        "roma_setting": args.roma_setting,
        "roma_min_overlap": args.roma_min_overlap,
        "roma_match_batch_size": int(roma_match_batch_size_base),
        "roma_match_oom_fallbacks": int(roma_match_oom_fallbacks),
        "roma_scale_align_enable": bool(roma_scale_align_enable),
        "roma_scale_align_cameras_root": str(roma_scale_align_cameras_root),
        "roma_scale_align_min_factor": float(args.roma_scale_align_min_factor),
        "roma_scale_align_max_factor": float(args.roma_scale_align_max_factor),
        "roma_scale_align_min_side": int(args.roma_scale_align_min_side),
        "roma_scale_align_diag_by_seed": roma_scale_align_diag_by_seed if roma_scale_align_enable else {},
        "min_visible_views": int(args.min_visible_views),
        "seed_bridge_view_ids": [str(v) for v in bridge_view_ids],
        "seed_bridge_min_visible": int(args.seed_bridge_min_visible),
        "seed_bridge_min_points_per_view": int(args.seed_bridge_min_points_per_view),
        "seed_consistency_max_error_px": float(args.seed_consistency_max_error_px),
        "seed_consistency_weight": float(args.seed_consistency_weight),
        "seed_bridge_consistency_max_error_px": float(args.seed_bridge_consistency_max_error_px),
        "seed_bridge_consistency_weight": float(args.seed_bridge_consistency_weight),
        "mask_mode": str(args.mask_mode),
        "masks_root": str(mask_dir) if mask_dir is not None else "",
        "tracker_backend": tracker_backend,
        "cotracker_ckpt": str(cotracker_ckpt) if cotracker_ckpt is not None else "",
        "cotracker_mode": str(args.cotracker_mode) if tracker_backend == "cotracker3" else "",
        "cotracker_online_no_chunk": bool(args.cotracker_online_no_chunk) if tracker_backend == "cotracker3" else False,
        "retracker_root": str(args.retracker_root) if tracker_backend == "retracker" else "",
        "retracker_ckpt": str(retracker_ckpt) if retracker_ckpt is not None else "",
        "retracker_dtype": str(args.retracker_dtype) if tracker_backend == "retracker" else "",
        "retracker_query_batch_size": int(args.retracker_query_batch_size) if tracker_backend == "retracker" else 0,
        "retracker_interp_shape": [int(x) for x in args.retracker_interp_shape] if tracker_backend == "retracker" else [],
        "alltracker_repo_root": str(args.alltracker_repo_root) if tracker_backend == "alltracker" else "",
        "alltracker_ckpt": str(alltracker_ckpt) if tracker_backend == "alltracker" else "",
        "alltracker_tiny": bool(args.alltracker_tiny) if tracker_backend == "alltracker" else False,
        "alltracker_image_size": int(args.alltracker_image_size) if tracker_backend == "alltracker" else 0,
        "alltracker_min_image_size": int(args.alltracker_min_image_size) if tracker_backend == "alltracker" else 0,
        "alltracker_downscale_step": int(args.alltracker_downscale_step) if tracker_backend == "alltracker" else 0,
        "alltracker_auto_downscale_on_oom": bool(args.alltracker_auto_downscale_on_oom) if tracker_backend == "alltracker" else False,
        "alltracker_inference_iters": int(args.alltracker_inference_iters) if tracker_backend == "alltracker" else 0,
        "alltracker_window_len": int(args.alltracker_window_len) if tracker_backend == "alltracker" else 0,
        "alltracker_stride": int(args.alltracker_stride) if tracker_backend == "alltracker" else 0,
        "alltracker_confidence_channel": int(args.alltracker_confidence_channel) if tracker_backend == "alltracker" else 0,
        "alltracker_conf_threshold": float(args.alltracker_conf_threshold) if tracker_backend == "alltracker" else 0.0,
        "alltracker_runtime_by_view": alltracker_runtime_by_view if tracker_backend == "alltracker" else {},
        "cowtracker_root": str(args.cowtracker_root) if tracker_backend == "cowtracker" else "",
        "cowtracker_ckpt": str(cowtracker_ckpt) if tracker_backend == "cowtracker" else "",
        "cowtracker_dtype": str(args.cowtracker_dtype) if tracker_backend == "cowtracker" else "",
        "cowtracker_image_size": int(args.cowtracker_image_size) if tracker_backend == "cowtracker" else 0,
        "cowtracker_visconf_threshold": float(args.cowtracker_visconf_threshold) if tracker_backend == "cowtracker" else 0.0,
        "cowtracker_runtime_by_view": cowtracker_runtime_by_view if tracker_backend == "cowtracker" else {},
        "raft_model": str(args.raft_model) if tracker_backend == "raft" else "",
        "raft_weights": str(raft_weights_source) if tracker_backend == "raft" else "",
        "raft_image_size": int(args.raft_image_size) if tracker_backend == "raft" else 0,
        "raft_iters": int(args.raft_iters) if tracker_backend == "raft" else 0,
        "raft_fb_consistency_threshold_px": float(args.raft_fb_consistency_threshold_px) if tracker_backend == "raft" else 0.0,
        "raft_conf_threshold": float(args.raft_conf_threshold) if tracker_backend == "raft" else 0.0,
        "raft_amp": bool(args.raft_amp) if tracker_backend == "raft" else False,
        "raft_runtime_by_view": raft_runtime_by_view if tracker_backend == "raft" else {},
        "tapir_repo_root": str(args.tapir_repo_root) if tracker_backend == "tapir" else "",
        "tapir_checkpoint": str(tapir_ckpt) if tracker_backend == "tapir" else "",
        "tapir_variant": str(args.tapir_variant) if tracker_backend == "tapir" else "",
        "tapir_image_size": int(args.tapir_image_size) if tracker_backend == "tapir" else 0,
        "tapir_query_chunk_size": int(args.tapir_query_chunk_size) if tracker_backend == "tapir" else 0,
        "tapir_conf_threshold": float(args.tapir_conf_threshold) if tracker_backend == "tapir" else 0.0,
        "tapir_amp": bool(args.tapir_amp) if tracker_backend == "tapir" else False,
        "tapir_runtime_by_view": tapir_runtime_by_view if tracker_backend == "tapir" else {},
        "cotracker_vis_threshold": float(args.cotracker_vis_threshold),
        "cotracker_vis_hysteresis_low": float(args.cotracker_vis_hysteresis_low),
        "cotracker_vis_hysteresis_high": float(args.cotracker_vis_hysteresis_high),
        "track_filter_border_margin_px": float(args.track_filter_border_margin_px),
        "track_filter_max_step_px": float(args.track_filter_max_step_px),
        "track_filter_reentry_dist_px": float(args.track_filter_reentry_dist_px),
        "track_filter_reentry_lost_frames": int(args.track_filter_reentry_lost_frames),
        "track_filter_stats_by_view": track_filter_stats_by_view,
        "track_quality_enable": bool(args.track_quality_enable),
        "track_quality_vis_weight": float(args.track_quality_vis_weight),
        "track_quality_border_margin_px": float(args.track_quality_border_margin_px),
        "track_quality_step_px": float(args.track_quality_step_px),
        "track_quality_reentry_dist_px": float(args.track_quality_reentry_dist_px),
        "track_quality_reentry_lost_frames": int(args.track_quality_reentry_lost_frames),
        "track_quality_min_weight": float(args.track_quality_min_weight),
        "track_quality_stats_by_view": track_quality_stats_by_view,
        "device": str(device),
    }
    meta.update(layout.as_meta())
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Optional visualization video.
    if args.save_video:
        n = total_num_points
        colors = _make_colors(n, seed=args.seed)
        video_path = out_dir / "tracks.mp4"
        try:
            writer = _open_video_writer(video_path, fps=int(args.fps), quality=8, codec=str(args.video_codec))
        except RuntimeError as exc:
            print(f"[WARN] Skipping track visualization video: {exc}", file=sys.stderr)
            writer = None
        if writer is not None:
            try:
                for t in range(int(num_frames)):
                    vis_frames = []
                    for vid in view_order:
                        fr = frames_by_view[vid][t]
                        fr_vis = _draw_tracks_rgb(fr, tracks_by_view[vid], vis_by_view[vid], colors, t=t)
                        # Label
                        fr_bgr = cv2.cvtColor(fr_vis, cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            fr_bgr,
                            f"view {vid}  frame {start_frame + t:06d}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        vis_frames.append(cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB))

                    combined = np.concatenate(vis_frames, axis=1)
                    writer.append_data(combined)
            finally:
                writer.close()

    print(f"[OK] Saved outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
