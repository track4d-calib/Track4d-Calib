from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch.Tensor / numpy array-like to numpy (CPU)."""
    if isinstance(x, np.ndarray):
        return x
    # Torch is an optional dependency for tests; import lazily.
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        torch = None  # type: ignore
    if torch is not None and isinstance(x, torch.Tensor):  # type: ignore[attr-defined]
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalized_grid(h: int, w: int) -> np.ndarray:
    """Normalized (x, y) grid matching RoMaV2's get_normalized_grid convention."""
    ys = np.linspace(-1 + 1 / h, 1 - 1 / h, h, dtype=np.float32)
    xs = np.linspace(-1 + 1 / w, 1 - 1 / w, w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx, yy], axis=-1)  # (h, w, 2)


def _norm_to_pixel(xy_norm: np.ndarray, *, h: int, w: int) -> np.ndarray:
    """Convert normalized coords in [-1, 1] to pixel coords in [0, W/H]."""
    out = np.empty_like(xy_norm, dtype=np.float32)
    out[..., 0] = (xy_norm[..., 0] + 1.0) * 0.5 * w
    out[..., 1] = (xy_norm[..., 1] + 1.0) * 0.5 * h
    return out


def sample_triplet_points(
    *,
    ref_mask: Any,
    v1_mask: Any,
    v2_mask: Any,
    warp_ref_to_v1: Any,
    overlap_ref_to_v1: Any,
    warp_ref_to_v2: Any,
    overlap_ref_to_v2: Any,
    num_points: int,
    min_overlap: float,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Sample matched triplets (ref, v1, v2) from two RoMaV2-like warps.

    Args:
        ref_mask: (H, W) mask for the reference view (mask > 0 keeps).
        v1_mask:  (H, W) mask for target view 1.
        v2_mask:  (H, W) mask for target view 2.
        warp_ref_to_v1: (h, w, 2) normalized warp from ref -> v1 in [-1, 1].
        overlap_ref_to_v1: (h, w) overlap/confidence map for ref -> v1.
        warp_ref_to_v2: (h, w, 2) normalized warp from ref -> v2 in [-1, 1].
        overlap_ref_to_v2: (h, w) overlap/confidence map for ref -> v2.
        num_points: number of points to sample.
        min_overlap: minimum overlap threshold (applied to both pairs).
        seed: RNG seed for deterministic sampling.

    Returns:
        Dict with keys: ref_xy, v1_xy, v2_xy, each (N, 2) float32 pixel coords.
    """
    if num_points <= 0:
        raise ValueError("num_points must be > 0")

    ref_mask = _to_numpy(ref_mask)
    v1_mask = _to_numpy(v1_mask)
    v2_mask = _to_numpy(v2_mask)
    warp1 = _to_numpy(warp_ref_to_v1).astype(np.float32)
    warp2 = _to_numpy(warp_ref_to_v2).astype(np.float32)
    ov1 = _to_numpy(overlap_ref_to_v1).astype(np.float32)
    ov2 = _to_numpy(overlap_ref_to_v2).astype(np.float32)

    if warp1.ndim != 3 or warp1.shape[-1] != 2:
        raise ValueError(f"warp_ref_to_v1 must have shape (h, w, 2), got {warp1.shape}")
    if warp2.ndim != 3 or warp2.shape[-1] != 2:
        raise ValueError(f"warp_ref_to_v2 must have shape (h, w, 2), got {warp2.shape}")
    if warp2.shape[:2] != warp1.shape[:2]:
        raise ValueError("warp_ref_to_v1 and warp_ref_to_v2 must have the same (h, w)")

    h_warp, w_warp = warp1.shape[:2]
    if ov1.shape[:2] != (h_warp, w_warp):
        ov1 = ov1.reshape(h_warp, w_warp)
    if ov2.shape[:2] != (h_warp, w_warp):
        ov2 = ov2.reshape(h_warp, w_warp)

    h_ref, w_ref = ref_mask.shape[:2]
    h_v1, w_v1 = v1_mask.shape[:2]
    h_v2, w_v2 = v2_mask.shape[:2]

    # Grid positions in the ref view, at warp resolution.
    grid_ref = _normalized_grid(h_warp, w_warp)  # (h, w, 2) in (x, y)

    # Convert all normalized coords to pixel coords in their *image* resolutions.
    ref_xy = _norm_to_pixel(grid_ref, h=h_ref, w=w_ref)  # (h, w, 2)
    v1_xy = _norm_to_pixel(warp1, h=h_v1, w=w_v1)  # (h, w, 2)
    v2_xy = _norm_to_pixel(warp2, h=h_v2, w=w_v2)  # (h, w, 2)

    # Basic validity: overlap threshold + warp in-bounds.
    valid = (ov1 >= float(min_overlap)) & (ov2 >= float(min_overlap))
    valid &= (np.abs(warp1).max(axis=-1) <= 1.0) & (np.abs(warp2).max(axis=-1) <= 1.0)

    # Mask filtering (nearest-neighbor lookup).
    def in_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
        hh, ww = mask.shape[:2]
        x = np.clip(xy[..., 0].astype(np.int64), 0, ww - 1)
        y = np.clip(xy[..., 1].astype(np.int64), 0, hh - 1)
        return mask[y, x] > 0

    valid &= in_mask(ref_mask, ref_xy)
    valid &= in_mask(v1_mask, v1_xy)
    valid &= in_mask(v2_mask, v2_xy)

    cand = np.flatnonzero(valid.reshape(-1))
    if cand.size < num_points:
        raise ValueError(
            f"Not enough candidates after filtering: need {num_points}, have {cand.size}"
        )

    scores = (ov1 * ov2).reshape(-1)[cand]
    if float(scores.sum()) > 0:
        p = scores / scores.sum()
    else:  # fallback: uniform
        p = None

    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(cand, size=int(num_points), replace=False, p=p)

    ref_xy_f = ref_xy.reshape(-1, 2)[chosen].astype(np.float32)
    v1_xy_f = v1_xy.reshape(-1, 2)[chosen].astype(np.float32)
    v2_xy_f = v2_xy.reshape(-1, 2)[chosen].astype(np.float32)

    return {"ref_xy": ref_xy_f, "v1_xy": v1_xy_f, "v2_xy": v2_xy_f}


def sample_multiview_points(
    *,
    ref_mask: Any,
    view_masks: Dict[str, Any],
    warps_ref_to_view: Dict[str, Any],
    overlaps_ref_to_view: Dict[str, Any],
    num_points: int,
    min_overlap: float,
    min_visible_views: int = 3,
    bridge_view_ids: Sequence[str] = (),
    bridge_min_visible: int = 0,
    bridge_min_points_per_view: int = 0,
    consistency_error_ref_to_view: Optional[Dict[str, Any]] = None,
    consistency_max_error_px: float = 0.0,
    consistency_weight: float = 0.0,
    bridge_consistency_max_error_px: float = 0.0,
    bridge_consistency_weight: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Sample matched points from a reference view to multiple target views.

    This generalizes :func:`sample_triplet_points` to N views by sampling points on the
    reference grid that are valid in at least ``min_visible_views`` total views
    (including the reference view).

    Optional bridge constraints allow enforcing extra support on a designated set of
    views (e.g., long-focal views):
      - ``bridge_min_visible``: each sampled point must be valid in at least this many
        bridge views (0 disables).
      - ``bridge_min_points_per_view``: soft per-view quota; we first reserve points
        for each bridge view, then fill the remaining budget globally (0 disables).

    Optional geometric consistency terms:
      - ``consistency_error_ref_to_view``: per-view error map (h,w), e.g., RoMa cycle
        consistency in ref-pixel units.
      - ``consistency_max_error_px``: hard gate applied to all views (0 disables).
      - ``consistency_weight``: downweight high-error observations in global sampling
        score (0 disables).
      - ``bridge_consistency_max_error_px`` and ``bridge_consistency_weight``:
        stricter counterparts used only for bridge quota ranking.

    Returns:
        Dict with:
          - ref_xy: (N, 2) float32 pixel coords in ref image
          - xy_by_view: Dict[view_id, (N, 2)] pixel coords for each target view
          - valid_by_view: Dict[view_id, (N,)] bool validity for each target view
    """
    if num_points <= 0:
        raise ValueError("num_points must be > 0")
    if not view_masks:
        raise ValueError("view_masks must be non-empty")
    if set(view_masks.keys()) != set(warps_ref_to_view.keys()) or set(view_masks.keys()) != set(
        overlaps_ref_to_view.keys()
    ):
        raise ValueError("view_masks/warps_ref_to_view/overlaps_ref_to_view must have identical keys")
    if min_visible_views < 2:
        raise ValueError("min_visible_views must be >= 2 (including reference view)")
    if bridge_min_visible < 0:
        raise ValueError("bridge_min_visible must be >= 0")
    if bridge_min_points_per_view < 0:
        raise ValueError("bridge_min_points_per_view must be >= 0")
    if float(consistency_max_error_px) < 0.0:
        raise ValueError("consistency_max_error_px must be >= 0")
    if float(consistency_weight) < 0.0:
        raise ValueError("consistency_weight must be >= 0")
    if float(bridge_consistency_max_error_px) < 0.0:
        raise ValueError("bridge_consistency_max_error_px must be >= 0")
    if float(bridge_consistency_weight) < 0.0:
        raise ValueError("bridge_consistency_weight must be >= 0")

    ref_mask = _to_numpy(ref_mask)
    h_ref, w_ref = ref_mask.shape[:2]

    # Convert and validate warp shapes (all warps must share the same resolution).
    first_vid = next(iter(view_masks.keys()))
    warp0 = _to_numpy(warps_ref_to_view[first_vid]).astype(np.float32)
    if warp0.ndim != 3 or warp0.shape[-1] != 2:
        raise ValueError(f"warp_ref_to_view[{first_vid}] must have shape (h, w, 2), got {warp0.shape}")
    h_warp, w_warp = warp0.shape[:2]

    for vid, warp in warps_ref_to_view.items():
        w = _to_numpy(warp)
        if w.ndim != 3 or w.shape[-1] != 2 or w.shape[:2] != (h_warp, w_warp):
            raise ValueError(
                f"All warps must have shape (h, w, 2) with the same h,w; got warp[{vid}]={w.shape}"
            )

    # Grid positions in the ref view, at warp resolution.
    grid_ref = _normalized_grid(h_warp, w_warp)  # (h, w, 2) in (x, y)
    ref_xy_all = _norm_to_pixel(grid_ref, h=h_ref, w=w_ref)  # (h, w, 2)

    # Mask filtering (nearest-neighbor lookup).
    def in_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
        hh, ww = mask.shape[:2]
        x = np.clip(xy[..., 0].astype(np.int64), 0, ww - 1)
        y = np.clip(xy[..., 1].astype(np.int64), 0, hh - 1)
        return mask[y, x] > 0

    base_valid = in_mask(ref_mask, ref_xy_all)  # (h, w)

    xy_all_by_view: Dict[str, np.ndarray] = {}
    valid_all_by_view: Dict[str, np.ndarray] = {}
    consistency_flat_by_view: Dict[str, np.ndarray] = {}
    score_map = np.zeros((h_warp, w_warp), dtype=np.float32)

    for vid in view_masks.keys():
        vmask = _to_numpy(view_masks[vid])
        warp = _to_numpy(warps_ref_to_view[vid]).astype(np.float32)
        ov = _to_numpy(overlaps_ref_to_view[vid]).astype(np.float32)
        if ov.ndim == 3 and ov.shape[-1] == 1:
            ov = ov[..., 0]
        if ov.shape[:2] != (h_warp, w_warp):
            ov = ov.reshape(h_warp, w_warp)

        h_v, w_v = vmask.shape[:2]
        xy_all = _norm_to_pixel(warp, h=h_v, w=w_v)  # (h, w, 2)

        cons_map: Optional[np.ndarray] = None
        if consistency_error_ref_to_view is not None and vid in consistency_error_ref_to_view:
            cons_map = _to_numpy(consistency_error_ref_to_view[vid]).astype(np.float32)
            if cons_map.ndim == 3 and cons_map.shape[-1] == 1:
                cons_map = cons_map[..., 0]
            if cons_map.shape[:2] != (h_warp, w_warp):
                cons_map = cons_map.reshape(h_warp, w_warp)

        if cons_map is None:
            cons_map_valid = np.ones((h_warp, w_warp), dtype=bool)
            cons_map_safe = np.zeros((h_warp, w_warp), dtype=np.float32)
        else:
            cons_map_valid = np.isfinite(cons_map)
            cons_map_safe = np.where(cons_map_valid, cons_map, 1e6).astype(np.float32)

        # Per-view validity for each ref-grid cell.
        v_valid = (ov >= float(min_overlap)) & (np.abs(warp).max(axis=-1) <= 1.0)
        v_valid &= in_mask(vmask, xy_all)
        v_valid &= base_valid
        v_valid &= cons_map_valid
        if float(consistency_max_error_px) > 0.0:
            v_valid &= cons_map_safe <= float(consistency_max_error_px)

        xy_all_by_view[vid] = xy_all
        valid_all_by_view[vid] = v_valid
        consistency_flat_by_view[vid] = cons_map_safe.reshape(-1)

        score_contrib = ov * v_valid.astype(np.float32)
        if float(consistency_weight) > 0.0:
            score_contrib = score_contrib / (
                1.0 + float(consistency_weight) * np.clip(cons_map_safe, 0.0, 1e6)
            )
        score_map += score_contrib

    # Filter candidates by visibility count (including reference view).
    visibility_count = base_valid.astype(np.int32)
    for vid in view_masks.keys():
        visibility_count += valid_all_by_view[vid].astype(np.int32)

    candidate_mask = visibility_count >= int(min_visible_views)
    bridge_ids: List[str] = [str(v) for v in bridge_view_ids if str(v) in view_masks]
    bridge_visibility_count = None
    if bridge_ids:
        bridge_visibility_count = np.zeros((h_warp, w_warp), dtype=np.int32)
        for vid in bridge_ids:
            bridge_visibility_count += valid_all_by_view[vid].astype(np.int32)
        if int(bridge_min_visible) > 0:
            candidate_mask &= bridge_visibility_count >= int(bridge_min_visible)

    cand = np.flatnonzero(candidate_mask.reshape(-1))
    if cand.size < int(num_points):
        raise ValueError(f"Not enough candidates after filtering: need {num_points}, have {cand.size}")

    score_flat = score_map.reshape(-1)
    scores = score_flat[cand]
    if float(scores.sum()) > 0.0:
        p = scores / scores.sum()
    else:
        p = None
    rng = np.random.default_rng(int(seed))

    chosen: np.ndarray
    if int(bridge_min_points_per_view) > 0 and bridge_ids:
        mandatory: List[int] = []
        mandatory_set = set()
        cand_mask_flat = np.zeros((score_flat.size,), dtype=bool)
        cand_mask_flat[cand] = True
        bridge_cons_thr = float(bridge_consistency_max_error_px)
        if bridge_cons_thr <= 0.0:
            bridge_cons_thr = float(consistency_max_error_px)
        for vid in bridge_ids:
            v_flat = valid_all_by_view[vid].reshape(-1)
            eligible_mask = cand_mask_flat & v_flat
            cons_flat = consistency_flat_by_view.get(vid)
            if cons_flat is not None and bridge_cons_thr > 0.0:
                eligible_mask &= cons_flat <= bridge_cons_thr
            eligible = np.flatnonzero(eligible_mask)
            if eligible.size == 0:
                continue
            bridge_score = score_flat[eligible]
            if cons_flat is not None and float(bridge_consistency_weight) > 0.0:
                bridge_score = bridge_score / (
                    1.0 + float(bridge_consistency_weight) * np.clip(cons_flat[eligible], 0.0, 1e6)
                )
            ranked = eligible[np.argsort(bridge_score)[::-1]]
            need = int(min(int(bridge_min_points_per_view), ranked.size))
            for idx in ranked[:need]:
                idx_int = int(idx)
                if idx_int not in mandatory_set:
                    mandatory_set.add(idx_int)
                    mandatory.append(idx_int)

        if len(mandatory) > int(num_points):
            mandatory_arr = np.asarray(mandatory, dtype=np.int64)
            mandatory = mandatory_arr[np.argsort(score_flat[mandatory_arr])[::-1][: int(num_points)]].astype(np.int64).tolist()
            mandatory_set = set(int(x) for x in mandatory)

        remaining_need = int(num_points) - len(mandatory)
        if remaining_need > 0:
            remaining_cand = np.asarray([idx for idx in cand if int(idx) not in mandatory_set], dtype=np.int64)
            if remaining_cand.size < remaining_need:
                raise ValueError(
                    f"Not enough remaining candidates after bridge quotas: need {remaining_need}, have {remaining_cand.size}"
                )
            remaining_scores = score_flat[remaining_cand]
            if float(remaining_scores.sum()) > 0.0:
                p_rem = remaining_scores / remaining_scores.sum()
            else:
                p_rem = None
            sampled_rem = rng.choice(remaining_cand, size=remaining_need, replace=False, p=p_rem)
            chosen = np.concatenate([np.asarray(mandatory, dtype=np.int64), sampled_rem.astype(np.int64)], axis=0)
        else:
            chosen = np.asarray(mandatory, dtype=np.int64)
    else:
        chosen = rng.choice(cand, size=int(num_points), replace=False, p=p)

    ref_xy = ref_xy_all.reshape(-1, 2)[chosen].astype(np.float32)
    xy_by_view = {vid: xy_all_by_view[vid].reshape(-1, 2)[chosen].astype(np.float32) for vid in view_masks.keys()}
    valid_by_view = {vid: valid_all_by_view[vid].reshape(-1)[chosen].astype(bool) for vid in view_masks.keys()}

    return {"ref_xy": ref_xy, "xy_by_view": xy_by_view, "valid_by_view": valid_by_view}
