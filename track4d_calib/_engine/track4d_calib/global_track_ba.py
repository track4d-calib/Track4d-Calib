from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from track4d_calib.calibration import _torch_available, rodrigues_torch
from track4d_calib.cameras import CameraParams
from track4d_calib.triangulation import triangulate_point_dlt


@dataclass
class GlobalTrackBAConfig:
    """Configuration for track-augmented global BA refinement."""

    frame_stride: int = 1
    max_points_per_frame: int = 0
    min_views_per_point: int = 3
    triangulation_reproj_threshold: float = 5.0
    min_triangulation_angle_deg: float = 0.5
    max_tracks_total: int = 0

    # Track quality filtering (for more stable BA observations).
    min_track_support_frames: int = 2
    min_track_mean_visible_views: float = 0.0

    # Initial observation outlier filtering (before BA solve).
    init_obs_filter_enable: bool = True
    init_obs_max_reproj_error_px: float = 20.0
    init_obs_keep_quantile: float = 0.9
    init_obs_min_keep_ratio: float = 0.5

    # Observation weighting from tracking confidence.
    obs_weight_enable: bool = False
    obs_weight_min: float = 0.25
    obs_weight_power: float = 1.0

    # Layer2.5: track-driven camera-graph closure reweighting.
    graph_closure_enable: bool = False
    graph_closure_min_shared_obs: int = 64
    graph_closure_max_pair_obs: int = 2000
    graph_closure_ransac_threshold: float = 1e-3
    graph_closure_ransac_prob: float = 0.999
    graph_closure_min_inliers: int = 24
    graph_closure_min_inlier_ratio: float = 0.25
    graph_closure_cycle_tau_deg: float = 1.0
    graph_closure_min_weight: float = 0.05
    graph_closure_weight_power: float = 1.0
    graph_closure_no_cycle_weight: float = 0.5
    graph_closure_loss_weight: float = 0.1
    graph_closure_stage1_loss_scale: float = 0.5
    graph_closure_loss_huber_delta_deg: float = 5.0

    max_steps: int = 200
    batch_size: int = 8192
    lr: float = 1e-2
    huber_delta: float = 3.0

    # Two-stage optimization (warmup -> refine).
    stage1_steps_ratio: float = 0.4
    stage1_lr_scale: float = 0.5
    stage1_reg_scale: float = 5.0
    stage1_point_reg_scale: float = 10.0
    stage1_depth_penalty_scale: float = 2.0
    stage1_huber_delta_scale: float = 0.75

    reg_weight: float = 1e-3
    point_reg_weight: float = 1e-5
    depth_penalty_weight: float = 1e-2

    # Internal acceptance gate (not baseline fallback): reject unstable updates.
    accept_gate_enable: bool = True
    accept_holdout_ratio: float = 0.1
    accept_min_holdout_obs: int = 128
    accept_max_error_increase_abs: float = 0.05
    accept_max_error_increase_rel: float = 0.01
    accept_per_view_holdout_gate_enable: bool = True
    accept_per_view_holdout_min_obs: int = 128
    accept_per_view_holdout_max_error_increase_abs: float = 0.05
    accept_per_view_holdout_max_error_increase_rel: float = 0.02
    accept_require_global_improvement: bool = True
    accept_min_error_drop_abs: float = 1e-4

    seed: int = 0
    device: str = "cuda"
    fix_ref_extrinsic: bool = True
    fixed_views: Sequence[str] = ()


def _huber_torch(x, delta: float):
    import torch

    ax = torch.abs(x)
    d = torch.tensor(float(delta), dtype=ax.dtype, device=ax.device)
    quad = torch.minimum(ax, d)
    lin = ax - quad
    return 0.5 * quad * quad + d * lin


def _max_triangulation_angle_deg(
    *,
    cameras: Dict[str, CameraParams],
    view_candidates: Sequence[str],
    X: np.ndarray,
) -> float:
    """Return the maximum triangulation angle (deg) across view pairs for a point."""
    if len(view_candidates) < 2:
        return 0.0
    X = np.asarray(X, dtype=np.float64).reshape(3)
    rays: Dict[str, np.ndarray] = {}
    for vid in view_candidates:
        cam = cameras[vid]
        ray = X - cam.camera_center
        n = float(np.linalg.norm(ray))
        if n <= 1e-12:
            continue
        rays[vid] = ray / n
    vids = list(rays.keys())
    if len(vids) < 2:
        return 0.0

    best = 0.0
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            a = rays[vids[i]]
            b = rays[vids[j]]
            c = float(np.clip(np.dot(a, b), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(c)))
            if angle > best:
                best = angle
    return float(best)


def _triangulate_point_multi_view_undistorted(
    *,
    cameras: Dict[str, CameraParams],
    points_2d: Dict[str, np.ndarray],
    visibility: Dict[str, bool],
    reprojection_error_threshold: float,
    min_views_for_triangulation: int,
    min_triangulation_angle_deg: float,
) -> Tuple[np.ndarray | None, float, int, str]:
    """Triangulate from undistorted observations and reject low-parallax points."""
    visible_views = [vid for vid in cameras.keys() if bool(visibility.get(vid, False)) and vid in points_2d]
    if len(visible_views) < int(min_views_for_triangulation):
        return None, float("inf"), len(visible_views), "insufficient_views"

    proj = []
    pts_undist = []
    for vid in visible_views:
        cam = cameras[vid]
        uv = np.asarray(points_2d[vid], dtype=np.float64).reshape(1, 1, 2)
        d = np.asarray(cam.D, dtype=np.float64).reshape(-1)
        if d.size < 5:
            d2 = np.zeros((5,), dtype=np.float64)
            d2[: d.size] = d
            d = d2
        uv_u = cv2.undistortPoints(uv, cam.K, d[:5], P=cam.K).reshape(2)
        proj.append(cam.P)
        pts_undist.append(uv_u)

    X = triangulate_point_dlt(proj_matrices=proj, pts_2d=pts_undist)
    if X is None:
        return None, float("inf"), len(proj), "dlt_failed"

    for vid in visible_views:
        cam = cameras[vid]
        Xc = cam.R @ X + cam.T
        if float(Xc[2]) <= 1e-6:
            return None, float("inf"), len(proj), "behind_camera"

    total = 0.0
    for P, pt in zip(proj, pts_undist):
        x = P @ np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
        if float(x[2]) <= 1e-12:
            return None, float("inf"), len(proj), "behind_camera"
        uv = x[:2] / x[2]
        total += float(np.linalg.norm(uv - pt))
    mean_err = total / float(len(proj))

    if mean_err > float(reprojection_error_threshold):
        return None, mean_err, len(proj), "high_reprojection"

    min_angle = float(max(0.0, min_triangulation_angle_deg))
    if min_angle > 0.0:
        max_angle = _max_triangulation_angle_deg(cameras=cameras, view_candidates=visible_views, X=X)
        if (not np.isfinite(max_angle)) or float(max_angle) < min_angle:
            return None, mean_err, len(proj), "low_parallax"

    return np.asarray(X, dtype=np.float64), float(mean_err), len(proj), "ok"


def _triangulate_point_best_pair(
    *,
    cameras: Dict[str, CameraParams],
    points_2d: Dict[str, np.ndarray],
    view_candidates: Sequence[str],
    reproj_threshold: float,
    min_triangulation_angle_deg: float,
) -> np.ndarray | None:
    """Fallback triangulation from the strongest baseline pair among visible views."""
    if len(view_candidates) < 2:
        return None

    best_pair = None
    best_baseline = -1.0
    for i in range(len(view_candidates)):
        for j in range(i + 1, len(view_candidates)):
            va, vb = view_candidates[i], view_candidates[j]
            ca = cameras[va]
            cb = cameras[vb]
            b = float(np.linalg.norm(ca.camera_center - cb.camera_center))
            if b > best_baseline:
                best_baseline = b
                best_pair = (va, vb)

    if best_pair is None:
        return None

    va, vb = best_pair
    ca, cb = cameras[va], cameras[vb]

    ua = np.asarray(points_2d[va], dtype=np.float64).reshape(1, 1, 2)
    ub = np.asarray(points_2d[vb], dtype=np.float64).reshape(1, 1, 2)

    ua_u = cv2.undistortPoints(ua, ca.K, np.asarray(ca.D, dtype=np.float64).reshape(-1)[:5], P=ca.K).reshape(2)
    ub_u = cv2.undistortPoints(ub, cb.K, np.asarray(cb.D, dtype=np.float64).reshape(-1)[:5], P=cb.K).reshape(2)

    Xh = cv2.triangulatePoints(ca.P, cb.P, ua_u.reshape(2, 1), ub_u.reshape(2, 1)).reshape(4)
    if abs(float(Xh[3])) < 1e-12:
        return None
    X = np.asarray(Xh[:3] / Xh[3], dtype=np.float64).reshape(3)

    Xa = ca.R @ X + ca.T
    Xb = cb.R @ X + cb.T
    if float(Xa[2]) <= 1e-6 or float(Xb[2]) <= 1e-6:
        return None

    min_angle = float(max(0.0, min_triangulation_angle_deg))
    if min_angle > 0.0:
        angle = _max_triangulation_angle_deg(cameras=cameras, view_candidates=[va, vb], X=X)
        if (not np.isfinite(angle)) or float(angle) < min_angle:
            return None

    def _reproj(cam: CameraParams, uv_obs: np.ndarray) -> float:
        rvec, _ = cv2.Rodrigues(np.asarray(cam.R, dtype=np.float64))
        uv_pred, _ = cv2.projectPoints(X.reshape(1, 3), rvec, cam.T.reshape(3, 1), cam.K, cam.D)
        uv_pred = uv_pred.reshape(2)
        return float(np.linalg.norm(uv_pred - uv_obs.reshape(2)))

    err = 0.5 * (_reproj(ca, ua.reshape(2)) + _reproj(cb, ub.reshape(2)))
    if err > float(reproj_threshold):
        return None
    return X


def _gather_triangulated_observations(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    cameras: Dict[str, CameraParams],
    cfg: GlobalTrackBAConfig,
) -> Dict[str, Any]:
    """Build BA observations and point initialization via per-frame triangulation."""
    rng = np.random.default_rng(int(cfg.seed))
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)

    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must have shape (V,T,N,2), got {tracks.shape}")
    if visibility.shape != tracks.shape[:3]:
        raise ValueError(f"visibility shape mismatch: {visibility.shape} vs {tracks.shape[:3]}")

    _, T, N, _ = tracks.shape
    stride = max(int(cfg.frame_stride), 1)
    min_views = max(int(cfg.min_views_per_point), 2)

    points_init: List[np.ndarray] = []
    point_meta: List[Tuple[int, int]] = []  # (frame_idx, point_id)
    point_confidence: List[float] = []

    obs_view_idx: List[int] = []
    obs_point_idx: List[int] = []
    obs_uv: List[np.ndarray] = []

    n_candidates = 0
    n_candidates_kept = 0
    n_failed_triangulation = 0
    n_failed_low_parallax = 0

    sampled_ts = list(range(0, int(T), stride))
    if sampled_ts:
        vis_sub = np.take(visibility, sampled_ts, axis=1)
        trk_sub = np.take(tracks, sampled_ts, axis=1)
        valid_sampled = vis_sub & np.isfinite(trk_sub[..., 0]) & np.isfinite(trk_sub[..., 1])
        num_visible_sampled = np.sum(valid_sampled, axis=0)  # (Ts, N)
        support_frames = np.sum(num_visible_sampled >= min_views, axis=0).astype(np.int32)
        mean_visible_views = np.mean(num_visible_sampled, axis=0).astype(np.float64)
    else:
        support_frames = np.zeros((N,), dtype=np.int32)
        mean_visible_views = np.zeros((N,), dtype=np.float64)

    support_mask = np.ones((N,), dtype=bool)
    if int(cfg.min_track_support_frames) > 0:
        support_mask &= support_frames >= int(cfg.min_track_support_frames)
    if float(cfg.min_track_mean_visible_views) > 0.0:
        support_mask &= mean_visible_views >= float(cfg.min_track_mean_visible_views)

    n_filtered_by_support = int(np.sum(~support_mask))
    track_quality = support_frames.astype(np.float64) + 0.25 * mean_visible_views
    max_support = float(max(1, len(sampled_ts)))
    max_visible = float(max(1, len(view_ids)))
    support_norm = np.clip(support_frames.astype(np.float64) / max_support, 0.0, 1.0)
    visible_norm = np.clip(mean_visible_views.astype(np.float64) / max_visible, 0.0, 1.0)
    track_confidence = np.clip(0.6 * support_norm + 0.4 * visible_norm, 0.0, 1.0)

    for t in sampled_ts:
        valid = (
            visibility[:, t, :]
            & np.isfinite(tracks[:, t, :, 0])
            & np.isfinite(tracks[:, t, :, 1])
        )
        num_visible = np.sum(valid, axis=0)
        candidate_ids = np.flatnonzero(num_visible >= min_views)
        if candidate_ids.size == 0:
            continue

        n_candidates += int(candidate_ids.size)
        candidate_ids = candidate_ids[support_mask[candidate_ids]]
        if candidate_ids.size == 0:
            continue

        if int(cfg.max_points_per_frame) > 0 and candidate_ids.size > int(cfg.max_points_per_frame):
            q = track_quality[candidate_ids]
            # Stable ranking with tiny deterministic jitter to break ties.
            q = q + rng.normal(scale=1e-9, size=q.shape)
            order = np.argsort(-q)
            candidate_ids = candidate_ids[order[: int(cfg.max_points_per_frame)]]

        n_candidates_kept += int(candidate_ids.size)

        for pid in candidate_ids:
            points_2d: Dict[str, np.ndarray] = {}
            vis: Dict[str, bool] = {}
            local_view_indices: List[int] = []

            for vidx, vid in enumerate(view_ids):
                ok = bool(valid[vidx, pid])
                vis[vid] = ok
                if not ok:
                    continue
                points_2d[vid] = tracks[vidx, t, pid]
                local_view_indices.append(vidx)

            X, _, _, tri_reason = _triangulate_point_multi_view_undistorted(
                cameras=cameras,
                points_2d=points_2d,
                visibility=vis,
                reprojection_error_threshold=float(cfg.triangulation_reproj_threshold),
                min_views_for_triangulation=min_views,
                min_triangulation_angle_deg=float(getattr(cfg, "min_triangulation_angle_deg", 0.0)),
            )
            if X is None:
                if tri_reason == "low_parallax":
                    n_failed_low_parallax += 1
                # Fallback: use the best-baseline two-view pair for a looser init.
                X = _triangulate_point_best_pair(
                    cameras=cameras,
                    points_2d=points_2d,
                    view_candidates=[view_ids[i] for i in local_view_indices],
                    reproj_threshold=float(max(20.0, 4.0 * float(cfg.triangulation_reproj_threshold))),
                    min_triangulation_angle_deg=float(getattr(cfg, "min_triangulation_angle_deg", 0.0)),
                )
            if X is None:
                n_failed_triangulation += 1
                continue

            pidx = len(points_init)
            points_init.append(np.asarray(X, dtype=np.float64).reshape(3))
            point_meta.append((int(t), int(pid)))
            point_confidence.append(float(track_confidence[int(pid)]))

            for vidx in local_view_indices:
                obs_view_idx.append(int(vidx))
                obs_point_idx.append(int(pidx))
                obs_uv.append(np.asarray(tracks[vidx, t, pid], dtype=np.float64).reshape(2))

    if int(cfg.max_tracks_total) > 0 and len(points_init) > int(cfg.max_tracks_total):
        keep = rng.choice(np.arange(len(points_init)), size=int(cfg.max_tracks_total), replace=False)
        keep = np.sort(keep)
        keep_set = set(int(x) for x in keep)

        remap = {old: new for new, old in enumerate(keep.tolist())}
        points_init = [points_init[i] for i in keep.tolist()]
        point_meta = [point_meta[i] for i in keep.tolist()]
        point_confidence = [point_confidence[i] for i in keep.tolist()]

        new_obs_view_idx: List[int] = []
        new_obs_point_idx: List[int] = []
        new_obs_uv: List[np.ndarray] = []
        for v, p, uv in zip(obs_view_idx, obs_point_idx, obs_uv):
            if int(p) not in keep_set:
                continue
            new_obs_view_idx.append(int(v))
            new_obs_point_idx.append(int(remap[int(p)]))
            new_obs_uv.append(np.asarray(uv, dtype=np.float64))
        obs_view_idx, obs_point_idx, obs_uv = new_obs_view_idx, new_obs_point_idx, new_obs_uv

    if not points_init or not obs_uv:
        return {
            "points_init": np.zeros((0, 3), dtype=np.float64),
            "obs_view_idx": np.zeros((0,), dtype=np.int64),
            "obs_point_idx": np.zeros((0,), dtype=np.int64),
            "obs_uv": np.zeros((0, 2), dtype=np.float64),
            "point_meta": [],
            "point_confidence": np.zeros((0,), dtype=np.float64),
            "n_candidates": int(n_candidates),
            "n_candidates_kept": int(n_candidates_kept),
            "n_failed_triangulation": int(n_failed_triangulation),
            "n_failed_low_parallax": int(n_failed_low_parallax),
            "n_filtered_by_support": int(n_filtered_by_support),
            "n_frames_used": int(len(sampled_ts)),
            "support_frames_mean": float(np.mean(support_frames)) if support_frames.size else 0.0,
            "support_frames_median": float(np.median(support_frames)) if support_frames.size else 0.0,
        }

    return {
        "points_init": np.asarray(points_init, dtype=np.float64),
        "obs_view_idx": np.asarray(obs_view_idx, dtype=np.int64),
        "obs_point_idx": np.asarray(obs_point_idx, dtype=np.int64),
        "obs_uv": np.asarray(obs_uv, dtype=np.float64),
        "point_meta": point_meta,
        "point_confidence": np.asarray(point_confidence, dtype=np.float64),
        "n_candidates": int(n_candidates),
        "n_candidates_kept": int(n_candidates_kept),
        "n_failed_triangulation": int(n_failed_triangulation),
        "n_failed_low_parallax": int(n_failed_low_parallax),
        "n_filtered_by_support": int(n_filtered_by_support),
        "n_frames_used": int(len(sampled_ts)),
        "support_frames_mean": float(np.mean(support_frames)) if support_frames.size else 0.0,
        "support_frames_median": float(np.median(support_frames)) if support_frames.size else 0.0,
    }


def _project_observations_torch(
    *,
    X_points,
    obs_point_idx,
    obs_view_idx,
    rvec_all,
    t_all,
    fx,
    fy,
    cx,
    cy,
    dist,
):
    import torch

    R_all = rodrigues_torch(rvec_all)  # (V,3,3)

    X = X_points[obs_point_idx]  # (M,3)
    R = R_all[obs_view_idx]  # (M,3,3)
    t = t_all[obs_view_idx]  # (M,3)

    Xc = torch.einsum("mij,mj->mi", R, X) + t
    z = Xc[:, 2]
    z_safe = torch.clamp(z, min=1e-6)

    x = Xc[:, 0] / z_safe
    y = Xc[:, 1] / z_safe

    d = dist[obs_view_idx]
    k1, k2, p1, p2, k3 = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_rad = x * radial
    y_rad = y * radial

    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    xd = x_rad + x_tan
    yd = y_rad + y_tan

    u = fx[obs_view_idx] * xd + cx[obs_view_idx]
    v = fy[obs_view_idx] * yd + cy[obs_view_idx]

    uv = torch.stack([u, v], dim=-1)
    return uv, z


def _mean_reprojection_error_torch(
    *,
    X_points,
    obs_uv,
    obs_point_idx,
    obs_view_idx,
    rvec_all,
    t_all,
    fx,
    fy,
    cx,
    cy,
    dist,
) -> float:
    import torch

    if int(obs_uv.shape[0]) == 0:
        return float("nan")

    with torch.no_grad():
        uv_pred, _ = _project_observations_torch(
            X_points=X_points,
            obs_point_idx=obs_point_idx,
            obs_view_idx=obs_view_idx,
            rvec_all=rvec_all,
            t_all=t_all,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist=dist,
        )
        err = torch.linalg.norm(uv_pred - obs_uv, dim=-1).mean()
        return float(err.detach().cpu().numpy())


def _per_view_mean_reprojection_error_torch(
    *,
    X_points,
    obs_uv,
    obs_point_idx,
    obs_view_idx,
    rvec_all,
    t_all,
    fx,
    fy,
    cx,
    cy,
    dist,
    n_views: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    means = np.full((int(n_views),), np.nan, dtype=np.float64)
    counts = np.zeros((int(n_views),), dtype=np.int64)
    if int(obs_uv.shape[0]) == 0:
        return means, counts

    with torch.no_grad():
        uv_pred, _ = _project_observations_torch(
            X_points=X_points,
            obs_point_idx=obs_point_idx,
            obs_view_idx=obs_view_idx,
            rvec_all=rvec_all,
            t_all=t_all,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist=dist,
        )
        err = torch.linalg.norm(uv_pred - obs_uv, dim=-1)
        err_np = err.detach().cpu().numpy().astype(np.float64)
        view_np = obs_view_idx.detach().cpu().numpy().astype(np.int64)

    for vidx in np.unique(view_np):
        if int(vidx) < 0 or int(vidx) >= int(n_views):
            continue
        mask = view_np == int(vidx)
        cnt = int(np.sum(mask))
        if cnt <= 0:
            continue
        counts[int(vidx)] = cnt
        means[int(vidx)] = float(np.mean(err_np[mask]))

    return means, counts


def _evaluate_per_view_holdout_gate(
    *,
    err_before_per_view: np.ndarray,
    err_after_per_view: np.ndarray,
    obs_counts_per_view: np.ndarray,
    min_obs: int,
    max_error_increase_abs: float,
    max_error_increase_rel: float,
) -> Dict[str, Any]:
    n = int(err_before_per_view.shape[0])
    considered: List[int] = []
    degraded: List[int] = []

    min_obs = max(1, int(min_obs))
    abs_tol = float(max(0.0, max_error_increase_abs))
    rel_tol = float(max(0.0, max_error_increase_rel))

    for i in range(n):
        cnt = int(obs_counts_per_view[i])
        before = float(err_before_per_view[i])
        after = float(err_after_per_view[i])
        if cnt < min_obs:
            continue
        if not np.isfinite(before):
            continue

        considered.append(i)
        allowed = float(before) * (1.0 + rel_tol) + abs_tol
        if (not np.isfinite(after)) or float(after) > float(allowed):
            degraded.append(i)

    return {
        "ok": bool(len(degraded) == 0),
        "considered": considered,
        "degraded": degraded,
    }


def _split_train_holdout_indices(
    *,
    n_obs: int,
    holdout_ratio: float,
    min_holdout_obs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_obs = int(n_obs)
    if n_obs <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    if float(holdout_ratio) <= 0.0:
        return np.arange(n_obs, dtype=np.int64), np.zeros((0,), dtype=np.int64)

    # Keep enough training observations for stable optimization.
    min_train = 64
    max_holdout = max(0, n_obs - min_train)
    if max_holdout <= 0:
        return np.arange(n_obs, dtype=np.int64), np.zeros((0,), dtype=np.int64)

    holdout_count = int(round(float(holdout_ratio) * n_obs))
    holdout_count = max(holdout_count, int(min_holdout_obs))
    holdout_count = min(holdout_count, max_holdout)
    holdout_count = min(holdout_count, n_obs // 3)
    if holdout_count <= 0:
        return np.arange(n_obs, dtype=np.int64), np.zeros((0,), dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    holdout_idx = np.sort(rng.choice(np.arange(n_obs, dtype=np.int64), size=holdout_count, replace=False))
    train_mask = np.ones((n_obs,), dtype=bool)
    train_mask[holdout_idx] = False
    train_idx = np.flatnonzero(train_mask).astype(np.int64)
    return train_idx, holdout_idx


def _compute_reprojection_residuals_numpy(
    *,
    points_init: np.ndarray,
    obs_uv: np.ndarray,
    obs_point_idx: np.ndarray,
    obs_view_idx: np.ndarray,
    rvec_all: np.ndarray,
    t_all: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-observation reprojection residual and depth at initialization."""
    if int(obs_uv.shape[0]) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    R_all = np.zeros((rvec_all.shape[0], 3, 3), dtype=np.float64)
    for i in range(rvec_all.shape[0]):
        R_i, _ = cv2.Rodrigues(np.asarray(rvec_all[i], dtype=np.float64).reshape(3, 1))
        R_all[i] = np.asarray(R_i, dtype=np.float64)

    X = points_init[obs_point_idx]  # (M,3)
    R = R_all[obs_view_idx]  # (M,3,3)
    t = t_all[obs_view_idx]  # (M,3)

    Xc = np.einsum("mij,mj->mi", R, X) + t
    z = Xc[:, 2]
    z_safe = np.clip(z, 1e-6, None)

    x = Xc[:, 0] / z_safe
    y = Xc[:, 1] / z_safe

    d = dist[obs_view_idx]
    k1, k2, p1, p2, k3 = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_rad = x * radial
    y_rad = y * radial

    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    xd = x_rad + x_tan
    yd = y_rad + y_tan

    u = fx[obs_view_idx] * xd + cx[obs_view_idx]
    v = fy[obs_view_idx] * yd + cy[obs_view_idx]

    uv_hat = np.stack([u, v], axis=-1)
    residual = np.linalg.norm(uv_hat - obs_uv, axis=-1)
    return residual.astype(np.float64), z.astype(np.float64)


def _filter_observations_by_init_reprojection(
    *,
    points_init: np.ndarray,
    point_confidence: np.ndarray,
    point_meta: Optional[Sequence[Tuple[int, int]]] = None,
    obs_uv: np.ndarray,
    obs_point_idx: np.ndarray,
    obs_view_idx: np.ndarray,
    rvec_all: np.ndarray,
    t_all: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    dist: np.ndarray,
    min_views_per_point: int,
    max_reproj_error_px: float,
    keep_quantile: float,
) -> Dict[str, Any]:
    """Filter high-residual initialization observations and remap valid points."""
    n_obs_before = int(obs_uv.shape[0])
    n_points_before = int(points_init.shape[0])
    point_confidence = np.asarray(point_confidence, dtype=np.float64).reshape(-1)
    if point_confidence.shape[0] != n_points_before:
        point_confidence = np.ones((n_points_before,), dtype=np.float64)
    point_meta_list: List[Tuple[int, int]] = []
    if point_meta is not None:
        point_meta_list = [tuple(map(int, m)) for m in point_meta]
        if len(point_meta_list) != n_points_before:
            point_meta_list = []

    if n_obs_before == 0 or n_points_before == 0:
        return {
            "points_init": points_init,
            "point_confidence": np.asarray(point_confidence, dtype=np.float64),
            "point_meta": list(point_meta_list),
            "obs_uv": obs_uv,
            "obs_point_idx": obs_point_idx,
            "obs_view_idx": obs_view_idx,
            "stats": {
                "n_observations_before_init_filter": int(n_obs_before),
                "n_observations_after_init_filter": int(n_obs_before),
                "n_points_before_init_filter": int(n_points_before),
                "n_points_after_init_filter": int(n_points_before),
                "init_obs_filter_keep_ratio": 1.0,
                "init_obs_filter_quantile_threshold_px": float("nan"),
                "init_obs_filter_applied_threshold_px": float("nan"),
                "init_obs_filter_residual_median_px": float("nan"),
                "init_obs_filter_residual_p90_px": float("nan"),
            },
        }

    residual, depth = _compute_reprojection_residuals_numpy(
        points_init=points_init,
        obs_uv=obs_uv,
        obs_point_idx=obs_point_idx,
        obs_view_idx=obs_view_idx,
        rvec_all=rvec_all,
        t_all=t_all,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist=dist,
    )

    finite = np.isfinite(residual) & np.isfinite(depth)
    keep = finite.copy()

    quantile_threshold = float("inf")
    q = float(keep_quantile)
    if 0.0 < q < 1.0 and bool(np.any(finite)):
        quantile_threshold = float(np.quantile(residual[finite], q))
        keep &= residual <= quantile_threshold

    max_err = float(max_reproj_error_px)
    if max_err > 0.0:
        keep &= residual <= max_err

    keep &= depth > 1e-6

    min_views = max(2, int(min_views_per_point))
    point_counts = np.bincount(obs_point_idx[keep], minlength=n_points_before)
    point_keep = point_counts >= min_views
    keep &= point_keep[obs_point_idx]

    keep_obs_idx = np.flatnonzero(keep)
    keep_point_idx = np.flatnonzero(point_keep)

    remap = -np.ones((n_points_before,), dtype=np.int64)
    if keep_point_idx.size > 0:
        remap[keep_point_idx] = np.arange(keep_point_idx.size, dtype=np.int64)

    obs_point_idx_new = remap[obs_point_idx[keep_obs_idx]]
    valid_obs = obs_point_idx_new >= 0
    keep_obs_idx = keep_obs_idx[valid_obs]
    obs_point_idx_new = obs_point_idx_new[valid_obs]

    points_new = np.asarray(points_init[keep_point_idx], dtype=np.float64) if keep_point_idx.size > 0 else np.zeros((0, 3), dtype=np.float64)
    conf_new = np.asarray(point_confidence[keep_point_idx], dtype=np.float64) if keep_point_idx.size > 0 else np.zeros((0,), dtype=np.float64)
    if point_meta_list and keep_point_idx.size > 0:
        point_meta_new = [point_meta_list[int(i)] for i in keep_point_idx.tolist()]
    elif point_meta_list:
        point_meta_new = []
    else:
        point_meta_new = []
    obs_uv_new = np.asarray(obs_uv[keep_obs_idx], dtype=np.float64) if keep_obs_idx.size > 0 else np.zeros((0, 2), dtype=np.float64)
    obs_view_new = np.asarray(obs_view_idx[keep_obs_idx], dtype=np.int64) if keep_obs_idx.size > 0 else np.zeros((0,), dtype=np.int64)

    n_obs_after = int(obs_uv_new.shape[0])
    keep_ratio = float(n_obs_after / max(1, n_obs_before))

    finite_residual = residual[finite]
    if finite_residual.size > 0:
        residual_median = float(np.median(finite_residual))
        residual_p90 = float(np.quantile(finite_residual, 0.9))
    else:
        residual_median = float("nan")
        residual_p90 = float("nan")

    applied_threshold = float("inf")
    if np.isfinite(quantile_threshold):
        applied_threshold = min(applied_threshold, quantile_threshold)
    if max_err > 0.0:
        applied_threshold = min(applied_threshold, max_err)

    return {
        "points_init": points_new,
        "point_confidence": conf_new,
        "point_meta": point_meta_new,
        "obs_uv": obs_uv_new,
        "obs_point_idx": obs_point_idx_new.astype(np.int64),
        "obs_view_idx": obs_view_new,
        "stats": {
            "n_observations_before_init_filter": int(n_obs_before),
            "n_observations_after_init_filter": int(n_obs_after),
            "n_points_before_init_filter": int(n_points_before),
            "n_points_after_init_filter": int(points_new.shape[0]),
            "init_obs_filter_keep_ratio": float(keep_ratio),
            "init_obs_filter_quantile_threshold_px": float(quantile_threshold),
            "init_obs_filter_applied_threshold_px": float(applied_threshold),
            "init_obs_filter_residual_median_px": float(residual_median),
            "init_obs_filter_residual_p90_px": float(residual_p90),
        },
    }


def _is_finite_scalar(x: float) -> bool:
    return bool(np.isfinite(float(x)))


def _rotation_angle_deg(R_delta: np.ndarray) -> float:
    R_delta = np.asarray(R_delta, dtype=np.float64).reshape(3, 3)
    c = float((np.trace(R_delta) - 1.0) * 0.5)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _relative_pose_world_to_cam(
    *,
    R_i: np.ndarray,
    t_i: np.ndarray,
    R_j: np.ndarray,
    t_j: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return relative pose (i->j) from two world->cam extrinsics."""
    R_i = np.asarray(R_i, dtype=np.float64).reshape(3, 3)
    t_i = np.asarray(t_i, dtype=np.float64).reshape(3)
    R_j = np.asarray(R_j, dtype=np.float64).reshape(3, 3)
    t_j = np.asarray(t_j, dtype=np.float64).reshape(3)

    R_ij = R_j @ R_i.T
    t_ij = t_j - R_ij @ t_i
    return np.asarray(R_ij, dtype=np.float64), np.asarray(t_ij, dtype=np.float64)


def _estimate_pair_relative_rotation_ransac(
    *,
    uv_i: np.ndarray,
    uv_j: np.ndarray,
    cam_i: CameraParams,
    cam_j: CameraParams,
    threshold: float,
    prob: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Estimate relative rotation (i->j) from 2D-2D correspondences via essential RANSAC."""
    uv_i = np.asarray(uv_i, dtype=np.float64).reshape(-1, 2)
    uv_j = np.asarray(uv_j, dtype=np.float64).reshape(-1, 2)
    if uv_i.shape != uv_j.shape or uv_i.shape[0] < 8:
        return None, {"ok": False, "reason": "insufficient_points", "n_points_used": int(min(uv_i.shape[0], uv_j.shape[0]))}

    D_i = np.asarray(cam_i.D, dtype=np.float64).reshape(-1)
    D_j = np.asarray(cam_j.D, dtype=np.float64).reshape(-1)
    if D_i.size < 5:
        d2 = np.zeros((5,), dtype=np.float64)
        d2[: D_i.size] = D_i
        D_i = d2
    if D_j.size < 5:
        d2 = np.zeros((5,), dtype=np.float64)
        d2[: D_j.size] = D_j
        D_j = d2

    p1 = cv2.undistortPoints(uv_i.reshape(-1, 1, 2), np.asarray(cam_i.K, dtype=np.float64), D_i[:5]).reshape(-1, 2)
    p2 = cv2.undistortPoints(uv_j.reshape(-1, 1, 2), np.asarray(cam_j.K, dtype=np.float64), D_j[:5]).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        p1,
        p2,
        focal=1.0,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=float(prob),
        threshold=float(threshold),
    )
    if E is None:
        return None, {"ok": False, "reason": "findEssentialMat_failed", "n_points_used": int(p1.shape[0])}

    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        candidates = [E]
    elif E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        candidates = [E[i * 3 : (i + 1) * 3, :] for i in range(int(E.shape[0] // 3))]
    else:
        candidates = []

    if not candidates:
        return None, {"ok": False, "reason": "invalid_essential_shape", "n_points_used": int(p1.shape[0])}

    best_R = None
    best_inliers = -1
    for Ei in candidates:
        try:
            n_inliers, R_ij, _, _ = cv2.recoverPose(Ei, p1, p2, mask=mask)
        except Exception:
            continue
        if int(n_inliers) > int(best_inliers):
            best_inliers = int(n_inliers)
            best_R = np.asarray(R_ij, dtype=np.float64)

    if best_R is None:
        return None, {"ok": False, "reason": "recoverPose_failed", "n_points_used": int(p1.shape[0])}

    inlier_ratio = float(best_inliers / max(int(p1.shape[0]), 1))
    return best_R, {
        "ok": True,
        "reason": "ok",
        "n_inliers": int(best_inliers),
        "n_points_used": int(p1.shape[0]),
        "inlier_ratio": float(inlier_ratio),
    }


def _edge_rotation_for_order(
    *,
    edge_rotations: Dict[Tuple[int, int], np.ndarray],
    a: int,
    b: int,
) -> Optional[np.ndarray]:
    if int(a) == int(b):
        return np.eye(3, dtype=np.float64)
    key = (min(int(a), int(b)), max(int(a), int(b)))
    R_ab = edge_rotations.get(key)
    if R_ab is None:
        return None
    if int(a) < int(b):
        return np.asarray(R_ab, dtype=np.float64)
    return np.asarray(R_ab, dtype=np.float64).T


def _compute_camera_graph_closure_view_weights(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    cameras_init: Dict[str, CameraParams],
    cfg: GlobalTrackBAConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Compute per-view weights from track-induced camera-graph cycle consistency."""
    view_ids = [str(v) for v in view_ids]
    V = int(len(view_ids))
    view_weights = np.ones((V,), dtype=np.float64)
    edge_weight_matrix = np.full((V, V), np.nan, dtype=np.float64)
    np.fill_diagonal(edge_weight_matrix, 1.0)
    edge_constraints: Dict[str, np.ndarray] = {
        "edge_i_idx": np.zeros((0,), dtype=np.int64),
        "edge_j_idx": np.zeros((0,), dtype=np.int64),
        "edge_rotations": np.zeros((0, 3, 3), dtype=np.float64),
        "edge_weights": np.zeros((0,), dtype=np.float64),
    }
    base_stats: Dict[str, Any] = {
        "graph_closure_enabled": bool(getattr(cfg, "graph_closure_enable", False)),
        "graph_closure_ok": False,
        "graph_closure_reason": "not_run",
        "graph_closure_num_pairs_total": int(max(V * (V - 1) // 2, 0)),
        "graph_closure_num_pairs_shared_enough": 0,
        "graph_closure_num_edges_used": 0,
        "graph_closure_num_triangles": 0,
        "graph_closure_cycle_error_mean_deg": float("nan"),
        "graph_closure_cycle_error_p90_deg": float("nan"),
        "graph_closure_edge_weight_mean": 1.0,
        "graph_closure_edge_weight_median": 1.0,
        "graph_closure_edge_weight_min": 1.0,
        "graph_closure_edge_weight_max": 1.0,
        "graph_closure_view_weight_mean": 1.0,
        "graph_closure_view_weight_median": 1.0,
        "graph_closure_view_weight_min": 1.0,
        "graph_closure_view_weight_max": 1.0,
        "graph_closure_view_weights": {vid: 1.0 for vid in view_ids},
        "graph_closure_worst_edges": [],
    }

    if V < 3:
        base_stats["graph_closure_reason"] = "insufficient_views"
        return view_weights, edge_weight_matrix, edge_constraints, base_stats

    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    if tracks.ndim != 4 or tracks.shape[-1] != 2 or visibility.shape != tracks.shape[:3]:
        base_stats["graph_closure_reason"] = "invalid_track_shape"
        return view_weights, edge_weight_matrix, edge_constraints, base_stats

    stride = max(int(getattr(cfg, "frame_stride", 1)), 1)
    sampled_ts = np.arange(0, tracks.shape[1], stride, dtype=np.int64)
    if sampled_ts.size == 0:
        base_stats["graph_closure_reason"] = "no_sampled_frames"
        return view_weights, edge_weight_matrix, edge_constraints, base_stats

    min_shared_obs = max(8, int(getattr(cfg, "graph_closure_min_shared_obs", 64)))
    max_pair_obs = max(0, int(getattr(cfg, "graph_closure_max_pair_obs", 2000)))
    min_inliers = max(8, int(getattr(cfg, "graph_closure_min_inliers", 24)))
    min_inlier_ratio = float(np.clip(float(getattr(cfg, "graph_closure_min_inlier_ratio", 0.25)), 0.0, 1.0))
    ransac_threshold = float(max(1e-6, float(getattr(cfg, "graph_closure_ransac_threshold", 1e-3))))
    ransac_prob = float(np.clip(float(getattr(cfg, "graph_closure_ransac_prob", 0.999)), 1e-3, 0.999999))
    cycle_tau = float(max(1e-6, float(getattr(cfg, "graph_closure_cycle_tau_deg", 1.0))))
    min_weight = float(np.clip(float(getattr(cfg, "graph_closure_min_weight", 0.05)), 0.0, 1.0))
    weight_power = float(max(1e-6, float(getattr(cfg, "graph_closure_weight_power", 1.0))))
    no_cycle_weight = float(np.clip(float(getattr(cfg, "graph_closure_no_cycle_weight", 0.5)), 0.0, 1.0))

    rng = np.random.default_rng(int(cfg.seed) + 173)
    edge_rotations: Dict[Tuple[int, int], np.ndarray] = {}
    edge_entries: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for i in range(V):
        vid_i = view_ids[i]
        cam_i = cameras_init.get(vid_i)
        if cam_i is None:
            continue
        tr_i = tracks[i, sampled_ts, :, :]
        vis_i = visibility[i, sampled_ts, :]

        for j in range(i + 1, V):
            vid_j = view_ids[j]
            cam_j = cameras_init.get(vid_j)
            if cam_j is None:
                continue

            tr_j = tracks[j, sampled_ts, :, :]
            vis_j = visibility[j, sampled_ts, :]

            valid = (
                vis_i
                & vis_j
                & np.isfinite(tr_i[..., 0])
                & np.isfinite(tr_i[..., 1])
                & np.isfinite(tr_j[..., 0])
                & np.isfinite(tr_j[..., 1])
            )
            n_shared = int(np.sum(valid))
            if n_shared < min_shared_obs:
                continue

            base_stats["graph_closure_num_pairs_shared_enough"] = int(base_stats["graph_closure_num_pairs_shared_enough"]) + 1

            uv_i = tr_i[valid].reshape(-1, 2)
            uv_j = tr_j[valid].reshape(-1, 2)
            if uv_i.shape[0] < 8:
                continue

            if max_pair_obs > 0 and uv_i.shape[0] > max_pair_obs:
                keep = rng.choice(np.arange(uv_i.shape[0], dtype=np.int64), size=max_pair_obs, replace=False)
                keep = np.sort(keep)
                uv_i = uv_i[keep]
                uv_j = uv_j[keep]

            R_rel_init, _ = _relative_pose_world_to_cam(
                R_i=cam_i.R,
                t_i=cam_i.T,
                R_j=cam_j.R,
                t_j=cam_j.T,
            )

            R_rel_meas, est_stats = _estimate_pair_relative_rotation_ransac(
                uv_i=uv_i,
                uv_j=uv_j,
                cam_i=cam_i,
                cam_j=cam_j,
                threshold=ransac_threshold,
                prob=ransac_prob,
            )
            if R_rel_meas is None:
                continue

            n_inliers = int(est_stats.get("n_inliers", 0))
            inlier_ratio = float(est_stats.get("inlier_ratio", 0.0))
            if n_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
                continue

            key = (i, j)
            edge_rotations[key] = np.asarray(R_rel_meas, dtype=np.float64)
            edge_entries[key] = {
                "view_i": str(vid_i),
                "view_j": str(vid_j),
                "n_shared_obs": int(n_shared),
                "n_pair_obs_used": int(uv_i.shape[0]),
                "n_inliers": int(n_inliers),
                "inlier_ratio": float(inlier_ratio),
                "delta_vs_init_deg": float(_rotation_angle_deg(np.asarray(R_rel_meas, dtype=np.float64) @ R_rel_init.T)),
            }

    if not edge_entries:
        base_stats["graph_closure_reason"] = "no_valid_edges"
        return view_weights, edge_weight_matrix, edge_constraints, base_stats

    edge_cycle_errors: Dict[Tuple[int, int], List[float]] = {k: [] for k in edge_entries.keys()}
    cycle_errors_all: List[float] = []
    triangle_count = 0
    for i in range(V):
        for j in range(i + 1, V):
            if (i, j) not in edge_entries:
                continue
            for k in range(j + 1, V):
                if (j, k) not in edge_entries or (i, k) not in edge_entries:
                    continue

                R_ij = _edge_rotation_for_order(edge_rotations=edge_rotations, a=i, b=j)
                R_jk = _edge_rotation_for_order(edge_rotations=edge_rotations, a=j, b=k)
                R_ki = _edge_rotation_for_order(edge_rotations=edge_rotations, a=k, b=i)
                if R_ij is None or R_jk is None or R_ki is None:
                    continue

                cyc = np.asarray(R_ij, dtype=np.float64) @ np.asarray(R_jk, dtype=np.float64) @ np.asarray(R_ki, dtype=np.float64)
                cyc_deg = float(_rotation_angle_deg(cyc))
                edge_cycle_errors[(i, j)].append(cyc_deg)
                edge_cycle_errors[(j, k)].append(cyc_deg)
                edge_cycle_errors[(i, k)].append(cyc_deg)
                cycle_errors_all.append(cyc_deg)
                triangle_count += 1

    edge_weights: List[float] = []
    for key, row in edge_entries.items():
        cyc_vals = edge_cycle_errors.get(key, [])
        if cyc_vals:
            cyc_mean = float(np.mean(cyc_vals))
            cyc_term = float(np.exp(-((cyc_mean / cycle_tau) ** 2)))
        else:
            cyc_mean = 0.0
            cyc_term = float(no_cycle_weight)

        inlier_ratio = float(np.clip(float(row.get("inlier_ratio", 0.0)), 0.0, 1.0))
        quality_term = float(
            np.clip((inlier_ratio - min_inlier_ratio) / max(1.0 - min_inlier_ratio, 1e-6), 0.0, 1.0)
        )

        n_inliers = float(max(0.0, float(row.get("n_inliers", 0.0))))
        n_pair = float(max(1.0, float(row.get("n_pair_obs_used", 1.0))))
        support_term = float(np.clip(np.log1p(n_inliers) / max(np.log1p(n_pair), 1e-6), 0.0, 1.0))

        quality_mix = float(0.5 * quality_term + 0.5 * support_term)
        alpha_raw = float(cyc_term * (0.5 + 0.5 * quality_mix))
        alpha_raw = float(np.clip(alpha_raw, 0.0, 1.0))
        alpha = float(np.clip(alpha_raw**weight_power, min_weight, 1.0))

        row["cycle_error_mean_deg"] = float(cyc_mean)
        row["cycle_support"] = int(len(cyc_vals))
        row["cycle_term"] = float(cyc_term)
        row["quality_term"] = float(quality_term)
        row["support_term"] = float(support_term)
        row["edge_weight"] = float(alpha)
        edge_weights.append(alpha)

    acc = np.zeros((V,), dtype=np.float64)
    wsum = np.zeros((V,), dtype=np.float64)
    for (i, j), row in edge_entries.items():
        w = float(max(1.0, float(row.get("n_inliers", 1))))
        alpha = float(row.get("edge_weight", 1.0))
        acc[i] += w * alpha
        wsum[i] += w
        acc[j] += w * alpha
        wsum[j] += w

    for i in range(V):
        if wsum[i] > 0.0:
            view_weights[i] = float(acc[i] / wsum[i])
        else:
            view_weights[i] = 1.0
    view_weights = np.clip(view_weights, min_weight, 1.0)

    for (i, j), row in edge_entries.items():
        w = float(np.clip(float(row.get("edge_weight", 1.0)), min_weight, 1.0))
        edge_weight_matrix[int(i), int(j)] = w
        edge_weight_matrix[int(j), int(i)] = w

    edge_keys_sorted = sorted(edge_entries.keys())
    if edge_keys_sorted:
        edge_constraints = {
            "edge_i_idx": np.asarray([int(k[0]) for k in edge_keys_sorted], dtype=np.int64),
            "edge_j_idx": np.asarray([int(k[1]) for k in edge_keys_sorted], dtype=np.int64),
            "edge_rotations": np.asarray([edge_rotations[k] for k in edge_keys_sorted], dtype=np.float64),
            "edge_weights": np.asarray([float(edge_entries[k].get("edge_weight", 1.0)) for k in edge_keys_sorted], dtype=np.float64),
        }

    worst_edges = sorted(edge_entries.values(), key=lambda x: float(x.get("edge_weight", 1.0)))
    worst_edges = worst_edges[: min(len(worst_edges), 8)]

    base_stats.update(
        {
            "graph_closure_ok": True,
            "graph_closure_reason": "ok",
            "graph_closure_num_edges_used": int(len(edge_entries)),
            "graph_closure_num_triangles": int(triangle_count),
            "graph_closure_cycle_error_mean_deg": float(np.mean(cycle_errors_all)) if cycle_errors_all else float("nan"),
            "graph_closure_cycle_error_p90_deg": float(np.percentile(np.asarray(cycle_errors_all, dtype=np.float64), 90.0)) if cycle_errors_all else float("nan"),
            "graph_closure_edge_weight_mean": float(np.mean(edge_weights)) if edge_weights else 1.0,
            "graph_closure_edge_weight_median": float(np.median(edge_weights)) if edge_weights else 1.0,
            "graph_closure_edge_weight_min": float(np.min(edge_weights)) if edge_weights else 1.0,
            "graph_closure_edge_weight_max": float(np.max(edge_weights)) if edge_weights else 1.0,
            "graph_closure_view_weight_mean": float(np.mean(view_weights)) if view_weights.size else 1.0,
            "graph_closure_view_weight_median": float(np.median(view_weights)) if view_weights.size else 1.0,
            "graph_closure_view_weight_min": float(np.min(view_weights)) if view_weights.size else 1.0,
            "graph_closure_view_weight_max": float(np.max(view_weights)) if view_weights.size else 1.0,
            "graph_closure_view_weights": {vid: float(view_weights[i]) for i, vid in enumerate(view_ids)},
            "graph_closure_worst_edges": worst_edges,
        }
    )
    return view_weights, edge_weight_matrix, edge_constraints, base_stats


def _compute_observation_graph_weights(
    *,
    obs_view_idx: np.ndarray,
    obs_point_idx: np.ndarray,
    point_meta: Sequence[Tuple[int, int]],
    visibility: np.ndarray,
    edge_weight_matrix: np.ndarray,
    view_weights: np.ndarray,
    min_weight: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build per-observation graph weights from edge-level closure weights."""
    obs_view_idx = np.asarray(obs_view_idx, dtype=np.int64)
    obs_point_idx = np.asarray(obs_point_idx, dtype=np.int64)
    n_obs = int(obs_view_idx.shape[0])
    if n_obs <= 0:
        return np.ones((0,), dtype=np.float64), {
            "graph_closure_obs_weight_mode": "edge_mean_per_observation",
            "graph_closure_obs_weight_num_with_edge": 0,
            "graph_closure_obs_weight_num_fallback_view": 0,
            "graph_closure_obs_weight_mean": 1.0,
            "graph_closure_obs_weight_median": 1.0,
            "graph_closure_obs_weight_min": 1.0,
            "graph_closure_obs_weight_max": 1.0,
        }

    visibility = np.asarray(visibility, dtype=bool)
    V = int(visibility.shape[0]) if visibility.ndim == 3 else int(view_weights.shape[0])
    T = int(visibility.shape[1]) if visibility.ndim == 3 else 0
    N = int(visibility.shape[2]) if visibility.ndim == 3 else 0

    min_weight = float(np.clip(float(min_weight), 0.0, 1.0))
    weights = np.ones((n_obs,), dtype=np.float64)

    vis_cache: Dict[int, Optional[np.ndarray]] = {}
    num_with_edge = 0
    num_fallback = 0

    for m in range(n_obs):
        v_i = int(obs_view_idx[m])
        p_i = int(obs_point_idx[m])
        base_w = float(view_weights[v_i]) if 0 <= v_i < int(view_weights.shape[0]) else 1.0

        if p_i not in vis_cache:
            if p_i < 0 or p_i >= len(point_meta):
                vis_cache[p_i] = None
            else:
                t_idx, track_id = point_meta[p_i]
                t_idx = int(t_idx)
                track_id = int(track_id)
                if (
                    visibility.ndim == 3
                    and 0 <= t_idx < T
                    and 0 <= track_id < N
                ):
                    vis_cache[p_i] = np.flatnonzero(visibility[:, t_idx, track_id]).astype(np.int64)
                else:
                    vis_cache[p_i] = None

        vis_views = vis_cache[p_i]
        if vis_views is None or int(vis_views.size) <= 1:
            num_fallback += 1
            weights[m] = base_w
            continue

        edge_vals: List[float] = []
        for v_j in vis_views.tolist():
            v_j = int(v_j)
            if v_j == v_i or v_j < 0 or v_j >= V:
                continue
            w_ij = float(edge_weight_matrix[v_i, v_j]) if (0 <= v_i < edge_weight_matrix.shape[0] and 0 <= v_j < edge_weight_matrix.shape[1]) else float("nan")
            if np.isfinite(w_ij):
                edge_vals.append(float(w_ij))

        if edge_vals:
            num_with_edge += 1
            weights[m] = float(np.mean(np.asarray(edge_vals, dtype=np.float64)))
        else:
            num_fallback += 1
            weights[m] = base_w

    weights = np.clip(weights, min_weight, 1.0)
    stats = {
        "graph_closure_obs_weight_mode": "edge_mean_per_observation",
        "graph_closure_obs_weight_num_with_edge": int(num_with_edge),
        "graph_closure_obs_weight_num_fallback_view": int(num_fallback),
        "graph_closure_obs_weight_mean": float(np.mean(weights)) if weights.size else 1.0,
        "graph_closure_obs_weight_median": float(np.median(weights)) if weights.size else 1.0,
        "graph_closure_obs_weight_min": float(np.min(weights)) if weights.size else 1.0,
        "graph_closure_obs_weight_max": float(np.max(weights)) if weights.size else 1.0,
    }
    return weights, stats


def refine_cameras_global_ba(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    cameras_init: Dict[str, CameraParams],
    ref_view: str,
    cfg: GlobalTrackBAConfig,
) -> Tuple[Dict[str, CameraParams], Dict[str, Any]]:
    """Refine camera extrinsics with track-augmented global BA (points are per-frame variables)."""
    if not _torch_available():
        raise RuntimeError("global_ba requires PyTorch")

    import torch

    view_ids = [str(v) for v in view_ids]
    if len(view_ids) < 2:
        raise ValueError("global_ba requires at least 2 views")

    if ref_view not in view_ids:
        ref_view = view_ids[0]

    for vid in view_ids:
        if vid not in cameras_init:
            raise ValueError(f"Missing camera init for view {vid}")

    obs_data = _gather_triangulated_observations(
        tracks=tracks,
        visibility=visibility,
        view_ids=view_ids,
        cameras=cameras_init,
        cfg=cfg,
    )

    fallback_reproj_th = float(max(50.0, float(cfg.triangulation_reproj_threshold) * 8.0))
    used_reproj_th = float(cfg.triangulation_reproj_threshold)
    if float(obs_data["points_init"].shape[0]) < 16 and fallback_reproj_th > float(cfg.triangulation_reproj_threshold):
        relaxed_cfg = replace(cfg, triangulation_reproj_threshold=fallback_reproj_th)
        obs_data_relaxed = _gather_triangulated_observations(
            tracks=tracks,
            visibility=visibility,
            view_ids=view_ids,
            cameras=cameras_init,
            cfg=relaxed_cfg,
        )
        if int(obs_data_relaxed["points_init"].shape[0]) > int(obs_data["points_init"].shape[0]):
            obs_data = obs_data_relaxed
            used_reproj_th = fallback_reproj_th

    points_init = np.asarray(obs_data["points_init"], dtype=np.float64)
    point_confidence = np.asarray(obs_data.get("point_confidence", np.ones((points_init.shape[0],), dtype=np.float64)), dtype=np.float64)
    if point_confidence.shape[0] != points_init.shape[0]:
        point_confidence = np.ones((points_init.shape[0],), dtype=np.float64)
    point_meta_raw = obs_data.get("point_meta", [])
    point_meta: List[Tuple[int, int]] = []
    if isinstance(point_meta_raw, list):
        point_meta = [tuple(map(int, m)) for m in point_meta_raw if isinstance(m, (list, tuple)) and len(m) >= 2]
    if len(point_meta) != int(points_init.shape[0]):
        point_meta = [(-1, -1)] * int(points_init.shape[0])
    obs_view_idx = np.asarray(obs_data["obs_view_idx"], dtype=np.int64)
    obs_point_idx = np.asarray(obs_data["obs_point_idx"], dtype=np.int64)
    obs_uv = np.asarray(obs_data["obs_uv"], dtype=np.float64)

    init_filter_stats: Dict[str, Any] = {
        "init_obs_filter_enabled": bool(getattr(cfg, "init_obs_filter_enable", True)),
        "init_obs_filter_applied": False,
        "init_obs_filter_reason": "not_run",
        "n_observations_before_init_filter": int(obs_uv.shape[0]),
        "n_observations_after_init_filter": int(obs_uv.shape[0]),
        "n_points_before_init_filter": int(points_init.shape[0]),
        "n_points_after_init_filter": int(points_init.shape[0]),
        "init_obs_filter_keep_ratio": 1.0,
        "init_obs_filter_quantile_threshold_px": float("nan"),
        "init_obs_filter_applied_threshold_px": float("nan"),
        "init_obs_filter_residual_median_px": float("nan"),
        "init_obs_filter_residual_p90_px": float("nan"),
    }

    device = str(cfg.device)
    dtype = torch.float64

    V = len(view_ids)

    rvec_init = np.zeros((V, 3), dtype=np.float64)
    t_init = np.zeros((V, 3), dtype=np.float64)
    fx = np.zeros((V,), dtype=np.float64)
    fy = np.zeros((V,), dtype=np.float64)
    cx = np.zeros((V,), dtype=np.float64)
    cy = np.zeros((V,), dtype=np.float64)
    dist = np.zeros((V, 5), dtype=np.float64)

    for i, vid in enumerate(view_ids):
        cam = cameras_init[vid]
        rv, _ = cv2.Rodrigues(np.asarray(cam.R, dtype=np.float64))
        rvec_init[i] = rv.reshape(3)
        t_init[i] = np.asarray(cam.T, dtype=np.float64).reshape(3)
        fx[i] = float(cam.K[0, 0])
        fy[i] = float(cam.K[1, 1])
        cx[i] = float(cam.K[0, 2])
        cy[i] = float(cam.K[1, 2])
        d = np.asarray(cam.D, dtype=np.float64).reshape(-1)
        if d.size < 5:
            d2 = np.zeros((5,), dtype=np.float64)
            d2[: d.size] = d
            d = d2
        dist[i] = d[:5]

    if bool(getattr(cfg, "init_obs_filter_enable", True)):
        filtered_obs = _filter_observations_by_init_reprojection(
            points_init=points_init,
            point_confidence=point_confidence,
            point_meta=point_meta,
            obs_uv=obs_uv,
            obs_point_idx=obs_point_idx,
            obs_view_idx=obs_view_idx,
            rvec_all=rvec_init,
            t_all=t_init,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist=dist,
            min_views_per_point=int(cfg.min_views_per_point),
            max_reproj_error_px=float(getattr(cfg, "init_obs_max_reproj_error_px", 20.0)),
            keep_quantile=float(getattr(cfg, "init_obs_keep_quantile", 0.9)),
        )
        init_filter_stats.update(filtered_obs.get("stats", {}))

        keep_ratio = float(filtered_obs.get("stats", {}).get("init_obs_filter_keep_ratio", 0.0))
        min_keep_ratio = float(max(0.0, getattr(cfg, "init_obs_min_keep_ratio", 0.5)))
        if (
            int(filtered_obs["points_init"].shape[0]) >= 16
            and int(filtered_obs["obs_uv"].shape[0]) >= 64
            and keep_ratio >= min_keep_ratio
        ):
            points_init = np.asarray(filtered_obs["points_init"], dtype=np.float64)
            point_confidence = np.asarray(filtered_obs.get("point_confidence", np.ones((points_init.shape[0],), dtype=np.float64)), dtype=np.float64)
            if point_confidence.shape[0] != points_init.shape[0]:
                point_confidence = np.ones((points_init.shape[0],), dtype=np.float64)
            point_meta_f = filtered_obs.get("point_meta", [])
            if isinstance(point_meta_f, list) and len(point_meta_f) == int(points_init.shape[0]):
                point_meta = [tuple(map(int, m)) for m in point_meta_f]
            else:
                point_meta = [(-1, -1)] * int(points_init.shape[0])
            obs_uv = np.asarray(filtered_obs["obs_uv"], dtype=np.float64)
            obs_point_idx = np.asarray(filtered_obs["obs_point_idx"], dtype=np.int64)
            obs_view_idx = np.asarray(filtered_obs["obs_view_idx"], dtype=np.int64)
            init_filter_stats["init_obs_filter_applied"] = True
            init_filter_stats["init_obs_filter_reason"] = "applied"
        else:
            init_filter_stats["init_obs_filter_applied"] = False
            init_filter_stats["init_obs_filter_reason"] = "too_aggressive"
    else:
        init_filter_stats["init_obs_filter_applied"] = False
        init_filter_stats["init_obs_filter_reason"] = "disabled"

    if points_init.shape[0] < 16 or obs_uv.shape[0] < 64:
        stats = {
            "method": "global_ba",
            "ok": False,
            "accepted": False,
            "reason": "insufficient_observations",
            "n_points": int(points_init.shape[0]),
            "n_observations": int(obs_uv.shape[0]),
            "n_candidates": int(obs_data.get("n_candidates", 0)),
            "n_candidates_kept": int(obs_data.get("n_candidates_kept", 0)),
            "n_failed_triangulation": int(obs_data.get("n_failed_triangulation", 0)),
            "n_failed_low_parallax": int(obs_data.get("n_failed_low_parallax", 0)),
            "n_filtered_by_support": int(obs_data.get("n_filtered_by_support", 0)),
            "n_frames_used": int(obs_data.get("n_frames_used", 0)),
            "triangulation_reproj_threshold_used": float(used_reproj_th),
            **init_filter_stats,
        }
        return dict(cameras_init), stats

    ref_idx = view_ids.index(ref_view)
    optimize_mask = np.ones((V,), dtype=bool)
    if bool(cfg.fix_ref_extrinsic):
        optimize_mask[ref_idx] = False
    for fv in cfg.fixed_views:
        if fv in view_ids:
            optimize_mask[view_ids.index(fv)] = False
    optimize_indices = np.flatnonzero(optimize_mask)

    train_idx_np, holdout_idx_np = _split_train_holdout_indices(
        n_obs=int(obs_uv.shape[0]),
        holdout_ratio=float(cfg.accept_holdout_ratio),
        min_holdout_obs=int(cfg.accept_min_holdout_obs),
        seed=int(cfg.seed),
    )
    if train_idx_np.size < 64:
        stats = {
            "method": "global_ba",
            "ok": False,
            "accepted": False,
            "reason": "insufficient_training_observations",
            "n_points": int(points_init.shape[0]),
            "n_observations": int(obs_uv.shape[0]),
            "n_observations_train": int(train_idx_np.size),
            "n_observations_holdout": int(holdout_idx_np.size),
            "n_candidates": int(obs_data.get("n_candidates", 0)),
            "n_candidates_kept": int(obs_data.get("n_candidates_kept", 0)),
            "n_failed_triangulation": int(obs_data.get("n_failed_triangulation", 0)),
            "n_failed_low_parallax": int(obs_data.get("n_failed_low_parallax", 0)),
            "n_filtered_by_support": int(obs_data.get("n_filtered_by_support", 0)),
            "n_frames_used": int(obs_data.get("n_frames_used", 0)),
            "triangulation_reproj_threshold_used": float(used_reproj_th),
            **init_filter_stats,
        }
        return dict(cameras_init), stats

    point_conf_clip = np.clip(np.asarray(point_confidence, dtype=np.float64), 0.0, 1.0)
    if point_conf_clip.shape[0] != points_init.shape[0]:
        point_conf_clip = np.ones((points_init.shape[0],), dtype=np.float64)

    graph_closure_enabled = bool(getattr(cfg, "graph_closure_enable", False))
    if graph_closure_enabled:
        view_graph_weight_np, edge_graph_weight_np, graph_edge_constraints, graph_closure_stats = _compute_camera_graph_closure_view_weights(
            tracks=tracks,
            visibility=visibility,
            view_ids=view_ids,
            cameras_init=cameras_init,
            cfg=cfg,
        )
    else:
        view_graph_weight_np = np.ones((V,), dtype=np.float64)
        edge_graph_weight_np = np.full((V, V), np.nan, dtype=np.float64)
        np.fill_diagonal(edge_graph_weight_np, 1.0)
        graph_edge_constraints = {
            "edge_i_idx": np.zeros((0,), dtype=np.int64),
            "edge_j_idx": np.zeros((0,), dtype=np.int64),
            "edge_rotations": np.zeros((0, 3, 3), dtype=np.float64),
            "edge_weights": np.zeros((0,), dtype=np.float64),
        }
        graph_closure_stats = {
            "graph_closure_enabled": False,
            "graph_closure_ok": False,
            "graph_closure_reason": "disabled",
            "graph_closure_num_pairs_total": int(max(V * (V - 1) // 2, 0)),
            "graph_closure_num_pairs_shared_enough": 0,
            "graph_closure_num_edges_used": 0,
            "graph_closure_num_triangles": 0,
            "graph_closure_cycle_error_mean_deg": float("nan"),
            "graph_closure_cycle_error_p90_deg": float("nan"),
            "graph_closure_edge_weight_mean": 1.0,
            "graph_closure_edge_weight_median": 1.0,
            "graph_closure_edge_weight_min": 1.0,
            "graph_closure_edge_weight_max": 1.0,
            "graph_closure_view_weight_mean": 1.0,
            "graph_closure_view_weight_median": 1.0,
            "graph_closure_view_weight_min": 1.0,
            "graph_closure_view_weight_max": 1.0,
            "graph_closure_view_weights": {vid: 1.0 for vid in view_ids},
            "graph_closure_worst_edges": [],
        }

    obs_weight_enable = bool(getattr(cfg, "obs_weight_enable", False))
    obs_weight_min = float(np.clip(float(getattr(cfg, "obs_weight_min", 0.25)), 0.0, 1.0))
    obs_weight_power = float(max(1e-6, float(getattr(cfg, "obs_weight_power", 1.0))))

    if obs_weight_enable:
        raw_obs_conf = np.power(np.clip(point_conf_clip[obs_point_idx], 0.0, 1.0), obs_weight_power)
        obs_weight_all_np = obs_weight_min + (1.0 - obs_weight_min) * raw_obs_conf
    else:
        obs_weight_all_np = np.ones((obs_uv.shape[0],), dtype=np.float64)

    min_graph_weight = float(np.clip(float(getattr(cfg, "graph_closure_min_weight", 0.05)), 0.0, 1.0))
    obs_graph_weight_all_np, graph_closure_obs_stats = _compute_observation_graph_weights(
        obs_view_idx=obs_view_idx,
        obs_point_idx=obs_point_idx,
        point_meta=point_meta,
        visibility=visibility,
        edge_weight_matrix=edge_graph_weight_np,
        view_weights=view_graph_weight_np,
        min_weight=min_graph_weight,
    )
    obs_weight_all_np = np.clip(obs_weight_all_np * obs_graph_weight_all_np, 1e-6, None)

    obs_weight_train_np = obs_weight_all_np[train_idx_np]

    # Torch constants.
    obs_view_all_t = torch.tensor(obs_view_idx, dtype=torch.long, device=device)
    obs_point_all_t = torch.tensor(obs_point_idx, dtype=torch.long, device=device)
    obs_uv_all_t = torch.tensor(obs_uv, dtype=dtype, device=device)

    obs_view_train_t = torch.tensor(obs_view_idx[train_idx_np], dtype=torch.long, device=device)
    obs_point_train_t = torch.tensor(obs_point_idx[train_idx_np], dtype=torch.long, device=device)
    obs_uv_train_t = torch.tensor(obs_uv[train_idx_np], dtype=dtype, device=device)
    obs_weight_train_t = torch.tensor(obs_weight_train_np, dtype=dtype, device=device)

    if holdout_idx_np.size > 0:
        obs_view_holdout_t = torch.tensor(obs_view_idx[holdout_idx_np], dtype=torch.long, device=device)
        obs_point_holdout_t = torch.tensor(obs_point_idx[holdout_idx_np], dtype=torch.long, device=device)
        obs_uv_holdout_t = torch.tensor(obs_uv[holdout_idx_np], dtype=dtype, device=device)
    else:
        obs_view_holdout_t = torch.zeros((0,), dtype=torch.long, device=device)
        obs_point_holdout_t = torch.zeros((0,), dtype=torch.long, device=device)
        obs_uv_holdout_t = torch.zeros((0, 2), dtype=dtype, device=device)

    rvec_init_t = torch.tensor(rvec_init, dtype=dtype, device=device)
    t_init_t = torch.tensor(t_init, dtype=dtype, device=device)
    fx_t = torch.tensor(fx, dtype=dtype, device=device)
    fy_t = torch.tensor(fy, dtype=dtype, device=device)
    cx_t = torch.tensor(cx, dtype=dtype, device=device)
    cy_t = torch.tensor(cy, dtype=dtype, device=device)
    dist_t = torch.tensor(dist, dtype=dtype, device=device)

    points_init_t = torch.tensor(points_init, dtype=dtype, device=device)

    X_param = torch.nn.Parameter(points_init_t.clone())

    params = [X_param]
    if optimize_indices.size > 0:
        opt_idx_t = torch.tensor(optimize_indices, dtype=torch.long, device=device)
        rvec_opt = torch.nn.Parameter(rvec_init_t[opt_idx_t].clone())
        t_opt = torch.nn.Parameter(t_init_t[opt_idx_t].clone())
        params.extend([rvec_opt, t_opt])
        rvec_opt_init = rvec_init_t[opt_idx_t].detach().clone()
        t_opt_init = t_init_t[opt_idx_t].detach().clone()
    else:
        opt_idx_t = None
        rvec_opt = None
        t_opt = None
        rvec_opt_init = None
        t_opt_init = None

    def compose_extrinsics():
        r_all = rvec_init_t.clone()
        t_all = t_init_t.clone()
        if opt_idx_t is not None:
            r_all[opt_idx_t] = rvec_opt
            t_all[opt_idx_t] = t_opt
        return r_all, t_all

    graph_loss_weight_cfg = float(max(0.0, float(getattr(cfg, "graph_closure_loss_weight", 0.1))))
    graph_stage1_scale_cfg = float(max(0.0, float(getattr(cfg, "graph_closure_stage1_loss_scale", 0.5))))
    graph_loss_huber_delta_deg_cfg = float(max(1e-6, float(getattr(cfg, "graph_closure_loss_huber_delta_deg", 5.0))))

    edge_i_np = np.asarray(graph_edge_constraints.get("edge_i_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    edge_j_np = np.asarray(graph_edge_constraints.get("edge_j_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    edge_R_np = np.asarray(graph_edge_constraints.get("edge_rotations", np.zeros((0, 3, 3), dtype=np.float64)), dtype=np.float64)
    edge_w_np = np.asarray(graph_edge_constraints.get("edge_weights", np.zeros((0,), dtype=np.float64)), dtype=np.float64)

    graph_loss_active = bool(
        graph_closure_enabled
        and graph_loss_weight_cfg > 0.0
        and edge_i_np.size > 0
        and edge_j_np.size == edge_i_np.size
        and edge_w_np.size == edge_i_np.size
        and edge_R_np.shape[0] == edge_i_np.size
    )

    graph_loss_quality_scale = 0.0
    if graph_loss_active:
        edge_weight_mean = float(np.mean(np.clip(edge_w_np, 0.0, 1.0)))
        cycle_tau_deg = float(max(1e-6, float(getattr(cfg, "graph_closure_cycle_tau_deg", 1.0))))
        cycle_mean_deg = float(graph_closure_stats.get("graph_closure_cycle_error_mean_deg", float("nan")))
        if np.isfinite(cycle_mean_deg):
            cycle_quality = float(np.exp(-((cycle_mean_deg / cycle_tau_deg) ** 2)))
        else:
            cycle_quality = float(np.clip(float(getattr(cfg, "graph_closure_no_cycle_weight", 0.5)), 0.0, 1.0))

        edge_density = float(np.clip(float(edge_i_np.size) / max(float(V - 1), 1.0), 0.0, 1.0))
        graph_loss_quality_scale = float(
            np.clip(edge_weight_mean * (0.5 + 0.5 * edge_density) * (0.5 + 0.5 * cycle_quality), 0.0, 1.0)
        )

    graph_loss_weight_effective = float(graph_loss_weight_cfg) * float(graph_loss_quality_scale)
    if not graph_loss_active or graph_loss_weight_effective <= 0.0:
        graph_loss_active = False
        graph_loss_weight_effective = 0.0
    graph_stage1_loss_weight_effective = float(graph_loss_weight_effective) * float(graph_stage1_scale_cfg)

    if graph_loss_active:
        edge_i_t = torch.tensor(edge_i_np, dtype=torch.long, device=device)
        edge_j_t = torch.tensor(edge_j_np, dtype=torch.long, device=device)
        edge_R_t = torch.tensor(edge_R_np, dtype=dtype, device=device)
        edge_w_t = torch.tensor(np.clip(edge_w_np, 1e-6, 1.0), dtype=dtype, device=device)
    else:
        edge_i_t = None
        edge_j_t = None
        edge_R_t = None
        edge_w_t = None

    def _graph_rotation_loss_torch(r_all):
        if not graph_loss_active:
            return torch.tensor(0.0, dtype=dtype, device=device)
        R_all = rodrigues_torch(r_all)
        R_i = R_all[edge_i_t]
        R_j = R_all[edge_j_t]
        R_pred = torch.einsum("eij,ejk->eik", R_j, R_i.transpose(-1, -2))
        R_delta = torch.einsum("eij,ejk->eik", R_pred, edge_R_t.transpose(-1, -2))
        trace = R_delta[:, 0, 0] + R_delta[:, 1, 1] + R_delta[:, 2, 2]
        cos_theta = torch.clamp(0.5 * (trace - 1.0), -1.0 + 1e-7, 1.0 - 1e-7)
        theta_deg = torch.rad2deg(torch.acos(cos_theta))
        per_edge = _huber_torch(theta_deg, float(graph_loss_huber_delta_deg_cfg))
        return (per_edge * edge_w_t).sum() / torch.clamp(edge_w_t.sum(), min=1e-12)

    def run_stage(
        *,
        num_steps: int,
        lr: float,
        huber_delta: float,
        reg_weight: float,
        point_reg_weight: float,
        depth_penalty_weight: float,
        graph_loss_weight: float,
    ) -> Tuple[float, float]:
        if int(num_steps) <= 0:
            return float("nan"), float("nan")
        opt = torch.optim.Adam(params, lr=float(lr))
        n_obs_train = int(obs_uv_train_t.shape[0])
        batch_size = min(max(32, int(cfg.batch_size)), n_obs_train)
        last = float("nan")
        last_graph = float("nan")

        for _ in range(int(num_steps)):
            if batch_size == n_obs_train:
                b_view = obs_view_train_t
                b_point = obs_point_train_t
                b_uv = obs_uv_train_t
                b_weight = obs_weight_train_t
            else:
                batch_idx = torch.randint(0, n_obs_train, size=(batch_size,), device=device)
                b_view = obs_view_train_t[batch_idx]
                b_point = obs_point_train_t[batch_idx]
                b_uv = obs_uv_train_t[batch_idx]
                b_weight = obs_weight_train_t[batch_idx]

            r_all, t_all = compose_extrinsics()
            uv_pred, depth = _project_observations_torch(
                X_points=X_param,
                obs_point_idx=b_point,
                obs_view_idx=b_view,
                rvec_all=r_all,
                t_all=t_all,
                fx=fx_t,
                fy=fy_t,
                cx=cx_t,
                cy=cy_t,
                dist=dist_t,
            )

            reproj_loss_per_obs = _huber_torch(uv_pred - b_uv, float(huber_delta)).mean(dim=-1)
            data_loss = (reproj_loss_per_obs * b_weight).sum() / torch.clamp(b_weight.sum(), min=1e-12)

            reg_ext = torch.tensor(0.0, dtype=dtype, device=device)
            if rvec_opt is not None:
                reg_ext = reg_ext + (rvec_opt - rvec_opt_init).pow(2).mean()
                reg_ext = reg_ext + (t_opt - t_opt_init).pow(2).mean()

            reg_points = (X_param - points_init_t).pow(2).mean()
            depth_pen = torch.relu(1e-3 - depth).mean()
            graph_rot = _graph_rotation_loss_torch(r_all)

            loss = (
                data_loss
                + float(reg_weight) * reg_ext
                + float(point_reg_weight) * reg_points
                + float(depth_penalty_weight) * depth_pen
                + float(max(0.0, graph_loss_weight)) * graph_rot
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                X_param.clamp_(-1e4, 1e4)
                if rvec_opt is not None:
                    rvec_opt.clamp_(-3.14, 3.14)
                    t_opt.clamp_(-1e4, 1e4)

            last = float(loss.detach().cpu().numpy())
            last_graph = float(graph_rot.detach().cpu().numpy())

        return last, last_graph

    err_before_all = _mean_reprojection_error_torch(
        X_points=points_init_t,
        obs_uv=obs_uv_all_t,
        obs_point_idx=obs_point_all_t,
        obs_view_idx=obs_view_all_t,
        rvec_all=rvec_init_t,
        t_all=t_init_t,
        fx=fx_t,
        fy=fy_t,
        cx=cx_t,
        cy=cy_t,
        dist=dist_t,
    )
    err_before_train = _mean_reprojection_error_torch(
        X_points=points_init_t,
        obs_uv=obs_uv_train_t,
        obs_point_idx=obs_point_train_t,
        obs_view_idx=obs_view_train_t,
        rvec_all=rvec_init_t,
        t_all=t_init_t,
        fx=fx_t,
        fy=fy_t,
        cx=cx_t,
        cy=cy_t,
        dist=dist_t,
    )
    err_before_holdout = _mean_reprojection_error_torch(
        X_points=points_init_t,
        obs_uv=obs_uv_holdout_t,
        obs_point_idx=obs_point_holdout_t,
        obs_view_idx=obs_view_holdout_t,
        rvec_all=rvec_init_t,
        t_all=t_init_t,
        fx=fx_t,
        fy=fy_t,
        cx=cx_t,
        cy=cy_t,
        dist=dist_t,
    )
    err_before_holdout_per_view, holdout_obs_per_view = _per_view_mean_reprojection_error_torch(
        X_points=points_init_t,
        obs_uv=obs_uv_holdout_t,
        obs_point_idx=obs_point_holdout_t,
        obs_view_idx=obs_view_holdout_t,
        rvec_all=rvec_init_t,
        t_all=t_init_t,
        fx=fx_t,
        fy=fy_t,
        cx=cx_t,
        cy=cy_t,
        dist=dist_t,
        n_views=V,
    )

    total_steps = max(1, int(cfg.max_steps))
    stage1_steps = int(round(float(total_steps) * float(cfg.stage1_steps_ratio)))
    stage1_steps = min(max(stage1_steps, 0), max(0, total_steps - 1))
    stage2_steps = max(1, total_steps - stage1_steps)

    stage_logs: List[Dict[str, Any]] = []
    last_loss = float("nan")
    last_graph_loss = float("nan")

    graph_rot_loss_before = float("nan")
    if graph_loss_active:
        with torch.no_grad():
            graph_rot_loss_before = float(_graph_rotation_loss_torch(rvec_init_t).detach().cpu().numpy())

    if stage1_steps > 0:
        last_loss, last_graph_loss = run_stage(
            num_steps=stage1_steps,
            lr=float(cfg.lr) * float(cfg.stage1_lr_scale),
            huber_delta=float(cfg.huber_delta) * float(cfg.stage1_huber_delta_scale),
            reg_weight=float(cfg.reg_weight) * float(cfg.stage1_reg_scale),
            point_reg_weight=float(cfg.point_reg_weight) * float(cfg.stage1_point_reg_scale),
            depth_penalty_weight=float(cfg.depth_penalty_weight) * float(cfg.stage1_depth_penalty_scale),
            graph_loss_weight=float(graph_stage1_loss_weight_effective),
        )
        stage_logs.append(
            {
                "stage": "warmup",
                "steps": int(stage1_steps),
                "lr": float(cfg.lr) * float(cfg.stage1_lr_scale),
                "loss_last": float(last_loss),
                "graph_loss_weight": float(graph_stage1_loss_weight_effective),
                "graph_loss_last": float(last_graph_loss),
            }
        )

    last_loss, last_graph_loss = run_stage(
        num_steps=stage2_steps,
        lr=float(cfg.lr),
        huber_delta=float(cfg.huber_delta),
        reg_weight=float(cfg.reg_weight),
        point_reg_weight=float(cfg.point_reg_weight),
        depth_penalty_weight=float(cfg.depth_penalty_weight),
        graph_loss_weight=float(graph_loss_weight_effective),
    )
    stage_logs.append(
        {
            "stage": "refine",
            "steps": int(stage2_steps),
            "lr": float(cfg.lr),
            "loss_last": float(last_loss),
            "graph_loss_weight": float(graph_loss_weight_effective),
            "graph_loss_last": float(last_graph_loss),
        }
    )

    r_final, t_final = compose_extrinsics()
    graph_rot_loss_after = float("nan")
    if graph_loss_active:
        with torch.no_grad():
            graph_rot_loss_after = float(_graph_rotation_loss_torch(r_final).detach().cpu().numpy())
    err_after_all = _mean_reprojection_error_torch(
        X_points=X_param.detach(),
        obs_uv=obs_uv_all_t,
        obs_point_idx=obs_point_all_t,
        obs_view_idx=obs_view_all_t,
        rvec_all=r_final,
        t_all=t_final,
        fx=fx_t,
        fy=fy_t,
        cx=cx_t,
        cy=cy_t,
        dist=dist_t,
    )
    err_after_train = _mean_reprojection_error_torch(
        X_points=X_param.detach(),
        obs_uv=obs_uv_train_t,
        obs_point_idx=obs_point_train_t,
        obs_view_idx=obs_view_train_t,
        rvec_all=r_final,
        t_all=t_final,
        fx=fx_t,
        fy=fy_t,
        cx=cx_t,
        cy=cy_t,
        dist=dist_t,
    )
    err_after_holdout = _mean_reprojection_error_torch(
        X_points=X_param.detach(),
        obs_uv=obs_uv_holdout_t,
        obs_point_idx=obs_point_holdout_t,
        obs_view_idx=obs_view_holdout_t,
        rvec_all=r_final,
        t_all=t_final,
        fx=fx_t,
        fy=fy_t,
        cx=cx_t,
        cy=cy_t,
        dist=dist_t,
    )
    err_after_holdout_per_view, _ = _per_view_mean_reprojection_error_torch(
        X_points=X_param.detach(),
        obs_uv=obs_uv_holdout_t,
        obs_point_idx=obs_point_holdout_t,
        obs_view_idx=obs_view_holdout_t,
        rvec_all=r_final,
        t_all=t_final,
        fx=fx_t,
        fy=fy_t,
        cx=cx_t,
        cy=cy_t,
        dist=dist_t,
        n_views=V,
    )

    per_view_gate_eval = {
        "ok": True,
        "considered": [],
        "degraded": [],
    }

    accepted = True
    gate_reason = "accepted"
    if bool(cfg.accept_gate_enable):
        if _is_finite_scalar(err_before_all):
            allowed_all = float(err_before_all) * (1.0 + float(cfg.accept_max_error_increase_rel)) + float(cfg.accept_max_error_increase_abs)
            if (not _is_finite_scalar(err_after_all)) or float(err_after_all) > float(allowed_all):
                accepted = False
                gate_reason = "all_reprojection_degraded"

        if accepted and int(holdout_idx_np.size) > 0 and _is_finite_scalar(err_before_holdout):
            allowed_holdout = float(err_before_holdout) * (1.0 + float(cfg.accept_max_error_increase_rel)) + float(cfg.accept_max_error_increase_abs)
            if (not _is_finite_scalar(err_after_holdout)) or float(err_after_holdout) > float(allowed_holdout):
                accepted = False
                gate_reason = "holdout_reprojection_degraded"

        if accepted and bool(getattr(cfg, "accept_per_view_holdout_gate_enable", True)) and int(holdout_idx_np.size) > 0:
            per_view_gate_eval = _evaluate_per_view_holdout_gate(
                err_before_per_view=err_before_holdout_per_view,
                err_after_per_view=err_after_holdout_per_view,
                obs_counts_per_view=holdout_obs_per_view,
                min_obs=int(getattr(cfg, "accept_per_view_holdout_min_obs", 128)),
                max_error_increase_abs=float(getattr(cfg, "accept_per_view_holdout_max_error_increase_abs", cfg.accept_max_error_increase_abs)),
                max_error_increase_rel=float(getattr(cfg, "accept_per_view_holdout_max_error_increase_rel", cfg.accept_max_error_increase_rel)),
            )
            if per_view_gate_eval["considered"] and (not bool(per_view_gate_eval["ok"])):
                accepted = False
                gate_reason = "per_view_holdout_reprojection_degraded"

        if accepted and bool(cfg.accept_require_global_improvement):
            min_drop = float(max(0.0, float(cfg.accept_min_error_drop_abs)))
            improved = _is_finite_scalar(err_before_all) and _is_finite_scalar(err_after_all) and (
                float(err_after_all) <= float(err_before_all) - min_drop
            )
            if not improved:
                accepted = False
                gate_reason = "no_global_improvement"

    r_final_np = r_final.detach().cpu().numpy().astype(np.float64)
    t_final_np = t_final.detach().cpu().numpy().astype(np.float64)

    cams_opt: Dict[str, CameraParams] = {}
    for i, vid in enumerate(view_ids):
        cam0 = cameras_init[vid]
        R_i, _ = cv2.Rodrigues(r_final_np[i].reshape(3, 1))
        cams_opt[vid] = CameraParams(
            view_id=str(vid),
            K=np.asarray(cam0.K, dtype=np.float64),
            R=np.asarray(R_i, dtype=np.float64),
            T=np.asarray(t_final_np[i], dtype=np.float64).reshape(3),
            D=np.asarray(cam0.D, dtype=np.float64).reshape(-1)[:5],
            H=int(cam0.H),
            W=int(cam0.W),
        )

    cams_out = cams_opt if bool(accepted) else dict(cameras_init)
    obs_per_view = {vid: int(np.sum(obs_view_idx == i)) for i, vid in enumerate(view_ids)}

    holdout_obs_per_view_map = {
        vid: int(holdout_obs_per_view[i])
        for i, vid in enumerate(view_ids)
        if int(holdout_obs_per_view[i]) > 0
    }
    holdout_err_before_per_view_map = {
        vid: float(err_before_holdout_per_view[i])
        for i, vid in enumerate(view_ids)
        if int(holdout_obs_per_view[i]) > 0 and np.isfinite(err_before_holdout_per_view[i])
    }
    holdout_err_after_per_view_map = {
        vid: float(err_after_holdout_per_view[i])
        for i, vid in enumerate(view_ids)
        if int(holdout_obs_per_view[i]) > 0 and np.isfinite(err_after_holdout_per_view[i])
    }
    per_view_gate_considered = [view_ids[i] for i in per_view_gate_eval.get("considered", [])]
    per_view_gate_degraded = [view_ids[i] for i in per_view_gate_eval.get("degraded", [])]

    stats = {
        "method": "global_ba",
        "ok": bool(accepted),
        "accepted": bool(accepted),
        "gate_reason": str(gate_reason),
        "n_views": int(V),
        "n_points": int(points_init.shape[0]),
        "n_observations": int(obs_uv.shape[0]),
        "n_observations_train": int(train_idx_np.size),
        "n_observations_holdout": int(holdout_idx_np.size),
        "n_candidates": int(obs_data.get("n_candidates", 0)),
        "n_candidates_kept": int(obs_data.get("n_candidates_kept", 0)),
        "n_failed_triangulation": int(obs_data.get("n_failed_triangulation", 0)),
        "n_failed_low_parallax": int(obs_data.get("n_failed_low_parallax", 0)),
        "n_filtered_by_support": int(obs_data.get("n_filtered_by_support", 0)),
        "n_frames_used": int(obs_data.get("n_frames_used", 0)),
        "support_frames_mean": float(obs_data.get("support_frames_mean", 0.0)),
        "support_frames_median": float(obs_data.get("support_frames_median", 0.0)),
        "point_confidence_mean": float(np.mean(point_conf_clip)) if point_conf_clip.size else 0.0,
        "point_confidence_median": float(np.median(point_conf_clip)) if point_conf_clip.size else 0.0,
        "point_confidence_min": float(np.min(point_conf_clip)) if point_conf_clip.size else 0.0,
        "point_confidence_max": float(np.max(point_conf_clip)) if point_conf_clip.size else 0.0,
        "obs_weight_enabled": bool(obs_weight_enable),
        "obs_weight_min_cfg": float(obs_weight_min),
        "obs_weight_power_cfg": float(obs_weight_power),
        "obs_weight_mean": float(np.mean(obs_weight_all_np)) if obs_weight_all_np.size else 1.0,
        "obs_weight_median": float(np.median(obs_weight_all_np)) if obs_weight_all_np.size else 1.0,
        "obs_weight_min": float(np.min(obs_weight_all_np)) if obs_weight_all_np.size else 1.0,
        "obs_weight_max": float(np.max(obs_weight_all_np)) if obs_weight_all_np.size else 1.0,
        "obs_graph_weight_mean": float(np.mean(obs_graph_weight_all_np)) if obs_graph_weight_all_np.size else 1.0,
        "obs_graph_weight_median": float(np.median(obs_graph_weight_all_np)) if obs_graph_weight_all_np.size else 1.0,
        "obs_graph_weight_min": float(np.min(obs_graph_weight_all_np)) if obs_graph_weight_all_np.size else 1.0,
        "obs_graph_weight_max": float(np.max(obs_graph_weight_all_np)) if obs_graph_weight_all_np.size else 1.0,
        "graph_closure_loss_active": bool(graph_loss_active),
        "graph_closure_loss_num_edges": int(edge_i_np.size),
        "graph_closure_loss_weight_cfg": float(graph_loss_weight_cfg),
        "graph_closure_stage1_loss_scale_cfg": float(graph_stage1_scale_cfg),
        "graph_closure_loss_huber_delta_deg_cfg": float(graph_loss_huber_delta_deg_cfg),
        "graph_closure_loss_quality_scale": float(graph_loss_quality_scale),
        "graph_closure_loss_weight_effective": float(graph_loss_weight_effective),
        "graph_closure_stage1_loss_weight_effective": float(graph_stage1_loss_weight_effective),
        "graph_closure_loss_before": float(graph_rot_loss_before),
        "graph_closure_loss_after": float(graph_rot_loss_after),
        "graph_closure_loss_last": float(last_graph_loss),
        "triangulation_reproj_threshold_used": float(used_reproj_th),
        "mean_reproj_error_before": float(err_before_all),
        "mean_reproj_error_after": float(err_after_all),
        "mean_reproj_error_before_train": float(err_before_train),
        "mean_reproj_error_after_train": float(err_after_train),
        "mean_reproj_error_before_holdout": float(err_before_holdout),
        "mean_reproj_error_after_holdout": float(err_after_holdout),
        "loss_last": float(last_loss),
        "stage_logs": stage_logs,
        "obs_per_view": obs_per_view,
        "holdout_obs_per_view": holdout_obs_per_view_map,
        "holdout_reproj_error_before_per_view": holdout_err_before_per_view_map,
        "holdout_reproj_error_after_per_view": holdout_err_after_per_view_map,
        "per_view_holdout_gate_enable": bool(getattr(cfg, "accept_per_view_holdout_gate_enable", True)),
        "per_view_holdout_gate_min_obs": int(getattr(cfg, "accept_per_view_holdout_min_obs", 128)),
        "per_view_holdout_gate_considered_views": per_view_gate_considered,
        "per_view_holdout_gate_degraded_views": per_view_gate_degraded,
        "ref_view": str(ref_view),
        "fix_ref_extrinsic": bool(cfg.fix_ref_extrinsic),
        "optimized_views": [view_ids[i] for i in optimize_indices.tolist()],
        **graph_closure_stats,
        **graph_closure_obs_stats,
        **init_filter_stats,
    }

    return cams_out, stats


def anchored_local_ba(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    cameras_init: Dict[str, CameraParams],
    ref_view: str,
    fixed_views: Sequence[str],
    max_steps: int = 80,
    lr: float = 5e-3,
    device: str = "cuda",
    min_views_per_point: int = 2,
    triangulation_reproj_threshold: float = 5.0,
    huber_delta: float = 3.0,
    reg_weight: float = 5e-4,
    accept_max_error_increase_abs: float = 1.0,
    accept_max_error_increase_rel: float = 0.2,
    graph_closure_enable: bool = False,
    graph_closure_min_shared_obs: int = 64,
    graph_closure_max_pair_obs: int = 2000,
    graph_closure_ransac_threshold: float = 1e-3,
    graph_closure_ransac_prob: float = 0.999,
    graph_closure_min_inliers: int = 24,
    graph_closure_min_inlier_ratio: float = 0.25,
    graph_closure_cycle_tau_deg: float = 1.0,
    graph_closure_min_weight: float = 0.05,
    graph_closure_weight_power: float = 1.0,
    graph_closure_no_cycle_weight: float = 0.5,
    graph_closure_loss_weight: float = 0.1,
    graph_closure_stage1_loss_scale: float = 0.5,
    graph_closure_loss_huber_delta_deg: float = 5.0,
) -> Tuple[Dict[str, CameraParams], Dict[str, Any]]:
    """Run a short local BA with some views held fixed as anchors.

    This is designed for Layer-2 refinement: static-view (and confident dynamic-view)
    poses are fixed, and only uncertain long views are optimized.
    """
    cfg = GlobalTrackBAConfig(
        max_steps=int(max_steps),
        lr=float(lr),
        device=str(device),
        stage1_steps_ratio=0.0,
        fix_ref_extrinsic=True,
        fixed_views=tuple(str(v) for v in fixed_views),
        accept_gate_enable=True,
        accept_require_global_improvement=False,
        accept_holdout_ratio=0.1,
        accept_min_holdout_obs=32,
        accept_max_error_increase_abs=float(accept_max_error_increase_abs),
        accept_max_error_increase_rel=float(accept_max_error_increase_rel),
        accept_per_view_holdout_gate_enable=False,
        huber_delta=float(huber_delta),
        reg_weight=float(reg_weight),
        point_reg_weight=1e-5,
        depth_penalty_weight=1e-2,
        min_views_per_point=int(min_views_per_point),
        triangulation_reproj_threshold=float(triangulation_reproj_threshold),
        batch_size=8192,
        init_obs_filter_enable=False,
        init_obs_max_reproj_error_px=200.0,
        graph_closure_enable=bool(graph_closure_enable),
        graph_closure_min_shared_obs=int(graph_closure_min_shared_obs),
        graph_closure_max_pair_obs=int(graph_closure_max_pair_obs),
        graph_closure_ransac_threshold=float(graph_closure_ransac_threshold),
        graph_closure_ransac_prob=float(graph_closure_ransac_prob),
        graph_closure_min_inliers=int(graph_closure_min_inliers),
        graph_closure_min_inlier_ratio=float(graph_closure_min_inlier_ratio),
        graph_closure_cycle_tau_deg=float(graph_closure_cycle_tau_deg),
        graph_closure_min_weight=float(graph_closure_min_weight),
        graph_closure_weight_power=float(graph_closure_weight_power),
        graph_closure_no_cycle_weight=float(graph_closure_no_cycle_weight),
        graph_closure_loss_weight=float(graph_closure_loss_weight),
        graph_closure_stage1_loss_scale=float(graph_closure_stage1_loss_scale),
        graph_closure_loss_huber_delta_deg=float(graph_closure_loss_huber_delta_deg),
    )
    return refine_cameras_global_ba(
        tracks=tracks,
        visibility=visibility,
        view_ids=view_ids,
        cameras_init=cameras_init,
        ref_view=ref_view,
        cfg=cfg,
    )
