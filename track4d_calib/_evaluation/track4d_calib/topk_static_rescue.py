from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from track4d_calib.cameras import CameraParams


@dataclass(frozen=True)
class TopKRescueDecision:
    trigger: bool
    reasons: tuple[str, ...]


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def should_trigger_topk_rescue(
    info: Mapping[str, object],
    *,
    min_center_shift: float,
    max_fixed_inliers: int,
    min_free_inliers: int,
    require_free_rescue: bool = True,
) -> TopKRescueDecision:
    """Decide whether a tracking-PnP result needs independent TopK static rescue."""
    reasons: list[str] = []
    if not bool(info.get("success", True)):
        return TopKRescueDecision(False, ("not_successful",))

    if bool(require_free_rescue):
        selected = str(info.get("pnp_fix_center_selected", ""))
        rescue_applied = bool(info.get("pnp_fix_center_rescue_applied", False))
        if selected != "free_rescue" and not rescue_applied:
            return TopKRescueDecision(False, ("not_free_rescue",))
        reasons.append("free_rescue")

    center_shift = _as_float(
        info.get("pnp_fix_center_free_center_shift", info.get("risk_guard_center_shift_from_stable", 0.0))
    )
    if center_shift < float(min_center_shift):
        return TopKRescueDecision(False, ("center_shift_below_threshold",))
    reasons.append("center_shift")

    fixed_inliers = _as_int(info.get("pnp_fix_center_fixed_inliers", 0))
    if fixed_inliers > int(max_fixed_inliers):
        return TopKRescueDecision(False, ("fixed_center_has_support",))
    reasons.append("fixed_center_low_support")

    free_inliers = _as_int(info.get("pnp_fix_center_free_inliers", info.get("num_inliers", 0)))
    if free_inliers < int(min_free_inliers):
        return TopKRescueDecision(False, ("free_solution_low_support",))
    reasons.append("free_solution_supported")

    return TopKRescueDecision(True, tuple(reasons))


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float((np.trace(rotation) - 1.0) * 0.5)
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _estimate_static_orientation_rotation(
    candidate_cams: Mapping[str, CameraParams],
    target_cams: Mapping[str, CameraParams],
    static_views: Sequence[str],
) -> tuple[np.ndarray, dict[str, float | int]]:
    accum = np.zeros((3, 3), dtype=np.float64)
    before: list[float] = []
    used = 0
    for view in static_views:
        candidate = candidate_cams.get(str(view))
        target = target_cams.get(str(view))
        if candidate is None or target is None:
            continue
        candidate_c2w = np.asarray(candidate.R, dtype=np.float64).reshape(3, 3).T
        target_c2w = np.asarray(target.R, dtype=np.float64).reshape(3, 3).T
        accum += target_c2w @ candidate_c2w.T
        before.append(_rotation_angle_deg(candidate_c2w @ target_c2w.T))
        used += 1

    if used <= 0:
        raise ValueError("No common static views for TopK rescue gauge alignment")

    U, _, Vt = np.linalg.svd(accum)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1.0
        rotation = U @ Vt

    after: list[float] = []
    for view in static_views:
        candidate = candidate_cams.get(str(view))
        target = target_cams.get(str(view))
        if candidate is None or target is None:
            continue
        candidate_c2w = np.asarray(candidate.R, dtype=np.float64).reshape(3, 3).T
        target_c2w = np.asarray(target.R, dtype=np.float64).reshape(3, 3).T
        after.append(_rotation_angle_deg((rotation @ candidate_c2w) @ target_c2w.T))

    return rotation, {
        "static_orientation_error_before_mean_deg": float(np.mean(before)) if before else 0.0,
        "static_orientation_error_before_max_deg": float(np.max(before)) if before else 0.0,
        "static_orientation_error_after_mean_deg": float(np.mean(after)) if after else 0.0,
        "static_orientation_error_after_max_deg": float(np.max(after)) if after else 0.0,
        "static_alignment_num_views": int(used),
    }


def _fit_scale_translation_with_fixed_rotation(
    src: np.ndarray,
    dst: np.ndarray,
    rotation: np.ndarray,
    *,
    with_scale: bool = True,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if src.shape != dst.shape or src.shape[0] < 1:
        raise ValueError("src and dst must have matching shape (N,3) with N>=1")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_rot = (rotation @ (src - mu_src).T).T
    dst_centered = dst - mu_dst
    if bool(with_scale):
        denom = float(np.sum(src_rot * src_rot))
        scale = float(np.sum(dst_centered * src_rot) / denom) if denom > float(eps) else 1.0
    else:
        scale = 1.0
    translation = mu_dst - float(scale) * (rotation @ mu_src)
    return float(scale), np.asarray(translation, dtype=np.float64).reshape(3)


def transform_camera_world(
    cam: CameraParams,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> CameraParams:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    center_old = np.asarray(cam.camera_center, dtype=np.float64).reshape(3)
    c2w_rot_old = np.asarray(cam.R, dtype=np.float64).reshape(3, 3).T

    center_new = float(scale) * (rotation @ center_old) + translation
    c2w_rot_new = rotation @ c2w_rot_old
    R_new = c2w_rot_new.T
    T_new = -R_new @ center_new
    return CameraParams(
        view_id=str(cam.view_id),
        K=np.asarray(cam.K, dtype=np.float64).copy(),
        R=R_new,
        T=T_new,
        D=np.asarray(cam.D, dtype=np.float64).copy(),
        H=int(cam.H),
        W=int(cam.W),
    )


def align_candidate_cameras_to_target_static_gauge(
    *,
    candidate_cams: Mapping[str, CameraParams],
    target_cams: Mapping[str, CameraParams],
    static_views: Sequence[str],
    views: Iterable[str],
) -> tuple[dict[str, CameraParams], dict[str, object]]:
    """Align candidate cameras into target static-camera gauge using static views."""
    static = [str(view) for view in static_views]
    rotation, orient_summary = _estimate_static_orientation_rotation(candidate_cams, target_cams, static)
    src_static = np.stack([candidate_cams[view].camera_center for view in static if view in candidate_cams], axis=0)
    dst_static = np.stack([target_cams[view].camera_center for view in static if view in target_cams], axis=0)
    if src_static.shape != dst_static.shape:
        raise ValueError("candidate and target static view sets differ")
    scale, translation = _fit_scale_translation_with_fixed_rotation(src_static, dst_static, rotation, with_scale=True)

    aligned = {
        str(view): transform_camera_world(
            candidate_cams[str(view)],
            scale=float(scale),
            rotation=rotation,
            translation=translation,
        )
        for view in views
        if str(view) in candidate_cams
    }

    aligned_static = (float(scale) * (rotation @ src_static.T)).T + translation.reshape(1, 3)
    static_err = np.linalg.norm(aligned_static - dst_static, axis=1)
    summary: dict[str, object] = {
        **orient_summary,
        "static_alignment_scale": float(scale),
        "static_alignment_translation": translation.tolist(),
        "static_center_error_after_mean": float(np.mean(static_err)) if static_err.size else 0.0,
        "static_center_error_after_max": float(np.max(static_err)) if static_err.size else 0.0,
    }
    return aligned, summary
