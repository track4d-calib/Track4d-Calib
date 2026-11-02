#!/usr/bin/env python3
"""Jointly refine a same-time dynamic camera band from shared tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import CameraLoader, CameraParams, save_cameras
from track4d_calib.global_track_ba import anchored_local_ba
from track4d_calib.multiview_epipolar_verify import sampson_distances_px
from track4d_calib.same_time_multiview_pnp import (
    SameTimePnPConfig,
    score_same_time_multiview_poses,
)
from track4d_calib.track_common_rig_rotation import (
    apply_common_world_rotation,
    estimate_common_world_rotation_from_pose_pairs,
)
from scripts.geometry.export_track_geometry_verified_hybrid import (
    paired_reprojection_confidence,
    stratified_crossfit_point_id_folds,
)
from scripts.geometry.run_track_closure import (
    build_node_observations,
    seed_point_ranges,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_track_freeze(*, freeze_path: Path, track_root: Path) -> dict[str, str]:
    freeze_path = freeze_path.resolve()
    root = track_root.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_pose_gt_or_nvs":
        raise RuntimeError("track inputs were not frozen before readout")
    if freeze.get("gt_used") or freeze.get("nvs_used"):
        raise RuntimeError("track freeze used a forbidden readout")
    if Path(str(freeze.get("track_root", ""))).resolve() != root:
        raise RuntimeError("track freeze points to a different root")
    declared = {
        str(name): str(digest)
        for name, digest in dict(freeze.get("file_sha256") or {}).items()
    }
    required = {"meta.json", "tracks.npy", "visibility.npy", "views.txt"}
    missing = sorted(required - set(declared))
    if missing:
        raise RuntimeError(f"track freeze is missing required files: {missing}")
    actual = {name: _sha256(root / name) for name in sorted(declared)}
    if actual != declared:
        raise RuntimeError("track files changed after freeze")
    return actual


def _camera_hashes(root: Path) -> dict[str, str]:
    return {
        name: _sha256(root / name) for name in ("intri.yml", "extri.yml")
    }


def verify_code_freeze(
    *, manifest_path: Path, code_root: Path
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = code_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_execution":
        raise RuntimeError("method code was not frozen before execution")
    if manifest.get("gt_used") or manifest.get("nvs_used"):
        raise RuntimeError("method code freeze used a forbidden readout")
    if Path(str(manifest.get("code_root", ""))).resolve() != root:
        raise RuntimeError("method code freeze points to a different code root")
    declared = {
        str(name): str(digest)
        for name, digest in dict(manifest.get("files") or {}).items()
    }
    current_files = {
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if set(declared) != current_files:
        raise RuntimeError("method code manifest does not cover the Python snapshot")
    actual = {name: _sha256(root / name) for name in sorted(declared)}
    if actual != declared:
        raise RuntimeError("method code files changed after freeze")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "code_root": str(root),
        "files": actual,
    }


def _verify_base_attestation(
    *, attestation_path: Path, base_pose_root: Path, frames: Sequence[int]
) -> dict[str, Any]:
    attestation_path = attestation_path.resolve()
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    prediction_path = Path(str(attestation.get("path", ""))).resolve()
    if _sha256(prediction_path) != str(attestation.get("sha256", "")):
        raise RuntimeError("base prediction attestation hash mismatch")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if prediction.get("gt_used") or prediction.get("nvs_used"):
        raise RuntimeError("base prediction used a forbidden readout")
    if Path(str(prediction.get("pose_root", ""))).resolve() != base_pose_root.resolve():
        raise RuntimeError("base prediction pose root mismatch")
    declared = dict(prediction.get("output_camera_hashes") or {})
    selected_names = [f"f{int(frame):06d}" for frame in frames]
    for frame_name in selected_names:
        expected = {
            str(name): str(digest)
            for name, digest in dict(declared.get(frame_name) or {}).items()
        }
        actual = _camera_hashes(base_pose_root / frame_name / "cameras_pred")
        if actual != expected:
            raise RuntimeError(f"base cameras changed after prediction: {frame_name}")
    return {
        "attestation": str(attestation_path),
        "attestation_sha256": _sha256(attestation_path),
        "prediction": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "protocol": str(prediction.get("protocol", "")),
        "runtime_tree_digest": str(prediction.get("runtime_tree_digest", "")),
        "camera_hashes": {
            frame_name: _camera_hashes(
                base_pose_root / frame_name / "cameras_pred"
            )
            for frame_name in selected_names
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_visibility_stratified_folds(
    *,
    meta: Mapping[str, Any],
    visibility: np.ndarray,
    view_ids: Sequence[str],
    target_views: Sequence[str],
    frame_local: int,
) -> tuple[list[int], list[int]]:
    """Split eligible points by seed bank and dynamic visibility signature."""

    visible = np.asarray(visibility, dtype=bool)
    views = [str(view) for view in view_ids]
    targets = [str(view) for view in target_views]
    if visible.ndim != 3 or visible.shape[0] != len(views):
        raise ValueError("visibility must have shape [view, frame, point]")
    if frame_local < 0 or frame_local >= visible.shape[1]:
        raise IndexError("frame_local is outside visibility")
    view_to_index = {view: index for index, view in enumerate(views)}
    if not targets or set(targets) - set(views):
        raise ValueError("target_views must be a non-empty subset of view_ids")
    target_indices = np.asarray([view_to_index[view] for view in targets], dtype=np.int64)
    frame_visibility = visible[:, int(frame_local), :]
    target_visibility = frame_visibility[target_indices]
    eligible = np.flatnonzero(
        (np.sum(frame_visibility, axis=0) >= 2)
        & (np.sum(target_visibility, axis=0) >= 1)
    ).astype(np.int64)
    strata = {
        int(point_id): tuple(
            target
            for target, is_visible in zip(
                targets, target_visibility[:, int(point_id)].tolist()
            )
            if bool(is_visible)
        )
        for point_id in eligible.tolist()
    }
    ranges = seed_point_ranges(meta=meta, total_points=int(visible.shape[2]))
    return stratified_crossfit_point_id_folds(
        point_ids=eligible.tolist(),
        point_ranges=ranges,
        strata_by_point_id=strata,
    )


def build_visibility_stratified_fit_select_verify_folds(
    *,
    meta: Mapping[str, Any],
    visibility: np.ndarray,
    view_ids: Sequence[str],
    target_views: Sequence[str],
    frame_local: int,
) -> tuple[list[int], list[int], list[int]]:
    """Preserve the fit fold and split its heldout complement in half."""

    fit_ids, heldout_ids = build_visibility_stratified_folds(
        meta=meta,
        visibility=visibility,
        view_ids=view_ids,
        target_views=target_views,
        frame_local=frame_local,
    )
    visible = np.asarray(visibility, dtype=bool)
    views = [str(view) for view in view_ids]
    targets = [str(view) for view in target_views]
    view_to_index = {view: index for index, view in enumerate(views)}
    target_indices = np.asarray(
        [view_to_index[view] for view in targets], dtype=np.int64
    )
    target_visibility = visible[target_indices, int(frame_local), :]
    strata = {
        int(point_id): tuple(
            target
            for target, is_visible in zip(
                targets, target_visibility[:, int(point_id)].tolist()
            )
            if bool(is_visible)
        )
        for point_id in heldout_ids
    }
    selector_ids, final_verify_ids = stratified_crossfit_point_id_folds(
        point_ids=heldout_ids,
        point_ranges=seed_point_ranges(
            meta=meta, total_points=int(visible.shape[2])
        ),
        strata_by_point_id=strata,
    )
    if set(fit_ids) & (set(selector_ids) | set(final_verify_ids)):
        raise RuntimeError("nested fit and verification folds overlap")
    if set(selector_ids) & set(final_verify_ids):
        raise RuntimeError("selector and final verification folds overlap")
    if set(selector_ids) | set(final_verify_ids) != set(heldout_ids):
        raise RuntimeError("nested verification folds do not cover heldout points")
    return fit_ids, selector_ids, final_verify_ids


def select_supported_target_band(
    *,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    target_views: Sequence[str],
    point_ids: np.ndarray,
    min_target_observations: int = 24,
    min_shared_observations: int = 8,
) -> dict[str, Any]:
    """Select the largest fit-fold target component with enough track support."""

    visible = np.asarray(visibility, dtype=bool)
    views = [str(view) for view in view_ids]
    targets = [str(view) for view in target_views]
    selected = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    if visible.ndim != 3 or visible.shape[0] != len(views) or visible.shape[1] != 1:
        raise ValueError("visibility must have shape [view, 1, point]")
    if not targets or set(targets) - set(views):
        raise ValueError("target_views must be present in view_ids")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("point_ids must be unique")
    if len(selected) and (
        int(np.min(selected)) < 0 or int(np.max(selected)) >= visible.shape[2]
    ):
        raise IndexError("point_ids are outside visibility")
    minimum_target = max(1, int(min_target_observations))
    minimum_shared = max(1, int(min_shared_observations))
    view_to_index = {view: index for index, view in enumerate(views)}
    counts = {
        target: int(visible[view_to_index[target], 0, selected].sum())
        for target in targets
    }
    eligible = [target for target in targets if counts[target] >= minimum_target]
    neighbors = {target: set() for target in eligible}
    supported_edges: list[tuple[str, str]] = []
    for left, right in combinations(eligible, 2):
        shared = int(
            np.sum(
                visible[view_to_index[left], 0, selected]
                & visible[view_to_index[right], 0, selected]
            )
        )
        if shared < minimum_shared:
            continue
        neighbors[left].add(right)
        neighbors[right].add(left)
        supported_edges.append((left, right))

    components: list[list[str]] = []
    unseen = set(eligible)
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        frontier = [seed]
        component: list[str] = []
        while frontier:
            current = frontier.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current] & unseen, reverse=True):
                unseen.remove(neighbor)
                frontier.append(neighbor)
        components.append(sorted(component))
    components.sort(
        key=lambda component: (
            -len(component),
            -sum(counts[target] for target in component),
            component,
        )
    )
    active = components[0] if components and len(components[0]) >= 2 else []
    return {
        "active_target_views": list(active),
        "fallback_target_views": [
            target for target in targets if target not in set(active)
        ],
        "observation_count_by_target": counts,
        "connected_components": components,
        "supported_edges": [list(edge) for edge in supported_edges],
        "min_target_observations": minimum_target,
        "min_shared_observations": minimum_shared,
        "reason": "ok" if active else "no_supported_multi_target_component",
    }


def build_world_right_fallback_candidate(
    *,
    base_cameras: Mapping[str, CameraParams],
    candidate_cameras: Mapping[str, CameraParams],
    contributor_views: Sequence[str],
    fallback_views: Sequence[str],
) -> tuple[dict[str, CameraParams], dict[str, Any]]:
    """Let unsupported cameras inherit a track-supported world-right rotation."""

    fallbacks = list(dict.fromkeys(str(view) for view in fallback_views))
    missing = [view for view in fallbacks if view not in base_cameras]
    if missing:
        raise KeyError(f"fallback cameras are missing from base_cameras: {missing}")
    estimate = estimate_common_world_rotation_from_pose_pairs(
        base_cameras=base_cameras,
        candidate_cameras=candidate_cameras,
        contributor_views=contributor_views,
        min_views=3,
    )
    output = dict(candidate_cameras)
    if bool(estimate["success"]):
        correction = np.asarray(estimate["correction"], dtype=np.float64)
        for view in fallbacks:
            output[view] = apply_common_world_rotation(
                base_cameras[view], correction
            )
    return output, {
        **estimate,
        "fallback_views": fallbacks,
        "inherited_fallback_count": (
            len(fallbacks) if bool(estimate["success"]) else 0
        ),
        "model": "world-right R_candidate = R_base Q; center preserved",
    }


def select_verified_band_cameras(
    *,
    base_cameras: Mapping[str, CameraParams],
    candidate_cameras: Mapping[str, CameraParams],
    active_views: Sequence[str],
    fallback_views: Sequence[str],
    active_commit: bool,
    full_band_commit: bool,
) -> tuple[dict[str, CameraParams], str]:
    """Materialize only the independently verified parts of a band candidate."""

    active = list(dict.fromkeys(str(view) for view in active_views))
    fallbacks = list(dict.fromkeys(str(view) for view in fallback_views))
    if set(active) & set(fallbacks):
        raise ValueError("active and fallback view sets must be disjoint")
    if bool(full_band_commit) and not bool(active_commit):
        raise ValueError("a full-band commit requires an accepted active component")
    selected = dict(base_cameras)
    if not bool(active_commit):
        return selected, "static_map_same_frame_pnp"
    missing = [
        view for view in active + fallbacks if view not in candidate_cameras
    ]
    if missing:
        raise KeyError(f"candidate cameras are missing selected views: {missing}")
    selected.update({view: candidate_cameras[view] for view in active})
    if not bool(full_band_commit):
        return selected, "track_supported_component"
    selected.update({view: candidate_cameras[view] for view in fallbacks})
    return (
        selected,
        "track_supported_component_plus_verified_world_right_fallback",
    )


def inherited_fallback_verification_passes(
    *,
    verification: Mapping[str, Any],
    fallback_views: Sequence[str],
) -> dict[str, Any]:
    """Require full-band acceptance and significant fallback/static improvement."""

    fallbacks = list(dict.fromkeys(str(view) for view in fallback_views))
    if not bool(verification.get("commit", False)):
        return {
            "accepted": False,
            "reason": "full_band_geometry_rejected",
            "fallback_views": fallbacks,
        }
    per_view = dict(verification.get("per_view_static_map") or {})
    rejected = []
    for view in fallbacks:
        confidence = dict((per_view.get(view) or {}).get("confidence") or {})
        lower_bound = float(
            confidence.get("gain_lower_bound_one_sided_95_px", -math.inf)
        )
        if (
            not bool(confidence.get("valid", False))
            or not math.isfinite(lower_bound)
            or lower_bound <= 0.0
        ):
            rejected.append(view)
    if rejected:
        return {
            "accepted": False,
            "reason": "fallback_static_geometry_not_significantly_better",
            "fallback_views": fallbacks,
            "rejected_fallback_views": rejected,
        }
    return {
        "accepted": True,
        "reason": "full_band_and_fallback_static_geometry_verified",
        "fallback_views": fallbacks,
    }


def fit_joint_band_candidate(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    cameras_init: Mapping[str, CameraParams],
    static_views: Sequence[str],
    point_ids: np.ndarray,
    max_steps: int = 80,
    device: str = "cuda",
    init_obs_filter_enable: bool = False,
    optimize_points: bool = True,
    static_only_triangulation: bool = False,
) -> tuple[dict[str, CameraParams], dict[str, Any]]:
    """Fit all non-static cameras together while holding static anchors fixed."""

    views = [str(view) for view in view_ids]
    fixed = [str(view) for view in static_views]
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError("tracks must have shape [view, frame, point, 2]")
    if visibility.shape != tracks.shape[:3]:
        raise ValueError("visibility shape differs from tracks")
    if tracks.shape[0] != len(views):
        raise ValueError("view_ids length differs from tracks")
    if not fixed or set(fixed) - set(views):
        raise ValueError("static_views must be a non-empty subset of view_ids")
    if set(views) - set(cameras_init):
        raise ValueError("cameras_init does not cover all track views")
    selected = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    if selected.size == 0 or len(np.unique(selected)) != selected.size:
        raise ValueError("point_ids must be non-empty and unique")
    if int(np.min(selected)) < 0 or int(np.max(selected)) >= tracks.shape[2]:
        raise IndexError("point_ids are outside the track tensor")
    tracks_selected = np.asarray(tracks[:, :, selected, :], dtype=np.float32)
    visibility_selected = np.asarray(visibility[:, :, selected], dtype=bool)
    return anchored_local_ba(
        tracks=tracks_selected,
        visibility=visibility_selected,
        view_ids=views,
        cameras_init={view: cameras_init[view] for view in views},
        ref_view=fixed[0],
        fixed_views=fixed,
        max_steps=int(max_steps),
        lr=5e-3,
        device=str(device),
        min_views_per_point=2,
        min_track_support_frames=1,
        triangulation_reproj_threshold=5.0,
        triangulation_views=fixed if static_only_triangulation else (),
        huber_delta=3.0,
        reg_weight=5e-4,
        optimize_points=bool(optimize_points),
        init_obs_filter_enable=bool(init_obs_filter_enable),
    )


def fit_component_conditioned_candidate(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    base_cameras: Mapping[str, CameraParams],
    precursor_cameras: Mapping[str, CameraParams],
    static_views: Sequence[str],
    component_views: Sequence[str],
    boundary_views: Sequence[str],
    point_ids: np.ndarray,
    max_steps: int = 80,
    device: str = "cuda",
    init_obs_filter_enable: bool = False,
    optimize_points: bool = True,
    static_only_triangulation: bool = False,
) -> tuple[dict[str, CameraParams], dict[str, Any]]:
    """Refit one commit component in the exact mixed-camera geometry.

    Static cameras and one-hop target boundaries remain at their base poses. Only
    component cameras inherit the precursor initialization and remain optimizable.
    """

    views = [str(view) for view in view_ids]
    static = list(dict.fromkeys(str(view) for view in static_views))
    component = list(dict.fromkeys(str(view) for view in component_views))
    boundary = list(dict.fromkeys(str(view) for view in boundary_views))
    if not static or not component:
        raise ValueError("static_views and component_views must be non-empty")
    if (
        (set(static) & set(component))
        or (set(component) & set(boundary))
        or (set(static) & set(boundary))
    ):
        raise ValueError("static, component, and boundary view sets must be disjoint")
    requested = set(static) | set(component) | set(boundary)
    missing = sorted(requested - set(views))
    if missing:
        raise ValueError(f"component-conditioned views are absent: {missing}")
    if requested - set(base_cameras):
        raise ValueError("base_cameras does not cover component-conditioned views")
    if set(component) - set(precursor_cameras):
        raise ValueError("precursor_cameras does not cover component views")

    selected_views = [view for view in views if view in requested]
    selected_indices = [views.index(view) for view in selected_views]
    fixed = [view for view in selected_views if view in set(static) | set(boundary)]
    cameras_init = {view: base_cameras[view] for view in selected_views}
    cameras_init.update({view: precursor_cameras[view] for view in component})
    selected = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    if selected.size == 0 or len(np.unique(selected)) != selected.size:
        raise ValueError("point_ids must be non-empty and unique")

    fitted, raw_stats = anchored_local_ba(
        tracks=np.asarray(tracks)[selected_indices][:, :, selected, :],
        visibility=np.asarray(visibility)[selected_indices][:, :, selected],
        view_ids=selected_views,
        cameras_init=cameras_init,
        ref_view=fixed[0],
        fixed_views=fixed,
        max_steps=int(max_steps),
        lr=5e-3,
        device=str(device),
        min_views_per_point=2,
        min_track_support_frames=1,
        triangulation_reproj_threshold=5.0,
        triangulation_views=static if static_only_triangulation else (),
        huber_delta=3.0,
        reg_weight=5e-4,
        optimize_points=bool(optimize_points),
        init_obs_filter_enable=bool(init_obs_filter_enable),
    )
    fitted = dict(fitted)
    fitted.update({view: base_cameras[view] for view in fixed})
    stats = {
        **dict(raw_stats),
        "protocol": "component-conditioned-exact-mixed-geometry-refit-v1",
        "selected_views": selected_views,
        "component_views": component,
        "boundary_views": boundary,
        "fixed_views": fixed,
        "fit_point_count": int(len(selected)),
        "fit_point_ids_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "requested_max_steps": int(max_steps),
        "static_only_triangulation": bool(static_only_triangulation),
    }
    return fitted, stats


def allocate_component_step_budget(
    *, total_steps: int, component_count: int
) -> list[int]:
    """Split a fixed per-frame refit budget without scene-dependent tuning."""

    count = int(component_count)
    if count < 0:
        raise ValueError("component_count must be non-negative")
    if count == 0:
        return []
    total = int(total_steps)
    if total <= 0:
        raise ValueError("component refit total_steps must be positive")
    if total < count:
        raise ValueError("component refit total_steps must cover every component")
    quotient, remainder = divmod(total, count)
    return [quotient + (index < remainder) for index in range(count)]


def _paired_confidence(
    base_values: Sequence[float], candidate_values: Sequence[float]
) -> dict[str, Any]:
    base = np.asarray(base_values, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_values, dtype=np.float64).reshape(-1)
    return paired_reprojection_confidence(
        {
            "success": bool(len(base) > 0),
            "truncated_reprojection_values_px": base.tolist(),
        },
        {
            "success": bool(len(candidate) > 0),
            "truncated_reprojection_values_px": candidate.tolist(),
        },
    )


def _supported_graph_connected(
    target_views: Sequence[str], supported_edges: Sequence[tuple[str, str]]
) -> bool:
    targets = [str(view) for view in target_views]
    if len(targets) <= 1:
        return True
    neighbors = {view: set() for view in targets}
    for left, right in supported_edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    visited = {targets[0]}
    frontier = [targets[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in neighbors[current] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    return len(visited) == len(targets)


def select_positive_static_track_components(
    *,
    verification: Mapping[str, Any],
    target_views: Sequence[str],
    allow_boundary_supported_singletons: bool = False,
) -> dict[str, Any]:
    """Find positive-evidence target components joined by heldout track edges."""

    targets = list(dict.fromkeys(str(view) for view in target_views))
    per_view = dict(verification.get("per_view_static_map") or {})
    positive = []
    for view in targets:
        confidence = dict((per_view.get(view) or {}).get("confidence") or {})
        lower = float(
            confidence.get("gain_lower_bound_one_sided_95_px", -math.inf)
        )
        if bool(confidence.get("valid", False)) and math.isfinite(lower) and lower > 0.0:
            positive.append(view)
    positive_set = set(positive)
    target_set = set(targets)
    neighbors = {view: set() for view in positive}
    boundary_supported_positive = set()
    induced_edges = []
    for raw_left, raw_right in verification.get("supported_target_edges") or []:
        left, right = str(raw_left), str(raw_right)
        if left == right or left not in target_set or right not in target_set:
            continue
        if left in positive_set and right not in positive_set:
            boundary_supported_positive.add(left)
        if right in positive_set and left not in positive_set:
            boundary_supported_positive.add(right)
        if left not in positive_set or right not in positive_set:
            continue
        neighbors[left].add(right)
        neighbors[right].add(left)
        induced_edges.append([left, right])

    components = []
    unseen = set(positive)
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        frontier = [root]
        component = []
        while frontier:
            current = frontier.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current] & unseen, reverse=True):
                unseen.remove(neighbor)
                frontier.append(neighbor)
        if len(component) >= 2 or (
            allow_boundary_supported_singletons
            and component[0] in boundary_supported_positive
        ):
            components.append(sorted(component))
    components.sort(key=lambda component: (-len(component), component))
    selected = {view for component in components for view in component}
    return {
        "protocol": "positive-static-evidence-track-components-v1",
        "positive_target_views": positive,
        "components": components,
        "selected_target_views": [view for view in targets if view in selected],
        "isolated_positive_target_views": [
            view for view in positive if view not in selected
        ],
        "induced_supported_edges": induced_edges,
        "allow_boundary_supported_singletons": bool(
            allow_boundary_supported_singletons
        ),
        "minimum_component_size": (
            1 if allow_boundary_supported_singletons else 2
        ),
        "minimum_component_rationale": (
            "candidate_internal_or_candidate_to_base_boundary_track_edge"
            if allow_boundary_supported_singletons
            else "at_least_one_candidate_candidate_target_track_edge"
        ),
        "gt_used": False,
        "nvs_used": False,
    }


def select_component_track_boundary(
    *,
    component: Sequence[str],
    target_views: Sequence[str],
    supported_target_edges: Sequence[Sequence[str]],
) -> list[str]:
    """Return one-hop target-track neighbors outside a selected component."""

    members = {str(view) for view in component}
    targets = list(dict.fromkeys(str(view) for view in target_views))
    target_set = set(targets)
    boundary = set()
    for raw_edge in supported_target_edges:
        if len(raw_edge) != 2:
            raise ValueError("supported target edges must contain two views")
        left, right = (str(raw_edge[0]), str(raw_edge[1]))
        if left not in target_set or right not in target_set:
            continue
        if left in members and right not in members:
            boundary.add(right)
        elif right in members and left not in members:
            boundary.add(left)
    return [view for view in targets if view in boundary]


def build_mixed_component_candidate(
    *,
    base_cameras: Mapping[str, Any],
    candidate_cameras: Mapping[str, Any],
    component: Sequence[str],
    boundary_views: Sequence[str],
    target_views: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    """Commit a component in-memory while retaining base poses on its boundary."""

    members = list(dict.fromkeys(str(view) for view in component))
    boundary = list(dict.fromkeys(str(view) for view in boundary_views))
    if not members or set(members) & set(boundary):
        raise ValueError("component must be non-empty and disjoint from its boundary")
    targets = list(dict.fromkeys(str(view) for view in target_views))
    verification_set = set(members) | set(boundary)
    missing_base = [view for view in targets if view not in base_cameras]
    missing_candidate = [view for view in members if view not in candidate_cameras]
    if missing_base or missing_candidate or not verification_set <= set(targets):
        raise KeyError(
            "mixed component cameras or verification targets are incomplete"
        )
    mixed = dict(base_cameras)
    mixed.update({view: candidate_cameras[view] for view in members})
    return mixed, [view for view in targets if view in verification_set]


def build_combined_component_candidate(
    *,
    base_cameras: Mapping[str, Any],
    component_candidates: Sequence[tuple[Sequence[str], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Merge disjoint component candidates into the exact export proposal."""

    combined = dict(base_cameras)
    claimed: set[str] = set()
    for raw_component, cameras in component_candidates:
        component = list(dict.fromkeys(str(view) for view in raw_component))
        if not component:
            raise ValueError("component candidate must be non-empty")
        overlap = claimed & set(component)
        if overlap:
            raise ValueError(f"component candidates overlap: {sorted(overlap)}")
        missing = set(component) - set(base_cameras)
        missing_candidate = set(component) - set(cameras)
        if missing or missing_candidate:
            raise KeyError("component candidate camera roster is incomplete")
        combined.update({view: cameras[view] for view in component})
        claimed.update(component)
    return combined


def component_final_verification_passes(
    *,
    verification: Mapping[str, Any],
    require_static_improvement_significance: bool = True,
) -> dict[str, Any]:
    """Accept a component only with independent aggregate gain and node safety."""

    if not bool(verification.get("supported_target_graph_connected", False)):
        return {"accepted": False, "reason": "final_target_graph_disconnected"}
    static_confidence = dict(verification.get("static_map_confidence") or {})
    static_lower = float(
        static_confidence.get("gain_lower_bound_one_sided_95_px", -math.inf)
    )
    static_upper = float(
        static_confidence.get("gain_upper_bound_one_sided_95_px", -math.inf)
    )
    if (
        not bool(static_confidence.get("valid", False))
        or not math.isfinite(static_lower)
        or not math.isfinite(static_upper)
    ):
        return {"accepted": False, "reason": "static_component_invalid"}
    if require_static_improvement_significance and static_lower <= 0.0:
        return {"accepted": False, "reason": "static_component_not_improved"}
    if not require_static_improvement_significance and static_upper < 0.0:
        return {
            "accepted": False,
            "reason": "static_component_significantly_worse",
        }
    rejected_static = []
    for view, payload in dict(verification.get("per_view_static_map") or {}).items():
        confidence = dict((payload or {}).get("confidence") or {})
        upper = float(
            confidence.get("gain_upper_bound_one_sided_95_px", -math.inf)
        )
        if (
            not bool(confidence.get("valid", False))
            or not math.isfinite(upper)
            or upper < 0.0
        ):
            rejected_static.append(str(view))
    if rejected_static:
        return {
            "accepted": False,
            "reason": "static_node_significantly_worse",
            "rejected_static_views": sorted(rejected_static),
        }
    pair_confidence = dict(verification.get("target_pair_confidence") or {})
    pair_lower = float(
        pair_confidence.get("gain_lower_bound_one_sided_95_px", -math.inf)
    )
    if (
        not bool(pair_confidence.get("valid", False))
        or not math.isfinite(pair_lower)
        or pair_lower <= 0.0
    ):
        return {"accepted": False, "reason": "target_pair_component_not_improved"}
    rejected_pair = []
    for view, confidence_payload in dict(
        verification.get("target_pair_confidence_by_view") or {}
    ).items():
        confidence = dict(confidence_payload or {})
        upper = float(
            confidence.get("gain_upper_bound_one_sided_95_px", -math.inf)
        )
        if (
            not bool(confidence.get("valid", False))
            or not math.isfinite(upper)
            or upper < 0.0
        ):
            rejected_pair.append(str(view))
    if rejected_pair:
        return {
            "accepted": False,
            "reason": "target_pair_node_significantly_worse",
            "rejected_pair_views": sorted(rejected_pair),
        }
    return {
        "accepted": True,
        "reason": (
            "independent_component_geometry_improved"
            if require_static_improvement_significance
            else "independent_component_geometry_confirmed"
        ),
        "require_static_improvement_significance": bool(
            require_static_improvement_significance
        ),
    }


def score_heldout_band_geometry(
    *,
    tracks: np.ndarray,
    visibility: np.ndarray,
    view_ids: Sequence[str],
    static_views: Sequence[str],
    target_views: Sequence[str],
    base_cameras: Mapping[str, CameraParams],
    candidate_cameras: Mapping[str, CameraParams],
    point_ids: np.ndarray,
) -> dict[str, Any]:
    """Verify a fitted band on points that were excluded from BA."""

    views = [str(view) for view in view_ids]
    static = [str(view) for view in static_views]
    targets = [str(view) for view in target_views]
    selected = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    if tracks.shape[1] != 1:
        raise ValueError("heldout band scoring expects one frame")
    if not targets or set(static + targets) - set(views):
        raise ValueError("static and target views must be present in the track tensor")
    config = SameTimePnPConfig(
        min_support_views=2,
        support_reprojection_error_px=2.5,
        pnp_reprojection_error_px=4.0,
        min_pnp_inliers=12,
        rng_seed=0,
    )
    support_cameras = {view: base_cameras[view] for view in static}
    static_base_values: list[float] = []
    static_candidate_values: list[float] = []
    per_view: dict[str, Any] = {}
    per_view_nonregression = True
    for target in targets:
        support_observations, target_observations, observation = build_node_observations(
            tracks=tracks,
            visibility=visibility,
            view_ids=views,
            static_views=static,
            target_view=target,
            frame_local=0,
            min_support_views=2,
            max_points=0,
            allowed_point_ids=selected,
        )
        scores = score_same_time_multiview_poses(
            support_cameras=support_cameras,
            support_observations=support_observations,
            target_observations=target_observations,
            target_cameras={
                "base": base_cameras[target],
                "candidate": candidate_cameras[target],
            },
            config=config,
        )
        base_score = scores["base"]
        candidate_score = scores["candidate"]
        valid = bool(base_score.get("success") and candidate_score.get("success"))
        base_values = np.asarray(
            base_score.get("truncated_reprojection_values_px", []), dtype=np.float64
        )
        candidate_values = np.asarray(
            candidate_score.get("truncated_reprojection_values_px", []),
            dtype=np.float64,
        )
        valid = bool(valid and len(base_values) == len(candidate_values) and len(base_values))
        base_mean = float(np.mean(base_values)) if valid else None
        candidate_mean = float(np.mean(candidate_values)) if valid else None
        if valid:
            static_base_values.extend(base_values.tolist())
            static_candidate_values.extend(candidate_values.tolist())
        view_confidence = _paired_confidence(base_values, candidate_values)
        if not valid or float(candidate_mean) > float(base_mean):
            per_view_nonregression = False
        per_view[target] = {
            "valid": valid,
            "observation": observation,
            "count": int(len(base_values)) if valid else 0,
            "base_mean_px": base_mean,
            "candidate_mean_px": candidate_mean,
            "gain_mean_px": (
                None if not valid else float(base_mean) - float(candidate_mean)
            ),
            "confidence": view_confidence,
        }

    view_to_index = {view: index for index, view in enumerate(views)}
    pair_base_values: list[float] = []
    pair_candidate_values: list[float] = []
    pair_rows: dict[str, Any] = {}
    supported_edges: list[tuple[str, str]] = []
    pair_base_values_by_view = {target: [] for target in targets}
    pair_candidate_values_by_view = {target: [] for target in targets}
    cap = 4.0
    for left, right in combinations(targets, 2):
        left_index = view_to_index[left]
        right_index = view_to_index[right]
        valid = (
            visibility[left_index, 0, selected]
            & visibility[right_index, 0, selected]
            & np.isfinite(tracks[left_index, 0, selected]).all(axis=1)
            & np.isfinite(tracks[right_index, 0, selected]).all(axis=1)
        )
        shared_ids = selected[valid]
        key = f"{left}:{right}"
        if len(shared_ids) < 8:
            pair_rows[key] = {
                "supported": False,
                "shared_point_count": int(len(shared_ids)),
            }
            continue
        left_points = np.asarray(tracks[left_index, 0, shared_ids], dtype=np.float64)
        right_points = np.asarray(tracks[right_index, 0, shared_ids], dtype=np.float64)
        base_errors = sampson_distances_px(
            target_camera=base_cameras[left],
            support_camera=base_cameras[right],
            target_points=left_points,
            support_points=right_points,
        )
        candidate_errors = sampson_distances_px(
            target_camera=candidate_cameras[left],
            support_camera=candidate_cameras[right],
            target_points=left_points,
            support_points=right_points,
        )
        pair_valid = bool(
            len(base_errors) == len(candidate_errors) == len(shared_ids)
            and np.isfinite(base_errors).all()
            and np.isfinite(candidate_errors).all()
        )
        if not pair_valid:
            pair_rows[key] = {
                "supported": False,
                "shared_point_count": int(len(shared_ids)),
                "reason": "nonfinite_pair_score",
            }
            continue
        base_values = np.minimum(base_errors, cap)
        candidate_values = np.minimum(candidate_errors, cap)
        supported_edges.append((left, right))
        pair_base_values.extend(base_values.tolist())
        pair_candidate_values.extend(candidate_values.tolist())
        for target in (left, right):
            pair_base_values_by_view[target].extend(base_values.tolist())
            pair_candidate_values_by_view[target].extend(candidate_values.tolist())
        pair_rows[key] = {
            "supported": True,
            "shared_point_count": int(len(shared_ids)),
            "base_mean_px": float(np.mean(base_values)),
            "candidate_mean_px": float(np.mean(candidate_values)),
            "gain_mean_px": float(np.mean(base_values - candidate_values)),
        }

    static_confidence = _paired_confidence(
        static_base_values, static_candidate_values
    )
    pair_confidence = _paired_confidence(pair_base_values, pair_candidate_values)
    pair_confidence_by_view = {
        target: _paired_confidence(
            pair_base_values_by_view[target], pair_candidate_values_by_view[target]
        )
        for target in targets
    }
    graph_connected = _supported_graph_connected(targets, supported_edges)
    static_map_not_significantly_worse = bool(
        static_confidence.get("valid", False)
        and float(static_confidence["gain_upper_bound_one_sided_95_px"]) >= 0.0
    )
    target_pair_node_nonregression = bool(
        all(
            confidence.get("valid", False)
            and float(confidence["gain_upper_bound_one_sided_95_px"]) >= 0.0
            for confidence in pair_confidence_by_view.values()
        )
    )
    commit = bool(
        graph_connected
        and static_map_not_significantly_worse
        and pair_confidence.get("valid", False)
        and float(pair_confidence["gain_lower_bound_one_sided_95_px"]) > 0.0
        and target_pair_node_nonregression
    )
    return {
        "protocol": "cross-dynamic-band-heldout-dual-geometry-v2",
        "commit": commit,
        "static_map_confidence": static_confidence,
        "target_pair_confidence": pair_confidence,
        "per_view_static_map": per_view,
        "per_view_static_map_nonregression": bool(per_view_nonregression),
        "static_map_not_significantly_worse": static_map_not_significantly_worse,
        "target_pair_confidence_by_view": pair_confidence_by_view,
        "target_pair_node_nonregression": target_pair_node_nonregression,
        "supported_target_graph_connected": bool(graph_connected),
        "supported_target_edges": [list(edge) for edge in supported_edges],
        "target_pair_rows": pair_rows,
        "gt_used": False,
        "nvs_used": False,
    }


def _parse_views(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _parse_frames(value: str) -> list[int]:
    return sorted({int(item) for item in value.replace(",", " ").split()})


def validate_joint_band_track_builder(meta: Mapping[str, Any]) -> str:
    """Accept only shared target tracks without transitive relay merges."""

    builder = str(meta.get("seed_builder", ""))
    if builder in {"spsg_target_closure", "spsg_target_verified_closure"}:
        return builder
    if builder != "spsg_target_relay":
        raise RuntimeError(
            "joint band BA requires target-closure, geometry-verified "
            "target-closure, or direct target-relay tracks"
        )
    diagnostics = meta.get("seed_builder_diag_by_seed")
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise RuntimeError("target-relay tracks lack per-seed builder diagnostics")
    valid = all(
        isinstance(row, Mapping)
        and row.get("direct_support_policy")
        == "focus_centered_direct_edges_only"
        and int(row.get("transitive_merge_count", -1)) == 0
        for row in diagnostics.values()
    )
    if not valid:
        raise RuntimeError(
            "joint band BA requires direct nontransitive target-relay tracks"
        )
    return builder


def run_closure(
    *,
    runtime_scene: Path,
    track_root: Path,
    track_freeze: Path,
    base_pose_root: Path,
    base_prediction_attestation: Path,
    output_root: Path,
    static_views: Sequence[str],
    target_views: Sequence[str],
    frames: Sequence[int],
    max_steps: int,
    device: str,
    init_obs_filter_enable: bool = False,
    candidate_optimize_points: bool = True,
    component_verified_commit: bool = False,
    component_boundary_verify: bool = False,
    component_conditioned_refit: bool = False,
    component_refit_total_steps: int = 0,
    static_only_triangulation: bool = False,
    code_manifest: Path | None = None,
    component_require_static_improvement_significance: bool = True,
) -> dict[str, Any]:
    runtime = runtime_scene.resolve()
    tracks_root = track_root.resolve()
    base_root = base_pose_root.resolve()
    output = output_root.resolve()
    static = [str(view) for view in static_views]
    targets = [str(view) for view in target_views]
    selected_frames = [int(frame) for frame in frames]
    if component_boundary_verify and not component_verified_commit:
        raise ValueError("component boundary verification requires component commit")
    if component_conditioned_refit and not (
        component_verified_commit and component_boundary_verify
    ):
        raise ValueError(
            "component-conditioned refit requires component and boundary verification"
        )
    if component_conditioned_refit and int(component_refit_total_steps) <= 0:
        raise ValueError("component-conditioned refit requires a positive refit budget")
    if not component_conditioned_refit and int(component_refit_total_steps) != 0:
        raise ValueError("component refit budget requires component-conditioned refit")
    if component_conditioned_refit and code_manifest is None:
        raise ValueError("component-conditioned refit requires a frozen code manifest")
    if output.exists():
        raise FileExistsError(output)
    if not static or not targets or set(static) & set(targets):
        raise ValueError("static and target view sets must be non-empty and disjoint")
    if not selected_frames:
        raise ValueError("at least one frame is required")

    code_lineage = (
        verify_code_freeze(
            manifest_path=Path(code_manifest),
            code_root=PROJECT_ROOT,
        )
        if code_manifest is not None
        else None
    )

    runtime_manifest_path = runtime / "runtime_manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    if runtime_manifest.get("gt_materialized") or runtime_manifest.get(
        "heldout_images_materialized"
    ):
        raise RuntimeError("runtime materialized a forbidden readout")
    expected_views = static + targets
    track_hashes = verify_track_freeze(
        freeze_path=track_freeze, track_root=tracks_root
    )
    meta = json.loads((tracks_root / "meta.json").read_text(encoding="utf-8"))
    track_builder = validate_joint_band_track_builder(meta)
    views = (tracks_root / "views.txt").read_text(encoding="utf-8").split()
    if views != expected_views:
        raise RuntimeError("track view roster differs from static plus target views")
    tracks = np.load(tracks_root / "tracks.npy", mmap_mode="r")
    visibility = np.load(tracks_root / "visibility.npy", mmap_mode="r").astype(bool)
    if tracks.shape[:3] != visibility.shape or tracks.shape[0] != len(views):
        raise RuntimeError("track tensor metadata mismatch")
    effective_visibility = visibility & np.isfinite(tracks).all(axis=-1)
    start_frame = int(meta["start_frame"])
    num_frames = int(meta["num_frames"])
    if any(
        frame < start_frame or frame >= start_frame + num_frames
        for frame in selected_frames
    ):
        raise ValueError("requested frame lies outside the track bundle")
    base_lineage = _verify_base_attestation(
        attestation_path=base_prediction_attestation,
        base_pose_root=base_root,
        frames=selected_frames,
    )

    output.mkdir(parents=True)
    pose_root = output / "pose_cross_dynamic_band_ba"
    candidate_pose_root = output / "candidate_cross_dynamic_band_ba"
    component_candidate_pose_root = output / "candidate_component_conditioned"
    combined_candidate_pose_root = output / "candidate_component_conditioned_combined"
    rows: list[dict[str, Any]] = []
    output_camera_hashes: dict[str, dict[str, str]] = {}
    precursor_candidate_camera_hashes: dict[str, dict[str, str]] = {}
    component_candidate_camera_hashes: dict[str, dict[str, dict[str, str]]] = {}
    combined_candidate_camera_hashes: dict[str, dict[str, str]] = {}
    for frame in selected_frames:
        frame_name = f"f{frame:06d}"
        frame_local = frame - start_frame
        base_camera_root = base_root / frame_name / "cameras_pred"
        base_cameras = CameraLoader(base_camera_root).load_cameras(expected_views)
        tracks_frame = np.asarray(
            tracks[:, frame_local : frame_local + 1, :, :], dtype=np.float32
        )
        visibility_frame = np.asarray(
            effective_visibility[:, frame_local : frame_local + 1, :], dtype=bool
        )
        if component_verified_commit:
            fit_ids, selector_ids, final_verify_ids = (
                build_visibility_stratified_fit_select_verify_folds(
                    meta=meta,
                    visibility=effective_visibility,
                    view_ids=views,
                    target_views=targets,
                    frame_local=frame_local,
                )
            )
        else:
            fit_ids, final_verify_ids = build_visibility_stratified_folds(
                meta=meta,
                visibility=effective_visibility,
                view_ids=views,
                target_views=targets,
                frame_local=frame_local,
            )
            selector_ids = []
        band_selection = select_supported_target_band(
            visibility=visibility_frame,
            view_ids=views,
            target_views=targets,
            point_ids=np.asarray(fit_ids, dtype=np.int64),
            min_target_observations=24,
            min_shared_observations=8,
        )
        active_targets = list(band_selection["active_target_views"])
        fallback_targets = list(band_selection["fallback_target_views"])
        candidate_cameras = dict(base_cameras)
        ba_stats: dict[str, Any] = {
            "accepted": False,
            "reason": "insufficient_crossfit_points",
        }
        verification: dict[str, Any] = {
            "commit": False,
            "reason": "insufficient_crossfit_points",
            "gt_used": False,
            "nvs_used": False,
        }
        selector_verification: dict[str, Any] = {
            "commit": False,
            "reason": "component_selector_not_enabled",
            "gt_used": False,
            "nvs_used": False,
        }
        component_selection: dict[str, Any] = {
            "protocol": "positive-static-evidence-track-components-v1",
            "positive_target_views": [],
            "components": [],
            "selected_target_views": [],
            "isolated_positive_target_views": [],
            "induced_supported_edges": [],
            "allow_boundary_supported_singletons": bool(
                component_boundary_verify
            ),
            "minimum_component_size": 1 if component_boundary_verify else 2,
            "minimum_component_rationale": (
                "candidate_internal_or_candidate_to_base_boundary_track_edge"
                if component_boundary_verify
                else "at_least_one_candidate_candidate_target_track_edge"
            ),
            "gt_used": False,
            "nvs_used": False,
        }
        component_verifications: list[dict[str, Any]] = []
        committed_components: list[list[str]] = []
        committed_component_camera_updates: dict[str, CameraParams] = {}
        provisional_component_candidates: list[dict[str, Any]] = []
        combined_export_verifications: list[dict[str, Any]] = []
        combined_export_gate: dict[str, Any] = {
            "accepted": False,
            "reason": "no_provisional_component",
        }
        component_candidate_camera_hashes[frame_name] = {}
        verification_budget_ready = bool(
            len(final_verify_ids) >= 8
            and (
                not component_verified_commit
                or len(selector_ids) >= 8
            )
        )
        if len(fit_ids) >= 8 and verification_budget_ready and active_targets:
            fit_views = static + active_targets
            fit_view_indices = [views.index(view) for view in fit_views]
            fitted_cameras, ba_stats = fit_joint_band_candidate(
                tracks=tracks_frame[fit_view_indices],
                visibility=visibility_frame[fit_view_indices],
                view_ids=fit_views,
                cameras_init={view: base_cameras[view] for view in fit_views},
                static_views=static,
                point_ids=np.asarray(fit_ids, dtype=np.int64),
                max_steps=int(max_steps),
                device=str(device),
                init_obs_filter_enable=bool(init_obs_filter_enable),
                optimize_points=bool(candidate_optimize_points),
                static_only_triangulation=bool(static_only_triangulation),
            )
            candidate_cameras.update(
                {view: fitted_cameras[view] for view in active_targets}
            )
            if component_verified_commit:
                selector_verification = score_heldout_band_geometry(
                    tracks=tracks_frame,
                    visibility=visibility_frame,
                    view_ids=views,
                    static_views=static,
                    target_views=active_targets,
                    base_cameras=base_cameras,
                    candidate_cameras=candidate_cameras,
                    point_ids=np.asarray(selector_ids, dtype=np.int64),
                )
                verification = selector_verification
                if bool(ba_stats.get("accepted", False)):
                    component_selection = select_positive_static_track_components(
                        verification=selector_verification,
                        target_views=active_targets,
                        allow_boundary_supported_singletons=bool(
                            component_boundary_verify
                        ),
                    )
                    component_step_budgets = allocate_component_step_budget(
                        total_steps=int(component_refit_total_steps),
                        component_count=len(component_selection["components"]),
                    ) if component_conditioned_refit else [
                        0 for _component in component_selection["components"]
                    ]
                    for component_index, component in enumerate(
                        component_selection["components"]
                    ):
                        boundary_views = (
                            select_component_track_boundary(
                                component=component,
                                target_views=active_targets,
                                supported_target_edges=selector_verification.get(
                                    "supported_target_edges", []
                                ),
                            )
                            if component_boundary_verify
                            else []
                        )
                        component_fit_stats: dict[str, Any] = {
                            "accepted": bool(ba_stats.get("accepted", False)),
                            "reason": "whole_band_precursor_reused",
                            "protocol": "whole-band-precursor-splice-v1",
                        }
                        component_candidate_output = None
                        if component_conditioned_refit:
                            component_fitted, component_fit_stats = (
                                fit_component_conditioned_candidate(
                                    tracks=tracks_frame,
                                    visibility=visibility_frame,
                                    view_ids=views,
                                    base_cameras=base_cameras,
                                    precursor_cameras=candidate_cameras,
                                    static_views=static,
                                    component_views=component,
                                    boundary_views=boundary_views,
                                    point_ids=np.asarray(fit_ids, dtype=np.int64),
                                    max_steps=int(
                                        component_step_budgets[component_index]
                                    ),
                                    device=str(device),
                                    init_obs_filter_enable=bool(
                                        init_obs_filter_enable
                                    ),
                                    optimize_points=bool(candidate_optimize_points),
                                    static_only_triangulation=bool(
                                        static_only_triangulation
                                    ),
                                )
                            )
                            final_candidate_cameras = dict(base_cameras)
                            final_candidate_cameras.update(
                                {
                                    view: component_fitted[view]
                                    for view in component
                                }
                            )
                            final_target_views = [
                                view
                                for view in active_targets
                                if view in set(component) | set(boundary_views)
                            ]
                            component_name = f"component_{component_index:03d}"
                            component_candidate_output = (
                                component_candidate_pose_root
                                / frame_name
                                / component_name
                                / "cameras_pred"
                            )
                            save_cameras(
                                component_candidate_output,
                                final_candidate_cameras,
                            )
                            component_candidate_camera_hashes[frame_name][
                                component_name
                            ] = _camera_hashes(component_candidate_output)
                        elif component_boundary_verify:
                            final_candidate_cameras, final_target_views = (
                                build_mixed_component_candidate(
                                    base_cameras=base_cameras,
                                    candidate_cameras=candidate_cameras,
                                    component=component,
                                    boundary_views=boundary_views,
                                    target_views=active_targets,
                                )
                            )
                        else:
                            final_candidate_cameras = candidate_cameras
                            final_target_views = component
                        if bool(component_fit_stats.get("accepted", False)):
                            final_verification = score_heldout_band_geometry(
                                tracks=tracks_frame,
                                visibility=visibility_frame,
                                view_ids=views,
                                static_views=static,
                                target_views=final_target_views,
                                base_cameras=base_cameras,
                                candidate_cameras=final_candidate_cameras,
                                point_ids=np.asarray(
                                    final_verify_ids, dtype=np.int64
                                ),
                            )
                            gate = component_final_verification_passes(
                                verification=final_verification,
                                require_static_improvement_significance=bool(
                                    component_require_static_improvement_significance
                                ),
                            )
                        else:
                            final_verification = {
                                "commit": False,
                                "reason": "component_conditioned_refit_rejected",
                                "gt_used": False,
                                "nvs_used": False,
                            }
                            gate = {
                                "accepted": False,
                                "reason": "component_conditioned_refit_rejected",
                            }
                        component_verifications.append(
                            {
                                "target_views": component,
                                "boundary_target_views": boundary_views,
                                "verification_target_views": final_target_views,
                                "candidate_policy": (
                                    "component_conditioned_refit_plus_static_pnp_boundary"
                                    if component_conditioned_refit
                                    else (
                                        "candidate_component_plus_static_pnp_boundary"
                                        if component_boundary_verify
                                        else "candidate_component_only"
                                    )
                                ),
                                "component_fit": component_fit_stats,
                                "component_candidate_output": (
                                    str(component_candidate_output)
                                    if component_candidate_output is not None
                                    else None
                                ),
                                "verification": final_verification,
                                "gate": gate,
                            }
                        )
                        if bool(gate["accepted"]):
                            if component_conditioned_refit:
                                provisional_component_candidates.append(
                                    {
                                        "target_views": list(component),
                                        "boundary_target_views": list(boundary_views),
                                        "verification_target_views": list(
                                            final_target_views
                                        ),
                                        "cameras": final_candidate_cameras,
                                    }
                                )
                            else:
                                committed_components.append(component)
                                committed_component_camera_updates.update(
                                    {
                                        view: final_candidate_cameras[view]
                                        for view in component
                                    }
                                )
            else:
                verification = score_heldout_band_geometry(
                    tracks=tracks_frame,
                    visibility=visibility_frame,
                    view_ids=views,
                    static_views=static,
                    target_views=active_targets,
                    base_cameras=base_cameras,
                    candidate_cameras=candidate_cameras,
                    point_ids=np.asarray(final_verify_ids, dtype=np.int64),
                )
        combined_candidate_cameras = dict(base_cameras)
        if component_conditioned_refit and provisional_component_candidates:
            combined_candidate_cameras = build_combined_component_candidate(
                base_cameras=base_cameras,
                component_candidates=[
                    (record["target_views"], record["cameras"])
                    for record in provisional_component_candidates
                ],
            )
            combined_ok = True
            for record in provisional_component_candidates:
                exact_verification = score_heldout_band_geometry(
                    tracks=tracks_frame,
                    visibility=visibility_frame,
                    view_ids=views,
                    static_views=static,
                    target_views=record["verification_target_views"],
                    base_cameras=base_cameras,
                    candidate_cameras=combined_candidate_cameras,
                    point_ids=np.asarray(final_verify_ids, dtype=np.int64),
                )
                exact_gate = component_final_verification_passes(
                    verification=exact_verification,
                    require_static_improvement_significance=bool(
                        component_require_static_improvement_significance
                    ),
                )
                combined_export_verifications.append(
                    {
                        "target_views": record["target_views"],
                        "boundary_target_views": record[
                            "boundary_target_views"
                        ],
                        "verification_target_views": record[
                            "verification_target_views"
                        ],
                        "verification": exact_verification,
                        "gate": exact_gate,
                    }
                )
                combined_ok = bool(combined_ok and exact_gate["accepted"])
            combined_export_gate = {
                "accepted": bool(combined_ok),
                "reason": (
                    "exact_combined_export_candidate_verified"
                    if combined_ok
                    else "exact_combined_export_candidate_rejected"
                ),
                "component_count": len(provisional_component_candidates),
            }
            if combined_ok:
                committed_components = [
                    list(record["target_views"])
                    for record in provisional_component_candidates
                ]
                committed_component_camera_updates = {
                    view: combined_candidate_cameras[view]
                    for component in committed_components
                    for view in component
                }
        commit = bool(
            ba_stats.get("accepted", False)
            and (
                bool(committed_components)
                if component_verified_commit
                else verification["commit"]
            )
        )
        inheritance: dict[str, Any] = {
            "success": False,
            "reason": "active_component_not_committed",
            "fallback_views": fallback_targets,
            "inherited_fallback_count": 0,
        }
        full_band_verification: dict[str, Any] = {
            "commit": False,
            "reason": "fallback_inheritance_not_attempted",
            "gt_used": False,
            "nvs_used": False,
        }
        fallback_gate: dict[str, Any] = {
            "accepted": False,
            "reason": "fallback_inheritance_not_attempted",
            "fallback_views": fallback_targets,
        }
        full_band_commit = False
        if commit and fallback_targets and not component_verified_commit:
            candidate_cameras, inheritance = (
                build_world_right_fallback_candidate(
                    base_cameras=base_cameras,
                    candidate_cameras=candidate_cameras,
                    contributor_views=active_targets,
                    fallback_views=fallback_targets,
                )
            )
            if bool(inheritance["success"]):
                full_band_verification = score_heldout_band_geometry(
                    tracks=tracks_frame,
                    visibility=visibility_frame,
                    view_ids=views,
                    static_views=static,
                    target_views=targets,
                    base_cameras=base_cameras,
                    candidate_cameras=candidate_cameras,
                    point_ids=np.asarray(final_verify_ids, dtype=np.int64),
                )
                fallback_gate = inherited_fallback_verification_passes(
                    verification=full_band_verification,
                    fallback_views=fallback_targets,
                )
                full_band_commit = bool(fallback_gate["accepted"])
        if component_verified_commit:
            committed_target_views = [
                view for component in committed_components for view in component
            ]
            if len(set(committed_target_views)) != len(committed_target_views):
                raise RuntimeError("committed target components overlap")
            selected_cameras = dict(base_cameras)
            selected_cameras.update(committed_component_camera_updates)
            source = (
                (
                    "track_component_conditioned_exact_mixed_geometry"
                    if component_conditioned_refit
                    else (
                        "track_connected_positive_geometry_components_with_static_boundary"
                        if component_boundary_verify
                        else "track_connected_positive_geometry_components"
                    )
                )
                if committed_target_views
                else "static_map_same_frame_pnp"
            )
        else:
            committed_target_views = active_targets if commit else []
            selected_cameras, source = select_verified_band_cameras(
                base_cameras=base_cameras,
                candidate_cameras=candidate_cameras,
                active_views=active_targets,
                fallback_views=fallback_targets,
                active_commit=commit,
                full_band_commit=full_band_commit,
            )
        candidate_frame_output = (
            candidate_pose_root / frame_name / "cameras_pred"
        )
        save_cameras(candidate_frame_output, candidate_cameras)
        precursor_candidate_camera_hashes[frame_name] = _camera_hashes(
            candidate_frame_output
        )
        if component_conditioned_refit:
            combined_frame_output = (
                combined_candidate_pose_root / frame_name / "cameras_pred"
            )
            save_cameras(combined_frame_output, combined_candidate_cameras)
            combined_candidate_camera_hashes[frame_name] = _camera_hashes(
                combined_frame_output
            )
        frame_output = pose_root / frame_name / "cameras_pred"
        save_cameras(frame_output, selected_cameras)
        output_camera_hashes[frame_name] = _camera_hashes(frame_output)
        rows.append(
            {
                "frame": frame,
                "source": source,
                "commit": commit,
                "active_component_commit": commit,
                "full_band_inheritance_commit": full_band_commit,
                "active_target_views": active_targets,
                "fallback_target_views": fallback_targets,
                "committed_target_views": committed_target_views,
                "committed_fallback_target_views": (
                    fallback_targets if full_band_commit else []
                ),
                "band_selection": band_selection,
                "fit_point_count": len(fit_ids),
                "selector_point_count": len(selector_ids),
                "final_verify_point_count": len(final_verify_ids),
                "verify_point_count": len(final_verify_ids),
                "fit_point_ids_sha256": hashlib.sha256(
                    np.asarray(fit_ids, dtype=np.int64).tobytes()
                ).hexdigest(),
                "selector_point_ids_sha256": hashlib.sha256(
                    np.asarray(selector_ids, dtype=np.int64).tobytes()
                ).hexdigest(),
                "final_verify_point_ids_sha256": hashlib.sha256(
                    np.asarray(final_verify_ids, dtype=np.int64).tobytes()
                ).hexdigest(),
                "verify_point_ids_sha256": hashlib.sha256(
                    np.asarray(final_verify_ids, dtype=np.int64).tobytes()
                ).hexdigest(),
                "ba_stats": ba_stats,
                "verification": verification,
                "selector_verification": selector_verification,
                "component_selection": component_selection,
                "component_verifications": component_verifications,
                "combined_export_verifications": combined_export_verifications,
                "combined_export_gate": combined_export_gate,
                "committed_components": committed_components,
                "component_conditioned_refit": bool(component_conditioned_refit),
                "optimizer_step_budget": {
                    "precursor_steps": int(max_steps),
                    "component_refit_total_steps": (
                        int(component_refit_total_steps)
                        if component_conditioned_refit
                        else 0
                    ),
                    "component_refit_allocated_steps": (
                        component_step_budgets
                        if component_conditioned_refit
                        and bool(ba_stats.get("accepted", False))
                        else []
                    ),
                    "maximum_total_steps": int(max_steps)
                    + (
                        int(component_refit_total_steps)
                        if component_conditioned_refit
                        else 0
                    ),
                },
                "world_right_fallback_inheritance": inheritance,
                "full_band_verification": full_band_verification,
                "fallback_verification_gate": fallback_gate,
            }
        )

    manifest = {
        "protocol": (
            (
                "cross-dynamic-track-component-conditioned-pose-ba-closure-v5"
                if component_conditioned_refit
                else (
                    "cross-dynamic-track-component-boundary-pose-ba-closure-v4"
                    if component_boundary_verify
                    else "cross-dynamic-track-component-pose-ba-closure-v3"
                )
            )
            if component_verified_commit
            else "cross-dynamic-track-band-joint-ba-closure-v2"
        ),
        "runtime_scene": str(runtime),
        "runtime_manifest": str(runtime_manifest_path),
        "runtime_manifest_sha256": _sha256(runtime_manifest_path),
        "track_root": str(tracks_root),
        "track_builder": track_builder,
        "track_freeze": str(track_freeze.resolve()),
        "track_freeze_sha256": _sha256(track_freeze.resolve()),
        "track_file_sha256": track_hashes,
        "code_lineage": code_lineage,
        "base_pose_root": str(base_root),
        "base_lineage": base_lineage,
        "pose_root": str(pose_root),
        "candidate_pose_root": str(
            combined_candidate_pose_root
            if component_conditioned_refit
            else candidate_pose_root
        ),
        "precursor_candidate_pose_root": str(candidate_pose_root),
        "combined_candidate_pose_root": (
            str(combined_candidate_pose_root)
            if component_conditioned_refit
            else None
        ),
        "component_candidate_pose_root": (
            str(component_candidate_pose_root)
            if component_conditioned_refit
            else None
        ),
        "frames": selected_frames,
        "static_views": static,
        "target_views": targets,
        "method": {
            "candidate": (
                "same-frame-supported-target-band-joint-ba"
                if candidate_optimize_points
                else "same-frame-supported-target-band-pose-only-ba"
            ),
            "fixed_views": "all observed static views",
            "point_model": "per-frame triangulated shared tracks",
            "optimize_points": bool(candidate_optimize_points),
            "triangulation_policy": (
                "fixed-static-views-only"
                if static_only_triangulation
                else "all-observed-views"
            ),
            "fit_fold": 0,
            "selector_fold": (
                "heldout-0" if component_verified_commit else None
            ),
            "final_verify_fold": (
                "heldout-1" if component_verified_commit else 1
            ),
            "verify_fold": 1 if not component_verified_commit else None,
            "fold_split": (
                "fixed-fit-half-plus-nested-selector-final-heldout-quarters"
                if component_verified_commit
                else "seed-and-target-visibility-stratified"
            ),
            "commit_protocol": (
                (
                    "nested-heldout-component-conditioned-exact-mixed-geometry-v3"
                    if component_conditioned_refit
                    else (
                        "nested-heldout-positive-static-mixed-boundary-component-v2"
                        if component_boundary_verify
                        else "nested-heldout-positive-static-track-component-v1"
                    )
                )
                if component_verified_commit
                else "whole-supported-band-dual-geometry-v2"
            ),
            "component_boundary_verify": bool(component_boundary_verify),
            "component_conditioned_refit": bool(component_conditioned_refit),
            "component_refit_fixed_views": (
                "all static cameras plus one-hop Static+PnP target boundary"
                if component_conditioned_refit
                else None
            ),
            "component_refit_fold": 0 if component_conditioned_refit else None,
            "component_require_static_improvement_significance": bool(
                component_require_static_improvement_significance
            ),
            "precursor_max_steps": int(max_steps),
            "component_refit_total_steps_per_frame": (
                int(component_refit_total_steps)
                if component_conditioned_refit
                else 0
            ),
            "maximum_optimizer_steps_per_frame": int(max_steps)
            + (
                int(component_refit_total_steps)
                if component_conditioned_refit
                else 0
            ),
            "component_refit_budget_allocation": (
                "equal integer split across selector components; remainder to roster order"
                if component_conditioned_refit
                else None
            ),
            "max_steps": int(max_steps),
            "init_obs_filter_enable": bool(init_obs_filter_enable),
            "min_target_observations": 24,
            "min_shared_target_observations": 8,
            "lr": 0.005,
            "min_views_per_point": 2,
            "min_track_support_frames": 1,
            "triangulation_reprojection_px": 5.0,
            "verification": (
                (
                    "selector-static-map-lower95-positive-plus-track-connected-"
                    + (
                        "candidate-to-static-pnp-one-hop-boundary; "
                        if component_boundary_verify
                        else "component; "
                    )
                    + "independent-final-static-and-target-pair-"
                    "aggregate-lower95-positive-plus-node-upper95-nonnegative"
                )
                if component_verified_commit
                else (
                    "heldout-static-map-confidence95-plus-target-pair-confidence95-"
                    "plus-target-pair-node-nonregression"
                )
            ),
            "fallback_inheritance": {
                "model": "robust common world-right rotation from active BA pose pairs",
                "equation": "R_candidate = R_base Q",
                "minimum_contributor_views": 3,
                "center_policy": "preserve fallback Static+PnP camera center",
                "fit_source": "active target cameras accepted on fit-fold BA",
                "verification": (
                    "exact inherited candidate on heldout fold; full target graph and "
                    "aggregate target-pair gain plus no significant fallback/static "
                    "regression"
                ),
                "extra_tunable_thresholds": 0,
            },
        },
        "rows": rows,
        "track_commit_frame_count": sum(bool(row["commit"]) for row in rows),
        "track_commit_target_count": sum(
            len(row["committed_target_views"])
            + len(row["committed_fallback_target_views"])
            for row in rows
        ),
        "full_band_inheritance_commit_frame_count": sum(
            bool(row["full_band_inheritance_commit"]) for row in rows
        ),
        "base_fallback_frame_count": sum(not bool(row["commit"]) for row in rows),
        "output_camera_hashes": output_camera_hashes,
        "candidate_camera_hashes": (
            combined_candidate_camera_hashes
            if component_conditioned_refit
            else precursor_candidate_camera_hashes
        ),
        "precursor_candidate_camera_hashes": precursor_candidate_camera_hashes,
        "component_candidate_camera_hashes": component_candidate_camera_hashes,
        "gt_used": False,
        "nvs_used": False,
        "heldout_used": False,
    }
    manifest_path = output / "closure_manifest.json"
    _write_json(manifest_path, manifest)
    freeze = {
        "protocol": (
            (
                "cross-dynamic-track-component-conditioned-pose-ba-public-output-freeze-v5"
                if component_conditioned_refit
                else (
                    "cross-dynamic-track-component-boundary-pose-ba-public-output-freeze-v4"
                    if component_boundary_verify
                    else "cross-dynamic-track-component-pose-ba-public-output-freeze-v3"
                )
            )
            if component_verified_commit
            else "cross-dynamic-track-band-joint-ba-public-output-freeze-v2"
        ),
        "status": "frozen_before_pose_gt_or_nvs",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "pose_root": str(pose_root),
        "candidate_pose_root": str(
            combined_candidate_pose_root
            if component_conditioned_refit
            else candidate_pose_root
        ),
        "precursor_candidate_pose_root": str(candidate_pose_root),
        "combined_candidate_pose_root": (
            str(combined_candidate_pose_root)
            if component_conditioned_refit
            else None
        ),
        "component_candidate_pose_root": (
            str(component_candidate_pose_root)
            if component_conditioned_refit
            else None
        ),
        "output_camera_hashes": output_camera_hashes,
        "candidate_camera_hashes": (
            combined_candidate_camera_hashes
            if component_conditioned_refit
            else precursor_candidate_camera_hashes
        ),
        "precursor_candidate_camera_hashes": precursor_candidate_camera_hashes,
        "component_candidate_camera_hashes": component_candidate_camera_hashes,
        "gt_used": False,
        "nvs_used": False,
        "heldout_used": False,
    }
    _write_json(output / "PUBLIC_OUTPUT_FREEZE.json", freeze)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--track-freeze", type=Path, required=True)
    parser.add_argument("--base-pose-root", type=Path, required=True)
    parser.add_argument("--base-prediction-attestation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--static-views", type=_parse_views, required=True)
    parser.add_argument("--target-views", type=_parse_views, required=True)
    parser.add_argument("--frames", type=_parse_frames, required=True)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-obs-filter", action="store_true")
    parser.add_argument("--pose-only-candidate", action="store_true")
    parser.add_argument("--component-verified-commit", action="store_true")
    parser.add_argument("--component-boundary-verify", action="store_true")
    parser.add_argument("--component-conditioned-refit", action="store_true")
    parser.add_argument("--component-refit-total-steps", type=int, default=0)
    parser.add_argument("--static-only-triangulation", action="store_true")
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument("--component-static-confirmation-only", action="store_true")
    args = parser.parse_args()
    manifest = run_closure(
        runtime_scene=args.runtime_scene,
        track_root=args.track_root,
        track_freeze=args.track_freeze,
        base_pose_root=args.base_pose_root,
        base_prediction_attestation=args.base_prediction_attestation,
        output_root=args.output_root,
        static_views=args.static_views,
        target_views=args.target_views,
        frames=args.frames,
        max_steps=args.max_steps,
        device=args.device,
        init_obs_filter_enable=args.init_obs_filter,
        candidate_optimize_points=not args.pose_only_candidate,
        component_verified_commit=args.component_verified_commit,
        component_boundary_verify=args.component_boundary_verify,
        component_conditioned_refit=args.component_conditioned_refit,
        component_refit_total_steps=args.component_refit_total_steps,
        static_only_triangulation=args.static_only_triangulation,
        code_manifest=args.code_manifest,
        component_require_static_improvement_significance=(
            not args.component_static_confirmation_only
        ),
    )
    print(
        json.dumps(
            {
                "track_commit_frame_count": manifest["track_commit_frame_count"],
                "base_fallback_frame_count": manifest["base_fallback_frame_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
