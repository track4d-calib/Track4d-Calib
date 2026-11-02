#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.cameras import (
    CameraLoader,
    CameraParams,
    save_cameras,
)
from track4d_calib.ptz_track_center_closure import (
    decide_fixed_center_commit,
    rank_consensus_center_candidates,
    rank_shared_center_candidates,
    robust_geometric_median_center,
)
from track4d_calib.public_runtime_protocols import (
    PUBLIC_NON_COMMON_CENTER_RUNTIME_PROTOCOLS,
)
from track4d_calib.track_crossview_verify import (
    CrossViewTrackVerifierConfig,
    build_context_point_bank,
    compare_focus_cameras,
    split_point_bank_for_validation,
)
from track4d_calib.track_pnp_repair import (
    crossfit_focus_track_pnp,
    solve_focus_track_rotation,
)
from track4d_calib.track_temporal_static_filter import (
    select_temporally_static_track_ids,
)
from scripts.geometry.audit_track_dualsource_stage_a import _tree_checksums
from scripts.geometry.evaluate_track_pose_after_stage_a_seal import (
    _verify_stage_a_seal,
)
from scripts.geometry.run_track_closure import (
    build_node_observations,
    camera_root_attestation,
    seed_point_ranges,
    track_file_attestation,
    verify_multiseed_inlier_support,
)


METHOD = "public_ptz_track_static_verify_geomedian_center_crossfit_rotation_v3"
INDEPENDENT_METHOD = (
    "public_ptz_independent_match_selected_center_crossfit_rotation_v1"
)
CONSENSUS_METHOD = "public_ptz_track_match_consensus_center_crossfit_rotation_v1"
OBSERVATION_SOURCES = {
    "temporal_tracks": METHOD,
    "independent_same_frame": INDEPENDENT_METHOD,
}
SELECTOR_SUBDIR = "dual_source_selector_missing_only"
COMMIT_POLICY = "confidence95"


def _observation_support_policy(
    observation_source: str,
    *,
    min_inliers: int,
) -> dict[str, int]:
    source = str(observation_source)
    if source == "temporal_tracks":
        return {"min_seed_banks": 2, "min_inliers_per_seed": 4}
    if source == "independent_same_frame":
        return {
            "min_seed_banks": 1,
            "min_inliers_per_seed": int(min_inliers),
        }
    raise ValueError(f"unsupported observation source: {source}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_center_consensus_if_available(
    *,
    primary_rows: Sequence[Mapping[str, Any]],
    auxiliary_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    if not primary_rows or not auxiliary_rows:
        return [], "one_observation_source_has_no_rankable_center"
    primary_frames = {int(row["source_frame"]) for row in primary_rows}
    auxiliary_frames = {int(row["source_frame"]) for row in auxiliary_rows}
    if not primary_frames.intersection(auxiliary_frames):
        return [], "observation_sources_have_no_common_center"
    return rank_consensus_center_candidates(primary_rows, auxiliary_rows), None


def _sealed_selector_protocol(seal_path: Path) -> tuple[str, str]:
    seal = _load_json(Path(seal_path).resolve())
    selector_subdir = str(seal.get("selector_subdir", ""))
    if (
        not selector_subdir
        or Path(selector_subdir).name != selector_subdir
        or selector_subdir in {".", ".."}
    ):
        raise ValueError("Stage-A seal has an invalid selector_subdir")
    commit_policy = str(seal.get("commit_policy", ""))
    if commit_policy != COMMIT_POLICY:
        raise ValueError(
            "Stage-A seal must use the confidence95 commit policy: "
            f"got {commit_policy!r}"
        )
    return selector_subdir, commit_policy


def _verify_runtime_tree(runtime: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(runtime / "runtime_manifest.json")
    declared = _load_json(runtime / "checksums.sha256.json")
    declared_files = declared.get("files")
    if not isinstance(declared_files, dict):
        raise ValueError("runtime checksum record has no file roster")
    excluded = {"checksums.sha256.json"}
    if "runtime_manifest.json" not in declared_files:
        excluded.add("runtime_manifest.json")
    actual = _tree_checksums(
        runtime,
        excluded_relative_paths=excluded,
    )
    if actual != declared:
        raise ValueError("runtime tree does not match checksums.sha256.json")
    runtime_protocol = str(manifest.get("protocol"))
    if runtime_protocol not in PUBLIC_NON_COMMON_CENTER_RUNTIME_PROTOCOLS:
        raise ValueError("runtime is not the public non-common-center protocol")
    if runtime_protocol == "dyn-neu3d-trackstress-calibration-runtime-v1":
        static_views = [str(view) for view in manifest.get("observed_short_views", [])]
        long_views = [str(view) for view in manifest.get("long_views", [])]
        runtime_views = [str(view) for view in manifest.get("runtime_views", [])]
        withheld_views = [str(view) for view in manifest.get("withheld_views", [])]
        rosters = [static_views, long_views, runtime_views, withheld_views]
        if any(len(values) != len(set(values)) for values in rosters):
            raise ValueError("Neu3D runtime view roster contains duplicates")
        if not static_views or not long_views:
            raise ValueError("Neu3D runtime has an empty calibration roster")
        if set(runtime_views) != set(static_views) | set(long_views):
            raise ValueError("Neu3D runtime roster is not static plus train-long")
        for view in withheld_views:
            if (runtime / "images" / view).exists():
                raise ValueError(f"Neu3D runtime contains withheld image view {view}")
        allowed_extrinsics = {(runtime / "cameras/static/extri.yml").resolve()}
        exposed_extrinsics = {
            path.resolve()
            for path in (runtime / "cameras").rglob("*extri*.yml")
            if path.is_file()
        }
        if exposed_extrinsics - allowed_extrinsics:
            raise ValueError("Neu3D runtime exposes non-static extrinsics")
        return manifest, declared
    if manifest.get("camera_layout", {}).get("long_extrinsics_available") is not False:
        raise ValueError("runtime exposes long-view extrinsics")
    if runtime_protocol == "sparse-anchor-partially-observed-parent-v1":
        exposed = manifest.get("sparse_anchor_layout", {}).get(
            "exposed_parent_views", []
        )
        if not exposed:
            raise ValueError("sparse-anchor runtime does not declare anchor parents")
    return manifest, declared


def _load_attested_prediction(
    *,
    attestation_path: Path,
    expected_prediction_path: Path,
    expected_pose_root: Path,
    runtime: Path,
    runtime_digest: str,
    frames: Sequence[int],
    target_views: Sequence[str],
    expected_track_mode: str,
) -> dict[str, Any]:
    attestation = _load_json(attestation_path)
    prediction_path = Path(str(attestation.get("path", ""))).resolve()
    if prediction_path != expected_prediction_path.resolve():
        raise ValueError("prediction attestation path mismatch")
    if _sha256(prediction_path) != str(attestation.get("sha256", "")):
        raise ValueError("prediction attestation hash mismatch")
    prediction = _load_json(prediction_path)
    if Path(str(prediction.get("runtime_scene", ""))).resolve() != runtime.resolve():
        raise ValueError("prediction runtime mismatch")
    if str(prediction.get("runtime_tree_digest")) != str(runtime_digest):
        raise ValueError("prediction runtime digest mismatch")
    if [int(value) for value in prediction.get("frames", [])] != [
        int(frame) for frame in frames
    ]:
        raise ValueError("prediction frame protocol mismatch")
    if [str(value) for value in prediction.get("target_views", [])] != [
        str(view) for view in target_views
    ]:
        raise ValueError("prediction target-view protocol mismatch")
    if str(prediction.get("track_source_mode")) != str(expected_track_mode):
        raise ValueError("prediction track-source protocol mismatch")
    if prediction.get("gt_used") is not False or prediction.get("nvs_used") is not False:
        raise ValueError("prediction used GT or NVS during selection")
    pose_root = Path(str(prediction.get("pose_root", ""))).resolve()
    if pose_root != expected_pose_root.resolve():
        raise ValueError("prediction pose-root mismatch")
    actual_hashes = {
        f"f{int(frame):06d}": camera_root_attestation(
            pose_root / f"f{int(frame):06d}" / "cameras_pred"
        )
        for frame in frames
    }
    if prediction.get("output_camera_hashes") != actual_hashes:
        raise ValueError("prediction camera payload changed after attestation")
    return prediction


def _load_attested_center_consensus(
    *,
    root: Path,
    runtime: Path,
    runtime_digest: str,
    stage: Path,
    base_pose_root: Path,
    frames: Sequence[int],
    target_views: Sequence[str],
) -> dict[str, Any]:
    source_root = Path(root).resolve()
    manifest_path = source_root / "closure_manifest.json"
    attestation = _load_json(source_root / "prediction_attestation.json")
    if Path(str(attestation.get("path", ""))).resolve() != manifest_path:
        raise ValueError("center-consensus attestation path mismatch")
    if str(attestation.get("sha256", "")) != _sha256(manifest_path):
        raise ValueError("center-consensus manifest hash mismatch")
    manifest = _load_json(manifest_path)
    if str(manifest.get("protocol")) != INDEPENDENT_METHOD:
        raise ValueError("center-consensus source is not independent same-frame geometry")
    if str(manifest.get("observation_source")) != "independent_same_frame":
        raise ValueError("center-consensus source observation mode mismatch")
    expected_paths = {
        "runtime_scene": runtime.resolve(),
        "stage_a_root": stage.resolve(),
        "base_pose_root": base_pose_root.resolve(),
    }
    for key, expected in expected_paths.items():
        if Path(str(manifest.get(key, ""))).resolve() != expected:
            raise ValueError(f"center-consensus {key} mismatch")
    if str(manifest.get("runtime_tree_digest")) != str(runtime_digest):
        raise ValueError("center-consensus runtime digest mismatch")
    if [int(value) for value in manifest.get("frames", [])] != [
        int(frame) for frame in frames
    ]:
        raise ValueError("center-consensus frame protocol mismatch")
    if [str(value) for value in manifest.get("target_views", [])] != [
        str(view) for view in target_views
    ]:
        raise ValueError("center-consensus target-view protocol mismatch")
    fairness = manifest.get("fairness", {})
    if any(
        fairness.get(key) is not False
        for key in ("GT_used", "NVS_used", "heldout_test_views_used")
    ):
        raise ValueError("center-consensus source used forbidden selection inputs")
    return manifest


def _load_track_bundle(
    *,
    track_root: Path,
    prediction: Mapping[str, Any],
    runtime_views: Sequence[str],
    static_views: Sequence[str],
    target_views: Sequence[str],
    frames: Sequence[int],
) -> dict[str, Any]:
    root = track_root.resolve()
    declared_roots = [Path(str(value)).resolve() for value in prediction["track_roots"]]
    if declared_roots != [root]:
        raise ValueError("temporal prediction does not attest the selected track root")
    declared_hashes = prediction.get("track_files_sha256", {}).get(str(root))
    if declared_hashes != track_file_attestation(root):
        raise ValueError("temporal track payload changed after prediction")
    return _load_track_bundle_payload(
        root=root,
        runtime_views=runtime_views,
        static_views=static_views,
        target_views=target_views,
        frames=frames,
    )


def _load_track_bundle_payload(
    *,
    root: Path,
    runtime_views: Sequence[str],
    static_views: Sequence[str],
    target_views: Sequence[str],
    frames: Sequence[int],
) -> dict[str, Any]:
    root = Path(root).resolve()
    meta = _load_json(root / "meta.json")
    views = (root / "views.txt").read_text(encoding="utf-8").split()
    expected_views = {str(view) for view in [*static_views, *target_views]}
    if set(views) != expected_views or len(views) != len(expected_views):
        raise ValueError("track views differ from the public calibration views")
    if set(views) - {str(view) for view in runtime_views}:
        raise ValueError("track bundle contains a non-runtime view")
    start = int(meta["start_frame"])
    count = int(meta["num_frames"])
    if any(int(frame) < start or int(frame) >= start + count for frame in frames):
        raise ValueError("requested frame lies outside the temporal track bundle")
    quality_path = root / "track_quality.npy"
    return {
        "root": root,
        "meta": meta,
        "views": views,
        "tracks": np.load(root / "tracks.npy", mmap_mode="r"),
        "visibility": np.load(root / "visibility.npy", mmap_mode="r"),
        "track_quality": (
            np.load(quality_path, mmap_mode="r") if quality_path.is_file() else None
        ),
    }


def _load_independent_track_bundles(
    *,
    prediction: Mapping[str, Any],
    runtime_views: Sequence[str],
    static_views: Sequence[str],
    target_views: Sequence[str],
    frames: Sequence[int],
) -> dict[int, dict[str, Any]]:
    requested = [int(frame) for frame in frames]
    roots = [Path(str(value)).resolve() for value in prediction["track_roots"]]
    if len(roots) != len(requested):
        raise ValueError("independent prediction must attest one bundle per frame")
    declared_hashes = prediction.get("track_files_sha256", {})
    bundles: dict[int, dict[str, Any]] = {}
    for root in roots:
        if declared_hashes.get(str(root)) != track_file_attestation(root):
            raise ValueError("independent track payload changed after prediction")
        meta = _load_json(root / "meta.json")
        frame = int(meta.get("start_frame", -1))
        if (
            int(meta.get("num_frames", 0)) != 1
            or int(meta.get("seed_frame", -1)) != frame
            or [int(value) for value in meta.get("seed_frames", [])] != [frame]
        ):
            raise ValueError("independent bundle is not a single same-frame bundle")
        if frame in bundles:
            raise ValueError("independent prediction repeats a frame bundle")
        bundles[frame] = _load_track_bundle_payload(
            root=root,
            runtime_views=runtime_views,
            static_views=static_views,
            target_views=target_views,
            frames=[frame],
        )
    if sorted(bundles) != sorted(requested):
        raise ValueError("independent bundle frames differ from prediction frames")
    return bundles


def _collect_successful_nodes(
    rows: Sequence[Mapping[str, Any]],
) -> set[tuple[int, str]]:
    return {
        (int(row["frame"]), str(row["view"]))
        for row in rows
        if bool(row.get("solver", {}).get("success", False))
    }


def _load_successful_cameras(
    *,
    pose_root: Path,
    successful_nodes: set[tuple[int, str]],
    frames: Sequence[int],
    target_views: Sequence[str],
) -> dict[tuple[int, str], CameraParams]:
    output: dict[tuple[int, str], CameraParams] = {}
    for frame in frames:
        loader = CameraLoader(
            pose_root / f"f{int(frame):06d}" / "cameras_pred"
        )
        for view in target_views:
            node = (int(frame), str(view))
            if node in successful_nodes:
                output[node] = loader.load_camera(str(view))
    return output


def _choose_rotation_template(
    *,
    frame: int,
    view: str,
    base_cameras: Mapping[tuple[int, str], CameraParams],
    temporal_cameras: Mapping[tuple[int, str], CameraParams],
) -> tuple[CameraParams | None, str | None]:
    node = (int(frame), str(view))
    if node in base_cameras:
        return base_cameras[node], "same_frame_static_pnp"
    if node in temporal_cameras:
        return temporal_cameras[node], "same_frame_temporal_track_pnp"
    same_view = [
        (abs(source_frame - int(frame)), source_frame, camera)
        for (source_frame, source_view), camera in base_cameras.items()
        if source_view == str(view)
    ]
    if not same_view:
        return None, None
    _distance, source_frame, camera = min(same_view, key=lambda row: (row[0], row[1]))
    return camera, f"nearest_static_pnp_frame_{source_frame}"


def _build_node_point_bank(
    *,
    frame: int,
    view: str,
    bundle: Mapping[str, Any],
    static_views: Sequence[str],
    support_cameras: Mapping[str, CameraParams],
    point_ranges: Sequence[Mapping[str, Any]],
    fit_budget: int,
    config: CrossViewTrackVerifierConfig,
    allowed_point_ids: np.ndarray | None = None,
    fixed_xyz_by_point_id: Mapping[int, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    local = int(frame) - int(bundle["meta"]["start_frame"])
    _support, targets, selection = build_node_observations(
        tracks=bundle["tracks"],
        visibility=bundle["visibility"],
        track_quality=bundle["track_quality"],
        view_ids=bundle["views"],
        static_views=static_views,
        target_view=str(view),
        frame_local=local,
        min_support_views=int(config.min_context_views),
        max_points=int(fit_budget),
        allowed_point_ids=allowed_point_ids,
        balanced_point_ranges=[
            tuple(int(value) for value in row["point_range"])
            for row in point_ranges
        ],
    )
    selected_ids = np.asarray(sorted(targets), dtype=np.int64)
    if fixed_xyz_by_point_id is None:
        bank = build_context_point_bank(
            tracks_frame=np.asarray(bundle["tracks"][:, local]),
            visibility_frame=np.asarray(bundle["visibility"][:, local]),
            view_ids=bundle["views"],
            cameras_by_view=support_cameras,
            context_views=static_views,
            point_ids=selected_ids,
            config=config,
        )
        landmark_policy = "per_frame_public_static_triangulation"
    else:
        kept_ids = np.asarray(
            [
                int(point_id)
                for point_id in selected_ids
                if int(point_id) in fixed_xyz_by_point_id
            ],
            dtype=np.int64,
        )
        xyz = np.asarray(
            [
                np.asarray(fixed_xyz_by_point_id[int(point_id)], dtype=np.float64)
                .reshape(3)
                for point_id in kept_ids
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        if not np.isfinite(xyz).all():
            raise ValueError("fixed static landmarks must be finite")
        bank = {
            "num_points_considered": int(len(selected_ids)),
            "num_points_valid": int(len(kept_ids)),
            "point_ids": kept_ids,
            "xyz": xyz,
            "context_views": [str(view) for view in static_views],
        }
        landmark_policy = "cross_frame_fixed_static_map_xyz"
    diagnostics = {
        **selection,
        "selected_track_id_count": int(len(selected_ids)),
        "triangulated_point_count": int(bank["num_points_valid"]),
        "landmark_policy": landmark_policy,
    }
    return bank, diagnostics


def _multiseed_oof_verification(
    *,
    comparison: Mapping[str, Any],
    point_ranges: Sequence[Mapping[str, Any]],
    focus_inlier_px: float,
    min_seed_banks: int,
    min_inliers_per_seed: int,
) -> dict[str, Any]:
    point_ids = np.asarray(comparison.get("point_ids", []), dtype=np.int64).reshape(-1)
    errors = np.asarray(
        comparison.get("candidate_errors_px", []), dtype=np.float64
    ).reshape(-1)
    if point_ids.shape != errors.shape:
        raise ValueError("OOF point IDs and candidate errors have different shapes")
    threshold = float(focus_inlier_px)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("focus_inlier_px must be finite and positive")
    inlier_ids = point_ids[np.isfinite(errors) & (errors <= threshold)]
    result = verify_multiseed_inlier_support(
        inlier_point_ids=inlier_ids,
        point_ranges=point_ranges,
        min_seed_banks=int(min_seed_banks),
        min_inliers_per_seed=int(min_inliers_per_seed),
    )
    return {
        **result,
        "oof_inlier_count": int(len(inlier_ids)),
        "oof_inlier_point_ids": [int(value) for value in inlier_ids.tolist()],
    }


def _center_score_row(
    *,
    source_frame: int,
    center: np.ndarray,
    view: str,
    frames: Sequence[int],
    point_banks: Mapping[tuple[int, str], Mapping[str, Any]],
    templates: Mapping[tuple[int, str], tuple[CameraParams | None, str | None]],
    bundles_by_frame: Mapping[int, Mapping[str, Any]],
    point_ranges_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    focus_reprojection_px: float,
    min_inliers: int,
    min_seed_banks: int,
    min_inliers_per_seed: int,
) -> dict[str, Any]:
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        if int(frame) == int(source_frame):
            frame_rows.append(
                {
                    "frame": int(frame),
                    "used_for_center_score": False,
                    "reason": "leave_source_frame_out",
                }
            )
            continue
        template, template_source = templates[(int(frame), str(view))]
        bank = point_banks[(int(frame), str(view))]
        if template is None:
            frame_rows.append(
                {
                    "frame": int(frame),
                    "used_for_center_score": False,
                    "reason": "rotation_template_unavailable",
                }
            )
            continue
        bundle = bundles_by_frame[int(frame)]
        point_ranges = point_ranges_by_frame[int(frame)]
        local = int(frame) - int(bundle["meta"]["start_frame"])
        result = solve_focus_track_rotation(
            point_bank=bank,
            tracks_frame=np.asarray(bundle["tracks"][:, local]),
            visibility_frame=np.asarray(bundle["visibility"][:, local]),
            view_ids=bundle["views"],
            focus_view=str(view),
            camera_template=template,
            fixed_center=np.asarray(center, dtype=np.float64),
            reprojection_error_px=float(focus_reprojection_px),
            min_inliers=int(min_inliers),
            direction_ransac_prefilter=True,
            iterations_count=512,
            rng_seed=int(frame),
        )
        multiseed: dict[str, Any] | None = None
        validated = False
        if bool(result["success"]):
            paired_ids = np.asarray(result["point_ids"], dtype=np.int64)
            inlier_ids = paired_ids[
                np.asarray(result["inlier_indices"], dtype=np.int64)
            ]
            multiseed = verify_multiseed_inlier_support(
                inlier_point_ids=inlier_ids,
                point_ranges=point_ranges,
                min_seed_banks=int(min_seed_banks),
                min_inliers_per_seed=int(min_inliers_per_seed),
            )
            validated = bool(multiseed["verified"])
        frame_rows.append(
            {
                "frame": int(frame),
                "used_for_center_score": bool(validated),
                "template_source": template_source,
                "solver_success": bool(result["success"]),
                "solver_reason": str(result["reason"]),
                "paired_count": int(result.get("paired_count", 0)),
                "inlier_count": int(result.get("inlier_count", 0)),
                "reprojection_median_px": result.get("reprojection_median_px"),
                "reprojection_p90_px": result.get("reprojection_p90_px"),
                "multiseed_verification": multiseed,
            }
        )
    validated_rows = [row for row in frame_rows if row["used_for_center_score"]]
    return {
        "source_frame": int(source_frame),
        "center": np.asarray(center, dtype=np.float64).reshape(3).tolist(),
        "validated_frame_count": int(len(validated_rows)),
        "total_inlier_count": int(
            sum(int(row["inlier_count"]) for row in validated_rows)
        ),
        "median_reprojection_px": (
            float(
                np.median(
                    [float(row["reprojection_median_px"]) for row in validated_rows]
                )
            )
            if validated_rows
            else None
        ),
        "frame_rows": frame_rows,
    }


def _crossfit_audit_row(result: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(result.get("success", False)):
        return {
            "success": False,
            "reason": str(result.get("reason", "unknown")),
            "folds": result.get("folds", []),
            "crossfit_camera_consistency": result.get(
                "crossfit_camera_consistency"
            ),
        }
    comparison = result["aggregate_oof_comparison"]
    return {
        "success": True,
        "reason": "ok",
        "camera_source": result["camera_source"],
        "commit_validation_fold": int(result["commit_validation_fold"]),
        "paired_count": int(result["paired_count"]),
        "inlier_count": int(result["inlier_count"]),
        "reprojection_median_px": float(result["reprojection_median_px"]),
        "folds": result["folds"],
        "crossfit_camera_consistency": result["crossfit_camera_consistency"],
        "aggregate_oof_comparison": {
            "paired_count": int(comparison["paired_count"]),
            "point_ids": np.asarray(comparison["point_ids"], dtype=np.int64),
            "base_errors_px": np.asarray(
                comparison["base_errors_px"], dtype=np.float64
            ),
            "candidate_errors_px": np.asarray(
                comparison["candidate_errors_px"], dtype=np.float64
            ),
            "base": comparison["base"],
            "candidate": comparison["candidate"],
            "gain_mean_px": comparison["gain_mean_px"],
            "gain_median_px": comparison["gain_median_px"],
            "gain_p90_px": comparison["gain_p90_px"],
        },
    }


def _camera_equal(first: CameraParams, second: CameraParams) -> bool:
    return (
        first.view_id == second.view_id
        and first.H == second.H
        and first.W == second.W
        and np.array_equal(first.K, second.K)
        and np.array_equal(first.R, second.R)
        and np.array_equal(first.T, second.T)
        and np.array_equal(first.D, second.D)
    )


def _camera_with_center(camera: CameraParams, center: np.ndarray) -> CameraParams:
    target = np.asarray(center, dtype=np.float64).reshape(3)
    if not np.isfinite(target).all():
        raise ValueError("camera center must contain three finite values")
    return CameraParams(
        view_id=camera.view_id,
        K=np.asarray(camera.K, dtype=np.float64).copy(),
        R=np.asarray(camera.R, dtype=np.float64).copy(),
        T=-np.asarray(camera.R, dtype=np.float64) @ target,
        D=np.asarray(camera.D, dtype=np.float64).copy(),
        H=int(camera.H),
        W=int(camera.W),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = args.runtime_scene.resolve()
    stage = args.stage_a_root.resolve()
    seal = args.stage_a_seal.resolve()
    output = args.output_root.resolve()
    frames = [int(frame) for frame in args.frames]
    target_views = [str(view) for view in args.target_views]
    observation_source = str(args.observation_source)
    if observation_source not in OBSERVATION_SOURCES:
        raise ValueError(f"unsupported observation source: {observation_source}")
    if len(frames) != len(set(frames)) or frames != sorted(frames):
        raise ValueError("frames must be sorted and unique")
    if len(target_views) != len(set(target_views)):
        raise ValueError("target views must be unique")
    if output.is_relative_to(stage) or output.is_relative_to(runtime):
        raise ValueError("closure output must be outside sealed input trees")

    selector_subdir, commit_policy = _sealed_selector_protocol(seal)
    seal_record = _verify_stage_a_seal(
        runtime_scene=runtime,
        stage_a_root=stage,
        stage_a_seal=seal,
        selector_subdir=selector_subdir,
        expected_commit_policy=commit_policy,
        frames=frames,
        target_views=target_views,
    )
    runtime_manifest, runtime_checksums = _verify_runtime_tree(runtime)
    static_views = [str(view) for view in runtime_manifest["observed_short_views"]]
    if set(target_views) - {str(view) for view in runtime_manifest["long_views"]}:
        raise ValueError("target view is absent from the public runtime")

    base_dir = stage / "base_same_frame_pnp"
    temporal_dir = stage / "temporal_pose_fit150"
    base_pose_root = base_dir / "pose_track_same_time_pnp"
    temporal_pose_root = temporal_dir / "pose_track_same_time_pnp"
    base_prediction = _load_attested_prediction(
        attestation_path=base_dir / "prediction_attestation.json",
        expected_prediction_path=base_dir / "predictions_pre_gt.json",
        expected_pose_root=base_pose_root,
        runtime=runtime,
        runtime_digest=str(runtime_checksums["tree_digest"]),
        frames=frames,
        target_views=target_views,
        expected_track_mode="independent_per_frame_bundle",
    )
    temporal_prediction: dict[str, Any] | None = None
    track_root: Path | None = None
    if observation_source == "temporal_tracks":
        temporal_prediction = _load_attested_prediction(
            attestation_path=temporal_dir / "prediction_attestation.json",
            expected_prediction_path=temporal_dir / "predictions_pre_gt.json",
            expected_pose_root=temporal_pose_root,
            runtime=runtime,
            runtime_digest=str(runtime_checksums["tree_digest"]),
            frames=frames,
            target_views=target_views,
            expected_track_mode="shared_temporal_bundle",
        )
        track_root = stage / "temporal_targetstar_stride5_lkfb"
        temporal_bundle = _load_track_bundle(
            track_root=track_root,
            prediction=temporal_prediction,
            runtime_views=runtime_manifest["runtime_views"],
            static_views=static_views,
            target_views=target_views,
            frames=frames,
        )
        bundles_by_frame = {int(frame): temporal_bundle for frame in frames}
    else:
        bundles_by_frame = _load_independent_track_bundles(
            prediction=base_prediction,
            runtime_views=runtime_manifest["runtime_views"],
            static_views=static_views,
            target_views=target_views,
            frames=frames,
        )

    center_consensus: dict[str, Any] | None = None
    if args.center_consensus_root is not None:
        if observation_source != "temporal_tracks":
            raise ValueError(
                "center rank consensus is only valid for temporal-track candidates"
            )
        center_consensus = _load_attested_center_consensus(
            root=args.center_consensus_root,
            runtime=runtime,
            runtime_digest=str(runtime_checksums["tree_digest"]),
            stage=stage,
            base_pose_root=base_pose_root,
            frames=frames,
            target_views=target_views,
        )

    if output.exists():
        if not bool(args.overwrite):
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    base_success = _collect_successful_nodes(base_prediction["rows"])
    temporal_success = (
        _collect_successful_nodes(temporal_prediction["rows"])
        if temporal_prediction is not None
        else set()
    )
    base_cameras = _load_successful_cameras(
        pose_root=base_pose_root,
        successful_nodes=base_success,
        frames=frames,
        target_views=target_views,
    )
    temporal_cameras = (
        _load_successful_cameras(
            pose_root=temporal_pose_root,
            successful_nodes=temporal_success,
            frames=frames,
            target_views=target_views,
        )
        if temporal_prediction is not None
        else {}
    )
    support_cameras = CameraLoader(runtime / "cameras/static").load_cameras(
        static_views
    )
    point_ranges_by_frame = {
        int(frame): seed_point_ranges(
            meta=bundles_by_frame[int(frame)]["meta"],
            total_points=int(bundles_by_frame[int(frame)]["tracks"].shape[2]),
        )
        for frame in frames
    }
    support_policy = _observation_support_policy(
        observation_source,
        min_inliers=int(args.min_inliers),
    )
    config = CrossViewTrackVerifierConfig(
        min_context_views=2,
        context_inlier_px=float(args.support_reprojection_px),
        focus_inlier_px=float(args.focus_reprojection_px),
        max_pair_hypotheses=24,
    )

    temporal_static_filter: dict[str, Any] = {
        "enabled": False,
        "reason": (
            "disabled_ablation"
            if bool(args.disable_temporal_static_filter)
            else "not_applicable_to_independent_same_frame_observations"
        ),
    }
    static_validation_point_ids: np.ndarray | None = None
    static_validation_xyz_by_point_id: dict[int, np.ndarray] | None = None
    if (
        observation_source == "temporal_tracks"
        and not bool(args.disable_temporal_static_filter)
    ):
        temporal_bundle = bundles_by_frame[int(frames[0])]
        if any(
            Path(bundles_by_frame[int(frame)]["root"]).resolve()
            != Path(temporal_bundle["root"]).resolve()
            for frame in frames[1:]
        ):
            raise ValueError("temporal frames do not share one attested track bundle")
        seed_ranges = point_ranges_by_frame[int(frames[0])]
        point_ids_by_seed = {
            int(row["seed_frame"]): np.arange(
                int(row["point_range"][0]),
                int(row["point_range"][1]),
                dtype=np.int64,
            )
            for row in seed_ranges
        }
        temporal_static_filter = select_temporally_static_track_ids(
            tracks=np.asarray(temporal_bundle["tracks"]),
            visibility=np.asarray(temporal_bundle["visibility"]),
            view_ids=temporal_bundle["views"],
            cameras_by_frame={int(frame): support_cameras for frame in frames},
            context_views=static_views,
            point_ids_by_seed=point_ids_by_seed,
            track_start_frame=int(temporal_bundle["meta"]["start_frame"]),
            sample_frames=frames,
            config=config,
            reprojection_gate_px=float(args.support_reprojection_px),
        )
        temporal_static_filter["enabled"] = True
        static_validation_point_ids = np.asarray(
            temporal_static_filter["selected_point_ids"], dtype=np.int64
        )
        selected_xyz = np.asarray(
            temporal_static_filter["selected_landmark_xyz"], dtype=np.float64
        ).reshape(-1, 3)
        static_validation_xyz_by_point_id = {
            int(point_id): xyz
            for point_id, xyz in zip(
                static_validation_point_ids, selected_xyz, strict=True
            )
        }

    templates = {
        (int(frame), str(view)): _choose_rotation_template(
            frame=int(frame),
            view=str(view),
            base_cameras=base_cameras,
            temporal_cameras=temporal_cameras,
        )
        for frame in frames
        for view in target_views
    }
    point_banks: dict[tuple[int, str], dict[str, Any]] = {}
    static_validation_banks: dict[tuple[int, str], dict[str, Any]] = {}
    point_bank_rows: list[dict[str, Any]] = []
    for frame in frames:
        bundle = bundles_by_frame[int(frame)]
        point_ranges = point_ranges_by_frame[int(frame)]
        for view in target_views:
            bank, diagnostics = _build_node_point_bank(
                frame=int(frame),
                view=str(view),
                bundle=bundle,
                static_views=static_views,
                support_cameras=support_cameras,
                point_ranges=point_ranges,
                fit_budget=int(args.fit_budget),
                config=config,
            )
            node = (int(frame), str(view))
            point_banks[node] = bank
            validation_diagnostics: dict[str, Any] = {
                "enabled": False,
                "triangulated_point_count": 0,
            }
            if static_validation_xyz_by_point_id is not None:
                validation_bank, validation_diagnostics = _build_node_point_bank(
                    frame=int(frame),
                    view=str(view),
                    bundle=bundle,
                    static_views=static_views,
                    support_cameras=support_cameras,
                    point_ranges=point_ranges,
                    fit_budget=int(args.fit_budget),
                    config=config,
                    allowed_point_ids=static_validation_point_ids,
                    fixed_xyz_by_point_id=static_validation_xyz_by_point_id,
                )
                static_validation_banks[node] = validation_bank
                validation_diagnostics = {
                    **validation_diagnostics,
                    "enabled": True,
                }
            point_bank_rows.append(
                {
                    "frame": int(frame),
                    "view": str(view),
                    "generation": diagnostics,
                    "static_validation": validation_diagnostics,
                }
            )

    center_reports: dict[str, Any] = {}
    selected_centers: dict[str, np.ndarray] = {}
    for view in target_views:
        source_rows = []
        for frame in frames:
            node = (int(frame), str(view))
            if node not in base_cameras:
                continue
            source_rows.append(
                _center_score_row(
                    source_frame=int(frame),
                    center=base_cameras[node].camera_center,
                    view=str(view),
                    frames=frames,
                    point_banks=point_banks,
                    templates=templates,
                    bundles_by_frame=bundles_by_frame,
                    point_ranges_by_frame=point_ranges_by_frame,
                    focus_reprojection_px=float(args.focus_reprojection_px),
                    min_inliers=int(args.min_inliers),
                    min_seed_banks=int(support_policy["min_seed_banks"]),
                    min_inliers_per_seed=int(
                        support_policy["min_inliers_per_seed"]
                    ),
                )
            )
        rankable = [
            row
            for row in source_rows
            if int(row["validated_frame_count"]) > 0
            and row["median_reprojection_px"] is not None
        ]
        ranked = rank_shared_center_candidates(rankable) if rankable else []
        fused = robust_geometric_median_center(ranked) if ranked else None
        representative = None
        if fused is not None:
            selected_centers[str(view)] = np.asarray(
                fused["center"], dtype=np.float64
            )
            representative = min(
                ranked,
                key=lambda row: (
                    float(
                        np.linalg.norm(
                            np.asarray(row["center"], dtype=np.float64)
                            - selected_centers[str(view)]
                        )
                    ),
                    int(row["source_frame"]),
                ),
            )
        center_reports[str(view)] = {
            "base_center_candidate_count": int(len(source_rows)),
            "ranked_candidates": ranked,
            "unrankable_candidates": [
                row for row in source_rows if row not in rankable
            ],
            "selected_source_frame": (
                int(representative["source_frame"])
                if representative is not None
                else None
            ),
            "selected_center": None if fused is None else fused,
            "selected_center_policy": (
                "geometric_median_of_cross_time_track_validated_pnp_centers"
            ),
            "selection_uses_gt": False,
            "selection_uses_nvs": False,
            "selection_score_order": [
                "cross_time_track_validation_eligibility",
                "geometric_median_fusion",
            ],
        }

    if center_consensus is not None:
        auxiliary_reports = center_consensus.get("center_reports", {})
        if set(auxiliary_reports) != set(center_reports):
            raise ValueError("center-consensus report views mismatch")
        for view in target_views:
            report = center_reports[str(view)]
            auxiliary = auxiliary_reports[str(view)]
            report["single_source_selected_source_frame"] = report[
                "selected_source_frame"
            ]
            report["auxiliary_selected_source_frame"] = auxiliary.get(
                "selected_source_frame"
            )
            primary_ranked = report["ranked_candidates"]
            auxiliary_ranked = auxiliary.get("ranked_candidates", [])
            consensus_ranked, unavailable_reason = (
                _rank_center_consensus_if_available(
                    primary_rows=primary_ranked,
                    auxiliary_rows=auxiliary_ranked,
                )
            )
            if not consensus_ranked:
                report["consensus_ranked_candidates"] = []
                report["consensus_unavailable_reason"] = unavailable_reason
                report["consensus_fallback_policy"] = (
                    "retain_temporal_track_selected_center"
                )
                continue
            fused = robust_geometric_median_center(consensus_ranked)
            representative = min(
                consensus_ranked,
                key=lambda row: (
                    float(
                        np.linalg.norm(
                            np.asarray(row["center"], dtype=np.float64)
                            - np.asarray(fused["center"], dtype=np.float64)
                        )
                    ),
                    int(row["source_frame"]),
                ),
            )
            report["consensus_ranked_candidates"] = consensus_ranked
            report["selected_source_frame"] = int(
                representative["source_frame"]
            )
            report["selected_center"] = fused
            report["selected_center_policy"] = (
                "geometric_median_of_crosssource_track_validated_pnp_centers"
            )
            report["selection_score_order"] = [
                "primary_plus_auxiliary_rank_sum_asc",
                "worst_source_rank_asc",
                "temporal_track_rank_asc",
                "independent_same_frame_rank_asc",
                "source_frame_asc",
            ]
            selected_centers[str(view)] = np.asarray(
                fused["center"], dtype=np.float64
            )

    candidate_cameras: dict[tuple[int, str], CameraParams] = {}
    decisions: dict[tuple[int, str], dict[str, Any]] = {}
    node_rows: list[dict[str, Any]] = []
    for frame in frames:
        bundle = bundles_by_frame[int(frame)]
        point_ranges = point_ranges_by_frame[int(frame)]
        for view in target_views:
            node = (int(frame), str(view))
            template, template_source = templates[node]
            if str(view) not in selected_centers:
                decisions[node] = {"commit": False, "reason": "shared_center_unavailable"}
                node_rows.append(
                    {
                        "frame": int(frame),
                        "view": str(view),
                        "base_available": bool(node in base_cameras),
                        "template_source": template_source,
                        "decision": decisions[node],
                    }
                )
                continue
            if template is None:
                decisions[node] = {"commit": False, "reason": "rotation_template_unavailable"}
                node_rows.append(
                    {
                        "frame": int(frame),
                        "view": str(view),
                        "base_available": bool(node in base_cameras),
                        "template_source": None,
                        "decision": decisions[node],
                    }
                )
                continue
            local = int(frame) - int(bundle["meta"]["start_frame"])
            base_camera = base_cameras.get(node, template)
            solve = crossfit_focus_track_pnp(
                point_bank=point_banks[node],
                tracks_frame=np.asarray(bundle["tracks"][:, local]),
                visibility_frame=np.asarray(bundle["visibility"][:, local]),
                view_ids=bundle["views"],
                focus_view=str(view),
                base_camera=base_camera,
                camera_template=template,
                verifier_config=config,
                reprojection_error_px=float(args.focus_reprojection_px),
                min_inliers=int(args.min_inliers),
                num_folds=int(args.num_folds),
                iterations_count=512,
                rng_seed=int(frame),
                fixed_center=selected_centers[str(view)],
                fixed_center_direction_ransac_prefilter=True,
            )
            generation_audit = _crossfit_audit_row(solve)
            audit = generation_audit
            multiseed = None
            if bool(solve.get("success", False)):
                selection_comparison = solve["aggregate_oof_comparison"]
                validation_source = "candidate_generation_crossfit_oof"
                if node in static_validation_banks:
                    validation_fold = int(solve["commit_validation_fold"])
                    validation_bank = split_point_bank_for_validation(
                        static_validation_banks[node],
                        num_folds=int(args.num_folds),
                        validation_fold=validation_fold,
                    )["validation"]
                    selection_comparison = compare_focus_cameras(
                        point_bank=validation_bank,
                        tracks_frame=np.asarray(bundle["tracks"][:, local]),
                        visibility_frame=np.asarray(
                            bundle["visibility"][:, local]
                        ),
                        view_ids=bundle["views"],
                        focus_view=str(view),
                        base_camera=base_camera,
                        candidate_camera=solve["camera"],
                        config=config,
                    )
                    validation_source = (
                        "exact_candidate_disjoint_temporally_static_track_fold"
                    )
                verification_solve = {
                    **solve,
                    "aggregate_oof_comparison": selection_comparison,
                }
                audit = _crossfit_audit_row(verification_solve)
                audit["candidate_generation_aggregate_oof_comparison"] = (
                    generation_audit["aggregate_oof_comparison"]
                )
                audit["commit_validation_source"] = validation_source
                multiseed = _multiseed_oof_verification(
                    comparison=selection_comparison,
                    point_ranges=point_ranges,
                    focus_inlier_px=float(args.focus_reprojection_px),
                    min_seed_banks=int(support_policy["min_seed_banks"]),
                    min_inliers_per_seed=int(
                        support_policy["min_inliers_per_seed"]
                    ),
                )
                verification = {
                    **verification_solve,
                    "multiseed_verification": multiseed,
                    "minimum_oof_inliers": int(args.min_inliers),
                }
                decision = decide_fixed_center_commit(
                    base_available=node in base_cameras,
                    verification=verification,
                    focus_inlier_px=float(args.focus_reprojection_px),
                )
                candidate_cameras[node] = solve["camera"]
            else:
                decision = {"commit": False, "reason": "candidate_crossfit_failed"}
            decisions[node] = decision
            node_rows.append(
                {
                    "frame": int(frame),
                    "view": str(view),
                    "base_available": bool(node in base_cameras),
                    "template_source": template_source,
                    "selected_center_source_frame": center_reports[str(view)][
                        "selected_source_frame"
                    ],
                    "crossfit": audit,
                    "multiseed_verification": multiseed,
                    "decision": decision,
                }
            )

    candidate_root = output / "pose_track_center_candidates"
    pose_root = output / "pose_track_center_node_verified"
    committed_nodes = {node for node, decision in decisions.items() if decision["commit"]}
    center_repaired_nodes = {
        node
        for node in base_cameras
        if str(node[1]) in selected_centers
    }
    for frame in frames:
        base_frame = dict(support_cameras)
        candidate_frame = dict(support_cameras)
        selected_frame = dict(support_cameras)
        for view in target_views:
            node = (int(frame), str(view))
            if node in base_cameras:
                base_camera = base_cameras[node]
                base_frame[str(view)] = base_camera
                center_repaired = (
                    _camera_with_center(base_camera, selected_centers[str(view)])
                    if node in center_repaired_nodes
                    else base_camera
                )
                candidate_frame[str(view)] = center_repaired
                selected_frame[str(view)] = center_repaired
            if node in candidate_cameras:
                candidate_frame[str(view)] = candidate_cameras[node]
            if node in committed_nodes:
                selected_frame[str(view)] = candidate_cameras[node]
        save_cameras(
            candidate_root / f"f{int(frame):06d}" / "cameras_pred",
            candidate_frame,
        )
        save_cameras(
            pose_root / f"f{int(frame):06d}" / "cameras_pred",
            selected_frame,
        )

    passthrough_checks = []
    for node, source_camera in sorted(base_cameras.items()):
        if node in committed_nodes or node in center_repaired_nodes:
            continue
        frame, view = node
        written = CameraLoader(
            pose_root / f"f{frame:06d}" / "cameras_pred"
        ).load_camera(view)
        passthrough_checks.append(_camera_equal(source_camera, written))
    if not all(passthrough_checks):
        raise RuntimeError("non-committed Static+PnP camera changed in output")
    exact_passthrough = bool(
        all(passthrough_checks)
        and not (center_repaired_nodes - committed_nodes)
    )

    method = (
        CONSENSUS_METHOD
        if center_consensus is not None
        else OBSERVATION_SOURCES[observation_source]
    )
    center_selection_description = (
        "rank consensus of leave-source-frame-out temporal-track and independent same-frame support"
        if center_consensus is not None
        else (
            "leave-source-frame-out temporal-track support"
            if observation_source == "temporal_tracks"
            else "leave-source-frame-out independent same-frame match support"
        )
    )
    rotation_template_priority = [
        "same_frame_static_pnp",
        *(
            ["same_frame_temporal_track_pnp"]
            if observation_source == "temporal_tracks"
            else []
        ),
        "nearest_successful_static_pnp_rotation",
    ]
    manifest = {
        "protocol": method,
        "method": method,
        "observation_source": observation_source,
        "runtime_scene": str(runtime),
        "runtime_tree_digest": runtime_checksums["tree_digest"],
        "stage_a_root": str(stage),
        "stage_a_seal": seal_record,
        "frames": frames,
        "static_views": static_views,
        "target_views": target_views,
        "base_pose_root": str(base_pose_root),
        "config": {
            "fit_point_budget": int(args.fit_budget),
            "min_context_views": int(config.min_context_views),
            "support_reprojection_px": float(config.context_inlier_px),
            "focus_reprojection_px": float(config.focus_inlier_px),
            "min_inliers": int(args.min_inliers),
            "num_crossfit_folds": int(args.num_folds),
            "direction_ransac_iterations": 512,
            "min_seed_banks": int(support_policy["min_seed_banks"]),
            "min_inliers_per_seed": int(
                support_policy["min_inliers_per_seed"]
            ),
        },
        "candidate_generation": {
            "shared_center_candidates": "successful independent same-frame Static+PnP centers",
            "shared_center_selection": center_selection_description,
            "shared_center_estimator": (
                "geometric median of track-validated Static+PnP centers"
            ),
            "rotation_model": "fixed-center world-ray to camera-ray direction RANSAC plus robust refinement",
            "rotation_template_priority": rotation_template_priority,
            "fit_verify_split": "stable_track_id_modulo_crossfit",
            "generation_track_policy": "broad per-node track bank",
            "commit_track_policy": (
                "disjoint fold of temporally-static fixed landmarks"
                if temporal_static_filter.get("enabled", False)
                else "candidate-generation crossfit OOF ablation"
            ),
        },
        "point_bank_rows": point_bank_rows,
        "temporal_static_track_filter": temporal_static_filter,
        "center_reports": center_reports,
        "node_rows": node_rows,
        "base_success_count": int(len(base_success)),
        "temporal_success_count": int(len(temporal_success)),
        "candidate_success_count": int(len(candidate_cameras)),
        "commit_count": int(len(committed_nodes)),
        "center_commit_count": int(len(center_repaired_nodes)),
        "center_repaired_nodes": [
            {"frame": int(frame), "view": str(view)}
            for frame, view in sorted(center_repaired_nodes)
        ],
        "committed_nodes": [
            {"frame": int(frame), "view": str(view)}
            for frame, view in sorted(committed_nodes)
        ],
        "candidate_pose_root": str(candidate_root),
        "candidate_output_camera_hashes": {
            f"f{int(frame):06d}": camera_root_attestation(
                candidate_root / f"f{int(frame):06d}" / "cameras_pred"
            )
            for frame in frames
        },
        "pose_root": str(pose_root),
        "output_camera_hashes": {
            f"f{int(frame):06d}": camera_root_attestation(
                pose_root / f"f{int(frame):06d}" / "cameras_pred"
            )
            for frame in frames
        },
        "fairness": {
            "GT_used": False,
            "NVS_used": False,
            "heldout_test_views_used": False,
            "runtime_and_images_identical_to_static_pnp": True,
            "static_camera_map_identical_to_static_pnp": True,
            "fit_point_budget_matches_stage_a": bool(int(args.fit_budget) == 150),
            "noncommitted_base_cameras_preserved_exactly": exact_passthrough,
            "rotation_noncommitted_nodes_receive_center_only_repair": True,
            "temporal_tracks_are_additional_method_input": bool(
                observation_source == "temporal_tracks"
            ),
            "temporal_static_tracks_define_fixed_cross_frame_landmarks": bool(
                temporal_static_filter.get("enabled", False)
            ),
            "candidate_generation_and_static_commit_banks_are_decoupled": bool(
                temporal_static_filter.get("enabled", False)
            ),
            "independent_same_frame_matches_only": bool(
                observation_source == "independent_same_frame"
            ),
            "center_rank_consensus_uses_temporal_and_independent_geometry": bool(
                center_consensus is not None
            ),
            "track_or_same_center_source_images_used": False,
            "output_written_outside_sealed_stage_a": True,
        },
        "code": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "core": str((PROJECT_ROOT / "track4d_calib/ptz_track_center_closure.py").resolve()),
            "core_sha256": _sha256(
                PROJECT_ROOT / "track4d_calib/ptz_track_center_closure.py"
            ),
            "solver": str((PROJECT_ROOT / "track4d_calib/track_pnp_repair.py").resolve()),
            "solver_sha256": _sha256(PROJECT_ROOT / "track4d_calib/track_pnp_repair.py"),
            "temporal_static_filter": str(
                (PROJECT_ROOT / "track4d_calib/track_temporal_static_filter.py").resolve()
            ),
            "temporal_static_filter_sha256": _sha256(
                PROJECT_ROOT / "track4d_calib/track_temporal_static_filter.py"
            ),
        },
    }
    if observation_source == "temporal_tracks":
        assert track_root is not None
        manifest.update(
            {
                "temporal_pose_root": str(temporal_pose_root),
                "track_root": str(track_root),
                "track_files_sha256": track_file_attestation(track_root),
            }
        )
    else:
        observation_roots = [
            Path(bundles_by_frame[int(frame)]["root"]).resolve()
            for frame in frames
        ]
        manifest.update(
            {
                "independent_observation_roots": [
                    str(root) for root in observation_roots
                ],
                "independent_observation_files_sha256": {
                    str(root): track_file_attestation(root)
                    for root in observation_roots
                },
            }
        )
    if center_consensus is not None:
        consensus_root = Path(args.center_consensus_root).resolve()
        manifest["center_consensus_source"] = {
            "root": str(consensus_root),
            "manifest_sha256": _sha256(
                consensus_root / "closure_manifest.json"
            ),
            "protocol": str(center_consensus["protocol"]),
            "observation_source": str(center_consensus["observation_source"]),
            "new_numeric_hyperparameters": 0,
        }
    manifest_path = output / "closure_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        output / "prediction_attestation.json",
        {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a public PTZ shared center with temporal tracks, then "
            "cross-fit fixed-center rotations before geometry-only commit."
        )
    )
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--stage-a-seal", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-views", nargs="+", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--fit-budget", type=int, default=150)
    parser.add_argument("--support-reprojection-px", type=float, default=2.5)
    parser.add_argument("--focus-reprojection-px", type=float, default=4.0)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument(
        "--observation-source",
        choices=sorted(OBSERVATION_SOURCES),
        default="temporal_tracks",
    )
    parser.add_argument(
        "--disable-temporal-static-filter",
        action="store_true",
        help="ablation: retain per-frame point banks without cross-time filtering",
    )
    parser.add_argument(
        "--center-consensus-root",
        type=Path,
        help=(
            "Attested independent-same-frame closure used only to rank the "
            "same Static+PnP center candidates alongside temporal tracks."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "pose_root": result["pose_root"],
                "candidate_success_count": result["candidate_success_count"],
                "commit_count": result["commit_count"],
            },
            indent=2,
        )
    )
