from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from track4d_calib.cameras import CameraLoader
from track4d_calib.public_runtime_protocols import (
    PUBLIC_NON_COMMON_CENTER_RUNTIME_PROTOCOLS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_checksums(
    root: Path,
    excluded_relative_paths: set[str] | None = None,
) -> dict[str, Any]:
    excluded = excluded_relative_paths or set()
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }
    aggregate = hashlib.sha256()
    for relative, digest in files.items():
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "algorithm": "sha256(sorted_relative_path_nul_file_sha256_newline)_v1",
        "file_count": len(files),
        "tree_digest": aggregate.hexdigest(),
        "files": files,
    }


def _node_grid(frames: Iterable[int], views: Iterable[str]) -> set[tuple[int, str]]:
    return {(int(frame), str(view)) for frame in frames for view in views}


def _row_grid(rows: Iterable[dict[str, Any]]) -> set[tuple[int, str]]:
    return {(int(row["frame"]), str(row["view"])) for row in rows}


def _read_bundle_views(root: Path) -> list[str]:
    return [line.strip() for line in (root / "views.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def _camera_hashes(pose_root: Path, frames: list[int]) -> dict[str, dict[str, str]]:
    result = {}
    for frame in frames:
        camera_root = pose_root / f"f{frame:06d}" / "cameras_pred"
        result[f"f{frame:06d}"] = {
            name: _sha256(camera_root / name) for name in ("extri.yml", "intri.yml")
        }
    return result


def _add_check(
    checks: list[dict[str, Any]],
    condition: bool,
    name: str,
    details: Any,
    failure_message: str | None = None,
) -> None:
    checks.append({"name": name, "passed": bool(condition), "details": details})
    if not condition:
        raise ValueError(failure_message or f"Stage-A audit failed: {name}")


def _verify_attested_prediction(
    *,
    label: str,
    root: Path,
    runtime_scene: Path,
    runtime_digest: str,
    static_views: list[str],
    target_views: list[str],
    frames: list[int],
    fit_budget: int,
    expected_track_mode: str,
    checks: list[dict[str, Any]],
    allow_solver_failures: bool = False,
) -> dict[str, Any]:
    prediction_path = root / "predictions_pre_gt.json"
    attestation_path = root / "prediction_attestation.json"
    prediction = _load_json(prediction_path)
    attestation = _load_json(attestation_path)

    attestation_ok = (
        Path(attestation["path"]).resolve() == prediction_path.resolve()
        and str(attestation["sha256"]) == _sha256(prediction_path)
    )
    _add_check(
        checks,
        attestation_ok,
        f"{label}_prediction_attestation",
        attestation,
        f"{label} prediction attestation does not match the sealed prediction file",
    )
    _add_check(
        checks,
        Path(prediction["runtime_scene"]).resolve() == runtime_scene.resolve()
        and str(prediction["runtime_tree_digest"]) == runtime_digest,
        f"{label}_runtime_binding",
        {
            "runtime_scene": prediction["runtime_scene"],
            "runtime_tree_digest": prediction["runtime_tree_digest"],
        },
    )
    _add_check(
        checks,
        prediction.get("gt_used") is False and prediction.get("nvs_used") is False,
        f"{label}_pre_gt_pre_nvs",
        {"gt_used": prediction.get("gt_used"), "nvs_used": prediction.get("nvs_used")},
    )
    _add_check(
        checks,
        str(prediction["track_source_mode"]) == expected_track_mode,
        f"{label}_track_source_mode",
        prediction["track_source_mode"],
    )
    _add_check(
        checks,
        [str(view) for view in prediction["static_views"]] == static_views
        and [str(view) for view in prediction["target_views"]] == target_views
        and [int(frame) for frame in prediction["frames"]] == frames,
        f"{label}_split",
        {
            "static_views": prediction["static_views"],
            "target_views": prediction["target_views"],
            "frames": prediction["frames"],
        },
    )
    _add_check(
        checks,
        int(prediction["fit_point_budget"]) == fit_budget,
        f"{label}_fit_budget",
        prediction["fit_point_budget"],
    )

    expected_grid = _node_grid(frames, target_views)
    rows = prediction["rows"]
    _add_check(
        checks,
        len(rows) == len(expected_grid) and _row_grid(rows) == expected_grid,
        f"{label}_complete_node_grid",
        {"expected": len(expected_grid), "actual": len(rows)},
    )
    if label == "base":
        success_count = sum(
            bool((row.get("solver") or {}).get("success", False))
            for row in rows
        )
        _add_check(
            checks,
            bool(allow_solver_failures) or success_count == len(rows),
            (
                "base_solver_failures_explicitly_allowed"
                if allow_solver_failures
                else "base_all_nodes_solved"
            ),
            {
                "success": success_count,
                "total": len(rows),
            },
        )

    allowed_bundle_views = set(static_views + target_views)
    track_roots = [Path(path) for path in prediction["track_roots"]]
    expected_root_count = len(frames) if expected_track_mode == "independent_per_frame_bundle" else 1
    _add_check(
        checks,
        len(track_roots) == expected_root_count,
        f"{label}_track_root_count",
        {"expected": expected_root_count, "actual": len(track_roots)},
    )
    for track_root in track_roots:
        file_hashes = prediction["track_files_sha256"].get(str(track_root.resolve()))
        hashes_match = bool(file_hashes) and all(
            (track_root / name).is_file() and _sha256(track_root / name) == digest
            for name, digest in file_hashes.items()
        )
        _add_check(
            checks,
            hashes_match,
            f"{label}_track_hashes:{track_root.name}",
            file_hashes,
        )
        bundle_views = _read_bundle_views(track_root)
        _add_check(
            checks,
            set(bundle_views) == allowed_bundle_views and len(bundle_views) == len(allowed_bundle_views),
            f"{label}_legal_bundle_views:{track_root.name}",
            bundle_views,
            f"{label} bundle contains undeclared or held-out views: {track_root}",
        )

    pose_root = Path(prediction["pose_root"])
    actual_hashes = _camera_hashes(pose_root, frames)
    _add_check(
        checks,
        actual_hashes == prediction["output_camera_hashes"],
        f"{label}_camera_hashes",
        {"frame_count": len(actual_hashes)},
    )
    for row in rows:
        camera_root = pose_root / f"f{int(row['frame']):06d}" / "cameras_pred"
        if not bool((row.get("solver") or {}).get("success", False)):
            if allow_solver_failures:
                try:
                    CameraLoader(camera_root).load_camera(str(row["view"]))
                except ValueError:
                    continue
                raise ValueError(
                    f"{label} failed row unexpectedly contains a camera: "
                    f"frame={row['frame']} view={row['view']}"
                )
            continue
        CameraLoader(camera_root).load_camera(str(row["view"]))
    return prediction


def audit_stage_a(
    *,
    runtime_scene: Path,
    stage_a_root: Path,
    expected_target_views: list[str],
    expected_frames: list[int],
    expected_static_views: list[str] | None = None,
    expected_fit_budget: int = 150,
    selector_subdir: str = "dual_source_selector",
    expected_commit_policy: str = "mean",
    expected_missing_base_repair_protocol: str = "none",
    expected_successful_base_policy: str = "verify_commit",
    runtime_manifest_outside_payload_digest: bool = False,
    expected_runtime_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    runtime_scene = Path(runtime_scene).resolve()
    stage_a_root = Path(stage_a_root).resolve()
    target_views = [str(view) for view in expected_target_views]
    frames = [int(frame) for frame in expected_frames]
    checks: list[dict[str, Any]] = []
    if Path(selector_subdir).name != selector_subdir or selector_subdir in {"", ".", ".."}:
        raise ValueError(f"selector_subdir must be one directory name: {selector_subdir}")
    if expected_commit_policy not in {"mean", "confidence95"}:
        raise ValueError(f"unsupported expected commit policy: {expected_commit_policy}")
    if expected_missing_base_repair_protocol not in {"none", "attested_multiseed"}:
        raise ValueError(
            "unsupported expected missing-base repair protocol: "
            f"{expected_missing_base_repair_protocol}"
        )
    if expected_successful_base_policy not in {"verify_commit", "retain"}:
        raise ValueError(
            "unsupported expected successful-base policy: "
            f"{expected_successful_base_policy}"
        )
    if (
        runtime_manifest_outside_payload_digest
        and expected_runtime_manifest_sha256 is None
    ):
        raise ValueError(
            "expected_runtime_manifest_sha256 is required when "
            "runtime_manifest_outside_payload_digest is enabled"
        )

    runtime_manifest_path = runtime_scene / "runtime_manifest.json"
    runtime_manifest_sha256 = _sha256(runtime_manifest_path)
    runtime_manifest = _load_json(runtime_manifest_path)
    runtime_checksums = _load_json(runtime_scene / "checksums.sha256.json")
    runtime_payload_digest_excludes = {"checksums.sha256.json"}
    if runtime_manifest_outside_payload_digest:
        runtime_payload_digest_excludes.add("runtime_manifest.json")
    actual_runtime_checksums = _tree_checksums(
        runtime_scene,
        excluded_relative_paths=runtime_payload_digest_excludes,
    )
    runtime_digest = str(runtime_checksums["tree_digest"])
    observed_static_views = [
        str(view) for view in runtime_manifest["observed_short_views"]
    ]
    static_views = (
        observed_static_views
        if expected_static_views is None
        else [str(view) for view in expected_static_views]
    )
    static_subset_valid = bool(
        static_views
        and "00" in static_views
        and len(static_views) == len(set(static_views))
        and not (set(static_views) - set(observed_static_views))
    )
    _add_check(
        checks,
        static_subset_valid,
        "static_views_are_declared_observed_subset",
        {
            "observed": observed_static_views,
            "expected": static_views,
            "excluded": [
                view for view in observed_static_views if view not in static_views
            ],
        },
        "expected static views are not a valid observed subset",
    )
    _add_check(
        checks,
        actual_runtime_checksums == runtime_checksums,
        "runtime_tree_recomputed",
        {
            "declared": runtime_checksums.get("tree_digest"),
            "actual": actual_runtime_checksums.get("tree_digest"),
        },
        "runtime tree does not match checksums.sha256.json",
    )
    if expected_runtime_manifest_sha256 is not None:
        _add_check(
            checks,
            runtime_manifest_sha256 == expected_runtime_manifest_sha256,
            "runtime_manifest_attestation",
            {
                "expected": expected_runtime_manifest_sha256,
                "actual": runtime_manifest_sha256,
            },
            "runtime manifest SHA256 does not match the pre-run attestation",
        )
    runtime_protocol = str(runtime_manifest.get("protocol"))
    exposed_anchor_views = runtime_manifest.get("sparse_anchor_layout", {}).get(
        "exposed_parent_views", []
    )
    valid_sparse_anchor_layout = (
        runtime_protocol != "sparse-anchor-partially-observed-parent-v1"
        or bool(exposed_anchor_views)
    )
    valid_trackstress_isolation = (
        runtime_protocol != "dyn-neu3d-trackstress-calibration-runtime-v1"
        or (
            runtime_manifest.get("gt_materialized") is False
            and runtime_manifest.get("heldout_images_materialized") is False
        )
    )
    _add_check(
        checks,
        runtime_protocol in PUBLIC_NON_COMMON_CENTER_RUNTIME_PROTOCOLS
        and valid_sparse_anchor_layout
        and valid_trackstress_isolation,
        "public_non_common_center_runtime",
        {
            "protocol": runtime_protocol,
            "exposed_parent_views": exposed_anchor_views,
            "gt_materialized": runtime_manifest.get("gt_materialized"),
            "heldout_images_materialized": runtime_manifest.get(
                "heldout_images_materialized"
            ),
        },
    )
    _add_check(
        checks,
        not (set(target_views) - set(str(view) for view in runtime_manifest["long_views"])),
        "target_views_declared_by_runtime",
        target_views,
    )
    declared_frames = runtime_manifest.get("frames")
    if declared_frames is not None:
        frame_declaration_valid = frames == [int(frame) for frame in declared_frames]
    else:
        frame_declaration_valid = (
            bool(frames)
            and len(frames) == len(set(frames))
            and min(frames) >= 0
            and max(frames) < int(runtime_manifest["frame_count"])
        )
    _add_check(
        checks,
        frame_declaration_valid,
        "frame_set_declared_by_runtime",
        {
            "expected_frames": frames,
            "manifest_frames": declared_frames,
            "manifest_frame_count": runtime_manifest.get("frame_count"),
        },
    )

    base = _verify_attested_prediction(
        label="base",
        root=stage_a_root / "base_same_frame_pnp",
        runtime_scene=runtime_scene,
        runtime_digest=runtime_digest,
        static_views=static_views,
        target_views=target_views,
        frames=frames,
        fit_budget=expected_fit_budget,
        expected_track_mode="independent_per_frame_bundle",
        checks=checks,
        allow_solver_failures=(
            expected_missing_base_repair_protocol == "attested_multiseed"
        ),
    )
    temporal = _verify_attested_prediction(
        label="temporal",
        root=stage_a_root / "temporal_pose_fit150",
        runtime_scene=runtime_scene,
        runtime_digest=runtime_digest,
        static_views=static_views,
        target_views=target_views,
        frames=frames,
        fit_budget=expected_fit_budget,
        expected_track_mode="shared_temporal_bundle",
        checks=checks,
    )

    selector_root = stage_a_root / selector_subdir
    selector_path = selector_root / "hybrid_manifest.json"
    selector = _load_json(selector_path)
    expected_protocol = (
        "track-driven-public-static-pnp-missing-closure-v1"
        if (
            expected_missing_base_repair_protocol == "attested_multiseed"
            and expected_successful_base_policy == "retain"
        )
        else
        "hidden-parent-attested-multiseed-missing-base-closure-v5"
        if expected_missing_base_repair_protocol == "attested_multiseed"
        else "hidden-parent-dual-source-equal-budget-temporal-closure-v1"
    )
    expected_score_protocol = (
        "missing-base-attested-multiseed-inlier-closure-v1"
        if expected_missing_base_repair_protocol == "attested_multiseed"
        else "equal-fit-matcher-and-temporal-unseen-holdouts-v1"
    )
    _add_check(
        checks,
        selector.get("protocol") == expected_protocol
        and selector.get("score_protocol") == expected_score_protocol
        and selector.get("candidate_protocol") == "dual_source_holdout"
        and str(selector.get("missing_base_repair_protocol", "none"))
        == expected_missing_base_repair_protocol
        and str(selector.get("successful_base_policy", "verify_commit"))
        == expected_successful_base_policy,
        "selector_protocol",
        {
            "protocol": selector.get("protocol"),
            "score_protocol": selector.get("score_protocol"),
            "candidate_protocol": selector.get("candidate_protocol"),
            "missing_base_repair_protocol": selector.get(
                "missing_base_repair_protocol", "none"
            ),
            "successful_base_policy": selector.get(
                "successful_base_policy", "verify_commit"
            ),
        },
    )
    _add_check(
        checks,
        selector.get("commit_policy") == expected_commit_policy,
        "selector_commit_policy",
        selector.get("commit_policy"),
    )
    _add_check(
        checks,
        Path(selector["runtime_scene"]).resolve() == runtime_scene
        and str(selector["runtime_tree_digest"]) == runtime_digest
        and selector.get("gt_used") is False
        and selector.get("nvs_used") is False,
        "selector_pre_gt_runtime_binding",
        {
            "runtime_scene": selector.get("runtime_scene"),
            "runtime_tree_digest": selector.get("runtime_tree_digest"),
            "gt_used": selector.get("gt_used"),
            "nvs_used": selector.get("nvs_used"),
        },
    )
    _add_check(
        checks,
        selector.get("base_pose_method") == "static_map_same_frame_pnp"
        and selector.get("base_lineage_protocol") == base["protocol"],
        "selector_base_identity",
        {
            "method": selector.get("base_pose_method"),
            "lineage": selector.get("base_lineage_protocol"),
        },
    )
    _add_check(
        checks,
        [str(view) for view in selector["target_views"]] == target_views
        and [int(frame) for frame in selector["frames"]] == frames,
        "selector_split",
        {"target_views": selector["target_views"], "frames": selector["frames"]},
    )
    equal_budget = (
        int(selector["candidate_fit_budget_target"]) == expected_fit_budget
        and int(selector["fit_point_budget"]) == expected_fit_budget
        and int(selector["base_validation_budget_target"]) == expected_fit_budget
        and int(selector["temporal_validation_budget_target"]) == expected_fit_budget
        and selector.get("final_refit_point_budget") is None
    )
    _add_check(
        checks,
        equal_budget,
        "equal_fit_and_holdout_budgets_without_refit",
        {
            key: selector.get(key)
            for key in (
                "candidate_fit_budget_target",
                "fit_point_budget",
                "base_validation_budget_target",
                "temporal_validation_budget_target",
                "final_refit_point_budget",
            )
        },
    )
    _add_check(
        checks,
        selector.get("current_seed_leave_one_out") is True,
        "current_temporal_seed_left_out",
        selector.get("current_seed_leave_one_out"),
    )

    base_prediction_path = stage_a_root / "base_same_frame_pnp" / "predictions_pre_gt.json"
    temporal_prediction_path = stage_a_root / "temporal_pose_fit150" / "predictions_pre_gt.json"
    _add_check(
        checks,
        Path(selector["base_prediction_path"]).resolve() == base_prediction_path.resolve()
        and str(selector["base_prediction_sha256"]) == _sha256(base_prediction_path)
        and Path(selector["base_prediction_attestation"]).resolve()
        == (stage_a_root / "base_same_frame_pnp" / "prediction_attestation.json").resolve(),
        "selector_base_prediction_binding",
        selector.get("base_prediction_sha256"),
    )
    temporal_hashes = list(temporal["track_files_sha256"].values())
    _add_check(
        checks,
        len(temporal_hashes) == 1
        and selector["track_file_hashes"] == temporal_hashes[0]
        and str(selector["track_prediction_sha256"]) == _sha256(temporal_prediction_path),
        "selector_temporal_prediction_binding",
        selector.get("track_prediction_sha256"),
    )
    _add_check(
        checks,
        selector["input_camera_hashes"].get("base") == base["output_camera_hashes"],
        "selector_base_camera_binding",
        {"frame_count": len(selector["input_camera_hashes"].get("base", {}))},
    )

    expected_grid = _node_grid(frames, target_views)
    source_rows = selector["source_rows"]
    _add_check(
        checks,
        len(source_rows) == len(expected_grid) and _row_grid(source_rows) == expected_grid,
        "selector_complete_node_grid",
        {"expected": len(expected_grid), "actual": len(source_rows)},
    )
    allowed_sources = {"static_map_same_frame_pnp", "temporal_track_repair"}
    if expected_missing_base_repair_protocol == "attested_multiseed":
        allowed_sources.add("unresolved")
    _add_check(
        checks,
        {str(row["source"]) for row in source_rows} <= allowed_sources,
        "selector_source_labels",
        sorted({str(row["source"]) for row in source_rows}),
    )
    temporal_rows = [row for row in source_rows if row["source"] == "temporal_track_repair"]
    base_rows = [row for row in source_rows if row["source"] == "static_map_same_frame_pnp"]
    unresolved_rows = [row for row in source_rows if row["source"] == "unresolved"]
    missing_temporal_rows = [
        row for row in temporal_rows if bool(row.get("missing_base_repair", False))
    ]
    ordinary_temporal_rows = [
        row for row in temporal_rows if not bool(row.get("missing_base_repair", False))
    ]
    _add_check(
        checks,
        expected_successful_base_policy != "retain" or not ordinary_temporal_rows,
        "successful_base_policy_enforced",
        {
            "policy": expected_successful_base_policy,
            "ordinary_temporal_commits": len(ordinary_temporal_rows),
        },
    )
    _add_check(
        checks,
        int(selector["temporal_commit_count"]) == len(temporal_rows)
        and int(selector["base_retain_count"]) == len(base_rows)
        and int(selector.get("unresolved_count", 0)) == len(unresolved_rows)
        and len(temporal_rows) + len(base_rows) + len(unresolved_rows)
        == len(expected_grid),
        "selector_source_counts",
        {
            "temporal": len(temporal_rows),
            "base": len(base_rows),
            "unresolved": len(unresolved_rows),
        },
    )
    for row in ordinary_temporal_rows:
        fold_rows = row.get("crossfit_fold_rows", [])
        fold_sources = {str(fold["validation_source"]) for fold in fold_rows}
        valid_commit = (
            row.get("node_commit") is True
            and row.get("crossfit_valid") is True
            and row.get("both_directions_prefer_track") is True
            and int(row.get("fit_point_count", -1)) == expected_fit_budget
            and int(row.get("commit_candidate_fit_point_count", -1)) == expected_fit_budget
            and fold_sources == {"static_map_same_frame_pnp", "temporal_track"}
            and all(
                0 < int(fold["validation_point_count"]) <= expected_fit_budget
                and fold.get("validation_prefers_track") is True
                for fold in fold_rows
            )
        )
        _add_check(
            checks,
            valid_commit,
            f"verified_temporal_commit:f{int(row['frame']):06d}:v{row['view']}",
            {"fold_sources": sorted(fold_sources)},
        )
    for row in missing_temporal_rows:
        fold_rows = row.get("crossfit_fold_rows", [])
        valid_commit = (
            expected_missing_base_repair_protocol == "attested_multiseed"
            and row.get("node_commit") is True
            and row.get("crossfit_valid") is True
            and row.get("both_directions_prefer_track") is True
            and row.get("base_pose_available") is False
            and row.get("base_solver_success") is False
            and row.get("commit_candidate_protocol")
            == "attested-pooled-multiseed-track-pnp-v1"
            and row.get("commit_candidate_multiseed_verified") is True
            and 0 < int(row.get("commit_candidate_fit_point_count", 0))
            <= expected_fit_budget
            and len(fold_rows) == 1
            and fold_rows[0].get("validation_source")
            == "temporal_multiseed_inlier_support"
            and fold_rows[0].get("validation_prefers_track") is True
        )
        _add_check(
            checks,
            valid_commit,
            f"verified_missing_base_commit:f{int(row['frame']):06d}:v{row['view']}",
            {"protocol": row.get("commit_candidate_protocol")},
        )
    _add_check(
        checks,
        all(
            row.get("node_commit") is False
            and row.get("missing_base_repair") is True
            and row.get("base_pose_available") is False
            and row.get("base_solver_success") is False
            for row in unresolved_rows
        ),
        "unresolved_rows_are_explicit_missing_base_failures",
        {"count": len(unresolved_rows)},
    )
    failed_base_nodes = {
        (int(row["frame"]), str(row["view"]))
        for row in base["rows"]
        if not bool((row.get("solver") or {}).get("success", False))
    }
    reported_missing_nodes = {
        (int(row["frame"]), str(row["view"]))
        for row in missing_temporal_rows + unresolved_rows
    }
    _add_check(
        checks,
        failed_base_nodes == reported_missing_nodes,
        "missing_base_nodes_match_attested_solver_failures",
        {
            "base_failures": sorted(failed_base_nodes),
            "reported": sorted(reported_missing_nodes),
        },
    )
    _add_check(
        checks,
        all(row.get("node_commit") is False for row in base_rows),
        "base_rows_are_not_commits",
        {"count": len(base_rows)},
    )

    invocation_counts = {
        "base": int(selector["base_matcher_invocations"]),
        "temporal": int(selector["temporal_seed_matcher_invocations"]),
        "total": int(selector["total_matcher_invocations"]),
    }
    _add_check(
        checks,
        invocation_counts["base"] == len(frames)
        and invocation_counts["temporal"] > 0
        and invocation_counts["total"] == invocation_counts["base"] + invocation_counts["temporal"],
        "matcher_invocation_accounting",
        invocation_counts,
    )

    selector_pose_root = selector_root / "pose_geometry_verified_hybrid"
    selector_camera_hashes = _camera_hashes(selector_pose_root, frames)
    _add_check(
        checks,
        selector_camera_hashes == selector["output_camera_hashes"],
        "selector_output_camera_hashes",
        {"frame_count": len(selector_camera_hashes)},
    )
    for frame in frames:
        camera_loader = CameraLoader(selector_pose_root / f"f{frame:06d}" / "cameras_pred")
        expected_available_views = static_views + [
            view
            for view in target_views
            if not any(
                int(row["frame"]) == frame
                and str(row["view"]) == view
                and str(row["source"]) == "unresolved"
                for row in source_rows
            )
        ]
        declared_available = selector.get(
            "available_output_views_by_frame", {}
        ).get(f"f{frame:06d}", static_views + target_views)
        _add_check(
            checks,
            [str(view) for view in declared_available] == expected_available_views,
            f"selector_available_views:f{frame:06d}",
            declared_available,
        )
        camera_loader.load_cameras(expected_available_views)
        for unresolved_view in set(target_views) - set(expected_available_views):
            try:
                camera_loader.load_camera(unresolved_view)
            except ValueError:
                continue
            raise ValueError(
                "unresolved selector node unexpectedly has a target camera: "
                f"frame={frame} view={unresolved_view}"
            )

    return {
        "status": "ok",
        "protocol": selector["protocol"],
        "runtime_scene": str(runtime_scene),
        "runtime_tree_digest": runtime_digest,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "runtime_payload_digest_excludes": sorted(runtime_payload_digest_excludes),
        "stage_a_root": str(stage_a_root),
        "selector_subdir": selector_subdir,
        "commit_policy": expected_commit_policy,
        "missing_base_repair_protocol": expected_missing_base_repair_protocol,
        "successful_base_policy": expected_successful_base_policy,
        "stage_a_tree": _tree_checksums(stage_a_root),
        "target_views": target_views,
        "static_views": static_views,
        "observed_static_views": observed_static_views,
        "frames": frames,
        "fit_point_budget": expected_fit_budget,
        "node_count": len(source_rows),
        "temporal_commit_count": len(temporal_rows),
        "base_retain_count": len(base_rows),
        "missing_base_track_commit_count": len(missing_temporal_rows),
        "unresolved_count": len(unresolved_rows),
        "checks": checks,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal and audit an E185 pre-GT Stage-A run")
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--target-views", nargs="+", default=["80", "81", "83", "84"])
    parser.add_argument("--frames", nargs="+", type=int, default=list(range(30)))
    parser.add_argument("--fit-budget", type=int, default=150)
    parser.add_argument("--expected-static-views", nargs="+")
    parser.add_argument("--selector-subdir", default="dual_source_selector")
    parser.add_argument("--expected-commit-policy", choices=["mean", "confidence95"], default="mean")
    parser.add_argument(
        "--expected-missing-base-repair-protocol",
        choices=["none", "attested_multiseed"],
        default="none",
    )
    parser.add_argument(
        "--expected-successful-base-policy",
        choices=["verify_commit", "retain"],
        default="verify_commit",
    )
    parser.add_argument(
        "--runtime-manifest-outside-payload-digest",
        action="store_true",
        help=(
            "Recompute the declared runtime payload digest without "
            "runtime_manifest.json; requires a separately attested manifest SHA256"
        ),
    )
    parser.add_argument("--expected-runtime-manifest-sha256")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_stage_a(
        runtime_scene=args.runtime_scene,
        stage_a_root=args.stage_a_root,
        expected_target_views=args.target_views,
        expected_frames=args.frames,
        expected_static_views=args.expected_static_views,
        expected_fit_budget=args.fit_budget,
        selector_subdir=args.selector_subdir,
        expected_commit_policy=args.expected_commit_policy,
        expected_missing_base_repair_protocol=(
            args.expected_missing_base_repair_protocol
        ),
        expected_successful_base_policy=args.expected_successful_base_policy,
        runtime_manifest_outside_payload_digest=(
            args.runtime_manifest_outside_payload_digest
        ),
        expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"checks", "stage_a_tree"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
