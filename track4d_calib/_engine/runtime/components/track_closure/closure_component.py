#!/usr/bin/env python3
"""Run and verify the track_closure component-local-fallback calibration closure."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

from relay_contract import (
    BANDS,
    FRAMES,
    ROLES,
    RUNTIME_TARGETS,
    STATIC_VIEWS,
    TARGET_VIEWS,
    validate_relay_file,
    validate_runtime_xyz_cache_payload,
)


VERSION = "track_closure"
PROTOCOL = "track_closure-track-driven-component-local-fallback-closure-v1"
VERSION_ROOT = Path(__file__).resolve().parent
VERSION_MANIFEST = VERSION_ROOT / "VERSION_MANIFEST.json"
WRAPPER = VERSION_ROOT / "complete_closure.py"
GEOMETRY = VERSION_ROOT / "geometry"
DIRECT_WINDOW_SIZE = 5
DIRECT_MIN_SHARED_POINTS = 8
MAX_COMPONENT_VIEWS = 3
FROZEN_OPTIONS = {
    "arm": "D",
    "direct_max_nfev": 100,
    "direct_robust_loss": os.environ.get("TRACK4D_CALIB_CLOSURE_LOSS", "huber"),
    "direct_reprojection_weight": 1.0,
    "direct_epipolar_weight": 1.0,
    "direct_convergence_tolerance": 1e-4,
    "direct_x_scale": "jac",
    "workers": 9,
    "observation_executor": "process",
    "export_uncommitted_candidates": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path, files: Iterable[Path] | None = None) -> tuple[str, int]:
    base = root.resolve()
    selected = sorted(
        (path.resolve() for path in (files or base.rglob("*")) if path.is_file()),
        key=lambda path: path.relative_to(base).as_posix(),
    )
    digest = hashlib.sha256()
    for path in selected:
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest(), len(selected)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    if config.get("method_version") != VERSION:
        raise ValueError(f"config is not for {VERSION}: {path}")
    if str(config.get("scene", "")) == "":
        raise ValueError("scene must be non-empty")
    required = {
        "runtime_scene",
        "base_pose_root",
        "candidate_pose_root",
        "spatial_track_root",
        "temporal_track_root",
        "static_pnp_diagnostics",
        "relay_cache",
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or (missing := sorted(required - set(inputs))):
        raise ValueError(f"config inputs are incomplete: {missing}")
    config["inputs"] = {
        name: str(Path(value).expanduser().resolve()) for name, value in inputs.items()
    }
    return config


def _pose_files(root: Path) -> list[Path]:
    return [
        root / f"f{frame:06d}/cameras_pred/{name}"
        for frame in FRAMES
        for name in ("extri.yml", "intri.yml")
    ]


def _validate_pose_grid(root: Path) -> dict[str, Any]:
    if str(GEOMETRY) not in sys.path:
        sys.path.insert(0, str(GEOMETRY))
    from track4d_calib.cameras import CameraLoader

    missing: list[str] = []
    for frame in FRAMES:
        camera_root = root / f"f{frame:06d}/cameras_pred"
        loader = CameraLoader(camera_root)
        for view in TARGET_VIEWS:
            try:
                loader.load_camera(view)
            except ValueError:
                missing.append(f"{frame}:{view}")
    if missing:
        raise ValueError(
            f"pose grid is missing {len(missing)} dynamic cameras: {missing[:20]}"
        )
    return {
        "dynamic_camera_node_count": len(FRAMES) * len(TARGET_VIEWS),
        "dynamic_camera_roster_complete": True,
    }


def _spatial_track_files(root: Path) -> list[Path]:
    names = ("tracks.npy", "visibility.npy", "points_t0.npy", "views.txt", "meta.json")
    return [root / f"f{frame:06d}" / name for frame in FRAMES for name in names]


def _support_partition(
    component: Iterable[str],
    *,
    shared_count_by_edge: Mapping[str, Iterable[int]],
) -> list[list[str]]:
    """Mirror the direct-job support partition for contract validation."""

    views = [str(view) for view in component]
    order = {view: index for index, view in enumerate(TARGET_VIEWS)}
    if len(views) <= MAX_COMPONENT_VIEWS:
        return [views]
    clusters = {view: {view} for view in views}

    def root(view: str) -> str:
        for key, members in clusters.items():
            if view in members:
                return key
        raise KeyError(view)

    edges = []
    for left_index, left in enumerate(views):
        for right in views[left_index + 1 :]:
            key = f"{left}:{right}"
            reverse = f"{right}:{left}"
            counts = shared_count_by_edge.get(
                key, shared_count_by_edge.get(reverse, ())
            )
            support = int(sum(int(value) for value in counts))
            edges.append((-support, order[left], order[right], left, right))
    for _support, _left_order, _right_order, left, right in sorted(edges):
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            continue
        merged = clusters[left_root] | clusters[right_root]
        if len(merged) > MAX_COMPONENT_VIEWS:
            continue
        clusters[left_root] = merged
        del clusters[right_root]
    result = [sorted(members, key=order.get) for members in clusters.values()]
    result.sort(key=lambda members: min(order[view] for view in members))
    return result


def _frozen_direct_job_plan(spatial_track_root: Path) -> list[dict[str, Any]]:
    """Rebuild the exact public-track direct-job plan used by the solver."""

    root = spatial_track_root.resolve()
    plan: list[dict[str, Any]] = []
    for component_index, start in enumerate(range(0, len(FRAMES), DIRECT_WINDOW_SIZE)):
        window = list(FRAMES[start : start + DIRECT_WINDOW_SIZE])
        visibility_by_frame: list[dict[str, np.ndarray]] = []
        for frame in window:
            bundle = root / f"f{frame:06d}"
            views = (bundle / "views.txt").read_text(encoding="utf-8").split()
            visibility = np.load(
                bundle / "visibility.npy", mmap_mode="r", allow_pickle=False
            )
            if (
                visibility.ndim != 3
                or visibility.shape[0] != len(views)
                or visibility.shape[1] < 1
            ):
                raise ValueError(
                    f"invalid same-time visibility shape at frame {frame}: "
                    f"{visibility.shape}, views={len(views)}"
                )
            missing = sorted(set(TARGET_VIEWS) - set(views))
            if missing:
                raise ValueError(
                    f"same-time tracks at frame {frame} miss target views: {missing}"
                )
            index = {view: views.index(view) for view in TARGET_VIEWS}
            visibility_by_frame.append(
                {
                    view: np.asarray(visibility[index[view], 0], dtype=bool)
                    for view in TARGET_VIEWS
                }
            )

        edge_counts: dict[str, list[int]] = {}
        supported_edges: list[tuple[str, str]] = []
        for left_index, left in enumerate(TARGET_VIEWS):
            for right in TARGET_VIEWS[left_index + 1 :]:
                counts = [
                    int(np.count_nonzero(row[left] & row[right]))
                    for row in visibility_by_frame
                ]
                edge_counts[f"{left}:{right}"] = counts
                if any(count >= DIRECT_MIN_SHARED_POINTS for count in counts):
                    supported_edges.append((left, right))

        adjacency = {view: set() for view in TARGET_VIEWS}
        for left, right in supported_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        remaining = set(TARGET_VIEWS)
        source_components: list[list[str]] = []
        while remaining:
            seed = min(remaining, key=TARGET_VIEWS.index)
            remaining.remove(seed)
            queue = [seed]
            connected: list[str] = []
            while queue:
                current = queue.pop(0)
                connected.append(current)
                for neighbor in TARGET_VIEWS:
                    if neighbor in remaining and neighbor in adjacency[current]:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
            source_components.append(
                [view for view in TARGET_VIEWS if view in set(connected)]
            )

        components: list[list[str]] = []
        for source in source_components:
            components.extend(
                _support_partition(source, shared_count_by_edge=edge_counts)
            )
        plan.extend(
            {
                "component_index": int(component_index),
                "spatial_component_index": int(spatial_index),
                "window": list(window),
                "component": list(component),
            }
            for spatial_index, component in enumerate(components)
        )
    return plan


def _normalized_component_plan(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "component_index": int(row["component_index"]),
                "spatial_component_index": int(row["spatial_component_index"]),
                "window": [int(frame) for frame in row["window"]],
                "component": [str(view) for view in row["component"]],
            }
        )
    return normalized


def _validate_component_plan(
    expected_rows: Iterable[Mapping[str, Any]],
    observed_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate result rows against the data-derived direct-job plan."""

    expected = _normalized_component_plan(expected_rows)
    observed = _normalized_component_plan(observed_rows)
    expected_windows = [
        list(FRAMES[start : start + DIRECT_WINDOW_SIZE])
        for start in range(0, len(FRAMES), DIRECT_WINDOW_SIZE)
    ]
    rows_by_window: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(expected_windows))
    }
    valid_window_labels = True
    for row in observed:
        component_index = row["component_index"]
        if component_index not in rows_by_window:
            valid_window_labels = False
            continue
        if row["window"] != expected_windows[component_index]:
            valid_window_labels = False
        rows_by_window[component_index].append(row)

    components_are_nonempty = True
    component_views_are_known = True
    target_views_appear_once = True
    windows_are_disjoint = True
    windows_have_complete_union = True
    spatial_indices_are_dense = True
    components_are_unique = True
    for window_index, rows in rows_by_window.items():
        indices = [row["spatial_component_index"] for row in rows]
        spatial_indices_are_dense &= sorted(indices) == list(range(len(rows)))
        keys = [tuple(row["component"]) for row in rows]
        components_are_unique &= len(keys) == len(set(keys))

        flattened = [view for row in rows for view in row["component"]]
        components_are_nonempty &= all(bool(row["component"]) for row in rows)
        component_views_are_known &= all(view in TARGET_VIEWS for view in flattened)
        target_views_appear_once &= all(
            flattened.count(view) == 1 for view in TARGET_VIEWS
        ) and len(flattened) == len(TARGET_VIEWS)
        windows_are_disjoint &= len(flattened) == len(set(flattened))
        windows_have_complete_union &= set(flattened) == set(TARGET_VIEWS)

    checks = {
        "rows_match_frozen_direct_job_plan": observed == expected,
        "component_count_matches_frozen_direct_job_plan": len(observed)
        == len(expected),
        "five_frame_windows_are_exact": valid_window_labels
        and set(rows_by_window) == set(range(len(expected_windows)))
        and all(rows_by_window.values()),
        "spatial_component_indices_are_dense_per_window": spatial_indices_are_dense,
        "components_are_nonempty": components_are_nonempty,
        "components_are_unique": components_are_unique,
        "every_component_view_is_a_known_dynamic_target": component_views_are_known,
        "each_target_view_appears_exactly_once_per_window": target_views_appear_once,
        "components_do_not_overlap_within_window": windows_are_disjoint,
        "each_window_view_union_is_complete": windows_have_complete_union,
    }
    return {
        "protocol": "track_closure-data-derived-direct-job-plan-contract-v2",
        "expected_component_count": len(expected),
        "observed_component_count": len(observed),
        "window_component_counts": [
            len(rows_by_window[index]) for index in range(len(expected_windows))
        ],
        "expected_plan": expected,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _temporal_track_files(root: Path) -> list[Path]:
    names = ("tracks.npy", "visibility.npy", "points_t0.npy", "views.txt", "meta.json")
    return [root / name for name in names]


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[:20])


def _validate_runtime_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    static = [str(view) for view in payload.get("observed_short_views", [])]
    targets = [
        str(view) for view in payload.get("long_views", []) if str(view) in TARGET_VIEWS
    ]
    if static != list(STATIC_VIEWS) or targets != list(TARGET_VIEWS):
        raise ValueError({"static_count": len(static), "target_views": targets})
    return {
        "path": str(path),
        "sha256": sha256(path),
        "public_static_count": len(static),
        "dynamic_train_views": targets,
    }


def _validate_relay_cache(path: Path) -> dict[str, Any]:
    payload, audit = validate_relay_file(
        path, bands=BANDS, frames=FRAMES, roles=ROLES
    )
    attestation = payload.get("attestation", {})
    forbidden = {
        "GT_used": True,
        "NVS_used": True,
        "test_RGB_or_cameras_used": True,
        "reference_pose_used": True,
    }
    violated = [key for key, value in forbidden.items() if attestation.get(key) is value]
    if violated:
        raise ValueError(f"relay cache violates public-method boundary: {violated}")
    runtime = dict(payload.get("runtime", {}))
    inference = runtime.get(
        "single_model_equivalent_inference_seconds_per_frame",
        runtime.get("single_model_equivalent_seconds_per_frame"),
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "protocol": payload.get("protocol"),
        "observation_count": int(audit["node_role_count"]),
        "minimum_points": int(audit["minimum_points_per_node_role"]),
        "node_roles_at_least_12": int(audit["node_roles_at_least_minimum"]),
        **audit,
        "single_model_equivalent_inference_seconds_per_frame": float(inference),
        "public_only_attestation": True,
    }


def _algorithm_source_files() -> list[Path]:
    roots = (VERSION_ROOT / "chain", GEOMETRY)
    files = [
        Path(__file__).resolve(),
        WRAPPER,
        VERSION_ROOT / "relay_contract.py",
        VERSION_ROOT / "materialize_complete_candidate_grid.py",
        VERSION_ROOT / "build_conditioned_relay.py",
    ]
    for root in roots:
        files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(set(path.resolve() for path in files))


def algorithm_source_digest() -> tuple[str, int]:
    return tree_digest(VERSION_ROOT, _algorithm_source_files())


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = {name: Path(value) for name, value in config["inputs"].items()}
    runtime_manifest = inputs["runtime_scene"] / "runtime_manifest.json"
    base_files = _pose_files(inputs["base_pose_root"])
    candidate_files = _pose_files(inputs["candidate_pose_root"])
    spatial_files = _spatial_track_files(inputs["spatial_track_root"])
    temporal_files = _temporal_track_files(inputs["temporal_track_root"])
    _require_files(
        [runtime_manifest, inputs["static_pnp_diagnostics"], inputs["relay_cache"]]
        + base_files
        + candidate_files
        + spatial_files
        + temporal_files
    )
    diagnostics = read_json(inputs["static_pnp_diagnostics"])
    if diagnostics.get("gt_used") is not False or diagnostics.get("nvs_used") is not False:
        raise ValueError("Static+PnP diagnostics are not sealed GT/NVS-free")
    base_digest, base_count = tree_digest(inputs["base_pose_root"], base_files)
    candidate_digest, candidate_count = tree_digest(
        inputs["candidate_pose_root"], candidate_files
    )
    spatial_digest, spatial_count = tree_digest(inputs["spatial_track_root"], spatial_files)
    temporal_digest, temporal_count = tree_digest(
        inputs["temporal_track_root"], temporal_files
    )
    source_digest, source_count = algorithm_source_digest()
    base_grid_audit = _validate_pose_grid(inputs["base_pose_root"])
    candidate_grid_audit = _validate_pose_grid(inputs["candidate_pose_root"])
    direct_job_plan = _frozen_direct_job_plan(inputs["spatial_track_root"])
    direct_job_plan_audit = _validate_component_plan(
        direct_job_plan, direct_job_plan
    )
    if not direct_job_plan_audit["passed"]:
        raise RuntimeError(
            {"frozen_direct_job_plan_contract_failed": direct_job_plan_audit["checks"]}
        )
    version_manifest = read_json(VERSION_MANIFEST)
    expected_source = version_manifest.get("algorithm_source", {})
    if (
        source_digest != expected_source.get("tree_digest")
        or source_count != expected_source.get("file_count")
    ):
        raise RuntimeError(
            {
                "version_source_manifest_mismatch": {
                    "expected": expected_source,
                    "actual": {
                        "tree_digest": source_digest,
                        "file_count": source_count,
                    },
                }
            }
        )
    report = {
        "method_version": VERSION,
        "protocol": PROTOCOL,
        "status": "pass_before_public_geometry",
        "scene": config["scene"],
        "frozen_options": FROZEN_OPTIONS,
        "runtime": _validate_runtime_manifest(runtime_manifest),
        "inputs": {
            "base_pose": {
                "tree_digest": base_digest,
                "file_count": base_count,
                **base_grid_audit,
            },
            "candidate_pose": {
                "tree_digest": candidate_digest,
                "file_count": candidate_count,
                **candidate_grid_audit,
            },
            "spatial_tracks": {
                "tree_digest": spatial_digest,
                "file_count": spatial_count,
                "direct_job_plan": direct_job_plan_audit,
            },
            "temporal_tracks": {
                "tree_digest": temporal_digest,
                "file_count": temporal_count,
            },
            "static_pnp_diagnostics": {
                "path": str(inputs["static_pnp_diagnostics"]),
                "sha256": sha256(inputs["static_pnp_diagnostics"]),
            },
            "relay": _validate_relay_cache(inputs["relay_cache"]),
        },
        "algorithm_source": {
            "tree_digest": source_digest,
            "file_count": source_count,
            "manifest": str(VERSION_MANIFEST),
            "manifest_sha256": sha256(VERSION_MANIFEST),
            "verified": True,
        },
        "claim_boundary": {
            "GT_used": False,
            "NVS_used": False,
            "test_RGB_or_cameras_used": False,
            "reference_pose_used": False,
        },
    }
    expected = config.get("expected_input_digests", {})
    actual = {
        "base_pose": base_digest,
        "candidate_pose": candidate_digest,
        "spatial_tracks": spatial_digest,
        "temporal_tracks": temporal_digest,
        "static_pnp_diagnostics": report["inputs"]["static_pnp_diagnostics"]["sha256"],
        "relay_cache": report["inputs"]["relay"]["sha256"],
    }
    mismatches = {
        name: {"expected": value, "actual": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise RuntimeError({"sealed_input_digest_mismatch": mismatches})
    report["expected_input_digests_verified"] = bool(expected)
    return report


def command(config: Mapping[str, Any], output_root: Path) -> list[str]:
    inputs = config["inputs"]
    options = FROZEN_OPTIONS
    return [
        sys.executable,
        str(WRAPPER),
        "--arm",
        options["arm"],
        "--runtime-scene",
        inputs["runtime_scene"],
        "--base-pose-root",
        inputs["base_pose_root"],
        "--dynamic_track-pose-root",
        inputs["candidate_pose_root"],
        "--spatial-track-root",
        inputs["spatial_track_root"],
        "--temporal-track-root",
        inputs["temporal_track_root"],
        "--static-pnp-diagnostics",
        inputs["static_pnp_diagnostics"],
        "--output-root",
        str(output_root / "pose_closure"),
        "--direct-max-nfev",
        str(options["direct_max_nfev"]),
        "--direct-reprojection-weight",
        str(options["direct_reprojection_weight"]),
        "--direct-epipolar-weight",
        str(options["direct_epipolar_weight"]),
        "--workers",
        str(options["workers"]),
        "--observation-executor",
        options["observation_executor"],
        "--direct-convergence-tolerance",
        str(options["direct_convergence_tolerance"]),
        "--direct-x-scale",
        options["direct_x_scale"],
        "--export-uncommitted-candidates",
    ]


def _summary_from_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(audit["components"])
    runtime = audit.get("runtime_accounting", {})
    return {
        "method_version": VERSION,
        "protocol": PROTOCOL,
        "status": "complete_public_geometry_only",
        "component_count": len(rows),
        "candidate_success_components": int(audit["candidate_success_components"]),
        "commit_components": int(audit["commit_components"]),
        "committed_nodes": int(audit["committed_nodes"]),
        "fallback_nodes": int(audit["fallback_nodes"]),
        "exact_fallback_nodes": int(audit["exact_fallback_nodes"]),
        "reasons": dict(Counter(row.get("reason") for row in rows)),
        "runtime_seconds_per_frame": float(
            audit["calibration_seconds_per_frame_including_roma_inference"]
        ),
        "runtime_accounting": runtime,
    }


def _validate_public_component_counts(
    audit: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, bool]:
    """Validate component counts against the partitions in the visibility graph."""

    try:
        declared_count = int(audit["component_count"])
        success_count = int(audit["candidate_success_components"])
    except (KeyError, TypeError, ValueError):
        declared_count = -1
        success_count = -1
    rows = audit.get("components")
    row_count = len(rows) if isinstance(rows, list) else -1
    summary_count = int(summary.get("component_count", -1))
    return {
        "public_component_count_positive": declared_count > 0,
        "public_component_count_matches_audit_rows": (
            declared_count == row_count == summary_count
        ),
        "candidate_success_component_count_valid": (
            0 <= success_count <= declared_count
        ),
    }


def verify_output(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    audit_path = output_root / "pose_closure/PUBLIC_GEOMETRY_AUDIT.json"
    audit = read_json(audit_path)
    summary = _summary_from_audit(audit)
    direct_job_plan = _frozen_direct_job_plan(
        Path(config["inputs"]["spatial_track_root"])
    )
    direct_job_plan_audit = _validate_component_plan(
        direct_job_plan, audit["components"]
    )
    checks = {
        "complete_dynamic_pose_grid": summary["committed_nodes"]
        + summary["fallback_nodes"]
        == len(FRAMES) * len(TARGET_VIEWS),
        "all_fallback_nodes_exact": summary["fallback_nodes"]
        == summary["exact_fallback_nodes"],
        "GT_not_used": audit.get("attestation", {}).get("runtime_GT_used", False)
        is False,
        "test_or_NVS_not_used": audit.get("attestation", {}).get(
            "runtime_test_or_NVS_used", False
        )
        is False,
        "different_PTZ_centers_independent": bool(
            audit.get("attestation", {}).get(
                "different_physical_PTZ_centers_remain_independent"
            )
        ),
        **direct_job_plan_audit["checks"],
        **_validate_public_component_counts(audit, summary),
    }
    pose_root = output_root / "pose_closure/pose_arm_D"
    pose_digest, pose_file_count = tree_digest(pose_root)
    golden = config.get("golden_output_contract")
    golden_checks: dict[str, bool] = {}
    if isinstance(golden, dict):
        actual = {
            "candidate_success_components": summary["candidate_success_components"],
            "commit_components": summary["commit_components"],
            "committed_nodes": summary["committed_nodes"],
            "fallback_nodes": summary["fallback_nodes"],
            "exact_fallback_nodes": summary["exact_fallback_nodes"],
            "pose_tree_digest": pose_digest,
        }
        golden_checks = {
            name: actual.get(name) == expected for name, expected in golden.items()
        }
    checks.update({f"golden_{name}": value for name, value in golden_checks.items()})
    result = {
        **summary,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "pose_tree": {"digest": pose_digest, "file_count": pose_file_count},
        "direct_job_plan": direct_job_plan_audit,
        "public_geometry_audit": {
            "path": str(audit_path),
            "sha256": sha256(audit_path),
        },
    }
    if not result["passed"]:
        raise RuntimeError({"track_closure_output_contract_failed": checks})
    return result


def run(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    preflight_report = preflight(config)
    write_json(output / "PREFLIGHT_AUDIT.json", preflight_report)
    frozen_command = command(config, output)
    freeze = {
        "method_version": VERSION,
        "protocol": PROTOCOL,
        "status": "configured_before_geometry_evaluation",
        "frozen_options": FROZEN_OPTIONS,
        "command": frozen_command,
        "preflight_sha256": sha256(output / "PREFLIGHT_AUDIT.json"),
        "algorithm_source": preflight_report["algorithm_source"],
        "method_contract": {
            "track_driven_detect_repair_verify_commit": True,
            "frame_local_dynamic_XYZ": True,
            "one_unbounded_fit_only_30frame_center_per_physical_PTZ": True,
            "different_PTZ_centers_independent": True,
            "direct_per_node_SO3_residual_BA": True,
            "fit_confirm_final_physical_IDs_disjoint": True,
            "five_frame_three_PTZ_components": True,
            "confirm_rollback": True,
            "independent_dual_static_confirm_final": True,
            "component_atomic_commit": True,
            "recoverable_low_support_is_component_local_fallback": True,
            "preflight_runtime_share_relay_validator": True,
            "exact_StaticPnP_fallback": True,
        },
    }
    write_json(output / "METHOD_FREEZE.json", freeze)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(GEOMETRY), environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
            "TRACK4D_CALIB_SCENE": str(config["scene"]),
            "TRACK4D_CALIB_RELAY_CACHE": config["inputs"]["relay_cache"],
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.time()
    with (output / "stdout.log").open("wb") as stdout, (
        output / "stderr.log"
    ).open("wb") as stderr:
        completed = subprocess.run(
            frozen_command,
            cwd=GEOMETRY,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
    write_json(
        output / "RUN_STATUS.json",
        {
            "command": frozen_command,
            "exit_code": completed.returncode,
            "started_unix": started,
            "finished_unix": time.time(),
        },
    )
    if completed.returncode:
        raise RuntimeError(
            {"exit_code": completed.returncode, "stderr": str(output / "stderr.log")}
        )
    result = verify_output(config, output)
    write_json(output / "PUBLIC_COMPARISON.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "preflight":
        result = preflight(config)
    else:
        if args.output_root is None:
            parser.error("--output-root is required for run and verify")
        result = (
            run(config, args.output_root)
            if args.action == "run"
            else verify_output(config, args.output_root.resolve())
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
