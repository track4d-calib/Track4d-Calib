from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.geometry.audit_track_dualsource_stage_a import _tree_checksums
from scripts.geometry.evaluate_track_pose_root import evaluate_pose_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _verify_stage_a_seal(
    *,
    runtime_scene: Path,
    stage_a_root: Path,
    stage_a_seal: Path,
    selector_subdir: str,
    expected_commit_policy: str,
    frames: list[int],
    target_views: list[str],
) -> dict[str, Any]:
    runtime = Path(runtime_scene).resolve()
    stage = Path(stage_a_root).resolve()
    seal_path = Path(stage_a_seal).resolve()
    if seal_path.is_relative_to(stage):
        raise ValueError("Stage-A seal must be outside the sealed Stage-A tree")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "ok":
        raise ValueError("Stage-A seal status is not ok")
    if Path(seal["runtime_scene"]).resolve() != runtime:
        raise ValueError("runtime scene does not match Stage-A seal")
    if Path(seal["stage_a_root"]).resolve() != stage:
        raise ValueError("Stage-A root does not match seal")
    if str(seal.get("selector_subdir")) != selector_subdir:
        raise ValueError("selector subdirectory does not match Stage-A seal")
    if str(seal.get("commit_policy")) != expected_commit_policy:
        raise ValueError("commit policy does not match Stage-A seal")
    if [int(frame) for frame in seal["frames"]] != [int(frame) for frame in frames]:
        raise ValueError("frame set does not match Stage-A seal")
    if [str(view) for view in seal["target_views"]] != [str(view) for view in target_views]:
        raise ValueError("target views do not match Stage-A seal")
    allowed_runtime_excludes = {
        "checksums.sha256.json",
        "runtime_manifest.json",
    }
    runtime_excludes = {
        str(path) for path in seal.get("runtime_payload_digest_excludes", [])
    }
    if not runtime_excludes.issubset(allowed_runtime_excludes):
        raise ValueError("Stage-A seal has unsafe runtime digest exclusions")
    runtime_manifest = runtime / "runtime_manifest.json"
    expected_manifest_sha256 = str(seal.get("runtime_manifest_sha256", ""))
    if not expected_manifest_sha256 or _sha256(runtime_manifest) != expected_manifest_sha256:
        raise ValueError("runtime manifest changed after sealing")
    current_runtime_tree = _tree_checksums(
        runtime,
        excluded_relative_paths=runtime_excludes,
    )
    if current_runtime_tree["tree_digest"] != str(seal.get("runtime_tree_digest", "")):
        raise ValueError("runtime tree changed after sealing")
    current_tree = _tree_checksums(stage)
    if current_tree != seal["stage_a_tree"]:
        raise ValueError("Stage-A tree changed after sealing")
    return {
        "path": str(seal_path),
        "sha256": _sha256(seal_path),
        "tree_digest": current_tree["tree_digest"],
        "file_count": current_tree["file_count"],
        "selector_subdir": selector_subdir,
        "commit_policy": expected_commit_policy,
        "runtime_tree_digest": current_runtime_tree["tree_digest"],
        "runtime_manifest_sha256": expected_manifest_sha256,
        "runtime_payload_digest_excludes": sorted(runtime_excludes),
    }


def evaluate_after_stage_a_seal(
    *,
    runtime_scene: Path,
    stage_a_root: Path,
    stage_a_seal: Path,
    selector_subdir: str,
    expected_commit_policy: str,
    pose_relative_root: Path | None = None,
    sealed_reference: Path,
    frames: list[int],
    target_views: list[str],
) -> dict[str, Any]:
    stage = Path(stage_a_root).resolve()
    relative_pose = (
        Path(pose_relative_root)
        if pose_relative_root is not None
        else Path(selector_subdir) / "pose_geometry_verified_hybrid"
    )
    if relative_pose.is_absolute():
        raise ValueError("pose_relative_root must be relative to Stage-A root")
    pose_root = (stage / relative_pose).resolve()
    if not pose_root.is_relative_to(stage):
        raise ValueError("pose_relative_root escapes Stage-A root")
    seal_record = _verify_stage_a_seal(
        runtime_scene=runtime_scene,
        stage_a_root=stage,
        stage_a_seal=stage_a_seal,
        selector_subdir=selector_subdir,
        expected_commit_policy=expected_commit_policy,
        frames=frames,
        target_views=target_views,
    )
    result = evaluate_pose_root(
        runtime_scene=runtime_scene,
        pose_root=pose_root,
        sealed_reference=sealed_reference,
        frames=frames,
        target_views=target_views,
    )
    result["stage_a_seal"] = seal_record
    result["evaluated_pose_relative_root"] = relative_pose.as_posix()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a hidden-parent pose root only after Stage-A seal verification"
    )
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--stage-a-seal", type=Path, required=True)
    parser.add_argument("--selector-subdir", required=True)
    parser.add_argument("--expected-commit-policy", choices=["mean", "confidence95"], required=True)
    parser.add_argument("--pose-relative-root", type=Path)
    parser.add_argument("--sealed-reference", type=Path, required=True)
    parser.add_argument("--frames", type=_tokens, required=True)
    parser.add_argument("--target-views", type=_tokens, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    stage = args.stage_a_root.resolve()
    if output.is_relative_to(stage):
        raise ValueError("pose evaluation output must be outside the sealed Stage-A tree")
    result = evaluate_after_stage_a_seal(
        runtime_scene=args.runtime_scene,
        stage_a_root=stage,
        stage_a_seal=args.stage_a_seal,
        selector_subdir=args.selector_subdir,
        expected_commit_policy=args.expected_commit_policy,
        pose_relative_root=args.pose_relative_root,
        sealed_reference=args.sealed_reference,
        frames=[int(frame) for frame in args.frames],
        target_views=[str(view) for view in args.target_views],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
