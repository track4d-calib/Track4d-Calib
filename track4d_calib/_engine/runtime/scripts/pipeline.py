"""Schedule dependent geometry jobs and assemble complete frame shards."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence


def ordered_process_map(function: Callable, jobs: Sequence, workers: int):
    """Keep CPU-heavy Python geometry off the GIL, preserving result order.

    Spawn avoids inheriting a parent's CUDA/BLAS state. Callables must be module
    level and inputs pickleable. With one worker the exact serial call path is
    retained, without a child process or thread.
    """
    if workers < 1:
        raise ValueError("geometry workers must be positive")
    if workers == 1:
        yield from map(function, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        yield from pool.map(function, jobs)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Task:
    name: str
    lane: str
    action: Callable[[], Any]
    dependencies: tuple[str, ...] = ()


def execute(
    tasks: Sequence[Task], capacities: Mapping[str, int], *, journal: Path | None = None
) -> dict[str, Any]:
    """Run ready jobs within lane capacities; stop submitting after a failure.

All running jobs are drained before returning or raising. No consumer is
submitted until *every* dependency has succeeded. The journal uses the same
monotonic clock across all workers and includes scheduler/fill/drain time.
    """
    pending = {task.name: task for task in tasks}
    if len(pending) != len(tasks):
        raise ValueError("duplicate task name")
    if any(int(value) < 1 for value in capacities.values()):
        raise ValueError("lane capacities must be positive")
    for task in tasks:
        if task.lane not in capacities or set(task.dependencies) - set(pending):
            raise ValueError(f"undeclared lane or dependency: {task.name}")
    # Validate the entire DAG before causing any side effects.
    checked: set[str] = set()
    while len(checked) < len(tasks):
        ready = {task.name for task in tasks if set(task.dependencies) <= checked}
        if ready <= checked:
            raise ValueError("task dependency cycle")
        checked |= ready
    started = time.monotonic()
    results: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    active = dict.fromkeys(capacities, 0)
    failure: BaseException | None = None

    def invoke(task: Task) -> tuple[Any, dict[str, Any], BaseException | None]:
        begin = time.monotonic()
        value, error = None, None
        try:
            value = task.action()
        except BaseException as exc:
            error = exc
        end = time.monotonic()
        return value, {
            "task": task.name, "lane": task.lane,
            "dependencies": list(task.dependencies),
            "started_monotonic": begin, "finished_monotonic": end,
            "wall_seconds": end - begin, "success": error is None,
            "error": None if error is None else repr(error),
        }, error

    with ThreadPoolExecutor(max_workers=sum(capacities.values())) as pool:
        running = {}
        while pending or running:
            if failure is None:
                for name, task in list(pending.items()):
                    if (set(task.dependencies) <= results.keys()
                            and active[task.lane] < capacities[task.lane]):
                        active[task.lane] += 1
                        running[pool.submit(invoke, task)] = task
                        del pending[name]
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                task = running.pop(future)
                active[task.lane] -= 1
                value, event, error = future.result()
                events.append(event)
                if error is None:
                    results[task.name] = value
                elif failure is None:
                    failure = error
                    if journal is not None:
                        # Surface a failed worker immediately, even while a
                        # long independent GPU job is still being drained.
                        write_json(journal, {
                            "status": "failed_draining", "error": repr(failure),
                            "capacities": dict(capacities), "started_monotonic": started,
                            "wall_seconds": time.monotonic() - started,
                            "completed_tasks": len(events), "expected_tasks": len(tasks),
                            "running_tasks": sorted(task.name for task in running.values()),
                            "unsubmitted_tasks": sorted(pending), "events": list(events),
                        })
    summary = {
        "status": "complete" if failure is None else "failed_no_commit",
        "capacities": dict(capacities), "started_monotonic": started,
        "wall_seconds": time.monotonic() - started,
        "completed_tasks": len(events), "expected_tasks": len(tasks),
        "unsubmitted_tasks": sorted(pending),
        "events": sorted(events, key=lambda row: (row["started_monotonic"], row["task"])),
    }
    if journal is not None:
        write_json(journal, summary)
    if failure is not None:
        raise RuntimeError("pipeline task failed; dependent outputs were not committed") from failure
    return {"results": results, "timing": summary}


def frame_command(command: Sequence[str], frames: Sequence[int], output: Path) -> list[str]:
    """Change only the selected frame IDs and the new shard output path."""
    result = list(command)
    for flag, value in (("--frames", " ".join(map(str, frames))),
                        ("--output-root", str(output))):
        if result.count(flag) != 1:
            raise ValueError(f"expected exactly one {flag}")
        result[result.index(flag) + 1] = value
    return result


def merge_frame_predictions(
    roots: Sequence[Path], output: Path, frames: Sequence[int], *, run_root: Path,
    required_nodes: Sequence[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    """Assemble disjoint whole-frame predictions without reserializing cameras.

Every child retains the full view axis and point-bank IDs. Failed PnP nodes
remain failed; no missing camera is fabricated. Publish the merged manifest
only after coverage, configuration, input hashes and camera files agree.
    """
    run_root, output = run_root.resolve(), output.resolve()
    output.relative_to(run_root)
    if output.exists():
        raise FileExistsError(output)
    if not roots or not frames or len(frames) != len(set(frames)):
        raise ValueError("nonempty disjoint frame shards are required")
    payloads = []
    for root in roots:
        root = root.resolve()
        root.relative_to(run_root)
        payloads.append(json.loads((root / "predictions_pre_gt.json").read_text()))
    first = payloads[0]
    required = None if required_nodes is None else set(required_nodes)
    if required is not None:
        legal = {(int(frame), str(view)) for frame in frames for view in first["target_views"]}
        if (not required or len(required) != len(required_nodes)
                or required - legal or {frame for frame, _ in required} != set(frames)):
            raise ValueError("invalid sparse prediction grid")
    invariants = (
        "protocol", "runtime_scene", "runtime_tree_digest", "static_views", "target_views",
        "config", "fit_point_budget", "temporal_seed_policy", "verify_commit_used",
        "verification_protocol", "gt_used", "nvs_used", "center_frames", "centers",
        "base_pose_root", "center_pose_root", "temporal_track_root", "independent_track_root",
        "source_hashes", "claim_boundary",
    )
    seen_frames: set[int] = set()
    rows = {}
    camera_sources = {}
    track_hashes: dict[str, Any] = {}
    accepted = []
    for root, payload in zip(roots, payloads):
        for name in invariants:
            if payload.get(name) != first.get(name):
                raise ValueError(f"shard invariant changed: {name}")
        for flag in ("gt_used", "nvs_used"):
            if payload.get(flag, False) is not False:
                raise ValueError(f"forbidden shard input: {flag}")
        current = list(map(int, payload["frames"]))
        if len(current) != len(set(current)) or seen_frames & set(current):
            raise ValueError("duplicate frame shard")
        seen_frames.update(current)
        source_pose = Path(payload["pose_root"]).resolve()
        source_pose.relative_to(root.resolve())
        for frame in current:
            camera_sources[frame] = source_pose / f"f{frame:06d}"
        current_rows = {(int(row["frame"]), str(row["view"])): row for row in payload["rows"]}
        expected = {(frame, str(view)) for frame in current for view in first["target_views"]}
        if required is not None:
            expected &= required
            declared = [tuple(node) for node in payload.get("requested_nodes", [])]
            if len(declared) != len(set(declared)) or set(declared) != expected:
                raise ValueError("sparse shard request does not match required grid")
        if set(current_rows) != expected or len(current_rows) != len(payload["rows"]):
            raise ValueError("incomplete or duplicate shard prediction grid")
        rows.update(current_rows)
        for name, value in payload.get("track_files_sha256", {}).items():
            if name in track_hashes and track_hashes[name] != value:
                raise ValueError("track input changed across shards")
            track_hashes[name] = value
        accepted.extend(payload.get("accepted_nodes", []))
    if seen_frames != set(frames):
        raise ValueError("incomplete frame shard coverage")
    output.mkdir(parents=True, exist_ok=False)
    pose = output / Path(first["pose_root"]).name
    hashes = {}
    for frame in frames:
        source = camera_sources[frame]
        for name in ("intri.yml", "extri.yml"):
            if not (source / "cameras_pred" / name).is_file():
                raise FileNotFoundError(source / "cameras_pred" / name)
        shutil.copytree(source, pose / f"f{frame:06d}")
        hashes[f"f{frame:06d}"] = {
            name: sha256(pose / f"f{frame:06d}/cameras_pred" / name)
            for name in ("intri.yml", "extri.yml")
        }
    merged = {
        **first, "frames": list(frames), "pose_root": str(pose),
        "rows": [rows[(frame, str(view))] for frame in frames for view in first["target_views"]
                 if required is None or (frame, str(view)) in required],
        "track_files_sha256": track_hashes,
        "output_camera_hashes": hashes,
        "execution": {"mode": "disjoint_whole_frame_pipeline", "shards": [str(p.resolve()) for p in roots]},
    }
    if required is not None:
        merged["requested_nodes"] = [[row["frame"], row["view"]] for row in merged["rows"]]
    if "track_roots" in first:
        merged["track_roots"] = sorted({p for payload in payloads for p in payload["track_roots"]})
    if "accepted_nodes" in first:
        merged["accepted_nodes"] = sorted(accepted, key=lambda row: (int(row[0]), str(row[1])))
    prediction = output / "predictions_pre_gt.json"
    write_json(prediction, merged)
    write_json(output / "prediction_attestation.json", {"path": str(prediction), "sha256": sha256(prediction)})
    return merged
