"""One physical-GPU service lane with optional measured matching pacing.

At pace=0 this only serializes GPU services and records real intervals. At
pace=0.15 each matching request occupies the lane for at least its model
startup time plus pairs*0.15 seconds. CPU geometry may progress during that
interval. This is a
real paced run, not a max-stage throughput substituted for segment latency.
"""

from __future__ import annotations

import fcntl
import functools
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

try:
    from pipeline import write_json
except ModuleNotFoundError:  # imported as a package by tests
    from scripts.pipeline import write_json

ENVIRONMENT_KEY = "TRACK4D_CALIB_PIPELINE_CONFIG"
_recorder = None


class GPURecorder:
    def __init__(self, config: dict[str, Any]):
        self.root = Path(config["root"]).resolve()
        self.pace = float(config["matching_pace_seconds_per_pair"])
        if self.pace not in (0.0, 0.15):
            raise ValueError("matching pace must be 0 (actual) or 0.15")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "gpu.lock").open("a+")
        self.thread_lock = threading.RLock()
        self.local = threading.local()
        self.event_path = self.root / f"gpu_events_{os.getpid()}.jsonl"

    def model_startup(self, function: Callable, *args, **kwargs):
        """Keep model construction/device transfer outside the pair quota.

        HLOC constructs a model inside ``match_from_paths``. Without this
        nested timer a slow model load would be absorbed by matching padding.
        This timer does not acquire another flock or create overlapping events.
        """
        started = time.monotonic()
        try:
            result = function(*args, **kwargs)
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            return result
        finally:
            if getattr(self.local, "startup_seconds", None) is not None:
                self.local.startup_seconds += time.monotonic() - started

    def call(self, kind: str, count: int, function: Callable, *args, **kwargs):
        requested = time.monotonic()
        with self.thread_lock:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX)
            acquired = time.monotonic()
            error = None
            self.local.startup_seconds = 0.0
            try:
                result = function(*args, **kwargs)
                # Include asynchronous CUDA work in its owning service.
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                inference_end = time.monotonic()
                deadline = acquired + (
                    self.local.startup_seconds + count * self.pace
                    if kind == "match" else 0.0
                )
                while time.monotonic() < deadline:
                    time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            except BaseException as exc:
                inference_end = time.monotonic()
                error = repr(exc)
                raise
            finally:
                finished = time.monotonic()
                event = {
                    "pid": os.getpid(), "kind": kind, "count": count,
                    "requested_monotonic": requested, "acquired_monotonic": acquired,
                    "inference_finished_monotonic": inference_end,
                    "finished_monotonic": finished,
                    "queue_seconds": acquired - requested,
                    "inference_seconds": inference_end - acquired,
                    "model_startup_seconds_in_request": self.local.startup_seconds,
                    "service_seconds": finished - acquired,
                    "padding_seconds": finished - inference_end,
                    "matching_pace_seconds_per_pair": self.pace,
                    "success": error is None, "error": error,
                }
                self.local.startup_seconds = None
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
                with self.event_path.open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                self.last_event = event
        return result


def recorder() -> GPURecorder | None:
    global _recorder
    path = os.environ.get(ENVIRONMENT_KEY)
    if not path:
        return None
    if _recorder is None:
        _recorder = GPURecorder(json.loads(Path(path).read_text()))
    return _recorder


def install_hloc() -> None:
    service = recorder()
    if service is None:
        return
    from hloc import extract_features, match_features
    if getattr(match_features.match_from_paths, "_calibration_timed", False):
        return
    if os.environ.get("TRACK4D_CALIB_RESIDENT_MODELS") == "1":
        try:
            from resident_frontend import install_model_cache
        except ModuleNotFoundError:
            from scripts.resident_frontend import install_model_cache
        install_model_cache(extract_features)
        install_model_cache(match_features)
    original_extract, original_match = extract_features.main, match_features.match_from_paths
    original_load = match_features.dynamic_load

    @functools.wraps(original_load)
    def load(*args, **kwargs):
        model_class = service.model_startup(original_load, *args, **kwargs)

        @functools.wraps(model_class)
        def construct(*model_args, **model_kwargs):
            model = service.model_startup(model_class, *model_args, **model_kwargs)
            if getattr(model, "_calibration_startup_instrumented", False):
                return model
            for name in ("eval", "to"):
                original = getattr(model, name)

                @functools.wraps(original)
                def initialize(*init_args, _original=original, **init_kwargs):
                    return service.model_startup(_original, *init_args, **init_kwargs)

                setattr(model, name, initialize)
            model._calibration_startup_instrumented = True
            return model

        return construct

    @functools.wraps(original_extract)
    def extract(*args, **kwargs):
        return service.call("extract", 0, original_extract, *args, **kwargs)

    @functools.wraps(original_match)
    def match(*args, **kwargs):
        pairs = kwargs.get("pairs_path", kwargs.get("pairs", args[1] if len(args) > 1 else None))
        if pairs is None:
            raise ValueError("matching call omitted pair list")
        count = sum(bool(line.strip()) for line in Path(pairs).read_text().splitlines())
        return service.call("match", count, original_match, *args, **kwargs)

    match._calibration_timed = True
    extract_features.main, match_features.match_from_paths = extract, match
    match_features.dynamic_load = load


def install_cotracker() -> None:
    service = recorder()
    if service is None:
        return
    from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
    for cls in (CoTrackerPredictor, CoTrackerOnlinePredictor):
        original = cls.forward
        if getattr(original, "_calibration_timed", False):
            continue

        def forward(self, *args, _original=original, **kwargs):
            return service.call("tracking", 0, _original, self, *args, **kwargs)

        forward._calibration_timed = True
        cls.forward = forward


def summarize(root: Path, *, expected_pairs: int) -> dict[str, Any]:
    events = [json.loads(line) for path in sorted(root.glob("gpu_events_*.jsonl"))
              for line in path.read_text().splitlines()]
    events.sort(key=lambda row: row["acquired_monotonic"])
    if any(not row["success"] for row in events):
        raise ValueError("failed GPU service")
    if any(left["finished_monotonic"] > right["acquired_monotonic"] + 1e-6
           for left, right in zip(events, events[1:])):
        raise ValueError("overlapping services in the single-GPU lane")
    matches = [row for row in events if row["kind"] == "match"]
    count = sum(row["count"] for row in matches)
    if count != expected_pairs:
        raise ValueError(f"GPU matching ledger has {count} pairs, expected {expected_pairs}")
    result = {
        "status": "verified_single_gpu_service_ledger", "matching_pairs": count,
        "matching_equivalent_seconds": count * 0.15,
        "matching_pace_seconds_per_pair": matches[0]["matching_pace_seconds_per_pair"] if matches else 0.0,
        "service_seconds_by_kind": {
            kind: sum(row["service_seconds"] for row in events if row["kind"] == kind)
            for kind in sorted({row["kind"] for row in events})},
        "matching_padding_seconds": sum(row["padding_seconds"] for row in matches),
        "matching_model_startup_seconds": sum(row.get("model_startup_seconds_in_request", 0.0) for row in matches),
        "matching_slower_than_pace_seconds": sum(
            max(0.0, row["inference_seconds"] - row.get("model_startup_seconds_in_request", 0.0) - row["count"] * 0.15)
            for row in matches
        ),
        "events": events,
    }
    write_json(root / "GPU_SERVICE_SUMMARY.json", result)
    return result
