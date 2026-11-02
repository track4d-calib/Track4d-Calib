"""Serve matching and tracking models within one calibration run."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import functools
import importlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

try:
    from pipeline import sha256, write_json
except ModuleNotFoundError:
    from scripts.pipeline import sha256, write_json


MODEL_CACHE_ENV = "TRACK4D_CALIB_RESIDENT_MODELS"


def install_model_cache(module):
    """Retain exact eval models by constructor configuration in this process."""
    if getattr(module.dynamic_load, "_calibration_model_cache", False):
        return
    original = module.dynamic_load
    cache, stats = {}, {"constructed": 0, "reused": 0}

    @functools.wraps(original)
    def load(source, name):
        model_type = original(source, name)

        def construct(*args, **kwargs):
            key = (source.__name__, name, json.dumps([args, kwargs], sort_keys=True, allow_nan=False))
            if key not in cache:
                cache[key] = model_type(*args, **kwargs)
                stats["constructed"] += 1
            else:
                stats["reused"] += 1
            return cache[key]

        return construct

    load._calibration_model_cache = True
    module.dynamic_load = load
    module._calibration_model_cache_stats = stats


def model_cache_stats():
    return {name: dict(getattr(module, "_calibration_model_cache_stats"))
            for name in ("hloc.extract_features", "hloc.match_features")
            if (module := sys.modules.get(name)) is not None
            and hasattr(module, "_calibration_model_cache_stats")}


def initialize_worker(environment, cwd):
    # Must precede CUDA/BLAS imports. Spawn avoids inheriting a live CUDA context.
    os.environ.update(environment)
    os.environ[MODEL_CACHE_ENV] = "1"
    os.chdir(cwd)
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def worker_ready():
    """Exercise spawn/bootstrap before starting expensive independent jobs."""
    return {"pid": os.getpid(), "resident_models": os.environ[MODEL_CACHE_ENV]}


@contextmanager
def redirected_logs(stdout, stderr):
    """Capture Python and native library output in the owning frame's log."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    try:
        with stdout.open("wb") as out, stderr.open("wb") as err:
            os.dup2(out.fileno(), 1)
            os.dup2(err.fileno(), 2)
            yield
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        for target, descriptor in zip((1, 2), saved):
            os.dup2(descriptor, target)
            os.close(descriptor)


def run_spatial_command(name, command, root):
    """Run the explicit-graph command in a persistent process."""
    root = Path(root)
    logs, audit = root / "logs", root / "open_audit"
    logs.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    outer_audit = Path(os.environ["TRACK4D_CALIB_EXTERNAL_OPEN_AUDIT"])
    if not outer_audit.is_file():
        raise FileNotFoundError(outer_audit)
    manifest, trace = logs / f"{name}.command.json", audit / f"{name}.strace.log"
    out, err = logs / f"{name}.stdout.log", logs / f"{name}.stderr.log"
    write_json(manifest, {"name": name, "command": command, "cwd": os.getcwd(),
                          "execution": "same_run_resident_worker", "pid": os.getpid()})
    write_json(trace, {"external_process_tree_open_audit": str(outer_audit)})
    started, previous = time.time(), sys.argv
    exit_code = 1
    try:
        with redirected_logs(out, err):
            # The wrapper mutates this module's builders and TemporaryDirectory
            # per frame. Reset those adapters so old frame closures never stack.
            target_star = sys.modules.get("track4d_calib.target_star_correspondence")
            if target_star is not None:
                importlib.reload(target_star)
            spec = importlib.util.spec_from_file_location("calibration_resident_spatial", command[1])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.argv = list(command[1:])
            exit_code = int(module.main())
            if exit_code:
                raise RuntimeError(f"spatial worker exited {exit_code}: {name}")
    finally:
        sys.argv = previous
        finished = time.time()
        def record(path):
            return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        result = {
            "name": name, "pid": os.getpid(), "command_manifest": record(manifest),
            "stdout": record(out), "stderr": record(err), "strace": record(trace),
            "started_unix": started, "finished_unix": finished,
            "wall_seconds": finished - started, "exit_code": exit_code,
            "model_cache": model_cache_stats(), "execution": "same_run_resident_worker",
        }
        write_json(logs / f"{name}.run.json", result)
    return result


class SpatialWorkers:
    """Bounded, run-owned workers; drain jobs before joining and dropping models."""

    def __init__(self, *, count, environment, cwd, python):
        if count < 1 or not os.path.samefile(python, sys.executable):
            raise ValueError("resident workers require positive count and the controller's Python")
        audit = environment.get("TRACK4D_CALIB_EXTERNAL_OPEN_AUDIT")
        if not audit or not Path(audit).is_file():
            raise ValueError("resident workers require the outer process-tree native open audit")
        self.pool = ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(environment, str(cwd)),
        )

    def run(self, name, command, root):
        return self.pool.submit(run_spatial_command, name, list(command), str(root)).result()

    def __enter__(self):
        try:
            self.ready = self.pool.submit(worker_ready).result()
        except BaseException:
            self.pool.shutdown(wait=True, cancel_futures=True)
            raise
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)
