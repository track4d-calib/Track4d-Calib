from pathlib import Path
import json, subprocess, os, time
from concurrent.futures import ThreadPoolExecutor
import sys
from track_partition import fitting_track_root

root = Path(sys.argv[1])
base_command = json.loads((root / "PNP_COMMAND.json").read_text())
base_command[base_command.index("--per-frame-track-root") + 1] = str(fitting_track_root(root))
env = dict(
    os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
)
config = json.loads((root / "config.json").read_text())
frame_count = config["frames"]
worker_count = min(8, frame_count, len(os.sched_getaffinity(0)))
expected = frame_count * len(config["dynamic_views"])
started = time.perf_counter()


def run(i):
    command = base_command.copy()
    command[command.index("--frames") + 1] = " ".join(map(str, range(i, frame_count, worker_count)))
    command[command.index("--output-root") + 1] = str(root / f"pnp/shard_{i}")
    with (root / f"pnp_{i}.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((root / f"pnp/shard_{i}/predictions_pre_gt.json").read_text())


with ThreadPoolExecutor(max_workers=worker_count) as p:
    parts = list(p.map(run, range(worker_count)))
rows = sorted(
    [row for p in parts for row in p["rows"]], key=lambda x: (x["frame"], x["view"])
)
(root / "PNP_SUMMARY.json").write_text(
    json.dumps(
        {
            "rows": rows,
            "success": sum(bool(x["solver"]["success"]) for x in rows),
            "expected": expected,
            "pnp_wall_seconds": time.perf_counter() - started,
        },
        indent=2,
    )
)
print("Registered", sum(bool(x["solver"]["success"]) for x in rows), "of", expected)
