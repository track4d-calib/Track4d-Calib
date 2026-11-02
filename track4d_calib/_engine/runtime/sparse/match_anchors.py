from pathlib import Path
import importlib.util, json, os, sys, time, hashlib
import numpy as np

root = Path(sys.argv[1])
component_root = root / "code/runtime/components/track_closure"
sys.path.insert(0, str(component_root / "geometry"))
sys.path.insert(0, str(component_root))
sys.path.insert(0, str(root / "code/runtime/scripts"))
from rig_contract import make_rig_contract
from feature_cache import cache_image_features
from matching_cache import QueryMatchCache

config = json.loads((root / "config.json").read_text())
frame_count = config["frames"]
worker_count = min(8, frame_count, len(os.sched_getaffinity(0)))
dependencies = json.loads((root / "dependencies.json").read_text())
rig = make_rig_contract(
    config["static_views"],
    config["dynamic_views"],
    center_motion_radius=None,
    static_support_count=int(config.get("static_support_count", 8)),
)
(root / "RIG.json").write_text(json.dumps(rig))
os.environ["TRACK4D_CALIB_RIG_CONTRACT"] = str(root / "RIG.json")
module_spec = importlib.util.spec_from_file_location(
    "anchor", component_root / "chain/build_second_roma_component_anchor_cache.py"
)
matcher = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(matcher)
output = root / "anchor_tracks"
output.mkdir(exist_ok=False)
protocol = {
    "kind": "current-run sparse anchor correspondences",
    "anchors": np.rint(np.linspace(0, frame_count - 1, config.get("temporal_anchor_count", 5)))
    .astype(int)
    .tolist(),
    "matcher": "RoMaV2 fast bidirectional",
    "support_count": int(config.get("static_support_count", 8)),
    "points_per_band": config.get("anchor_points_per_band", 600),
    "query_stride": 8,
    "bands": rig["relay_bands"],
    "source_policy": "middle camera of declared band",
    "frame0_support_selection": "highest accepted RGB query counts",
    "fb_threshold": 1.5,
    "seed": 0,
}
(output / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
import torch

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
started = time.perf_counter()
model, _ = matcher._load_roma_model(
    source_root=Path(dependencies["romav2_src"]),
    weights=Path(dependencies["romav2_weights"]),
    device="cuda",
    setting="fast",
    force_bidirectional=True,
)
feature_cache = cache_image_features(model, capacity=32)
pair_cache = QueryMatchCache(
    root / "anchor_matches", weights=dependencies["romav2_weights"], source=matcher.__file__,
)
views = config["static_views"] + config["dynamic_views"]
view_indices = {v: i for i, v in enumerate(views)}
width, height = config["resolution"]
queries = matcher.query_grid(height=height, width=width, stride=8)
banks = {a: [] for a in protocol["anchors"]}
selection = {}
pair_count = 0


def match(source, target, frame):
    global pair_count
    result = pair_cache.match(
        matcher.match_queries, retain=target in config["static_views"],
        model=model,
        static_image=root / f"inputs/images/{source}/{frame:06d}.png",
        target_image=root / f"inputs/images/{target}/{frame:06d}.png",
        queries=queries,
        height=height,
        width=width,
        minimum_overlap=0.2,
        maximum_cycle_error_px=1.5,
    )
    pair_count += 1
    return result


work_bands = [list(band) for band in rig["relay_bands"]]
coverage_repairs = []
minimum_coverage = int(config.get("anchor_minimum_coverage", 24))
for band_index, band in enumerate(work_bands):
    source = band[len(band) // 2]
    initial = {v: match(source, v, 0) for v in config["static_views"]}
    support = sorted(initial, key=lambda v: (-len(initial[v]["point_ids"]), v))[
        : protocol["support_count"]
    ]
    selection[source] = support
    coverage = {view: [] for view in band}
    for frame in protocol["anchors"]:
        xy = np.zeros((len(views), len(queries), 2), np.float32)
        visible = np.zeros((len(views), len(queries)), bool)
        xy[view_indices[source]] = queries
        visible[view_indices[source]] = True
        for view in support + [v for v in band if v != source]:
            result = (
                initial[view]
                if frame == 0 and view in initial
                else match(source, view, frame)
            )
            ids = np.asarray(result["point_ids"], int)
            xy[view_indices[view], ids] = result["target_points"]
            visible[view_indices[view], ids] = True
        seed_visibility = visible
        keep = np.flatnonzero(seed_visibility[[view_indices[v] for v in support]].sum(0) >= 2)
        if len(keep) > protocol["points_per_band"]:
            keep = keep[np.linspace(0, len(keep) - 1, protocol["points_per_band"]).astype(int)]
        banks[frame].append((xy[:, keep], visible[:, keep]))
        for view in band:
            coverage[view].append(int(visible[view_indices[view], keep].sum()))
    if len(band) > 1:
        for view, counts in coverage.items():
            if min(counts) < minimum_coverage:
                work_bands.append([view])
                coverage_repairs.append({"view": view, "anchor_visible_counts": counts})
    (output / "STATUS.json").write_text(
        json.dumps({"completed_bands": band_index + 1, "pair_count": pair_count})
    )
for frame, blocks in banks.items():
    frame_output = output / f"f{frame:06d}"
    frame_output.mkdir()
    xy = np.concatenate([b[0] for b in blocks], axis=1)
    visibility = np.concatenate([b[1] for b in blocks], axis=1)
    np.save(frame_output / "tracks.npy", xy[:, None])
    np.save(frame_output / "visibility.npy", visibility[:, None])
    np.save(frame_output / "track_quality.npy", visibility[:, None].astype(np.float32))
    np.save(frame_output / "points_t0.npy", xy[0])
    (frame_output / "views.txt").write_text("\n".join(views) + "\n")
    (frame_output / "meta.json").write_text(
        json.dumps(
            {
                "views": views,
                "start_frame": frame,
                "num_frames": 1,
                "num_points": xy.shape[1],
                "seed_builder": "roma_component_anchors",
                "source_views": list(selection),
            },
            indent=2,
        )
    )
(output / "RESULT.json").write_text(
    json.dumps(
        {
            "wall_seconds": time.perf_counter() - started,
            "pairs": pair_count,
            "support": selection,
            "coverage_repairs": coverage_repairs,
            "minimum_anchor_coverage": minimum_coverage,
            "feature_cache": feature_cache,
            "query_cache": {"hits": pair_cache.hits, "misses": pair_cache.misses},
        },
        indent=2,
    )
)
print("Completed", pair_count, "pairs", time.perf_counter() - started)
