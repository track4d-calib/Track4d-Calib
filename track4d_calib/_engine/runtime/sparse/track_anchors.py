from pathlib import Path
import os, sys, json, time, shutil
import cv2, numpy as np, torch

root = Path(sys.argv[1])
dependencies = json.loads((root / "dependencies.json").read_text())
sys.path.insert(0, dependencies["cotracker"])
from cotracker.predictor import CoTrackerPredictor

output = root / "cotracker_cached_tracks"
output.mkdir(exist_ok=False)
config = json.loads((root / "config.json").read_text())
frame_count = config["frames"]
worker_count = min(8, frame_count, len(os.sched_getaffinity(0)))
anchors = tuple(
    np.rint(np.linspace(0, frame_count - 1, config.get("temporal_anchor_count", 5)))
    .astype(int)
    .tolist()
)
banks = []
for f in anchors:
    p = root / f"anchor_tracks/f{f:06d}"
    views = (p / "views.txt").read_text().split()
    banks.append((np.load(p / "tracks.npy")[:, 0], np.load(p / "visibility.npy")[:, 0]))
offset = np.r_[0, np.cumsum([x[0].shape[1] for x in banks])]
(output / "PROTOCOL.json").write_text(
    json.dumps(
        {
            "kind": "current-run anchor correspondences propagated to ordinary frames",
            "feature_cache": "exact-input reuse within one video",
            "anchors": anchors,
            "source": str(root / "anchor_tracks"),
            "query_batch": 512,
            "seed": 0,
            "frame_ids": list(range(frame_count)),
        },
        indent=2,
    )
)
cv2.setNumThreads(1)
torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)
start = time.perf_counter()
model = (
    CoTrackerPredictor(checkpoint=dependencies["cotracker_checkpoint"], offline=True)
    .cuda()
    .eval()
)
h, w = model.interp_shape
original_fnet = model.model.fnet.forward
feature_cache = []
cache_stats = {"hits": 0, "misses": 0}


def cached_fnet(video):
    for prior, result in feature_cache:
        if (
            prior.shape == video.shape
            and prior.dtype == video.dtype
            and torch.equal(prior, video)
        ):
            cache_stats["hits"] += 1
            return result.clone()
    result = original_fnet(video)
    cache_stats["misses"] += 1
    if len(feature_cache) == 2:
        feature_cache.pop(0)
    feature_cache.append((video.clone(), result.clone()))
    return result


model.model.fnet.forward = cached_fnet
xy = np.zeros((len(views), frame_count, offset[-1], 2), np.float32)
vis = np.zeros(xy.shape[:-1], bool)
for i, v in enumerate(views):
    feature_cache.clear()
    frames = [
        cv2.cvtColor(
            cv2.imread(str(root / f"inputs/images/{v}/{f:06d}.png")), cv2.COLOR_BGR2RGB
        )
        for f in range(frame_count)
    ]
    H, W = frames[0].shape[:2]
    video = (
        torch.from_numpy(np.stack([cv2.resize(im, (w, h)) for im in frames]))
        .permute(0, 3, 1, 2)[None]
        .float()
        .cuda()
    )
    ids = []
    queries = []
    for a, (f, (points, valid)) in enumerate(zip(anchors, banks)):
        keep = np.flatnonzero(valid[i])
        ids.extend((offset[a] + keep).tolist())
        queries.extend(
            np.c_[
                np.full(len(keep), f),
                points[i, keep] * [(w - 1) / (W - 1), (h - 1) / (H - 1)],
            ].tolist()
        )
    queries = np.asarray(queries, np.float32)
    ids = np.asarray(ids)
    for j in range(0, len(ids), 512):
        ix = ids[j : j + 512]
        q = torch.from_numpy(queries[j : j + 512])[None].cuda()
        with torch.inference_mode():
            p, m = model(video, queries=q, grid_size=0, backward_tracking=True)
        p = p[0].cpu().numpy() * [(W - 1) / (w - 1), (H - 1) / (h - 1)]
        m = m[0].cpu().numpy().reshape(frame_count, -1).astype(bool)
        for a, f in enumerate(anchors):
            inbank = (ix >= offset[a]) & (ix < offset[a + 1])
            p[f, inbank] = banks[a][0][i, ix[inbank] - offset[a]]
            m[f, inbank] = True
            if inbank.any():
                lo = anchors[max(a - 1, 0)]
                hi = anchors[min(a + 1, len(anchors) - 1)]
                m[:lo, inbank] = False
                m[hi + 1 :, inbank] = False
        m &= (
            np.isfinite(p).all(-1)
            & (p[..., 0] >= 0)
            & (p[..., 0] < W)
            & (p[..., 1] >= 0)
            & (p[..., 1] < H)
        )
        xy[i, :, ix] = p.transpose(1, 0, 2)
        vis[i, :, ix] = m.T
    print(v, len(ids), round(time.perf_counter() - start, 2), flush=True)
    del video
for f in range(frame_count):
    d = output / f"f{f:06d}"
    if f in anchors:
        shutil.copytree(root / f"anchor_tracks/f{f:06d}", d)
        continue
    d.mkdir()
    keep = np.flatnonzero(vis[:, f].sum(0) >= 3)
    np.save(d / "tracks.npy", xy[:, f : f + 1, keep])
    np.save(d / "visibility.npy", vis[:, f : f + 1, keep])
    np.save(d / "track_quality.npy", vis[:, f : f + 1, keep].astype(np.float32))
    np.save(d / "points_t0.npy", xy[0, f, keep])
    (d / "views.txt").write_text("\n".join(views) + "\n")
    (d / "meta.json").write_text(
        json.dumps(
            {
                "start_frame": f,
                "num_frames": 1,
                "views": views,
                "num_points": len(keep),
                "point_ids": keep.tolist(),
                "seed_builder": "cotracker_anchor_ids",
            }
        )
    )
(output / "RESULT.json").write_text(
    json.dumps(
        {
            "seconds": time.perf_counter() - start,
            "views": len(views),
            "feature_cache": cache_stats,
        },
        indent=2,
    )
)
