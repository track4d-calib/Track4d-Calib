"""Build disjoint frame-local RGB evidence for persistent camera tracks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from track_partition import frame_roles, persistent_ids, identity_roles
from static_triangulation import triangulate_queries

ROLES = ("fit", "confirm", "final")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spatial_sample(uv, ids, limit):
    """Choose broad image coverage without looking at poses or residuals."""
    order = np.argsort(ids)
    if len(order) <= limit:
        return order
    points = np.asarray(uv, dtype=float)[order]
    current = int(np.argmin(np.sum((points - np.median(points, axis=0)) ** 2, axis=1)))
    selected, distance = [], np.full(len(points), np.inf)
    for _ in range(limit):
        selected.append(current)
        distance = np.minimum(distance, np.sum((points - points[current]) ** 2, axis=1))
        distance[selected] = -1
        current = int(np.argmax(distance))
    return order[selected]


def separated(queries, forbidden, radius):
    if not len(forbidden):
        return np.ones(len(queries), dtype=bool)
    return cKDTree(forbidden).query(queries)[0] > radius


def query_groups(root, frame, view, resolution, radius):
    directory = root / f"cotracker_cached_tracks/f{frame:06d}"
    index = (directory / "views.txt").read_text().split().index(view)
    points = np.load(directory / "tracks.npy", mmap_mode="r")[index, 0]
    visible = np.load(directory / "visibility.npy", mmap_mode="r")[index, 0]
    fit_visible = np.load(root / f"fitting_tracks/f{frame:06d}/visibility.npy", mmap_mode="r")[index, 0]
    width, height = resolution
    eligible = visible & np.isfinite(points).all(-1)
    eligible &= (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    queries = points[eligible].astype(float)
    ids, roles = persistent_ids(root, frame)[eligible], frame_roles(root, frame)[eligible]
    fitting, all_crop = [points[fit_visible]], []
    for name in ("crop_matches", "nonanchor_matches"):
        path = root / name / f"f{frame:06d}_v{view}.npz"
        if path.exists():
            with np.load(path) as data:
                fitting.append(data["uv_0"])
                all_crop.extend(data[f"uv_{i}"].copy() for i in range(3))
    forbidden = np.concatenate(fitting)
    allowed = separated(queries, forbidden, radius)
    heldout = roles > 0
    allowed[~heldout] = separated(queries[~heldout], queries[heldout], radius)
    return queries, ids, roles, allowed, forbidden, all_crop


def acquire(root, output):
    """Triangulate using static cameras only, with no candidate-pose filtering."""
    import cv2
    import torch
    from track4d_calib.cameras import CameraLoader
    from track4d_calib.same_time_multiview_pnp import SameTimePnPConfig
    from feature_cache import cache_image_features

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    config, dependencies = read(root / "config.json"), read(root / "dependencies.json")
    radius = config["track_validation"]["exclusion_radius_px"]
    bands = read(root / "RIG.json")["relay_bands"]
    selected = read(root / "anchor_tracks/RESULT.json")["support"]
    supports = {view: selected[band[len(band) // 2]][:8] for band in bands for view in band}
    if set(supports) != set(config["dynamic_views"]):
        raise ValueError("Static support coverage differs from dynamic camera set")
    if any(len(values) < 2 or not set(values) <= set(config["static_views"]) for values in supports.values()):
        raise ValueError("At least two fixed static supports are required per camera")
    engine = Path(__file__).resolve().parents[1]
    matcher_path = engine / "components/track_closure/chain/build_second_roma_component_anchor_cache.py"
    spec = importlib.util.spec_from_file_location("rgb_matcher", matcher_path)
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)
    output.mkdir(exist_ok=False)
    destination = output / "observations"
    destination.mkdir()
    images = {root / f"inputs/images/{view}/{frame:06d}.png"
              for frame in range(config["frames"])
              for view in set(config["dynamic_views"] + sum(list(supports.values()), []))}
    paths = set(images)
    for frame in range(config["frames"]):
        for name in ("cotracker_cached_tracks", "fitting_tracks", "track_partition"):
            paths.update(p for p in (root / name / f"f{frame:06d}").iterdir() if p.is_file())
    for name in ("crop_matches", "nonanchor_matches"):
        paths.update((root / name).glob("*.npz"))
    paths.update((root / "runtime_scene/cameras/static").glob("*.yml"))
    paths.update(root / name for name in ("config.json", "dependencies.json", "RIG.json", "anchor_tracks/RESULT.json"))
    locks = {str(p): digest(p) for p in sorted(paths)}
    protocol = {"supports": supports, "frames": list(range(config["frames"])), "views": config["dynamic_views"],
        "input_sha256": locks, "source_sha256": {str(p): digest(p) for p in
            (Path(__file__), Path(__file__).with_name("static_triangulation.py"), matcher_path)},
        "weights_sha256": digest(Path(dependencies["romav2_weights"])),
        "matcher": "RoMaV2 fast bidirectional; overlap >= 0.2, cycle error <= 1.5 px",
        "triangulation": "At least two static supports, support error <= 2.5 px, angle >= 0.5 degrees",
        "query_roles": "Persistent SplitMix64 identities frozen before dynamic PnP; 60/20/20 fit/confirm/final",
        "exclusion_radius_px": radius, "coverage_repair": "Count < 12 in either heldout role triggers a stride-16 holdout-only grid; at most 100 points per role",
        "selection_uses_candidate_pose_or_GT": False, "gpu": torch.cuda.get_device_name(0)}
    write(output / "PROTOCOL.json", protocol)
    statics = CameraLoader(root / "runtime_scene/cameras/static").load_cameras(config["static_views"])
    model, _ = matcher._load_roma_model(source_root=Path(dependencies["romav2_src"]),
        weights=Path(dependencies["romav2_weights"]), device="cuda", setting="fast", force_bidirectional=True)
    cache = cache_image_features(model, capacity=16)
    tri = SameTimePnPConfig(support_reprojection_error_px=2.5, min_triangulation_angle_deg=.5)
    width, height = config["resolution"]
    started, records = time.monotonic(), []

    def triangulate(frame, view, queries, allowed):
        mappings, reports = [], []
        for support in supports[view]:
            result = matcher.match_queries(model=model,
                static_image=root / f"inputs/images/{view}/{frame:06d}.png",
                target_image=root / f"inputs/images/{support}/{frame:06d}.png",
                queries=queries, height=height, width=width,
                minimum_overlap=.2, maximum_cycle_error_px=1.5)
            mappings.append(dict(zip(result["point_ids"], result["target_points"])))
            reports.append({"support": support, "matches": len(result["point_ids"]), "seconds": result["inference_seconds"]})
        kept, points = triangulate_queries([statics[s] for s in supports[view]], mappings, allowed, tri)
        return kept, points, reports

    for frame in range(config["frames"]):
        for view in config["dynamic_views"]:
            queries, ids, roles, allowed, fitting, crop = query_groups(root, frame, view, config["resolution"], radius)
            kept, points, pairs = triangulate(frame, view, queries, allowed)
            arrays = {}
            for number, role in enumerate(ROLES):
                keep = roles[kept] == number
                arrays[role + "_xyz"] = points[keep]
                arrays[role + "_uv"] = queries[kept[keep]]
                arrays[role + "_ids"] = ids[kept[keep]]
            before = {role: len(arrays[role + "_ids"]) for role in ROLES}
            if min(before[role] for role in ROLES[1:]) < 12:
                grid = matcher.query_grid(height=height, width=width, stride=16)
                grid_ids = (10**12 + (frame * len(config["dynamic_views"]) + config["dynamic_views"].index(view))
                            * len(grid) + np.arange(len(grid), dtype=np.int64))
                grid_roles = identity_roles(grid_ids)
                forbidden = np.concatenate([fitting, *crop, *(arrays[role + "_uv"] for role in ROLES)])
                allowed_grid = (grid_roles > 0) & separated(grid, forbidden, radius)
                indices, xyz, extra_pairs = triangulate(frame, view, grid, allowed_grid)
                pairs.extend(extra_pairs)
                for number, role in enumerate(ROLES[1:], 1):
                    chosen = np.flatnonzero(grid_roles[indices] == number)
                    if len(chosen) > 100:
                        chosen = chosen[np.linspace(0, len(chosen) - 1, 100).astype(int)]
                    for key, values in (("xyz", xyz[chosen]), ("uv", grid[indices[chosen]]), ("ids", grid_ids[indices[chosen]])):
                        name = role + "_" + key
                        arrays[name] = np.concatenate([arrays[name], values])
            np.savez_compressed(destination / f"f{frame:06d}_v{view}.npz", **arrays)
            records.append({"frame": frame, "view": view, "before_counts": before,
                "after_counts": {role: len(arrays[role + "_ids"]) for role in ROLES}, "pairs": pairs})
        write(output / "STATE.json", {"frames_complete": frame + 1, "seconds": time.monotonic() - started})
        print("RGB frame", frame, "seconds", round(time.monotonic() - started, 2), flush=True)
    if any(digest(Path(path)) != value for path, value in locks.items()):
        raise ValueError("RGB evidence inputs changed during collection")
    write(output / "RESULT.json", {"rows": records, "seconds": time.monotonic() - started,
        "feature_cache": cache, "input_unchanged": True,
        "observations_sha256": {p.name: digest(p) for p in sorted(destination.glob("*.npz"))}})
