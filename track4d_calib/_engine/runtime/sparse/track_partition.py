"""Reserve persistent track identities before dynamic-camera registration."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


def anchor_offsets(root):
    anchors = json.loads((root / "anchor_tracks/PROTOCOL.json").read_text())["anchors"]
    offsets, offset = {}, 0
    for frame in anchors:
        offsets[frame] = offset
        offset += np.load(root / f"anchor_tracks/f{frame:06d}/tracks.npy", mmap_mode="r").shape[2]
    return offsets


def persistent_ids(root, frame, offsets=None):
    directory = root / f"cotracker_cached_tracks/f{frame:06d}"
    count = np.load(directory / "tracks.npy", mmap_mode="r").shape[2]
    offsets = anchor_offsets(root) if offsets is None else offsets
    ids = (np.arange(count, dtype=np.int64) + offsets[frame] if frame in offsets else
           np.asarray(json.loads((directory / "meta.json").read_text())["point_ids"], dtype=np.int64))
    if ids.shape != (count,) or len(np.unique(ids)) != count or np.any(ids < 0):
        raise ValueError("Invalid persistent track identities")
    return ids


def identity_roles(ids):
    """Assign identities, independently of coordinates, quality and pose errors."""
    # SplitMix64 gives stable assignments without depending on array order.
    x = np.asarray(ids, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        folds = (x ^ (x >> np.uint64(31))) % np.uint64(5)
    return np.where(folds < 3, 0, np.where(folds == 3, 1, 2)).astype(np.int8)


def frame_roles(root, frame):
    directory = root / f"track_partition/f{frame:06d}"
    ids, roles = np.load(directory / "ids.npy"), np.load(directory / "roles.npy")
    if not np.array_equal(ids, persistent_ids(root, frame)):
        raise ValueError("Track identities changed after partitioning")
    if not np.array_equal(roles, identity_roles(ids)):
        raise ValueError("Track role manifest was modified")
    return roles


def fitting_track_root(root):
    config = json.loads((root / "config.json").read_text())
    if config.get("track_validation"):
        if not (root / "track_partition/PROTOCOL.json").is_file():
            raise ValueError("Persistent track partition must precede PnP")
        return root / "fitting_tracks"
    return root / "cotracker_cached_tracks"


def exclude_reserved_queries(root, frame, view, queries):
    """Protect held-out dynamic observations from later dense RGB refinement."""
    config = json.loads((root / "config.json").read_text())
    if not config.get("track_validation"):
        return np.ones(len(queries), dtype=bool)
    directory = root / f"cotracker_cached_tracks/f{frame:06d}"
    index = (directory / "views.txt").read_text().split().index(view)
    visible = np.load(directory / "visibility.npy", mmap_mode="r")[index, 0]
    points = np.load(directory / "tracks.npy", mmap_mode="r")[index, 0]
    reserved = visible & (frame_roles(root, frame) != 0) & np.isfinite(points).all(-1)
    if not reserved.any():
        return np.ones(len(queries), dtype=bool)
    distance, _ = cKDTree(points[reserved]).query(queries, k=1)
    return distance > config["track_validation"]["exclusion_radius_px"]


def run(root):
    config = json.loads((root / "config.json").read_text())
    if not config.get("track_validation"):
        return
    destination, evidence = root / "fitting_tracks", root / "track_partition"
    destination.mkdir(exist_ok=False)
    evidence.mkdir(exist_ok=False)
    offsets = anchor_offsets(root)
    counts, hashes = [], {}
    for frame in range(config["frames"]):
        source = root / f"cotracker_cached_tracks/f{frame:06d}"
        ids = persistent_ids(root, frame, offsets)
        roles = identity_roles(ids)
        out, proof = destination / source.name, evidence / source.name
        out.mkdir(); proof.mkdir()
        np.save(proof / "ids.npy", ids)
        np.save(proof / "roles.npy", roles)
        for path in source.iterdir():
            if path.name != "visibility.npy":
                (out / path.name).symlink_to(path.resolve())
        visibility = np.load(source / "visibility.npy")
        visibility[..., roles != 0] = False
        np.save(out / "visibility.npy", visibility)
        counts.append({"frame": frame, "role_counts": np.bincount(roles, minlength=3).tolist(),
                       "reserved_visible_in_fitting": int(visibility[..., roles != 0].sum())})
        for path in (proof / "ids.npy", proof / "roles.npy", out / "visibility.npy"):
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {"split": "splitmix64_mod5_fit012_confirm3_final4", "rows": counts,
                "scope": "Reserved before dynamic PnP and subsequent pose fitting; static rig and image frontend are shared",
                "reserved_roles_used_by_closure": False, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files_sha256": hashes, "settings": config["track_validation"]}
    (evidence / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
