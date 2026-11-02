"""Verify pose repairs using disjoint static-camera track observations."""

import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from track_partition import exclude_reserved_queries


def evaluate(camera, xyz, image_points):
    if len(xyz) < 12:
        return {"count": len(xyz), "inliers": 0, "median_px": None}
    projected, _ = cv2.projectPoints(
        xyz, cv2.Rodrigues(camera.R)[0], camera.T, camera.K, camera.D
    )
    errors = np.linalg.norm(projected[:, 0] - image_points, axis=1)
    depths = (xyz @ camera.R.T + camera.T.reshape(1, 3))[:, 2]
    return {
        "count": len(xyz),
        "inliers": int(((errors <= 4) & (depths > 0)).sum()),
        "median_px": float(np.median(errors)),
    }


def improves(candidate, base):
    return (
        candidate["median_px"] is not None
        and base["median_px"] is not None
        and candidate["inliers"] >= max(12, base["inliers"])
        and candidate["median_px"] < base["median_px"] - 1e-6
    )


def main():
    from motion_constraints import solve_constrained_pose
    root = Path(sys.argv[1])
    crop_mode = "--anchor-crop" in sys.argv[2:]
    sys.path.insert(0, str(root / "code/runtime/scripts"))
    from roma_overlap import augment_queries, geometry_admitted_points
    from feature_cache import cache_image_features
    from matching_cache import QueryMatchCache
    config = json.loads((root / "config.json").read_text())
    centers = (json.loads((root / "motion_centers.json").read_text())
               if config["center_radius"] is not None else {})
    dependencies = json.loads((root / "dependencies.json").read_text())
    component = root / "code/runtime/components/track_closure"
    sys.path.insert(0, str(component / "geometry"))
    from track4d_calib.cameras import CameraLoader, save_cameras
    from track4d_calib.same_time_multiview_pnp import SameTimePnPConfig, _triangulate_robust
    from track4d_calib.track_crossview_verify import stable_point_id_validation_groups
    from track4d_calib.track_pnp_repair import solve_focus_track_pnp
    import torch

    spec = importlib.util.spec_from_file_location(
        "anchor_matcher", component / "chain/build_second_roma_component_anchor_cache.py"
    )
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    started = time.perf_counter()
    anchors = json.loads((root / "anchor_tracks/PROTOCOL.json").read_text())["anchors"]
    frame_count = config.get("frames", 30)
    support_count = config["nonanchor_static_supports"]
    static_views = config["static_views"]
    dynamic_views = config["dynamic_views"]
    statics = CameraLoader(root / "runtime_scene/cameras/static").load_cameras(static_views)
    bank = root / f"anchor_tracks/f{anchors[0]:06d}"
    views = (bank / "views.txt").read_text().split()
    visible = np.load(bank / "visibility.npy")[:, 0]
    indices = {view: i for i, view in enumerate(views)}
    supports = {}
    for target in dynamic_views:
        ranked = sorted(static_views, key=lambda view: (
            -int((visible[indices[target]] & visible[indices[view]]).sum()), view
        ))
        supports[target] = ranked[:support_count]
    if crop_mode:
        bands = json.loads((root / "RIG.json").read_text())["relay_bands"]
        selected = json.loads((root / "anchor_tracks/RESULT.json").read_text())["support"]
        supports = {view: selected[band[len(band) // 2]] for band in bands for view in band}
    output = root / ("crop_poses" if crop_mode else "nonanchor_poses")
    output.mkdir(exist_ok=False)
    evidence = root / ("crop_matches" if crop_mode else "nonanchor_matches")
    evidence.mkdir(exist_ok=False)
    protocol = {
        "anchor_crop": crop_mode,
        "crop_mode": "both" if crop_mode else None,
        "anchors": anchors,
        "frames": anchors if crop_mode else [f for f in range(frame_count) if f not in anchors],
        "support_selection": "frozen anchor band supports" if crop_mode else "highest common anchor visibility count; frozen at first anchor",
        "supports": supports,
        "matcher": "RoMaV2 fast bidirectional",
        "query_stride": 8,
        "maximum_points_per_role": [150 if crop_mode else config.get("pnp_max_points", 600), 150, 150],
        "acceptance": "both disjoint confirmation and final sets improve median error without losing inliers",
        "timing": "included in parent cold calibration",
    }
    (evidence / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
    model, _ = matcher._load_roma_model(
        source_root=Path(dependencies["romav2_src"]),
        weights=Path(dependencies["romav2_weights"]), device="cuda",
        setting="fast", force_bidirectional=True,
    )
    feature_cache = cache_image_features(model, capacity=32)
    pair_cache = QueryMatchCache(
        root / "anchor_matches", weights=dependencies["romav2_weights"], source=matcher.__file__,
    )
    width, height = config["resolution"]
    queries = matcher.query_grid(height=height, width=width, stride=8)
    folds = stable_point_id_validation_groups(np.arange(len(queries))) % 3
    triangulation = SameTimePnPConfig(
        support_reprojection_error_px=2.5, min_triangulation_angle_deg=0.5
    )
    rows = []
    pairs = 0
    for frame in range(frame_count):
        source = root / f"poses/f{frame:06d}/cameras_pred"
        cameras = CameraLoader(source).load_cameras(static_views + dynamic_views)
        if frame in protocol["frames"]:
            for target in dynamic_views:
                matches, augmented, reports = {}, {}, []
                for support in supports[target]:
                    result = pair_cache.match(
                        matcher.match_queries, retain=False,
                        model=model,
                        static_image=root / f"inputs/images/{target}/{frame:06d}.png",
                        target_image=root / f"inputs/images/{support}/{frame:06d}.png",
                        queries=queries, height=height, width=width,
                        minimum_overlap=0.2, maximum_cycle_error_px=1.5,
                    )
                    matches[support] = dict(zip(result["point_ids"], result["target_points"]))
                    if crop_mode:
                        aug = augment_queries(
                            matcher=matcher, model=model, first=result,
                            source_image=root / f"inputs/images/{target}/{frame:06d}.png",
                            target_image=root / f"inputs/images/{support}/{frame:06d}.png",
                            queries=queries, height=height, width=width,
                            crop_source=True,
                        )
                        augmented[support] = dict(zip(aug["point_ids"], aug["target_points"]))
                        reports.append({
                            "support": support,
                            "first_inference_seconds": float(result["inference_seconds"]),
                            **aug["crop_diagnostics"],
                        })
                    pairs += 1
                counts = np.zeros(len(queries), dtype=int)
                for mapping in matches.values():
                    counts[list(mapping)] += 1
                allowed = exclude_reserved_queries(root, frame, target, queries)
                shared = np.flatnonzero((counts >= 2) & allowed)
                roles = []
                saved = {}
                for role, limit in enumerate(protocol["maximum_points_per_role"]):
                    ids = shared[folds[shared] == role]
                    if len(ids) > limit:
                        ids = ids[np.linspace(0, len(ids) - 1, limit).astype(int)]
                    xyz, uv, accepted_ids = [], [], []
                    for point_id in ids:
                        observations = [
                            (view, statics[view], mapping[point_id])
                            for view, mapping in matches.items() if point_id in mapping
                        ]
                        result = _triangulate_robust(observations=observations, config=triangulation)
                        if result["success"]:
                            xyz.append(result["point"])
                            uv.append(queries[point_id])
                            accepted_ids.append(point_id)
                    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
                    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
                    roles.append((xyz, uv))
                    saved[f"xyz_{role}"] = xyz
                    saved[f"uv_{role}"] = uv
                    saved[f"ids_{role}"] = np.asarray(accepted_ids, dtype=int)
                np.savez_compressed(evidence / f"f{frame:06d}_v{target}.npz", **saved)
                if crop_mode and len(roles[0][0]):
                    repaired, admission = geometry_admitted_points(
                        saved["ids_0"], roles[0][0], matches, augmented, statics,
                        triangulate=lambda obs: _triangulate_robust(observations=obs, config=triangulation),
                    )
                    saved["crop_xyz"] = repaired
                    np.savez_compressed(evidence / f"f{frame:06d}_v{target}.npz", **saved)
                before = [evaluate(cameras[target], *role) for role in roles[1:]]
                row = {"frame": frame, "view": target, "before": before,
                       "role_counts": [len(xyz) for xyz, _ in roles], "committed": False}
                row["reserved_query_exclusions"] = int(((counts >= 2) & ~allowed).sum())
                row["matches"] = reports
                if all(len(xyz) >= 12 for xyz, _ in roles):
                    xyz, uv = roles[0]
                    if crop_mode:
                        xyz = repaired
                        row["admission"] = admission
                    def solve_points(xyz):
                        if centers:
                            return solve_constrained_pose(
                                xyz, uv, cameras[target], centers[target], config["center_radius"]
                            )
                        else:
                            return solve_focus_track_pnp(
                                point_bank={"point_ids": np.arange(len(xyz)), "xyz": xyz},
                                tracks_frame=uv[None], visibility_frame=np.ones((1, len(xyz)), bool),
                                view_ids=[target], focus_view=target, camera_template=cameras[target],
                                iterations_count=2000, rng_seed=0,
                            )
                    result = solve_points(xyz)
                    baseline_metrics = None
                    if crop_mode:
                        baseline = solve_points(roles[0][0])
                        if baseline["success"]:
                            baseline_metrics = [evaluate(baseline["camera"], *role) for role in roles[1:]]
                        row["single_pass"] = baseline_metrics
                    row["candidate_success"] = bool(result["success"])
                    if result["success"]:
                        candidate = result["camera"]
                        after = [evaluate(candidate, *role) for role in roles[1:]]
                        row["after"] = after
                        crop_verified = not crop_mode or (
                            baseline_metrics is not None
                            and all(improves(a, b) for a, b in zip(after, baseline_metrics))
                        )
                        if crop_verified and all(improves(a, b) for a, b in zip(after, before)):
                            cameras[target] = candidate
                            row["committed"] = True
                rows.append(row)
        save_cameras(output / f"f{frame:06d}/cameras_pred", cameras)
        print("frame", frame, "pairs", pairs, "commits", sum(r["committed"] for r in rows), flush=True)
    (evidence / "RESULT.json").write_text(json.dumps({
        "seconds": time.perf_counter() - started, "pairs": pairs,
        "committed": sum(r["committed"] for r in rows), "rows": rows,
        "feature_cache": feature_cache,
        "query_cache": {"hits": pair_cache.hits, "misses": pair_cache.misses},
    }, indent=2))


if __name__ == "__main__":
    main()
