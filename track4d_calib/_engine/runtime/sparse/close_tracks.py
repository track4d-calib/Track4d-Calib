from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys, json, time, os
import importlib.util
import numpy as np, cv2
from motion_constraints import prepare_centers, solve_constrained_pose
from track_partition import fitting_track_root

root = Path(sys.argv[1])
component_root = root / "code/runtime/components/track_closure"
sys.path.insert(0, str(component_root / "geometry"))
from track4d_calib.cameras import CameraLoader, CameraIntrinsicsLoader, save_cameras
from track4d_calib.same_time_multiview_pnp import SameTimePnPConfig, _triangulate_robust
from track4d_calib.track_pnp_repair import solve_focus_track_pnp
from track4d_calib.track_crossview_verify import stable_point_id_validation_groups

memo_path = root / "code/runtime/components/anchor_tracking/stage_code/track4d_calib/triangulation_memo.py"
memo_spec = importlib.util.spec_from_file_location("triangulation_memo", memo_path)
memo = importlib.util.module_from_spec(memo_spec)
memo_spec.loader.exec_module(memo)

config = json.loads((root / "config.json").read_text())
center_references = prepare_centers(root, config)
frame_count = config["frames"]
worker_count = min(8, frame_count, len(os.sched_getaffinity(0)))
statics = CameraLoader(root / "runtime_scene/cameras/static").load_cameras(
    config["static_views"]
)
triangulation_config = SameTimePnPConfig(
    support_reprojection_error_px=2.5, min_triangulation_angle_deg=0.5
)
output = root / "poses"
output.mkdir(exist_ok=False)


def evaluate(camera, role):
    xyz, uv = role
    if len(xyz) < 12:
        return {"count": len(xyz), "inliers": 0, "median_px": None}
    pred, _ = cv2.projectPoints(xyz, cv2.Rodrigues(camera.R)[0], camera.T, camera.K, camera.D)
    errors = np.linalg.norm(pred[:, 0] - uv, axis=1)
    depth = (xyz @ camera.R.T + camera.T.reshape(1, 3))[:, 2]
    return {
        "count": len(errors),
        "inliers": int(((errors <= 4) & (depth > 0)).sum()),
        "median_px": float(np.median(errors)),
    }


def better(candidate, base):
    return (
        candidate["median_px"] is not None
        and base["median_px"] is not None
        and candidate["inliers"] >= max(12, base["inliers"])
        and candidate["median_px"] < base["median_px"] - 1e-6
    )


def work(f):
    cv2.setNumThreads(1)
    cache_path = root / f"pnp/shard_{f % worker_count}/triangulation_cache.json.gz"
    triangulate = memo.load_cache(cache_path, _triangulate_robust)
    path = fitting_track_root(root) / f"f{f:06d}"
    views = (path / "views.txt").read_text().split()
    view_indices = {v: i for i, v in enumerate(views)}
    xy = np.load(path / "tracks.npy")[:, 0]
    visibility = np.load(path / "visibility.npy")[:, 0]
    quality = np.load(path / "track_quality.npy")[:, 0]
    base = CameraLoader(root / f"initial_poses/f{f:06d}/cameras_pred").load_cameras(
        config["static_views"] + config["dynamic_views"]
    )
    static_indices = [view_indices[v] for v in statics]
    support_count = visibility[static_indices].sum(0)
    selections = {}
    union = set()
    for v in config["dynamic_views"]:
        ids = np.flatnonzero(visibility[view_indices[v]] & (support_count >= 2))
        score = support_count[ids] + quality[view_indices[v], ids]
        ranked = ids[np.lexsort((ids, -score))]
        fit_limit = int(config.get("pnp_max_points", 150))
        fit = ranked[:fit_limit]
        rest = ranked[fit_limit:]
        fold = stable_point_id_validation_groups(rest) % 3
        roles = [
            np.r_[fit, rest[fold == 0][:150]],
            rest[fold == 1][:150],
            rest[fold == 2][:150],
        ]
        selections[v] = roles
        for role in roles:
            union.update(role.tolist())
    xyz = {}
    for i in sorted(union):
        observations = [
            (v, c, xy[view_indices[v], i]) for v, c in statics.items() if visibility[view_indices[v], i]
        ]
        result = triangulate(observations=observations, config=triangulation_config)
        if result["success"]:
            xyz[i] = result["point"]
    rows = []
    for v in config["dynamic_views"]:
        roles = []
        for subset in selections[v]:
            keep = [i for i in subset if i in xyz]
            roles.append(
                (np.asarray([xyz[i] for i in keep]).reshape(-1, 3), xy[view_indices[v], keep])
            )
        if center_references:
            fit_count = sum(i in xyz for i in selections[v][0][:int(config.get("pnp_max_points", 150))])
            constrained = solve_constrained_pose(
                roles[0][0][:fit_count], roles[0][1][:fit_count],
                base[v], center_references[v], config["center_radius"]
            )
            if not constrained["success"]:
                raise RuntimeError(f"motion-constrained pose failed: frame={f}, view={v}")
            base[v] = constrained["camera"]
        before = evaluate(base[v], roles[1])
        row = {
            "frame": f,
            "view": v,
            "base_confirm": before,
            "repair_attempted": False,
            "committed": False,
            "role_counts": [len(role[0]) for role in roles],
        }
        risk = (
            before["median_px"] is None
            or before["median_px"] > 4
            or before["inliers"] < 12
        )
        row["risk"] = risk
        if risk and before["median_px"] is not None:
            row["repair_attempted"] = True
            points, uv = roles[0]
            n = len(points)
            if center_references:
                result = solve_constrained_pose(
                    points, uv, base[v], center_references[v], config["center_radius"]
                )
            else:
                result = solve_focus_track_pnp(
                    point_bank={"point_ids": np.arange(n), "xyz": points},
                    tracks_frame=uv[None],
                    visibility_frame=np.ones((1, n), bool),
                    view_ids=[v], focus_view=v, camera_template=base[v],
                    iterations_count=2000,
                )
            row["candidate_success"] = bool(result["success"])
            if result["success"]:
                camera = result["camera"]
                after = evaluate(camera, roles[1])
                row["candidate_confirm"] = after
                if better(after, before):
                    original = evaluate(base[v], roles[2])
                    candidate = evaluate(camera, roles[2])
                    row["base_final"] = original
                    row["candidate_final"] = candidate
                    if better(candidate, original):
                        base[v] = camera
                        row["committed"] = True
        rows.append(row)
    save_cameras(output / f"f{f:06d}/cameras_pred", base)
    return rows, triangulate.cache_info()


if __name__ == "__main__":
    started = time.perf_counter()
    with ProcessPoolExecutor(worker_count) as pool:
        blocks = list(pool.map(work, range(frame_count)))
    rows = [row for block, _ in blocks for row in block]
    (root / "CLOSURE.json").write_text(
        json.dumps(
            {
                "policy": "risk-driven pose refinement with disjoint confirmation and final track observations",
                "track_source": str(fitting_track_root(root)),
                "base_fit_points_excluded_from_verification": int(
                    config.get("pnp_max_points", 150)
                ),
                "candidate_extra_fit_limit": 150,
                "confirm_point_limit": 150,
                "final_point_limit": 150,
                "risk_reprojection_px": 4,
                "risk_count": sum(x["risk"] for x in rows),
                "repair_attempted": sum(x["repair_attempted"] for x in rows),
                "committed": sum(x["committed"] for x in rows),
                "seconds": time.perf_counter() - started,
                "triangulation_cache": {
                    "hits": sum(stats["hits"] for _, stats in blocks),
                    "misses": sum(stats["misses"] for _, stats in blocks),
                    "solver_match": all(stats["solver_match"] for _, stats in blocks),
                },
                "rows": rows,
            },
            indent=2,
        )
    )
