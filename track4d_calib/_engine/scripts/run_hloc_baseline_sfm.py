#!/usr/bin/env python3
"""
Run an HLOC SfM baseline on a single multi-view keyframe to initialize calibration.

Why:
  - For wild data, we want a mature initializer (HLOC) before running our
    multi-frame tracking-based refinement.

Inputs:
  - data_root: multiview images root, containing per-view subfolders.
  - views: list of view ids (e.g. 00 01 07 ...)
  - frame_idx: which frame to pick from each view (default 0)

Outputs:
  - cameras_out/{extri.yml,intri.yml} in Track4D-Calib OpenCV-YAML format
  - output_dir/hloc_sfm/ : HLOC intermediate files + COLMAP model for debugging
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import shutil
import sqlite3
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycolmap

# Allow importing Track4D-Calib without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add HLOC (vendored under third_party/hloc) to sys.path.
HLOC_ROOT = PROJECT_ROOT / "third_party" / "hloc"
if not (HLOC_ROOT / "hloc").is_dir():
    raise SystemExit(
        "HLOC not found. Expected: third_party/hloc/hloc\n"
        "If you didn't download it yet, fetch it first."
    )
sys.path.insert(0, str(HLOC_ROOT))

from track4d_calib.cameras import (
    CameraIntrinsicsLoader,
    CameraLoader,
    CameraParams,
    save_cameras,
)
from track4d_calib.epipolar_utils_np import essential_from_Rt, sampson_distance
from track4d_calib.hloc_baseline_utils import (
    filter_hloc_features_by_masks,
    find_frame_mask_path,
    pairs_ring_knn,
)
from track4d_calib.io import list_image_paths
from track4d_calib.pycolmap_utils import colmap_camera_to_opencv_KD, normalize_cameras_to_ref
from track4d_calib.sfm_reproducibility import seeded_colmap_options

from hloc import extract_features, logger as hloc_logger, match_features, pairs_from_exhaustive, reconstruction


ROMA_SP_MATCHER_NAME = "roma_sp"


def _run_cmd(cmd: List[str], *, cwd: Path, log_path: Optional[Path] = None) -> None:
    if log_path is None:
        proc = subprocess.run(cmd, cwd=str(cwd), env=dict(os.environ))
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
            )
    if proc.returncode != 0:
        log_note = f" (log: {log_path})" if log_path is not None else ""
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}{log_note}")


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _write_png(dst: Path, bgr: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {dst}")


def _stage_image_file(*, src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            try:
                if dst.resolve() == src.resolve():
                    return
            except FileNotFoundError:
                pass
        if dst.is_dir():
            raise RuntimeError(f"Cannot stage image to directory path: {dst}")
        dst.unlink()
    if bool(use_symlink):
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError as exc:
            print(f"[WARN] Failed to symlink {dst} -> {src}: {exc}. Fallback to file copy.")
    shutil.copy2(src, dst)


def _read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(_read_bgr(path), cv2.COLOR_BGR2RGB)


def _is_h5_write_lock_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("already open for write" in msg) or ("unable to synchronously open file" in msg)


def _cleanup_hloc_h5(export_dir: Path, *, stem_prefix: str = "") -> int:
    count = 0
    pattern = f"{stem_prefix}*.h5" if stem_prefix else "*.h5"
    for p in export_dir.glob(pattern):
        try:
            p.unlink()
            count += 1
        except Exception:
            continue
    return count


def _infer_intrinsics_from_cameras(cameras_dir: Path, view_id: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    cam = CameraIntrinsicsLoader(Path(cameras_dir)).load_camera_template(str(view_id))
    K = np.asarray(cam.K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(cam.D, dtype=np.float64).reshape(-1)[:5]
    return K, D, int(cam.H), int(cam.W)


def _format_colmap_camera_params_pinhole(K: np.ndarray) -> str:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return f"{fx},{fy},{cx},{cy}"


def _colmap_camera_mode(name: str) -> pycolmap.CameraMode:
    name = str(name).upper()
    if name not in pycolmap.CameraMode.__members__:
        raise ValueError(f"Unknown camera_mode {name}, choices: {list(pycolmap.CameraMode.__members__.keys())}")
    return pycolmap.CameraMode.__members__[name]


def _infer_num_total_views(images_root: Path) -> int:
    # Typical multiview datasets use numeric view directory names (00..47).
    view_ids = []
    for p in Path(images_root).iterdir():
        if not p.is_dir():
            continue
        if p.name.isdigit():
            view_ids.append(int(p.name))
    if view_ids:
        return max(view_ids) + 1
    # Fallback: treat as a generic set of view folders.
    return len([p for p in Path(images_root).iterdir() if p.is_dir()])


def _add_roma_to_syspath() -> None:
    roma_src = PROJECT_ROOT / "third_party" / "RoMaV2" / "src"
    if not (roma_src / "romav2").is_dir():
        raise RuntimeError(
            "RoMaV2 not found. Expected: third_party/RoMaV2/src/romav2\n"
            "Please fetch submodules or clone RoMaV2 first."
        )
    if str(roma_src) not in sys.path:
        sys.path.insert(0, str(roma_src))


def _pixel_to_normalized(kps_xy: np.ndarray, *, h: int, w: int) -> np.ndarray:
    # grid_sample with align_corners=False maps x_norm to pixel as:
    # x_pix = ((x_norm + 1) * W - 1) / 2  =>  x_norm = (2*x_pix + 1)/W - 1
    kps_xy = np.asarray(kps_xy, dtype=np.float32).reshape(-1, 2)
    out = np.empty_like(kps_xy, dtype=np.float32)
    out[:, 0] = ((2.0 * kps_xy[:, 0] + 1.0) / float(max(1, w))) - 1.0
    out[:, 1] = ((2.0 * kps_xy[:, 1] + 1.0) / float(max(1, h))) - 1.0
    eps = 1e-6
    out[:, 0] = np.clip(out[:, 0], -1.0 + eps, 1.0 - eps)
    out[:, 1] = np.clip(out[:, 1], -1.0 + eps, 1.0 - eps)
    return out


def _parse_pairs_txt(pairs_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        a, b = s.split()
        pairs.append((a, b))
    return pairs


def _resolve_view_id_from_imported_image_name(
    image_name: str,
    image_name_to_view: Dict[str, str],
) -> Optional[str]:
    s = str(image_name)
    if s in image_name_to_view:
        return str(image_name_to_view[s])

    base = Path(s).name
    if base in image_name_to_view:
        return str(image_name_to_view[base])

    p = Path(s)
    parent = p.parent.name
    if parent.isdigit():
        return str(parent)

    parts = p.parts
    for i, tok in enumerate(parts[:-1]):
        if tok == "images" and (i + 1) < len(parts) and str(parts[i + 1]).isdigit():
            return str(parts[i + 1])

    stem = p.stem
    m = re.match(r"^view(.+)$", stem)
    if m:
        return str(m.group(1))
    return None


def _remap_image_ids_to_feature_names(
    image_ids_from_db: Dict[str, int],
    *,
    image_names: List[str],
    image_name_to_view: Dict[str, str],
) -> Dict[str, int]:
    if all(name in image_ids_from_db for name in image_names):
        return dict(image_ids_from_db)

    view_to_db_id: Dict[str, int] = {}
    for db_name, db_id in image_ids_from_db.items():
        vid = _resolve_view_id_from_imported_image_name(str(db_name), image_name_to_view)
        if vid is None:
            continue
        if str(vid) not in view_to_db_id:
            view_to_db_id[str(vid)] = int(db_id)

    out: Dict[str, int] = {}
    for name in image_names:
        vid = str(image_name_to_view.get(name, ""))
        if vid in view_to_db_id:
            out[name] = int(view_to_db_id[vid])
    return out


def _canonicalize_database_image_names(
    *,
    database_path: Path,
    image_names: List[str],
    image_name_to_view: Dict[str, str],
) -> None:
    """Keep COLMAP DB image names aligned with HLOC feature/pair names.

    pycolmap may resolve symlinked staged images to their real target paths when
    importing images. HLOC features and pairs still use the staged names
    (viewXX.png), so geometric verification needs the database names normalized
    back to the staged filenames.
    """
    desired_by_view = {str(view): str(name) for name, view in image_name_to_view.items()}
    with sqlite3.connect(str(database_path)) as conn:
        rows = conn.execute("SELECT image_id, name FROM images").fetchall()
        updates: List[Tuple[int, str]] = []
        for image_id, db_name in rows:
            view_id = _resolve_view_id_from_imported_image_name(str(db_name), image_name_to_view)
            if view_id is None:
                continue
            desired = desired_by_view.get(str(view_id))
            if desired is not None and str(db_name) != desired:
                updates.append((int(image_id), desired))
        if not updates:
            return
        target_names = [name for _, name in updates]
        if len(set(target_names)) != len(target_names):
            raise RuntimeError(f"Duplicate staged image names while canonicalizing database: {target_names}")
        for image_id, _ in updates:
            conn.execute("UPDATE images SET name=? WHERE image_id=?", (f"__track4d_calib_tmp_{image_id}.png", image_id))
        for image_id, desired in updates:
            conn.execute("UPDATE images SET name=? WHERE image_id=?", (desired, image_id))
        conn.commit()
    print(f"[INFO] Canonicalized COLMAP DB image names to staged HLOC names: {len(updates)}/{len(image_names)}")


def _import_matches_limited(
    *,
    image_ids: Dict[str, int],
    db: pycolmap.Database,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float],
    skip_geometric_verification: bool,
    max_matches_per_pair: int,
) -> None:
    from hloc.utils.io import get_matches

    pairs = _parse_pairs_txt(pairs_path)
    matched = set()
    max_keep = int(max_matches_per_pair)
    for name0, name1 in pairs:
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score is not None:
            keep = scores > float(min_match_score)
            matches = matches[keep]
            scores = scores[keep]
        if max_keep > 0 and int(matches.shape[0]) > max_keep:
            if scores is not None and int(scores.shape[0]) == int(matches.shape[0]):
                order = np.argsort(scores)[::-1][:max_keep]
                matches = matches[order]
            else:
                matches = matches[:max_keep]
        db.write_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}
        if bool(skip_geometric_verification):
            db.write_two_view_geometry(
                id0, id1, pycolmap.TwoViewGeometry(inlier_matches=matches)
            )


def _write_pairs_txt(pairs_path: Path, pairs: List[Tuple[str, str]]) -> None:
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")


def _select_colmap_model_dir(model_root: Path) -> Path:
    if (model_root / "cameras.bin").is_file() and (model_root / "images.bin").is_file():
        return model_root
    cands = []
    if model_root.is_dir():
        for path in sorted(model_root.iterdir()):
            if path.is_dir() and (path / "cameras.bin").is_file() and (path / "images.bin").is_file():
                cands.append(path)
    if not cands:
        raise RuntimeError(f"No COLMAP-compatible sparse model found under: {model_root}")
    return cands[0]


def _split_pairs_round_robin(pairs: List[Tuple[str, str]], num_shards: int) -> List[List[Tuple[str, str]]]:
    if int(num_shards) <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    shards: List[List[Tuple[str, str]]] = [[] for _ in range(int(num_shards))]
    for i, pair in enumerate(pairs):
        shards[i % int(num_shards)].append(pair)
    return [s for s in shards if s]


def _merge_match_h5(
    *,
    src_h5: Path,
    dst_h5: Path,
) -> None:
    import h5py

    pair_groups: List[str] = []

    def _collect_pair_groups(name: str, obj) -> None:
        if isinstance(obj, h5py.Group) and "matches0" in obj:
            pair_groups.append(name)

    with h5py.File(str(src_h5), "r") as src, h5py.File(str(dst_h5), "a", libver="latest") as dst:
        src.visititems(_collect_pair_groups)
        for group_path in pair_groups:
            if group_path in dst:
                del dst[group_path]
            src.copy(group_path, dst, name=group_path)


def _parse_match_gpu_ids(match_gpus: str) -> List[int]:
    text = str(match_gpus).strip()
    if not text:
        return []
    out: List[int] = []
    for tok in text.split(","):
        s = tok.strip()
        if not s:
            continue
        gid = int(s)
        out.append(gid)
    return out


def _match_features_multi_gpu(
    *,
    matcher_conf_name: str,
    pairs_path: Path,
    features_path: Path,
    matches_out: Path,
    overwrite: bool,
    gpu_ids: List[int],
    work_dir: Path,
    pair_batch_size: int = 1,
    match_batch_size: int = 1,
    num_workers: int = 0,
    writer_threads: int = 0,
    group_by_shape: bool = True,
) -> Path:
    gpu_ids = [int(x) for x in gpu_ids]
    if len(gpu_ids) <= 1:
        raise ValueError("Need at least 2 GPU ids for multi-gpu matching.")

    pairs_all = _parse_pairs_txt(pairs_path)
    pairs_todo = match_features.find_unique_new_pairs(
        pairs_all, None if bool(overwrite) else matches_out
    )
    if len(pairs_todo) == 0:
        print(f"[INFO] Matching skip: all {len(pairs_all)} pairs already exist in {matches_out}")
        return matches_out

    if bool(overwrite) and matches_out.exists():
        matches_out.unlink()
    matches_out.parent.mkdir(parents=True, exist_ok=True)

    shard_root = work_dir / f"match_shards_{matches_out.stem}"
    if bool(overwrite) and shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    num_shards = min(len(gpu_ids), len(pairs_todo))
    shards = _split_pairs_round_robin(pairs_todo, num_shards=num_shards)
    assert len(shards) > 0

    py_path_prev = os.environ.get("PYTHONPATH", "")
    proc_entries: List[Tuple[subprocess.Popen, Path, Path, int, int]] = []
    t_match_shards_start = time.perf_counter()
    for shard_idx, shard_pairs in enumerate(shards):
        gpu_id = int(gpu_ids[shard_idx % len(gpu_ids)])
        shard_pairs_path = shard_root / f"pairs_shard_{shard_idx:02d}.txt"
        shard_match_path = shard_root / f"matches_shard_{shard_idx:02d}.h5"
        shard_log_path = shard_root / f"match_shard_{shard_idx:02d}.log"
        _write_pairs_txt(shard_pairs_path, shard_pairs)
        if shard_match_path.exists():
            shard_match_path.unlink()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["HLOC_MATCH_PAIR_BATCH_SIZE"] = str(max(1, int(pair_batch_size)))
        env["HLOC_MATCH_BATCH_SIZE"] = str(max(1, int(match_batch_size)))
        if int(num_workers) > 0:
            env["HLOC_MATCH_NUM_WORKERS"] = str(int(num_workers))
        if int(writer_threads) > 0:
            env["HLOC_MATCH_WRITER_THREADS"] = str(int(writer_threads))
        env["HLOC_MATCH_GROUP_BY_KEYPOINTS"] = "1" if bool(group_by_shape) else "0"
        env["PYTHONPATH"] = (
            f"{HLOC_ROOT}{os.pathsep}{py_path_prev}" if py_path_prev else str(HLOC_ROOT)
        )
        cmd = [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "from pathlib import Path; "
                "from hloc import match_features; "
                "conf_name, pairs_p, feats_p, matches_p = sys.argv[1:5]; "
                "conf = dict(match_features.confs[conf_name]); "
                "pbs = int(os.environ.get('HLOC_MATCH_PAIR_BATCH_SIZE', '1')); "
                "bs = int(os.environ.get('HLOC_MATCH_BATCH_SIZE', '0')); "
                "nw = int(os.environ.get('HLOC_MATCH_NUM_WORKERS', '0')); "
                "wt = int(os.environ.get('HLOC_MATCH_WRITER_THREADS', '0')); "
                "gk = str(os.environ.get('HLOC_MATCH_GROUP_BY_KEYPOINTS', '')).strip().lower(); "
                "pbs > 1 and conf.update({'pair_batch_size': pbs}); "
                "bs > 0 and conf.update({'batch_size': bs}); "
                "nw > 0 and conf.update({'num_workers': nw}); "
                "wt > 0 and conf.update({'writer_threads': wt}); "
                "gk and conf.update({'group_by_keypoint_shape': gk not in {'0','false','no'}}); "
                "match_features.main("
                "conf, "
                "pairs=Path(pairs_p), "
                "features=Path(feats_p), "
                "matches=Path(matches_p), "
                "overwrite=True)"
            ),
            str(matcher_conf_name),
            str(shard_pairs_path),
            str(features_path),
            str(shard_match_path),
        ]
        with open(shard_log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        proc_entries.append((proc, shard_match_path, shard_log_path, len(shard_pairs), gpu_id))
        print(
            f"[INFO] Launch match shard {shard_idx + 1}/{len(shards)} "
            f"gpu={gpu_id} pairs={len(shard_pairs)} "
            f"batch={max(1, int(pair_batch_size))}"
        )

    failed = False
    for proc, _, shard_log_path, shard_pairs_count, gpu_id in proc_entries:
        ret = proc.wait()
        if ret != 0:
            failed = True
            log_tail = ""
            try:
                lines = shard_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                log_tail = "\n".join(lines[-80:])
            except Exception:
                pass
            raise RuntimeError(
                "Sharded matching failed: "
                f"gpu={gpu_id}, pairs={shard_pairs_count}, ret={ret}, log={shard_log_path}\n"
                f"{log_tail}"
            )

    if failed:
        raise RuntimeError("Sharded matching failed.")

    t_match_shards_end = time.perf_counter()
    print(
        f"[INFO] Sharded matching finished: shards={len(proc_entries)}, "
        f"pairs={len(pairs_todo)}, wall={t_match_shards_end - t_match_shards_start:.3f}s"
    )

    # Keep existing groups (when overwrite=False) and add/replace groups from shard outputs.
    if not matches_out.exists():
        matches_out.touch()
    for _, shard_match_path, _, _, _ in proc_entries:
        if not shard_match_path.exists():
            raise RuntimeError(f"Shard output missing: {shard_match_path}")
        _merge_match_h5(src_h5=shard_match_path, dst_h5=matches_out)

    return matches_out


def _merge_tracking_matches(
    *,
    tracks_dir: Path,
    frame_idx: int,
    ref_view: str,
    image_name_to_view: Dict[str, str],
    features_path: Path,
    matches_path: Path,
    pairs_path: Path,
    output_features: Path,
    output_matches: Path,
    track_score: float,
    min_visible_views_at_frame: int,
    min_ref_temporal_support: int,
    max_track_points: int,
    epipolar_cameras_dir: Optional[Path],
    epipolar_max_sampson: float,
    epipolar_min_keep_ratio: float,
    bridge_short_view_ids: Optional[List[str]],
    bridge_long_view_ids: Optional[List[str]],
    bridge_min_short_visible_at_frame: int,
    bridge_min_long_visible_at_frame: int,
    bridge_quality_weight: float,
    inject_pair_view_ids: Optional[List[str]],
    inject_pair_mode: str,
    graph_balance_mode: str,
    graph_short_view_ids: Optional[List[str]],
    graph_long_view_ids: Optional[List[str]],
    graph_exclude_view_ids: Optional[List[str]],
    graph_allowed_pair_types: Optional[List[str]],
    graph_max_injected_matches_per_pair: int,
    graph_max_injected_matches_per_long_view: int,
    graph_min_injected_matches_per_long_view: int,
    graph_soft_target_per_long_view: int,
    graph_soft_min_keep_ratio: float,
    graph_soft_decay: float,
    overwrite: bool,
) -> Tuple[Path, Path]:
    """Append tracked points as extra keypoints and merge their matches into HLOC files."""
    import h5py
    import numpy as np
    from hloc.utils.parsers import names_to_pair

    tracks_dir = Path(tracks_dir)
    meta_path = tracks_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    start_frame = int(meta.get("start_frame", 0))
    t = int(frame_idx) - start_frame

    views_path = tracks_dir / "views.txt"
    if not views_path.is_file():
        raise RuntimeError(f"tracks_dir missing views.txt: {tracks_dir}")
    track_views = [v.strip() for v in views_path.read_text(encoding="utf-8").splitlines() if v.strip()]
    if not track_views:
        raise RuntimeError(f"tracks_dir has empty views.txt: {tracks_dir}")

    tracks = np.load(tracks_dir / "tracks.npy", mmap_mode="r")
    visibility = np.load(tracks_dir / "visibility.npy", mmap_mode="r")
    if tracks.ndim != 4 or visibility.ndim != 3:
        raise RuntimeError(f"Unexpected tracks/visibility shapes: {tracks.shape} / {visibility.shape}")
    if tracks.shape[:3] != visibility.shape:
        raise RuntimeError(f"tracks/visibility shape mismatch: {tracks.shape} vs {visibility.shape}")

    if t < 0 or t >= int(tracks.shape[1]):
        raise RuntimeError(
            f"tracks frame_idx {frame_idx} out of range for start_frame={start_frame}, "
            f"num_frames={tracks.shape[1]}"
        )

    if output_features.exists() and output_matches.exists() and not bool(overwrite):
        return output_features, output_matches

    output_features.parent.mkdir(parents=True, exist_ok=True)
    output_matches.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(features_path, output_features)
    shutil.copy2(matches_path, output_matches)

    view_to_track_idx = {v: i for i, v in enumerate(track_views)}
    num_points = int(tracks.shape[2])

    point_keep = np.ones((num_points,), dtype=bool)
    vis_count_frame = np.sum(visibility[:, t, :], axis=0).astype(np.int32)
    if int(min_visible_views_at_frame) > 0:
        point_keep &= vis_count_frame >= int(min_visible_views_at_frame)

    ref_support = None
    if str(ref_view) in view_to_track_idx:
        ref_idx = view_to_track_idx[str(ref_view)]
        ref_support = np.sum(visibility[ref_idx, :, :], axis=0).astype(np.int32)
        if int(min_ref_temporal_support) > 0:
            point_keep &= ref_support >= int(min_ref_temporal_support)
    elif int(min_ref_temporal_support) > 0:
        print(f"[WARN] --tracks_min_ref_temporal_support set but ref_view {ref_view} not in tracks views.")

    bridge_short_set = {str(v) for v in (bridge_short_view_ids or []) if str(v).strip()}
    bridge_long_set = {str(v) for v in (bridge_long_view_ids or []) if str(v).strip()}

    def _visible_count_for_view_set(view_set: set[str]) -> Tuple[Optional[np.ndarray], int]:
        idxs = [view_to_track_idx[v] for v in sorted(view_set) if v in view_to_track_idx]
        if not idxs:
            return None, 0
        cnt = np.sum(visibility[idxs, t, :], axis=0).astype(np.int32)
        return cnt, len(idxs)

    bridge_short_vis_count, bridge_short_num_views = _visible_count_for_view_set(bridge_short_set)
    bridge_long_vis_count, bridge_long_num_views = _visible_count_for_view_set(bridge_long_set)

    if int(bridge_min_short_visible_at_frame) > 0:
        if bridge_short_vis_count is None:
            print(
                "[WARN] bridge static-view filter requested but no static views found in tracks. "
                f"short_views={sorted(bridge_short_set)}"
            )
        else:
            point_keep &= bridge_short_vis_count >= int(bridge_min_short_visible_at_frame)

    if int(bridge_min_long_visible_at_frame) > 0:
        if bridge_long_vis_count is None:
            print(
                "[WARN] bridge dynamic-view filter requested but no dynamic views found in tracks. "
                f"long_views={sorted(bridge_long_set)}"
            )
        else:
            point_keep &= bridge_long_vis_count >= int(bridge_min_long_visible_at_frame)

    track_point_quality = vis_count_frame.astype(np.float64)
    if ref_support is not None:
        track_point_quality = track_point_quality + (ref_support.astype(np.float64) / max(1.0, float(tracks.shape[1])))
    if (
        bridge_short_vis_count is not None
        and bridge_long_vis_count is not None
        and float(bridge_quality_weight) != 0.0
    ):
        bridge_strength = np.minimum(bridge_short_vis_count, bridge_long_vis_count).astype(np.float64)
        track_point_quality = track_point_quality + float(bridge_quality_weight) * bridge_strength

    if int(max_track_points) > 0:
        keep_ids = np.flatnonzero(point_keep)
        if keep_ids.size > int(max_track_points):
            order = np.argsort(track_point_quality[keep_ids])[::-1]
            selected = keep_ids[order[: int(max_track_points)]]
            new_keep = np.zeros_like(point_keep)
            new_keep[selected] = True
            point_keep = new_keep

    print(
        "[INFO] Tracking prior filtering: "
        f"kept_points={int(point_keep.sum())}/{num_points}, "
        f"min_visible_views_at_frame={int(min_visible_views_at_frame)}, "
        f"min_ref_temporal_support={int(min_ref_temporal_support)}, "
        f"max_track_points={int(max_track_points)}, "
        f"bridge_short_views={bridge_short_num_views}, "
        f"bridge_long_views={bridge_long_num_views}, "
        f"bridge_min_short_visible_at_frame={int(bridge_min_short_visible_at_frame)}, "
        f"bridge_min_long_visible_at_frame={int(bridge_min_long_visible_at_frame)}, "
        f"bridge_quality_weight={float(bridge_quality_weight):.3f}"
    )

    epipolar_enabled = bool(epipolar_cameras_dir) and float(epipolar_max_sampson) > 0.0
    epipolar_cam_cache: Dict[str, CameraParams] = {}
    if epipolar_enabled:
        epipolar_loader = CameraLoader(Path(epipolar_cameras_dir))
        for view_id in track_views:
            try:
                epipolar_cam_cache[view_id] = epipolar_loader.load_camera(str(view_id))
            except Exception:
                continue
        if len(epipolar_cam_cache) < 2:
            print(
                f"[WARN] Epipolar gating disabled: only {len(epipolar_cam_cache)} views loaded "
                f"from {epipolar_cameras_dir}."
            )
            epipolar_enabled = False

    epi_pairs_total = 0
    epi_pairs_with_camera = 0
    epi_matches_before = 0
    epi_matches_after = 0

    inject_pair_set = {str(v) for v in (inject_pair_view_ids or []) if str(v).strip()}
    inject_mode = str(inject_pair_mode).strip().lower()
    if inject_mode not in {"any", "both", "xor"}:
        raise ValueError(f"Unknown inject_pair_mode={inject_pair_mode}, expected any|both|xor")
    pair_policy_enabled = len(inject_pair_set) > 0
    pair_policy_total = 0
    pair_policy_selected = 0

    graph_mode = str(graph_balance_mode).strip().lower()
    if graph_mode not in {"off", "per_long", "per_long_soft"}:
        raise ValueError(f"Unknown graph_balance_mode={graph_balance_mode}, expected off|per_long|per_long_soft")

    graph_short_set = {str(v) for v in (graph_short_view_ids or []) if str(v).strip()}
    graph_long_set = {str(v) for v in (graph_long_view_ids or []) if str(v).strip()}
    if not graph_short_set and bridge_short_set:
        graph_short_set = set(bridge_short_set)
    if not graph_long_set and bridge_long_set:
        graph_long_set = set(bridge_long_set)

    graph_exclude_set = {str(v) for v in (graph_exclude_view_ids or []) if str(v).strip()}
    graph_allowed_set = {str(v).strip().upper() for v in (graph_allowed_pair_types or []) if str(v).strip()}
    valid_pair_types = {"SS", "SL", "LL", "OTHER"}
    if graph_allowed_set:
        invalid_pair_types = sorted(graph_allowed_set - valid_pair_types)
        if invalid_pair_types:
            raise ValueError(
                f"Unknown graph_allowed_pair_types={invalid_pair_types}, "
                f"expected subset of {sorted(valid_pair_types)}"
            )
    else:
        graph_allowed_set = set(valid_pair_types)

    graph_balance_enabled = graph_mode == "per_long"
    graph_soft_enabled = graph_mode == "per_long_soft"
    graph_balance_enabled = graph_balance_enabled or graph_soft_enabled
    graph_pairs_total = 0
    graph_pairs_selected = 0
    graph_pairs_filtered = 0
    graph_pairs_injected = 0
    graph_pair_type_counts: Dict[str, int] = defaultdict(int)
    graph_long_match_counts: Dict[str, int] = defaultdict(int)
    graph_max_pair = max(0, int(graph_max_injected_matches_per_pair))
    graph_max_long = max(0, int(graph_max_injected_matches_per_long_view))
    graph_min_long = max(0, int(graph_min_injected_matches_per_long_view))
    graph_soft_target = max(0, int(graph_soft_target_per_long_view))
    graph_soft_keep_ratio = float(np.clip(graph_soft_min_keep_ratio, 0.0, 1.0))
    graph_soft_decay_safe = max(1e-6, float(graph_soft_decay))
    graph_soft_ratios: List[float] = []
    graph_floor_boost_pairs = 0

    def _graph_soft_pair_ratio(long_endpoints: List[str]) -> float:
        if not long_endpoints:
            return 1.0
        raw_scores: List[float] = []
        for lv in long_endpoints:
            cur = float(graph_long_match_counts[str(lv)])
            if graph_soft_target > 0:
                score = max(0.0, (float(graph_soft_target) - cur) / max(1.0, float(graph_soft_target)))
            else:
                score = 1.0 / (1.0 + cur / graph_soft_decay_safe)
            raw_scores.append(float(score))
        mean_score = float(np.mean(raw_scores)) if raw_scores else 1.0
        ratio = graph_soft_keep_ratio + (1.0 - graph_soft_keep_ratio) * mean_score
        return float(np.clip(ratio, 0.0, 1.0))

    def _classify_pair(view0: str, view1: str) -> Tuple[str, List[str]]:
        v0 = str(view0)
        v1 = str(view1)
        in_short0 = v0 in graph_short_set
        in_short1 = v1 in graph_short_set
        in_long0 = v0 in graph_long_set
        in_long1 = v1 in graph_long_set

        long_endpoints: List[str] = []
        if in_long0:
            long_endpoints.append(v0)
        if in_long1 and v1 not in long_endpoints:
            long_endpoints.append(v1)

        if in_short0 and in_short1:
            return "SS", long_endpoints
        if in_long0 and in_long1:
            return "LL", long_endpoints
        if (in_short0 and in_long1) or (in_long0 and in_short1):
            return "SL", long_endpoints
        return "OTHER", long_endpoints

    def _filter_track_ids_by_epipolar(view0: str, view1: str, track_ids: np.ndarray) -> np.ndarray:
        nonlocal epi_pairs_total, epi_pairs_with_camera, epi_matches_before, epi_matches_after
        if track_ids.size == 0:
            return track_ids

        epi_pairs_total += 1
        epi_matches_before += int(track_ids.size)

        if not epipolar_enabled:
            epi_matches_after += int(track_ids.size)
            return track_ids

        cam0 = epipolar_cam_cache.get(str(view0))
        cam1 = epipolar_cam_cache.get(str(view1))
        if cam0 is None or cam1 is None:
            epi_matches_after += int(track_ids.size)
            return track_ids

        epi_pairs_with_camera += 1
        idx0 = view_to_track_idx[str(view0)]
        idx1 = view_to_track_idx[str(view1)]

        pts0 = np.asarray(tracks[idx0, t, track_ids], dtype=np.float64).reshape(-1, 1, 2)
        pts1 = np.asarray(tracks[idx1, t, track_ids], dtype=np.float64).reshape(-1, 1, 2)

        K0 = np.asarray(cam0.K, dtype=np.float64).reshape(3, 3)
        D0 = np.asarray(cam0.D, dtype=np.float64).reshape(-1)
        K1 = np.asarray(cam1.K, dtype=np.float64).reshape(3, 3)
        D1 = np.asarray(cam1.D, dtype=np.float64).reshape(-1)

        p0 = cv2.undistortPoints(pts0, K0, D0).reshape(-1, 2)
        p1 = cv2.undistortPoints(pts1, K1, D1).reshape(-1, 2)

        R0 = np.asarray(cam0.R, dtype=np.float64).reshape(3, 3)
        T0 = np.asarray(cam0.T, dtype=np.float64).reshape(3)
        R1 = np.asarray(cam1.R, dtype=np.float64).reshape(3, 3)
        T1 = np.asarray(cam1.T, dtype=np.float64).reshape(3)

        R_10 = R1 @ R0.T
        t_10 = T1 - (R_10 @ T0)
        E_10 = essential_from_Rt(R_10, t_10)
        d = sampson_distance(E_10, p0, p1)
        d = np.where(np.isfinite(d), d, np.inf)

        keep = d <= float(epipolar_max_sampson)
        min_keep_ratio = float(np.clip(epipolar_min_keep_ratio, 0.0, 1.0))
        if min_keep_ratio > 0.0:
            min_keep = int(np.ceil(min_keep_ratio * float(track_ids.size)))
            if int(keep.sum()) < min_keep:
                order = np.argsort(d)
                keep_idx = order[:min_keep]
                keep = np.zeros_like(keep)
                keep[keep_idx] = True

        kept = track_ids[keep]
        epi_matches_after += int(kept.size)
        return kept

    # Build per-view mapping from track id -> keypoint index after appending.
    track_kp_idx: Dict[str, np.ndarray] = {}
    kp_count_by_image: Dict[str, int] = {}

    with h5py.File(str(output_features), "r+") as f_feat:
        for image_name, view_id in image_name_to_view.items():
            if view_id not in view_to_track_idx:
                continue
            if image_name not in f_feat:
                continue
            g = f_feat[image_name]
            kps = np.asarray(g["keypoints"])
            n_old = int(kps.shape[0])

            coords = np.asarray(tracks[view_to_track_idx[view_id], t], dtype=np.float32).copy()
            vis = np.asarray(visibility[view_to_track_idx[view_id], t], dtype=bool)
            valid = vis & np.isfinite(coords).all(axis=1) & point_keep

            # Clamp to image bounds if available.
            if "image_size" in g:
                h, w = [int(x) for x in np.asarray(g["image_size"]).reshape(-1)[:2]]
                coords[:, 0] = np.clip(coords[:, 0], 0.0, max(0, w - 1))
                coords[:, 1] = np.clip(coords[:, 1], 0.0, max(0, h - 1))

            valid_ids = np.flatnonzero(valid)
            new_kps = coords[valid_ids] if valid_ids.size > 0 else np.zeros((0, 2), dtype=np.float32)

            kp_map = np.full((num_points,), -1, dtype=np.int32)
            if new_kps.shape[0] > 0:
                kp_map[valid_ids] = np.arange(int(new_kps.shape[0]), dtype=np.int32) + n_old

            # Append keypoints.
            kps_all = np.concatenate([kps, new_kps], axis=0)
            del g["keypoints"]
            g.create_dataset("keypoints", data=kps_all)

            if "scores" in g:
                scores = np.asarray(g["scores"])
                scores_new = np.ones((new_kps.shape[0],), dtype=scores.dtype) if new_kps.shape[0] > 0 else scores[:0]
                scores_all = np.concatenate([scores, scores_new], axis=0)
                del g["scores"]
                g.create_dataset("scores", data=scores_all)

            if "descriptors" in g:
                desc = np.asarray(g["descriptors"])
                if desc.ndim == 2 and desc.shape[1] == n_old:
                    pad = np.zeros((desc.shape[0], new_kps.shape[0]), dtype=desc.dtype)
                    desc_all = np.concatenate([desc, pad], axis=1)
                elif desc.ndim == 2 and desc.shape[0] == n_old:
                    pad = np.zeros((new_kps.shape[0], desc.shape[1]), dtype=desc.dtype)
                    desc_all = np.concatenate([desc, pad], axis=0)
                else:
                    raise RuntimeError(f"Unexpected descriptors shape for {image_name}: {desc.shape} (n_old={n_old})")
                del g["descriptors"]
                g.create_dataset("descriptors", data=desc_all)

            track_kp_idx[view_id] = kp_map
            kp_count_by_image[image_name] = int(kps_all.shape[0])

    # Merge tracking matches into matches h5.
    pairs = _parse_pairs_txt(pairs_path)
    with h5py.File(str(output_matches), "r+") as f_match:
        for name0, name1 in pairs:
            view0 = image_name_to_view.get(name0)
            view1 = image_name_to_view.get(name1)
            if view0 is None or view1 is None:
                continue

            pair_policy_total += 1
            if pair_policy_enabled:
                in0 = str(view0) in inject_pair_set
                in1 = str(view1) in inject_pair_set
                if inject_mode == "any":
                    allow = in0 or in1
                elif inject_mode == "both":
                    allow = in0 and in1
                else:  # inject_mode == "xor"
                    allow = (in0 != in1)
                if not allow:
                    continue
            pair_policy_selected += 1

            pair_type, long_endpoints = _classify_pair(str(view0), str(view1))
            if graph_balance_enabled:
                graph_pairs_total += 1
                if str(view0) in graph_exclude_set or str(view1) in graph_exclude_set:
                    graph_pairs_filtered += 1
                    continue
                if pair_type not in graph_allowed_set:
                    graph_pairs_filtered += 1
                    continue

                if graph_mode == "per_long" and graph_max_long > 0 and long_endpoints:
                    long_remaining = min(graph_max_long - int(graph_long_match_counts[lv]) for lv in long_endpoints)
                    if long_remaining <= 0:
                        graph_pairs_filtered += 1
                        continue

                graph_pairs_selected += 1
                graph_pair_type_counts[pair_type] += 1

            if view0 not in track_kp_idx or view1 not in track_kp_idx:
                continue
            kp0 = track_kp_idx[view0]
            kp1 = track_kp_idx[view1]
            valid_ids = np.flatnonzero((kp0 >= 0) & (kp1 >= 0))
            if valid_ids.size == 0:
                continue

            valid_ids = _filter_track_ids_by_epipolar(view0, view1, valid_ids)
            if valid_ids.size == 0:
                continue

            if valid_ids.size > 1:
                order = np.argsort(track_point_quality[valid_ids])[::-1]
                valid_ids = valid_ids[order]

            if graph_balance_enabled:
                available = int(valid_ids.size)
                max_allowed_by_pair = available
                if graph_max_pair > 0:
                    max_allowed_by_pair = min(max_allowed_by_pair, int(graph_max_pair))

                keep_count = int(max_allowed_by_pair)

                if graph_soft_enabled and long_endpoints:
                    keep_ratio = _graph_soft_pair_ratio(long_endpoints)
                    soft_cap = int(np.ceil(keep_ratio * float(available)))
                    soft_cap = max(0, min(int(available), int(soft_cap)))
                    graph_soft_ratios.append(float(keep_ratio))
                    keep_count = min(keep_count, int(soft_cap))

                if graph_min_long > 0 and long_endpoints:
                    deficits = [int(graph_min_long) - int(graph_long_match_counts[lv]) for lv in long_endpoints]
                    floor_needed = max(0, max(deficits)) if deficits else 0
                    floor_needed = min(int(max_allowed_by_pair), int(floor_needed))
                    if floor_needed > keep_count:
                        keep_count = int(floor_needed)
                        graph_floor_boost_pairs += 1

                if graph_max_long > 0 and long_endpoints:
                    long_remaining = min(graph_max_long - int(graph_long_match_counts[lv]) for lv in long_endpoints)
                    if long_remaining <= 0:
                        continue
                    keep_count = min(int(keep_count), int(long_remaining))

                if keep_count <= 0:
                    continue
                if keep_count < available:
                    valid_ids = valid_ids[:keep_count]

            idx0 = kp0[valid_ids].astype(np.int64)
            idx1 = kp1[valid_ids].astype(np.int64)

            pair = names_to_pair(name0, name1)
            if pair in f_match:
                grp = f_match[pair]
                matches0 = np.asarray(grp["matches0"])
                scores0 = np.asarray(grp["matching_scores0"])
                dtype_m = matches0.dtype
                dtype_s = scores0.dtype
            else:
                grp = f_match.create_group(pair)
                dtype_m = np.int16
                dtype_s = np.float16
                matches0 = np.full((0,), -1, dtype=dtype_m)
                scores0 = np.zeros((0,), dtype=dtype_s)

            n_kp0 = int(kp_count_by_image.get(name0, matches0.shape[0]))
            if matches0.shape[0] < n_kp0:
                matches0_new = np.full((n_kp0,), -1, dtype=dtype_m)
                scores0_new = np.zeros((n_kp0,), dtype=dtype_s)
                matches0_new[: matches0.shape[0]] = matches0
                scores0_new[: scores0.shape[0]] = scores0
                matches0, scores0 = matches0_new, scores0_new

            # Only fill empty entries (new keypoints).
            empty = matches0[idx0] == -1
            if np.any(empty):
                idx0 = idx0[empty]
                idx1 = idx1[empty]
            if idx0.size == 0:
                continue

            if graph_balance_enabled:
                graph_pairs_injected += 1

            if graph_balance_enabled and long_endpoints:
                for lv in long_endpoints:
                    graph_long_match_counts[lv] += int(idx0.size)

            matches0[idx0] = idx1.astype(dtype_m)
            scores0[idx0] = np.asarray(track_score, dtype=dtype_s)

            if "matches0" in grp:
                del grp["matches0"]
            if "matching_scores0" in grp:
                del grp["matching_scores0"]
            grp.create_dataset("matches0", data=matches0)
            grp.create_dataset("matching_scores0", data=scores0)

    if epi_pairs_total > 0:
        ratio = float(epi_matches_after) / float(max(1, epi_matches_before))
        print(
            "[INFO] Tracking epipolar gating: "
            f"pairs={epi_pairs_total}, "
            f"pairs_with_camera={epi_pairs_with_camera}, "
            f"kept_matches={epi_matches_after}/{epi_matches_before} ({ratio:.3f}), "
            f"max_sampson={float(epipolar_max_sampson):.6f}, "
            f"min_keep_ratio={float(np.clip(epipolar_min_keep_ratio, 0.0, 1.0)):.3f}"
        )

    if pair_policy_enabled:
        print(
            "[INFO] Tracking pair injection policy: "
            f"mode={inject_mode}, "
            f"selected_pairs={pair_policy_selected}/{pair_policy_total}, "
            f"views={sorted(inject_pair_set)}"
        )

    if graph_balance_enabled:
        long_counts_sorted = sorted(
            ((k, int(v)) for k, v in graph_long_match_counts.items()),
            key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]),
        )
        soft_ratio_mean = float(np.mean(graph_soft_ratios)) if graph_soft_ratios else float("nan")
        print(
            "[INFO] Tracking graph balance: "
            f"mode={graph_mode}, "
            f"selected_pairs={graph_pairs_selected}/{graph_pairs_total}, "
            f"injected_pairs={graph_pairs_injected}, "
            f"filtered_pairs={graph_pairs_filtered}, "
            f"pair_type_counts={dict(graph_pair_type_counts)}, "
            f"max_matches_per_pair={graph_max_pair}, "
            f"max_matches_per_long_view={graph_max_long}, "
            f"min_matches_per_long_view={graph_min_long}, "
            f"soft_target_per_long_view={graph_soft_target}, "
            f"soft_min_keep_ratio={graph_soft_keep_ratio:.3f}, "
            f"soft_decay={graph_soft_decay_safe:.3f}, "
            f"soft_keep_ratio_mean={soft_ratio_mean:.3f}, "
            f"floor_boost_pairs={graph_floor_boost_pairs}, "
            f"long_match_counts={long_counts_sorted}, "
            f"excluded_views={sorted(graph_exclude_set)}, "
            f"allowed_pair_types={sorted(graph_allowed_set)}"
        )

    return output_features, output_matches


def _match_features_roma_sp(
    *,
    image_dir: Path,
    pairs_path: Path,
    features_path: Path,
    matches_out: Path,
    overwrite: bool,
    roma_setting: str,
    roma_compile: bool,
    romav2_weights: str,
    roma_max_dist: float,
    roma_cert_th: float,
) -> Path:
    if matches_out.exists() and not bool(overwrite):
        import h5py

        try:
            with h5py.File(str(matches_out), "r") as f_match:
                num_groups = len(f_match.keys())
                image_size_order = str(f_match.attrs.get("roma_sp_image_size_order", ""))
            if num_groups > 0 and image_size_order == "wh":
                print(f"[INFO] Reusing RoMa-SP matches: {matches_out}")
                return matches_out
            if num_groups > 0:
                print(f"[WARN] Existing RoMa-SP matches use legacy image-size handling; rebuilding: {matches_out}")
            else:
                print(f"[WARN] Existing RoMa-SP matches are empty; rebuilding: {matches_out}")
        except OSError as exc:
            print(f"[WARN] Existing RoMa-SP matches are unreadable ({exc}); rebuilding: {matches_out}")

    _add_roma_to_syspath()
    import h5py
    import torch
    import torch.nn.functional as F

    from hloc.utils.parsers import names_to_pair
    from romav2 import RoMaV2

    default_weights = PROJECT_ROOT / "checkpoints" / "romav2.pt"
    if romav2_weights:
        wpath = Path(romav2_weights)
        if not wpath.is_file():
            raise RuntimeError(f"--romav2_weights not found: {wpath}")
        os.environ["ROMAV2_WEIGHTS"] = str(wpath)
    elif default_weights.is_file():
        os.environ["ROMAV2_WEIGHTS"] = str(default_weights)
    else:
        raise RuntimeError(
            "RoMaV2 weights not found.\n"
            f"Please download `romav2.pt` to: {default_weights}\n"
            "Or pass `--romav2_weights /path/to/romav2.pt`."
        )

    torch.set_float32_matmul_precision("highest")
    roma = RoMaV2(RoMaV2.Cfg(setting=str(roma_setting), compile=bool(roma_compile)))

    pairs = _parse_pairs_txt(pairs_path)
    if not pairs:
        raise RuntimeError(f"No pairs found: {pairs_path}")

    image_cache: Dict[str, np.ndarray] = {}
    matches_out.parent.mkdir(parents=True, exist_ok=True)
    total_matches = 0

    with h5py.File(str(features_path), "r") as f_feat, h5py.File(str(matches_out), "w") as f_match:
        f_match.attrs["roma_sp_image_size_order"] = "wh"

        def _load_feature_image(name: str) -> np.ndarray:
            if name in image_cache:
                return image_cache[name]
            img = _read_rgb(image_dir / name)
            if "image_size" in f_feat[name]:
                size = np.asarray(f_feat[name]["image_size"][:], dtype=np.int32).reshape(-1)
                if size.size >= 2:
                    # HLOC stores image_size as (width, height).
                    target_w, target_h = int(size[0]), int(size[1])
                    if img.shape[0] != target_h or img.shape[1] != target_w:
                        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            image_cache[name] = img
            return img

        for idx, (name0, name1) in enumerate(pairs):
            if name0 not in f_feat or name1 not in f_feat:
                raise RuntimeError(f"Pair image missing in features: ({name0}, {name1})")

            kps0 = np.asarray(f_feat[name0]["keypoints"], dtype=np.float32)
            kps1 = np.asarray(f_feat[name1]["keypoints"], dtype=np.float32)
            matches0 = np.full((int(kps0.shape[0]),), -1, dtype=np.int16)
            scores0 = np.zeros((int(kps0.shape[0]),), dtype=np.float16)

            if kps0.shape[0] > 0 and kps1.shape[0] > 0:
                img0 = _load_feature_image(name0)
                img1 = _load_feature_image(name1)

                preds = roma.match(img0, img1)
                warp = preds["warp_AB"][0]
                certainty = preds["overlap_AB"][0]
                if certainty.ndim == 3:
                    certainty = certainty[..., 0]

                h0, w0 = img0.shape[:2]
                h1, w1 = img1.shape[:2]
                x0 = _pixel_to_normalized(kps0, h=h0, w=w0)
                x1 = _pixel_to_normalized(kps1, h=h1, w=w1)
                x0_t = torch.from_numpy(x0).to(device=warp.device, dtype=warp.dtype)
                x1_t = torch.from_numpy(x1).to(device=warp.device, dtype=warp.dtype)

                idx0_t, idx1_t = RoMaV2.match_keypoints(
                    x_A=x0_t,
                    x_B=x1_t,
                    warp=warp,
                    certainty=certainty,
                    return_tuple=True,
                    return_inds=True,
                    max_dist=float(roma_max_dist),
                    cert_th=float(roma_cert_th),
                )

                if idx0_t.numel() > 0:
                    cert_at_x0 = F.grid_sample(
                        certainty[None, None],
                        x0_t[None, None],
                        align_corners=False,
                        mode="bilinear",
                    )[0, 0, 0]

                    idx0 = idx0_t.detach().cpu().numpy().astype(np.int64)
                    idx1 = idx1_t.detach().cpu().numpy().astype(np.int64)
                    score = cert_at_x0[idx0_t].detach().cpu().numpy().astype(np.float32)
                    score = np.clip(score, 0.0, 1.0).astype(np.float16)

                    matches0[idx0] = idx1.astype(np.int16)
                    scores0[idx0] = score
                    total_matches += int(idx0.shape[0])

            pair = names_to_pair(name0, name1)
            grp = f_match.create_group(pair)
            grp.create_dataset("matches0", data=matches0)
            grp.create_dataset("matching_scores0", data=scores0)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
                print(f"[INFO] RoMa-SP matched pairs: {idx + 1}/{len(pairs)}")

    print(f"[OK] RoMa-SP matches written: {matches_out} (total correspondences={total_matches})")
    return matches_out


def _build_cameras_from_reconstruction(
    rec: pycolmap.Reconstruction,
    *,
    image_name_to_view: Dict[str, str],
    ref_view: str,
    normalize_ref: bool,
    allow_partial: bool,
) -> Dict[str, CameraParams]:
    expected_view_ids = sorted({str(v) for v in image_name_to_view.values()}, key=lambda x: int(x) if str(x).isdigit() else str(x))
    images_by_view: Dict[str, pycolmap.Image] = {}
    for img in rec.images.values():
        vid = _resolve_view_id_from_imported_image_name(str(img.name), image_name_to_view)
        if vid is None:
            continue
        if str(vid) not in images_by_view:
            images_by_view[str(vid)] = img

    R_all = []
    t_all = []
    K_all: Dict[str, np.ndarray] = {}
    D_all: Dict[str, np.ndarray] = {}
    HW_all: Dict[str, Tuple[int, int]] = {}
    view_ids: List[str] = []
    missing: List[str] = []

    for vid in expected_view_ids:
        if vid not in images_by_view:
            missing.append(vid)
            continue
        img = images_by_view[vid]
        cam = rec.cameras[img.camera_id]

        K, D = colmap_camera_to_opencv_KD(cam)
        T_cw = img.cam_from_world()
        R = np.asarray(T_cw.rotation.matrix(), dtype=np.float64).reshape(3, 3)
        t = np.asarray(T_cw.translation, dtype=np.float64).reshape(3)

        K_all[vid] = K
        D_all[vid] = D
        HW_all[vid] = (int(cam.height), int(cam.width))
        R_all.append(R)
        t_all.append(t)
        view_ids.append(vid)

    if len(R_all) != len(expected_view_ids):
        if not bool(allow_partial):
            raise RuntimeError(
                f"Reconstruction registered {len(R_all)} / {len(expected_view_ids)} images. "
                "Try a different frame_idx, or use fewer views / different features."
            )
        print(
            f"[WARN] Reconstruction registered {len(R_all)} / {len(expected_view_ids)} images. "
            f"Proceeding with partial output. Missing views: {missing}"
        )

    R_arr = np.stack(R_all, axis=0)
    t_arr = np.stack(t_all, axis=0)

    # Normalize gauge so ref is identity (optional).
    if normalize_ref:
        if str(ref_view) not in view_ids:
            if not view_ids:
                raise RuntimeError("No registered views; cannot normalize gauge.")
            print(f"[WARN] ref_view {ref_view} not among reconstructed views; using {view_ids[0]} as ref.")
            ref_view = view_ids[0]
        ref_idx = view_ids.index(str(ref_view))
        R_arr, t_arr = normalize_cameras_to_ref(R_all=R_arr, t_all=t_arr, ref_idx=ref_idx)

    out: Dict[str, CameraParams] = {}
    for i, vid in enumerate(view_ids):
        H, W = HW_all[vid]
        out[str(vid)] = CameraParams(
            view_id=str(vid),
            K=np.asarray(K_all[vid], dtype=np.float64),
            R=np.asarray(R_arr[i], dtype=np.float64),
            T=np.asarray(t_arr[i], dtype=np.float64).reshape(3),
            D=np.asarray(D_all[vid], dtype=np.float64).reshape(5),
            H=int(H),
            W=int(W),
        )
    return out


def _resolve_database_camera_dimension(prior_value: int, imported_value: int) -> tuple[int, bool]:
    """Use the image-imported size when a calibration prior omits W/H."""
    prior_value = int(prior_value)
    imported_value = int(imported_value)
    if prior_value > 0:
        return prior_value, False
    if imported_value <= 0:
        raise ValueError(
            "Camera calibration omits image dimensions and the imported image size is invalid: "
            f"prior={prior_value}, imported={imported_value}"
        )
    return imported_value, True


def _set_database_intrinsics_per_view(
    *,
    database_path: Path,
    intrinsics_from_cameras: Path,
    image_name_to_view: Dict[str, str],
) -> None:
    """Overwrite COLMAP DB camera intrinsics with per-view priors.

    Requires images to be imported with PER_IMAGE mode so each image has its own
    camera entry in the database.
    """
    loader = CameraIntrinsicsLoader(Path(intrinsics_from_cameras))
    cam_cache: Dict[str, CameraParams] = {}
    updated_images = 0
    image_size_fallbacks = 0
    with pycolmap.Database.open(str(database_path)) as db:
        images = db.read_all_images()
        for image in images:
            image_name = str(image.name)
            view_id = _resolve_view_id_from_imported_image_name(image_name, image_name_to_view)
            if view_id is None:
                continue
            if view_id not in cam_cache:
                cam_cache[view_id] = loader.load_camera_template(str(view_id))
            cam_prior = cam_cache[view_id]
            K = np.asarray(cam_prior.K, dtype=np.float64).reshape(3, 3)

            db_cam = db.read_camera(int(image.camera_id))
            db_cam.model = "PINHOLE"
            width, width_fallback = _resolve_database_camera_dimension(cam_prior.W, db_cam.width)
            height, height_fallback = _resolve_database_camera_dimension(cam_prior.H, db_cam.height)
            db_cam.width = width
            db_cam.height = height
            image_size_fallbacks += int(width_fallback or height_fallback)
            db_cam.params = np.array(
                [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
                dtype=np.float64,
            )
            db_cam.has_prior_focal_length = True
            db.update_camera(db_cam)
            updated_images += 1

    if updated_images == 0:
        raise RuntimeError(
            "Failed to apply per-view intrinsics: no imported images matched image_name_to_view."
        )
    print(
        "[INFO] Applied per-view intrinsics priors to "
        f"{updated_images} images from {intrinsics_from_cameras}; "
        f"kept imported image size for {image_size_fallbacks} images"
    )


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root", type=str, default="", help="Images root containing per-view subfolders.")
    p.add_argument("--dataset_root", type=str, default="", help="Optional dataset root (contains images/).")
    p.add_argument(
        "--dataset_type",
        type=str,
        default="standard",
        choices=["standard", "custom", "diffuman4d", "panoptic"],
        help="Dataset layout type. Non-standard types expect a dataset config JSON.",
    )
    p.add_argument(
        "--dataset_config",
        type=str,
        default="",
        help="Optional dataset layout JSON (defaults to <dataset_root>/track4d_calib_dataset.json for non-standard layouts).",
    )
    p.add_argument("--views", type=str, nargs="+", required=True)
    p.add_argument("--frame_idx", type=int, default=0, help="Frame index inside each view folder.")
    p.add_argument("--ref_view", type=str, default="", help="Reference view for gauge normalization (optional).")

    p.add_argument("--output_dir", type=str, required=True, help="Directory for HLOC intermediate outputs.")
    p.add_argument("--cameras_out", type=str, required=True, help="Output cameras dir (extri.yml/intri.yml).")

    p.add_argument(
        "--intrinsics_from_cameras",
        type=str,
        default="",
        help="Optional cameras/ directory to read intrinsics priors from.",
    )
    p.add_argument(
        "--intrinsics_mode",
        type=str,
        default="per_view",
        choices=["per_view", "single"],
        help=(
            "How to apply --intrinsics_from_cameras priors: "
            "per_view=fixed per-view K (recommended for zoom/mixed-focal scenes); "
            "single=use K from the first view for all images (legacy behavior)."
        ),
    )
    p.add_argument("--overlap_crop", action="store_true")
    p.add_argument("--crop_resize_max", type=int, default=None,
                   help="Maximum crop side for the second SPSG pass; defaults to the feature extraction size.")
    p.add_argument("--crop_geometry_threshold_px", type=float, default=2.0)
    p.add_argument("--superglue_sinkhorn_iterations", type=int, default=0)
    p.add_argument("--feature_conf", type=str, default="superpoint_aachen")
    p.add_argument(
        "--max_keypoints",
        type=int,
        default=2048,
        help="Override feature extractor max_keypoints when the selected feature config supports it. <=0 disables override.",
    )
    p.add_argument(
        "--superpoint_keypoint_threshold",
        type=float,
        default=-1.0,
        help="Override SuperPoint keypoint_threshold for dense SPSG probes. <0 keeps the HLOC default.",
    )
    p.add_argument(
        "--superglue_match_threshold",
        type=float,
        default=-1.0,
        help="Override SuperGlue match_threshold for dense SPSG probes. <0 keeps the HLOC default.",
    )
    p.add_argument(
        "--feature_resize_max",
        type=int,
        default=0,
        help="Override feature preprocessing resize_max. <=0 keeps the selected HLOC config.",
    )
    p.add_argument(
        "--feature_resize_force",
        action="store_true",
        help="Force feature preprocessing resize even if the original image is smaller than resize_max.",
    )
    p.add_argument(
        "--matcher_conf",
        type=str,
        default="superglue",
        help=f"HLOC matcher config, or '{ROMA_SP_MATCHER_NAME}' for RoMa matches on SuperPoint keypoints.",
    )
    p.add_argument(
        "--match_gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids for sharded parallel matching (e.g. '0,1'). Empty disables sharded matching.",
    )
    p.add_argument(
        "--match_pair_batch_size",
        type=int,
        default=int(os.environ.get("HLOC_MATCH_PAIR_BATCH_SIZE", "1")),
        help=(
            "Pairs per SuperGlue forward pass (micro-batch). "
            "Only safe/effective when pairs in a batch share the same feature/image shapes."
        ),
    )
    p.add_argument(
        "--superglue_batch_size",
        type=int,
        default=0,
        help="Legacy alias for SuperGlue pair micro-batch size. Used when --match_pair_batch_size<=1.",
    )
    p.add_argument(
        "--superglue_num_workers",
        type=int,
        default=0,
        help="Optional override of hloc.match_features DataLoader workers for SuperGlue.",
    )
    p.add_argument(
        "--superglue_writer_threads",
        type=int,
        default=0,
        help="Optional override of match writer threads for SuperGlue.",
    )
    p.add_argument(
        "--superglue_group_by_shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group pairs by keypoint/image shapes before SuperGlue batching to preserve exact behavior.",
    )
    p.add_argument("--camera_mode", type=str, default="SINGLE", help="pycolmap.CameraMode (e.g. SINGLE, PER_IMAGE, AUTO)")
    p.add_argument(
        "--pairs_mode",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "ring_knn"],
        help="How to build image pairs for matching.",
    )
    p.add_argument("--pairs_k", type=int, default=2, help="KNN degree for pairs_mode=ring_knn.")
    p.add_argument(
        "--num_total_views",
        type=int,
        default=0,
        help="Total cameras in the rig (for ring distance). 0 means infer from --data_root.",
    )
    p.add_argument(
        "--masks_root",
        type=str,
        default="",
        help="Optional foreground masks root (e.g. .../fmasks). Used only when mask filtering is enabled.",
    )
    p.add_argument(
        "--mask_mode",
        type=str,
        default="auto",
        choices=["auto", "all", "background", "foreground"],
        help=(
            "Feature mask filtering mode. auto keeps legacy explicit --masks_root behavior "
            "but does not silently auto-detect dataset masks; all disables filtering; "
            "background removes masked foreground; foreground keeps only masked foreground."
        ),
    )
    p.add_argument(
        "--mask_thresh",
        type=int,
        default=0,
        help="Mask threshold: background keeps mask<=thresh; foreground keeps mask>thresh.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--link_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage per-view keyframe images via symlink (fallback to copy if symlink fails).",
    )
    p.add_argument(
        "--timing_json",
        type=str,
        default="",
        help="Optional path to write per-stage runtime summary JSON.",
    )
    p.add_argument("--normalize_ref", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow partial reconstructions (not all input views registered). "
        "Outputs cameras only for registered views and prints a warning.",
    )
    p.add_argument(
        "--skip_geometric_verification",
        action="store_true",
        help="Skip COLMAP geometric verification (RANSAC) and directly use imported matches as inliers.",
    )
    p.add_argument(
        "--max_db_matches_per_pair",
        type=int,
        default=0,
        help="Optional cap on matches imported to COLMAP per image pair (0 disables).",
    )
    p.add_argument(
        "--mapper_num_threads",
        type=int,
        default=0,
        help="Override mapper num_threads (0 keeps default behavior).",
    )
    p.add_argument(
        "--sfm_random_seed", type=int, default=None,
        help="Explicit COLMAP verification/mapping seed (per-view known intrinsics only). "
             "Omit to preserve historical random seeding; use one mapper thread for reproducibility.",
    )
    p.add_argument(
        "--mapper_fast_mode",
        action="store_true",
        help="Enable faster mapper settings (fewer BA iterations/refinements, single-model).",
    )
    p.add_argument(
        "--mapper_backend",
        type=str,
        default="colmap",
        choices=["colmap", "glomap"],
        help="SfM backend after HLOC feature/match import.",
    )
    p.add_argument("--glomap_exe", type=str, default="glomap", help="GLOMAP executable used when --mapper_backend=glomap.")
    p.add_argument("--glomap_ba_iterations", type=int, default=3, help="GLOMAP bundle-adjustment iterations.")
    p.add_argument(
        "--tracks_dir",
        type=str,
        default="",
        help="Optional tracking dir (outputs from build_multiview_tracks.py). "
        "If set, tracked points at --tracks_frame_idx are appended as extra keypoints and "
        "their matches are merged into the HLOC matches before SfM.",
    )
    p.add_argument(
        "--tracks_frame_idx",
        type=int,
        default=None,
        help="Frame index to sample tracked points from (default: --frame_idx).",
    )
    p.add_argument(
        "--tracks_score",
        type=float,
        default=1.0,
        help="Matching score assigned to propagated tracking matches.",
    )
    p.add_argument(
        "--tracks_min_visible_views_at_frame",
        type=int,
        default=0,
        help="Keep only tracked points visible in at least this many views at --tracks_frame_idx. 0 disables.",
    )
    p.add_argument(
        "--tracks_min_ref_temporal_support",
        type=int,
        default=0,
        help="Keep only tracked points visible in ref view for at least this many frames. 0 disables.",
    )
    p.add_argument(
        "--tracks_max_points",
        type=int,
        default=0,
        help="Maximum number of tracked points to inject (after filtering). 0 means all kept points.",
    )
    p.add_argument(
        "--tracks_bridge_short_view_ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional static-view ids for bridge observability filtering/scoring.",
    )
    p.add_argument(
        "--tracks_bridge_long_view_ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional dynamic-view ids for bridge observability filtering/scoring.",
    )
    p.add_argument(
        "--tracks_bridge_min_short_visible_at_frame",
        type=int,
        default=0,
        help="Keep only tracked points visible in at least this many static views at --tracks_frame_idx. 0 disables.",
    )
    p.add_argument(
        "--tracks_bridge_min_long_visible_at_frame",
        type=int,
        default=0,
        help="Keep only tracked points visible in at least this many dynamic views at --tracks_frame_idx. 0 disables.",
    )
    p.add_argument(
        "--tracks_bridge_quality_weight",
        type=float,
        default=0.0,
        help="Quality weight for bridge strength min(short_visible,long_visible) when selecting --tracks_max_points.",
    )
    p.add_argument(
        "--tracks_epipolar_cameras",
        type=str,
        default="",
        help="Optional cameras/ directory used to epipolar-filter injected track matches.",
    )
    p.add_argument(
        "--tracks_epipolar_max_sampson",
        type=float,
        default=0.0,
        help="If >0, keep injected track matches with Sampson distance <= this threshold.",
    )
    p.add_argument(
        "--tracks_epipolar_min_keep_ratio",
        type=float,
        default=0.0,
        help="When epipolar gating is enabled, keep at least this ratio of track matches per pair (best residual first).",
    )
    p.add_argument(
        "--tracks_inject_pair_view_ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional view ids to restrict which image pairs receive injected track matches.",
    )
    p.add_argument(
        "--tracks_inject_pair_mode",
        type=str,
        default="any",
        choices=["any", "both", "xor"],
        help="Policy for --tracks_inject_pair_view_ids: any=at least one endpoint in set, both=both endpoints in set, xor=exactly one endpoint in set.",
    )
    p.add_argument(
        "--tracks_graph_balance_mode",
        type=str,
        default="off",
        choices=["off", "per_long", "per_long_soft"],
        help="Optional ref-free graph-balance mode for injected tracking matches.",
    )
    p.add_argument(
        "--tracks_graph_short_view_ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional static-view ids used for graph pair typing in tracks_graph_balance_mode.",
    )
    p.add_argument(
        "--tracks_graph_long_view_ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional dynamic-view ids used for graph pair typing in tracks_graph_balance_mode.",
    )
    p.add_argument(
        "--tracks_graph_exclude_view_ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional view ids to exclude from injected tracking pairs in graph-balance mode.",
    )
    p.add_argument(
        "--tracks_graph_allowed_pair_types",
        type=str,
        nargs="*",
        default=[],
        help="Allowed pair types in graph-balance mode. Subset of {SS,SL,LL,OTHER}. Empty means all.",
    )
    p.add_argument(
        "--tracks_graph_max_injected_matches_per_pair",
        type=int,
        default=0,
        help="Max injected tracking matches per pair in graph-balance mode. 0 disables.",
    )
    p.add_argument(
        "--tracks_graph_max_injected_matches_per_long_view",
        type=int,
        default=0,
        help="Max injected tracking matches touching each long view in graph-balance mode. 0 disables.",
    )
    p.add_argument(
        "--tracks_graph_min_injected_matches_per_long_view",
        type=int,
        default=0,
        help="Min injected tracking matches touching each long view in graph-balance mode before soft decay suppresses pairs. 0 disables.",
    )
    p.add_argument(
        "--tracks_graph_soft_target_per_long_view",
        type=int,
        default=0,
        help="Soft target injected matches per long view for graph mode per_long_soft. 0 uses decay-only balancing.",
    )
    p.add_argument(
        "--tracks_graph_soft_min_keep_ratio",
        type=float,
        default=0.05,
        help="Minimum keep ratio per pair in graph mode per_long_soft (0..1).",
    )
    p.add_argument(
        "--tracks_graph_soft_decay",
        type=float,
        default=600.0,
        help="Decay scale for graph mode per_long_soft when soft target is 0.",
    )
    p.add_argument(
        "--roma_setting",
        type=str,
        default="precise",
        choices=["precise", "fast", "base", "turbo"],
        help=f"RoMaV2 setting used when --matcher_conf={ROMA_SP_MATCHER_NAME}.",
    )
    p.add_argument(
        "--roma_compile",
        action="store_true",
        help=f"Enable torch.compile in RoMaV2 when --matcher_conf={ROMA_SP_MATCHER_NAME}.",
    )
    p.add_argument(
        "--romav2_weights",
        type=str,
        default="",
        help=f"Optional local romav2.pt path when --matcher_conf={ROMA_SP_MATCHER_NAME}.",
    )
    p.add_argument(
        "--roma_max_dist",
        type=float,
        default=0.02,
        help=f"Max normalized keypoint distance for RoMa-SP matching ({ROMA_SP_MATCHER_NAME}).",
    )
    p.add_argument(
        "--roma_cert_th",
        type=float,
        default=0.0,
        help=f"Minimum RoMa overlap confidence for RoMa-SP matching ({ROMA_SP_MATCHER_NAME}).",
    )
    args = p.parse_args()
    if args.crop_resize_max is not None:
        if not args.overlap_crop:
            p.error("--crop_resize_max requires --overlap_crop")
        if not 64 <= args.crop_resize_max <= 4096:
            p.error("--crop_resize_max must be from 64 to 4096")
    crop_resize_max = (args.crop_resize_max if args.crop_resize_max is not None
                       else args.feature_resize_max if args.feature_resize_max > 0 else 1024)
    verification_options = None
    seeded_mapping = {}
    if args.sfm_random_seed is not None:
        if not (args.intrinsics_from_cameras and str(args.intrinsics_mode).lower() == "per_view"
                and str(args.mapper_backend) == "colmap"):
            raise ValueError("explicit SfM seeding requires COLMAP and per-view input intrinsics")
        verification_options, seeded_mapping = seeded_colmap_options(args.sfm_random_seed)

    from track4d_calib.dataset_layout import resolve_dataset_layout

    if not args.data_root and not args.dataset_root:
        raise SystemExit("Pass --data_root (images root) or --dataset_root (dataset root).")

    dataset_root_arg = Path(args.dataset_root) if args.dataset_root else None
    images_root_arg = Path(args.data_root) if args.data_root else None
    layout = resolve_dataset_layout(
        dataset_root=dataset_root_arg,
        images_root=images_root_arg,
        masks_root=Path(args.masks_root) if args.masks_root else None,
        dataset_type=str(args.dataset_type),
        dataset_config=Path(args.dataset_config) if args.dataset_config else None,
    )

    data_root = Path(layout.images_root)
    views = [str(v) for v in args.views]
    if len(views) < 2:
        raise SystemExit("--views must have at least 2 entries")

    ref_view = str(args.ref_view) if args.ref_view else views[len(views) // 2]
    if ref_view not in views:
        ref_view = views[0]

    output_dir = Path(args.output_dir)
    hloc_dir = output_dir / "hloc_sfm"
    hloc_dir.mkdir(parents=True, exist_ok=True)
    t_total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    # 1) Stage one keyframe per view to a flat image directory (prefer symlink).
    t_export_start = time.perf_counter()
    image_dir = hloc_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_names: List[str] = []
    image_name_to_view: Dict[str, str] = {}
    image_name_to_src: Dict[str, Path] = {}
    for vid in views:
        paths = list_image_paths(data_root / vid, start=int(args.frame_idx), count=1)
        if len(paths) != 1:
            raise RuntimeError(f"View {vid}: frame {args.frame_idx} not found under {data_root / vid}")
        src = paths[0]
        src_ext = str(Path(src).suffix).lower()
        if not src_ext:
            src_ext = ".png"
        desired_dst = image_dir / f"view{vid}{src_ext}"
        existing_variants = sorted(
            p for p in image_dir.glob(f"view{vid}.*") if (p.is_file() or p.is_symlink())
        )
        if bool(args.overwrite):
            for p in existing_variants:
                if p != desired_dst:
                    p.unlink()
            dst = desired_dst
        else:
            if desired_dst.is_file() or desired_dst.is_symlink():
                dst = desired_dst
            elif existing_variants:

                dst = existing_variants[0]
            else:
                dst = desired_dst
        if bool(args.overwrite) or not (dst.is_file() or dst.is_symlink()):
            _stage_image_file(src=Path(src), dst=dst, use_symlink=bool(args.link_images))
        name = str(dst.name)
        image_names.append(name)
        image_name_to_view[name] = vid
        image_name_to_src[name] = Path(src)
    timings["export_keyframe_images_sec"] = float(time.perf_counter() - t_export_start)

    # 2) HLOC pairs / features / matches.
    t_pairs_start = time.perf_counter()
    try:
        feat_conf = copy.deepcopy(extract_features.confs[str(args.feature_conf)])
    except KeyError as e:
        raise SystemExit(f"Unknown --feature_conf {args.feature_conf}. Available: {list(extract_features.confs.keys())}") from e
    max_keypoints_override = int(args.max_keypoints)
    is_superpoint_feature = str(args.feature_conf).startswith("superpoint")
    if max_keypoints_override > 0 and is_superpoint_feature:
        feat_model = feat_conf.get("model", None)
        if isinstance(feat_model, dict) and "max_keypoints" in feat_model:
            old_k = feat_model.get("max_keypoints")
            feat_model["max_keypoints"] = int(max_keypoints_override)
            print(
                f"[INFO] Override max_keypoints for feature_conf={args.feature_conf}: "
                f"{old_k} -> {max_keypoints_override}"
            )
        else:
            print(
                f"[INFO] --max_keypoints={max_keypoints_override} ignored for feature_conf={args.feature_conf} "
                "(no model.max_keypoints in config)."
            )
    elif max_keypoints_override > 0:
        print(
            f"[INFO] Keep feature_conf={args.feature_conf} default keypoint behavior "
            "(--max_keypoints override is only applied to SuperPoint/SPSG path)."
        )
    if is_superpoint_feature and float(args.superpoint_keypoint_threshold) >= 0.0:
        feat_model = feat_conf.setdefault("model", {})
        old_thr = feat_model.get("keypoint_threshold", "<default>")
        feat_model["keypoint_threshold"] = float(args.superpoint_keypoint_threshold)
        print(
            f"[INFO] Override SuperPoint keypoint_threshold for feature_conf={args.feature_conf}: "
            f"{old_thr} -> {float(args.superpoint_keypoint_threshold)}"
        )
    if int(args.feature_resize_max) > 0:
        preproc = feat_conf.setdefault("preprocessing", {})
        old_resize = preproc.get("resize_max", "<default>")
        preproc["resize_max"] = int(args.feature_resize_max)
        print(
            f"[INFO] Override feature preprocessing resize_max for feature_conf={args.feature_conf}: "
            f"{old_resize} -> {int(args.feature_resize_max)}"
        )
    if bool(args.feature_resize_force):
        preproc = feat_conf.setdefault("preprocessing", {})
        old_force = preproc.get("resize_force", False)
        preproc["resize_force"] = True
        print(
            f"[INFO] Override feature preprocessing resize_force for feature_conf={args.feature_conf}: "
            f"{old_force} -> True"
        )
    matcher_conf_name = str(args.matcher_conf)
    match_conf: Optional[Dict] = None
    effective_pair_batch_size = max(1, int(args.match_pair_batch_size))
    superglue_batch_alias = int(args.superglue_batch_size)
    if effective_pair_batch_size <= 1 and superglue_batch_alias > 0:
        effective_pair_batch_size = int(superglue_batch_alias)
    if matcher_conf_name != ROMA_SP_MATCHER_NAME:
        try:
            match_conf = copy.deepcopy(match_features.confs[matcher_conf_name])
        except KeyError as e:
            choices = sorted(list(match_features.confs.keys()) + [ROMA_SP_MATCHER_NAME])
            raise SystemExit(f"Unknown --matcher_conf {args.matcher_conf}. Available: {choices}") from e
        if (
            matcher_conf_name == "superglue"
            and float(args.superglue_match_threshold) >= 0.0
            and isinstance(match_conf.get("model"), dict)
        ):
            old_thr = match_conf["model"].get("match_threshold", "<default>")
            match_conf["model"]["match_threshold"] = float(args.superglue_match_threshold)
            print(
                f"[INFO] Override SuperGlue match_threshold: "
                f"{old_thr} -> {float(args.superglue_match_threshold)}"
            )
        if matcher_conf_name == "superglue" and args.superglue_sinkhorn_iterations > 0:
            match_conf["model"]["sinkhorn_iterations"] = args.superglue_sinkhorn_iterations
        if effective_pair_batch_size > 1:
            match_conf["pair_batch_size"] = int(effective_pair_batch_size)
            print(
                "[INFO] Override matcher pair batch size: "
                f"{effective_pair_batch_size}"
            )
        if int(args.superglue_num_workers) > 0:
            match_conf["num_workers"] = int(args.superglue_num_workers)
        if int(args.superglue_writer_threads) > 0:
            match_conf["writer_threads"] = int(args.superglue_writer_threads)

    pairs_path = hloc_dir / "pairs.txt"
    if args.pairs_mode == "exhaustive":
        pairs_from_exhaustive.main(pairs_path, image_list=image_names)
    elif args.pairs_mode == "ring_knn":
        num_total_views = int(args.num_total_views) if int(args.num_total_views) > 0 else _infer_num_total_views(data_root)
        pairs_views = pairs_ring_knn(views, k=int(args.pairs_k), num_total_views=num_total_views)
        view_to_image = {v: n for n, v in image_name_to_view.items()}
        with open(pairs_path, "w", encoding="utf-8") as f:
            for a, b in pairs_views:
                f.write(f"{view_to_image[a]} {view_to_image[b]}\n")
    else:
        raise SystemExit(f"Unknown pairs_mode: {args.pairs_mode}")
    timings["pair_generation_sec"] = float(time.perf_counter() - t_pairs_start)

    t_feature_start = time.perf_counter()
    feature_overwrite = bool(args.overwrite)
    try:
        features_path = extract_features.main(
            feat_conf,
            image_dir=image_dir,
            export_dir=hloc_dir,
            image_list=image_names,
            overwrite=feature_overwrite,
        )
    except OSError as e:
        # Recover from stale/unclean HDF5 files left by interrupted jobs.
        if (not feature_overwrite) and _is_h5_write_lock_error(e):
            stem_prefix = str(feat_conf.get("output", "")).strip()
            removed = _cleanup_hloc_h5(hloc_dir, stem_prefix=stem_prefix)
            print(
                "[WARN] Feature HDF5 open failed due to stale write lock. "
                f"Removed {removed} stale files under {hloc_dir}, retry with overwrite=True."
            )
            features_path = extract_features.main(
                feat_conf,
                image_dir=image_dir,
                export_dir=hloc_dir,
                image_list=image_names,
                overwrite=True,
            )
        else:
            raise
    timings["feature_extraction_sec"] = float(time.perf_counter() - t_feature_start)

    # Optional: filter HLOC features using per-frame foreground masks.
    t_mask_start = time.perf_counter()
    mask_mode = str(args.mask_mode).strip().lower()
    mask_filter_mode = None
    masks_root = None
    if mask_mode != "all":
        if args.masks_root:
            masks_root = Path(args.masks_root)
            mask_filter_mode = "background" if mask_mode == "auto" else mask_mode
        elif mask_mode in {"background", "foreground"}:
            if layout.masks_root is not None:
                masks_root = Path(layout.masks_root)
            else:
                masks_root = data_root.parent / "fmasks" if str(layout.dataset_type) == "standard" else None
            mask_filter_mode = mask_mode
        else:
            # auto mode intentionally avoids implicit dataset mask discovery. The
            # DynDNA masks are foreground masks, and enabling them silently changes
            # the SfM protocol.
            masks_root = None

    if mask_filter_mode is not None and masks_root is not None and masks_root.is_dir():
        image_masks = {}
        for name in image_names:
            vid = image_name_to_view[name]
            src = image_name_to_src[name]
            mpath = find_frame_mask_path(masks_root, vid, src)
            if mpath is None:
                continue
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            image_masks[name] = m
        if image_masks:
            stats = filter_hloc_features_by_masks(
                features_path,
                image_masks,
                mask_thresh=int(args.mask_thresh),
                mode=str(mask_filter_mode),
            )
            kept = sum(s["kept"] for s in stats.values())
            before = sum(s["before"] for s in stats.values())
            print(
                f"[INFO] Mask-filtered features ({mask_filter_mode}): kept {kept}/{before} "
                f"keypoints across {len(stats)}/{len(image_names)} images."
            )
    timings["mask_filtering_sec"] = float(time.perf_counter() - t_mask_start)

    t_match_start = time.perf_counter()
    if matcher_conf_name == ROMA_SP_MATCHER_NAME:
        matches_out = hloc_dir / f"{features_path.stem}_matches-roma-sp_{pairs_path.stem}.h5"
        matches_path = _match_features_roma_sp(
            image_dir=image_dir,
            pairs_path=pairs_path,
            features_path=features_path,
            matches_out=matches_out,
            overwrite=bool(args.overwrite),
            roma_setting=str(args.roma_setting),
            roma_compile=bool(args.roma_compile),
            romav2_weights=str(args.romav2_weights),
            roma_max_dist=float(args.roma_max_dist),
            roma_cert_th=float(args.roma_cert_th),
        )
    else:
        assert match_conf is not None
        match_conf_local = copy.deepcopy(match_conf)
        is_spsg_matcher = str(matcher_conf_name) in {"superglue", "superglue-fast"}
        if is_superpoint_feature and is_spsg_matcher:
            match_conf_local["batch_size"] = max(1, int(args.superglue_batch_size))
            match_conf_local["num_workers"] = max(0, int(args.superglue_num_workers))
            match_conf_local["writer_threads"] = max(1, int(args.superglue_writer_threads))
            match_conf_local["group_by_keypoint_shape"] = bool(args.superglue_group_by_shape)
        matches_out = hloc_dir / f"{features_path.stem}_{match_conf['output']}_{pairs_path.stem}.h5"
        match_gpu_ids = _parse_match_gpu_ids(str(args.match_gpus))
        use_multi_gpu_matching = (
            len(match_gpu_ids) > 1
            and is_superpoint_feature
            and is_spsg_matcher
        )
        if len(match_gpu_ids) > 1 and not use_multi_gpu_matching:
            print(
                "[INFO] Ignore --match_gpus for non-SPSG configuration: "
                f"feature_conf={args.feature_conf}, matcher_conf={matcher_conf_name}"
            )
        if use_multi_gpu_matching:
            matches_path = _match_features_multi_gpu(
                matcher_conf_name=matcher_conf_name,
                pairs_path=pairs_path,
                features_path=features_path,
                matches_out=matches_out,
                overwrite=bool(args.overwrite),
                gpu_ids=match_gpu_ids,
                work_dir=hloc_dir,
                pair_batch_size=int(effective_pair_batch_size),
                match_batch_size=max(1, int(args.superglue_batch_size)),
                num_workers=int(args.superglue_num_workers),
                writer_threads=int(args.superglue_writer_threads),
                group_by_shape=bool(args.superglue_group_by_shape),
            )
        else:
            match_overwrite = bool(args.overwrite)
            try:
                matches_path = match_features.main(
                    match_conf_local,
                    pairs=pairs_path,
                    features=features_path,
                    matches=matches_out,
                    overwrite=match_overwrite,
                )
            except OSError as e:
                if (not match_overwrite) and _is_h5_write_lock_error(e):
                    removed = 0
                    if matches_out.exists():
                        try:
                            matches_out.unlink()
                            removed = 1
                        except Exception:
                            pass
                    print(
                        "[WARN] Match HDF5 open failed due to stale write lock. "
                        f"Removed {removed} stale files and retry with overwrite=True."
                    )
                    matches_path = match_features.main(
                        match_conf_local,
                        pairs=pairs_path,
                        features=features_path,
                        matches=matches_out,
                        overwrite=True,
                    )
                else:
                    raise
    timings["feature_matching_sec"] = float(time.perf_counter() - t_match_start)

    if args.overlap_crop:
        if matcher_conf_name != "superglue" or not is_superpoint_feature:
            raise ValueError("Overlap refinement requires SuperPoint and SuperGlue")
        from overlap_matching import refine_overlap
        features_path, matches_path, crop_summary = refine_overlap(
            image_dir, pairs_path, features_path, matches_path, feat_conf, match_conf,
            resize_max=crop_resize_max,
            geometry_threshold_px=args.crop_geometry_threshold_px,
        )
        timings["overlap_matching_sec"] = crop_summary["seconds"]
        timings["first_pass_matching_sec"] = timings["feature_matching_sec"]
        timings["feature_matching_sec"] += crop_summary["seconds"]

    t_track_merge_start = time.perf_counter()
    if str(args.tracks_dir).strip():
        track_frame = int(args.tracks_frame_idx) if args.tracks_frame_idx is not None else int(args.frame_idx)
        tracks_dir = Path(args.tracks_dir)
        features_aug = hloc_dir / f"{features_path.stem}_tracks_f{track_frame:06d}.h5"
        matches_aug = hloc_dir / f"{matches_path.stem}_tracks_f{track_frame:06d}.h5"
        features_path, matches_path = _merge_tracking_matches(
            tracks_dir=tracks_dir,
            frame_idx=track_frame,
            ref_view=ref_view,
            image_name_to_view=image_name_to_view,
            features_path=features_path,
            matches_path=matches_path,
            pairs_path=pairs_path,
            output_features=features_aug,
            output_matches=matches_aug,
            track_score=float(args.tracks_score),
            min_visible_views_at_frame=int(args.tracks_min_visible_views_at_frame),
            min_ref_temporal_support=int(args.tracks_min_ref_temporal_support),
            max_track_points=int(args.tracks_max_points),
            epipolar_cameras_dir=(Path(args.tracks_epipolar_cameras) if str(args.tracks_epipolar_cameras).strip() else None),
            epipolar_max_sampson=float(args.tracks_epipolar_max_sampson),
            epipolar_min_keep_ratio=float(args.tracks_epipolar_min_keep_ratio),
            bridge_short_view_ids=[str(v) for v in args.tracks_bridge_short_view_ids],
            bridge_long_view_ids=[str(v) for v in args.tracks_bridge_long_view_ids],
            bridge_min_short_visible_at_frame=int(args.tracks_bridge_min_short_visible_at_frame),
            bridge_min_long_visible_at_frame=int(args.tracks_bridge_min_long_visible_at_frame),
            bridge_quality_weight=float(args.tracks_bridge_quality_weight),
            inject_pair_view_ids=[str(v) for v in args.tracks_inject_pair_view_ids],
            inject_pair_mode=str(args.tracks_inject_pair_mode),
            graph_balance_mode=str(args.tracks_graph_balance_mode),
            graph_short_view_ids=[str(v) for v in args.tracks_graph_short_view_ids],
            graph_long_view_ids=[str(v) for v in args.tracks_graph_long_view_ids],
            graph_exclude_view_ids=[str(v) for v in args.tracks_graph_exclude_view_ids],
            graph_allowed_pair_types=[str(v) for v in args.tracks_graph_allowed_pair_types],
            graph_max_injected_matches_per_pair=int(args.tracks_graph_max_injected_matches_per_pair),
            graph_max_injected_matches_per_long_view=int(args.tracks_graph_max_injected_matches_per_long_view),
            graph_min_injected_matches_per_long_view=int(args.tracks_graph_min_injected_matches_per_long_view),
            graph_soft_target_per_long_view=int(args.tracks_graph_soft_target_per_long_view),
            graph_soft_min_keep_ratio=float(args.tracks_graph_soft_min_keep_ratio),
            graph_soft_decay=float(args.tracks_graph_soft_decay),
            overwrite=bool(args.overwrite),
        )
    timings["track_merge_sec"] = float(time.perf_counter() - t_track_merge_start)

    # 3) SfM reconstruction.
    t_sfm_start = time.perf_counter()
    image_options = {}
    mapper_options = {}
    intrinsics_mode = str(args.intrinsics_mode).lower()
    if args.intrinsics_from_cameras:
        K, _, _, _ = _infer_intrinsics_from_cameras(Path(args.intrinsics_from_cameras), views[0])
        if intrinsics_mode == "single":
            image_options = {
                "camera_model": "PINHOLE",
                "camera_params": _format_colmap_camera_params_pinhole(K),
            }
        # Keep intrinsics fixed if we trust the priors.
        mapper_options = {
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
        }
    if int(args.mapper_num_threads) > 0:
        mapper_options["num_threads"] = int(args.mapper_num_threads)
    if args.sfm_random_seed is not None:
        mapper_options.update(seeded_mapping)
    if bool(args.mapper_fast_mode):
        mapper_options.update(
            {
                "multiple_models": False,
                "ba_local_max_num_iterations": 15,
                "ba_global_max_num_iterations": 20,
                "ba_local_max_refinements": 1,
                "ba_global_max_refinements": 2,
            }
        )

    sfm_dir = hloc_dir / "sfm"
    camera_mode = _colmap_camera_mode(args.camera_mode)
    if args.intrinsics_from_cameras and intrinsics_mode == "per_view":
        if camera_mode != pycolmap.CameraMode.PER_IMAGE:
            print(
                "[INFO] Overriding --camera_mode to PER_IMAGE for "
                "--intrinsics_mode per_view"
            )
        camera_mode = pycolmap.CameraMode.PER_IMAGE

    if args.intrinsics_from_cameras and intrinsics_mode == "per_view":
        t_sfm_db_prepare_start = time.perf_counter()
        sfm_dir.mkdir(parents=True, exist_ok=True)
        database = sfm_dir / "database.db"
        reconstruction.create_empty_db(database)
        reconstruction.import_images(
            image_dir=image_dir,
            database_path=database,
            camera_mode=camera_mode,
            image_list=image_names,
            options=None,
        )
        _canonicalize_database_image_names(
            database_path=database,
            image_names=image_names,
            image_name_to_view=image_name_to_view,
        )
        _set_database_intrinsics_per_view(
            database_path=database,
            intrinsics_from_cameras=Path(args.intrinsics_from_cameras),
            image_name_to_view=image_name_to_view,
        )
        image_ids_raw = reconstruction.get_image_ids(database)
        image_ids = _remap_image_ids_to_feature_names(
            image_ids_raw,
            image_names=image_names,
            image_name_to_view=image_name_to_view,
        )
        if len(image_ids) != len(image_names):
            missing_names = [n for n in image_names if n not in image_ids]
            print(
                "[WARN] DB image-id remap mismatch: "
                f"{len(image_ids)}/{len(image_names)} names resolved. Missing={missing_names[:8]}"
            )
        if not image_ids:
            raise RuntimeError("No image ids resolved for feature/match import.")
        with pycolmap.Database.open(str(database)) as db:
            with pycolmap.DatabaseTransaction(db):
                reconstruction.import_features(image_ids, db, features_path)
                if int(args.max_db_matches_per_pair) > 0:
                    print(
                        f"[INFO] Limit DB matches per pair: max_db_matches_per_pair={int(args.max_db_matches_per_pair)}"
                    )
                    _import_matches_limited(
                        image_ids=image_ids,
                        db=db,
                        pairs_path=pairs_path,
                        matches_path=matches_path,
                        min_match_score=None,
                        skip_geometric_verification=bool(args.skip_geometric_verification),
                        max_matches_per_pair=int(args.max_db_matches_per_pair),
                    )
                else:
                    reconstruction.import_matches(
                        image_ids,
                        db,
                        pairs_path,
                        matches_path,
                        min_match_score=None,
                        skip_geometric_verification=bool(args.skip_geometric_verification),
                    )
        if not bool(args.skip_geometric_verification):
            if verification_options is None:
                reconstruction.estimation_and_geometric_verification(database, pairs_path, verbose=False)
            else:
                # Use the same HLOC RANSAC thresholds, with an explicit C++ seed.
                pycolmap.verify_matches(str(database), str(pairs_path), options=verification_options)
        else:
            print("[INFO] Skip geometric verification: enabled by --skip_geometric_verification")
        timings["sfm_db_prepare_and_geom_verify_sec"] = float(time.perf_counter() - t_sfm_db_prepare_start)
        t_sfm_recon_start = time.perf_counter()
        if str(args.mapper_backend) == "glomap":
            glomap_dir = sfm_dir / "glomap_model"
            if glomap_dir.exists() and bool(args.overwrite):
                shutil.rmtree(glomap_dir)
            glomap_dir.mkdir(parents=True, exist_ok=True)
            glomap_cmd = [
                str(args.glomap_exe),
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(glomap_dir),
                "--output_format",
                "bin",
                "--ba_iteration_num",
                str(int(args.glomap_ba_iterations)),
            ]
            if args.intrinsics_from_cameras:
                glomap_cmd += [
                    "--BundleAdjustment.optimize_intrinsics",
                    "0",
                    "--BundleAdjustment.optimize_principal_point",
                    "0",
                ]
            _run_cmd(glomap_cmd, cwd=PROJECT_ROOT, log_path=hloc_dir / "glomap_mapper.log")
            model_dir = _select_colmap_model_dir(glomap_dir)
            rec = pycolmap.Reconstruction(str(model_dir))
        else:
            rec = reconstruction.run_reconstruction(
                sfm_dir=sfm_dir,
                database_path=database,
                image_dir=image_dir,
                verbose=False,
                options=mapper_options or None,
            )
        timings["sfm_reconstruction_sec"] = float(time.perf_counter() - t_sfm_recon_start)
    else:
        if str(args.mapper_backend) == "glomap":
            raise SystemExit("--mapper_backend=glomap currently requires --intrinsics_from_cameras with --intrinsics_mode per_view.")
        t_sfm_recon_start = time.perf_counter()
        rec = reconstruction.main(
            sfm_dir=sfm_dir,
            image_dir=image_dir,
            pairs=pairs_path,
            features=features_path,
            matches=matches_path,
            camera_mode=camera_mode,
            image_options=image_options or None,
            mapper_options=mapper_options or None,
        )
        timings["sfm_reconstruction_sec"] = float(time.perf_counter() - t_sfm_recon_start)
    if rec is None:
        raise SystemExit("HLOC reconstruction failed (no model). Try --frame_idx or use different configs.")
    timings["sfm_total_sec"] = float(time.perf_counter() - t_sfm_start)

    t_export_cameras_start = time.perf_counter()
    cams = _build_cameras_from_reconstruction(
        rec,
        image_name_to_view=image_name_to_view,
        ref_view=ref_view,
        normalize_ref=bool(args.normalize_ref),
        allow_partial=bool(args.allow_partial),
    )
    save_cameras(Path(args.cameras_out), cams)
    timings["export_pred_cameras_sec"] = float(time.perf_counter() - t_export_cameras_start)
    timings["total_sec"] = float(time.perf_counter() - t_total_start)

    timing_payload = {
        "frame_idx": int(args.frame_idx),
        "num_views": int(len(views)),
        "feature_conf": str(args.feature_conf),
        "max_keypoints": int(args.max_keypoints),
        "feature_resize_max": int(args.feature_resize_max),
        "superglue_sinkhorn_iterations": int(args.superglue_sinkhorn_iterations),
        "superglue_match_threshold": float(args.superglue_match_threshold),
        "feature_resize_force": bool(args.feature_resize_force),
        "superpoint_keypoint_threshold": float(args.superpoint_keypoint_threshold),
        "matcher_conf": str(args.matcher_conf),
        "overlap_crop": bool(args.overlap_crop),
        "crop_resize_max": crop_resize_max if args.overlap_crop else None,
        "crop_geometry_threshold_px": args.crop_geometry_threshold_px if args.overlap_crop else None,
        "match_gpus": str(args.match_gpus),
        "match_pair_batch_size": int(effective_pair_batch_size),
        "superglue_batch_size": int(args.superglue_batch_size),
        "superglue_num_workers": int(args.superglue_num_workers),
        "superglue_writer_threads": int(args.superglue_writer_threads),
        "superglue_group_by_shape": bool(args.superglue_group_by_shape),
        "pairs_mode": str(args.pairs_mode),
        "mask_mode": str(args.mask_mode),
        "mask_thresh": int(args.mask_thresh),
        "skip_geometric_verification": bool(args.skip_geometric_verification),
        "max_db_matches_per_pair": int(args.max_db_matches_per_pair),
        "mapper_num_threads": int(args.mapper_num_threads),
        "sfm_random_seed": args.sfm_random_seed,
        "mapper_fast_mode": bool(args.mapper_fast_mode),
        "mapper_backend": str(args.mapper_backend),
        "glomap_exe": str(args.glomap_exe),
        "glomap_ba_iterations": int(args.glomap_ba_iterations),
        "timings_sec": timings,
    }
    timing_json = Path(args.timing_json) if str(args.timing_json).strip() else (output_dir / "hloc_stage_timing.json")
    timing_json.parent.mkdir(parents=True, exist_ok=True)
    timing_json.write_text(json.dumps(timing_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Wrote timing summary to: {timing_json}")

    print(f"[OK] Wrote init cameras to: {args.cameras_out}")
    print(f"[OK] HLOC outputs: {hloc_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
