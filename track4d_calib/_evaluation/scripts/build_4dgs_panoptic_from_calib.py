#!/usr/bin/env python3
"""Build 4DGaussians PanopticSports-style dataset from per-frame calibrated cameras.

Expected calibration layout (from HLOC-SPSG / GLOMAP / unified per-frame calibration outputs):
  <calib_run_root>/{baseline_per_frame|glomap_per_frame}/f000001/cameras_pred/{extri.yml,intri.yml}

Expected render dataset layout:
  <render_scene_root>/images/<view>/<frame>.png
  <render_scene_root>/cameras/{extri.yml,intri.yml}
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

try:
    from plyfile import PlyData, PlyElement
    _HAS_PLYFILE = True
except Exception:
    PlyData = None
    PlyElement = None
    _HAS_PLYFILE = False

from track4d_calib.cameras import CameraLoader
from track4d_calib.topk_static_rescue import align_candidate_cameras_to_target_static_gauge


def _symlink_or_copy(src: Path, dst: Path) -> str:
    try:
        dst.symlink_to(src)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _natural_key(s: str) -> List[object]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(s))]


def _canon_view_id(view_id: str) -> str:
    token = str(view_id).strip()
    if token.isdigit():
        return token.zfill(2)
    return token


def _view_id_aliases(view_id: str) -> List[str]:
    token = str(view_id).strip()
    if token == "":
        return []
    aliases = [token]
    if token.isdigit():
        value = int(token)
        aliases.extend([str(value), f"{value:02d}", f"{value:03d}"])
    seen = set()
    out: List[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            out.append(alias)
    return out


def _view_dir_name(view_id: str) -> str:
    v = _canon_view_id(view_id)
    if v.isdigit():
        return v
    return v


def _resolve_source_view_dir(render_scene_root: Path, view_id: str) -> Tuple[Path | None, str]:
    for alias in _view_id_aliases(view_id):
        candidate = render_scene_root / "images" / alias
        if candidate.is_dir():
            return candidate, alias
    return None, _view_dir_name(view_id)


def _discover_views_from_extri(cameras_dir: Path) -> List[str]:
    extri = cameras_dir / "extri.yml"
    if not extri.is_file():
        return []
    text = extri.read_text(encoding="utf-8", errors="ignore")
    ids = re.findall(r"^Rot_(.+?):", text, flags=re.MULTILINE)
    uniq = sorted(set(v.strip() for v in ids if v.strip()), key=_natural_key)
    return [_canon_view_id(v) for v in uniq]


def _camera_lookup_aliases(view_id: str) -> List[str]:
    return _view_id_aliases(view_id)


def _load_camera_by_alias(loader: CameraLoader, view_id: str):
    for alias in _camera_lookup_aliases(view_id):
        try:
            return loader.load_camera(alias)
        except Exception:
            continue
    return None


def _per_frame_image_root_for_view(
    *,
    view: str,
    test_views: set[str],
    train_root: Path,
    test_root: Path | None,
) -> Path:
    if str(view) not in test_views:
        return train_root
    if test_root is None:
        raise ValueError(
            "source_images_from_per_frame requires a separate test per-frame "
            f"root for test view {view}"
        )
    return test_root


def _resolve_source_image(src_view_dir: Path, frame_idx: int) -> Path | None:
    stem = f"{int(frame_idx):06d}"
    for ext in (".png", ".webp", ".jpg", ".jpeg"):
        p = src_view_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _resolve_per_frame_source_image(per_frame_root: Path, frame_idx: int, view_id: str) -> Path | None:
    frame_dir = per_frame_root / f"f{int(frame_idx):06d}"
    candidates = [
        frame_dir / "undistort" / "images",
        frame_dir / "hloc_sfm" / "hloc_sfm" / "images",
    ]
    canon = _canon_view_id(view_id)
    aliases = []
    if str(canon).isdigit():
        value = int(canon)
        aliases.extend([f"view{value:03d}", f"view{value:02d}", f"view{value}"])
    else:
        aliases.append(str(canon))
    exts = ['.png', '.jpg', '.jpeg', '.webp']
    for image_root in candidates:
        if not image_root.is_dir():
            continue
        for stem in aliases:
            for ext in exts:
                p = image_root / f"{stem}{ext}"
                if p.is_file():
                    return p
    return None


def _load_grayscale_resized(image_path: Path, downsample_wh: Tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as im:
        im = im.convert("L")
        if int(downsample_wh[0]) > 0 and int(downsample_wh[1]) > 0 and tuple(im.size) != tuple(downsample_wh):
            im = im.resize(tuple(int(x) for x in downsample_wh), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return arr


def _score_temporal_variation_by_view(
    *,
    render_scene_root: Path,
    view_ids: Sequence[str],
    frames: Sequence[int],
    source_frame_index_offset: int,
    downsample_wh: Tuple[int, int] = (128, 128),
) -> Dict[str, float]:
    motion_scores: Dict[str, float] = {}
    for raw_view in view_ids:
        view_id = _canon_view_id(raw_view)
        src_view_dir, source_view_name = _resolve_source_view_dir(render_scene_root, view_id)
        if src_view_dir is None:
            raise FileNotFoundError(
                f"Missing source view dir for {view_id}; tried aliases {_view_id_aliases(view_id)} under {render_scene_root / 'images'}"
            )
        prev_gray: np.ndarray | None = None
        diffs: List[float] = []
        for frame in frames:
            src_frame = int(frame) + int(source_frame_index_offset)
            if src_frame < 0:
                raise FileNotFoundError(f"Negative source frame index for view {view_id}: {src_frame}")
            src_img = _resolve_source_image(src_view_dir, src_frame)
            if src_img is None:
                raise FileNotFoundError(
                    f"Missing source image for motion score: {src_view_dir}/{src_frame:06d}.(png|webp|jpg|jpeg)"
                )
            gray = _load_grayscale_resized(src_img, downsample_wh=downsample_wh)
            if prev_gray is not None:
                diffs.append(float(np.mean(np.abs(gray - prev_gray))))
            prev_gray = gray
        motion_scores[view_id] = float(np.mean(diffs)) if diffs else 0.0
    return motion_scores


def _select_test_views_by_temporal_variation(
    *,
    view_ids: Sequence[str],
    motion_scores: Dict[str, float],
    count: int,
) -> List[str]:
    uniq_views = []
    seen = set()
    for raw_view in view_ids:
        view_id = _canon_view_id(raw_view)
        if view_id not in seen:
            seen.add(view_id)
            uniq_views.append(view_id)
    ranked = sorted(
        uniq_views,
        key=lambda view_id: (-float(motion_scores.get(view_id, 0.0)), _natural_key(view_id)),
    )
    if int(count) <= 0:
        return []
    return ranked[: int(count)]


def _resolve_per_frame_root(calib_run_root: Path) -> Path:
    preferred = ["baseline_per_frame", "glomap_per_frame"]
    for name in preferred:
        p = calib_run_root / name
        if p.is_dir():
            return p

    direct_frame_dirs = sorted([p for p in calib_run_root.glob("f*") if p.is_dir()], key=lambda x: str(x))
    if direct_frame_dirs:
        if any(
            (p / camera_subdir).is_dir()
            for p in direct_frame_dirs
            for camera_subdir in ("cameras_pred", "cameras_gt")
        ):
            return calib_run_root
        if any((p / "intri.yml").is_file() and (p / "extri.yml").is_file() for p in direct_frame_dirs):
            return calib_run_root

    # Fallback: pick the *_per_frame directory with the most cameras_pred frame folders.
    cands = [p for p in calib_run_root.glob("*_per_frame") if p.is_dir()]
    if cands:
        scored = []
        for p in cands:
            n = len(list(p.glob("f*/cameras_pred")))
            scored.append((n, p))
        scored.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
        if scored[0][0] > 0:
            return scored[0][1]

    raise FileNotFoundError(
        "Could not locate per-frame calibrated cameras directory under "
        f"{calib_run_root} (expected baseline_per_frame, glomap_per_frame, or direct fXXXXXX camera dirs)"
    )


def _resolve_frame_cameras_dir(per_frame_root: Path, frame_idx: int) -> Path:
    direct = per_frame_root / f"f{int(frame_idx):06d}"
    for camera_subdir in ("cameras_pred", "cameras_gt"):
        nested = direct / camera_subdir
        if nested.is_dir():
            return nested
    if (direct / "intri.yml").is_file() and (direct / "extri.yml").is_file():
        return direct
    raise FileNotFoundError(f"Missing frame cameras dir under {per_frame_root} for frame {frame_idx:06d}")


def _read_ply_xyz_rgb(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAS_PLYFILE:
        raise ModuleNotFoundError('plyfile is required to read an external point_cloud_ply')
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    if {"red", "green", "blue"}.issubset(v.data.dtype.names):
        rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = np.ones((xyz.shape[0], 3), dtype=np.float32)
    return xyz, rgb


def _write_point_ply(path: Path, xyz: np.ndarray, rgb01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_f = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb_f = np.clip(np.asarray(rgb01, dtype=np.float32).reshape(-1, 3), 0.0, 1.0)
    with path.open('w', encoding='utf-8') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {int(xyz_f.shape[0])}\n')
        for name in ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']:
            f.write(f'property float {name}\n')
        f.write('end_header\n')
        for p, c in zip(xyz_f, rgb_f):
            f.write(
                f'{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} 0.0 0.0 0.0 '
                f'{float(c[0]):.6f} {float(c[1]):.6f} {float(c[2]):.6f}\n'
            )


def _camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
    R = pose[:3, :3]
    t = pose[:3, 3]
    return (-R.T @ t).reshape(3)


def _camera_axes_from_w2c(w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
    R = pose[:3, :3]
    right = R.T[:, 0]
    down = R.T[:, 1]
    forward = R.T[:, 2]
    return right.reshape(3), down.reshape(3), forward.reshape(3)


def _generate_synthetic_point_cloud(
    *,
    cams_by_frame_view: Dict[Tuple[int, str], Dict[str, np.ndarray]],
    train_views_by_frame: Dict[int, List[str]],
    num_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points_i = int(num_points)
    if num_points_i <= 0:
        raise ValueError('num_points must be > 0')
    keys: List[Tuple[int, str]] = []
    for frame, views in train_views_by_frame.items():
        for view in views:
            key = (int(frame), _canon_view_id(view))
            if key in cams_by_frame_view:
                keys.append(key)
    if not keys:
        keys = sorted(cams_by_frame_view.keys())
    if not keys:
        raise ValueError('No cameras available for synthetic point cloud generation')

    rng = np.random.default_rng(int(seed))
    xyz = np.zeros((num_points_i, 3), dtype=np.float32)
    rgb = np.zeros((num_points_i, 3), dtype=np.float32)
    for i in range(num_points_i):
        key = keys[i % len(keys)]
        item = cams_by_frame_view[key]
        center = _camera_center_from_w2c(item['w2c'])
        right, down, forward = _camera_axes_from_w2c(item['w2c'])
        depth = float(rng.uniform(1.5, 4.0))
        jitter_x = float(rng.uniform(-0.6, 0.6))
        jitter_y = float(rng.uniform(-0.6, 0.6))
        point = center + depth * forward + jitter_x * right + jitter_y * down
        xyz[i] = point.astype(np.float32)
        rgb[i] = np.clip(rng.uniform(0.1, 0.9, size=3), 0.0, 1.0).astype(np.float32)
    return xyz, rgb


def _build_split_meta(
    *,
    frames: Sequence[int],
    views_by_frame: Dict[int, List[str]],
    cams_by_frame_view: Dict[Tuple[int, str], Dict[str, np.ndarray]],
    image_names_by_frame_view: Dict[Tuple[int, str], str],
    out_h: int,
    out_w: int,
) -> Dict[str, object]:
    k_all: List[List[List[float]]] = []
    w2c_all: List[List[List[float]]] = []
    fn_all: List[List[str]] = []
    cam_id_all: List[List[str]] = []

    for f in frames:
        cur_k: List[List[float]] = []
        cur_w2c: List[List[float]] = []
        cur_fn: List[str] = []
        cur_ids: List[str] = []
        for v in views_by_frame.get(int(f), []):
            key = (int(f), _canon_view_id(v))
            if key not in cams_by_frame_view:
                continue
            item = cams_by_frame_view[key]
            cur_k.append(np.asarray(item["K"], dtype=np.float64).tolist())
            cur_w2c.append(np.asarray(item["w2c"], dtype=np.float64).tolist())
            cur_fn.append(str(image_names_by_frame_view[key]))
            cur_ids.append(str(key[1]))
        k_all.append(cur_k)
        w2c_all.append(cur_w2c)
        fn_all.append(cur_fn)
        cam_id_all.append(cur_ids)

    return {
        "w": int(out_w),
        "h": int(out_h),
        "k": k_all,
        "w2c": w2c_all,
        "fn": fn_all,
        "cam_id": cam_id_all,
    }


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--render_scene_root", type=str, required=True, help="Rendered dataset scene root with images/ and cameras/")
    parser.add_argument("--calib_run_root", type=str, required=True, help="SPSG baseline run root (<...>/baseline_runs/<scene>_..._HLOC-SPSG-...)")
    parser.add_argument(
        "--test_calib_run_root",
        type=str,
        default="",
        help=(
            "Optional separate per-frame calibration root for held-out test views. "
            "Use a shared GT root for pose-controlled NVS; train cameras still come from --calib_run_root."
        ),
    )
    parser.add_argument(
        "--test_calib_static_gauge_views",
        nargs="*",
        default=[],
        help=(
            "Static camera ids used to Sim(3)-align a separate test calibration root into the train gauge. "
            "Required for GT-pose NVS when the GT and method cameras use different world gauges."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output PanopticSports-format dataset dir")
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=120)
    parser.add_argument(
        "--source_frame_index_offset",
        type=int,
        default=-1,
        help="Map logical frame f to source image index (f + offset).",
    )
    parser.add_argument("--views", nargs="*", default=[], help="Optional all views (default: discover from render scene cameras)")
    parser.add_argument("--test_views", nargs="*", default=["06", "12", "18", "24", "30", "84", "88"])
    parser.add_argument(
        "--auto_test_view_strategy",
        type=str,
        default="none",
        choices=["none", "temporal_variation"],
        help="Optional automatic test-view selection strategy.",
    )
    parser.add_argument(
        "--auto_test_view_count",
        type=int,
        default=0,
        help="How many test views to auto-select. <=0 means keep the same count as --test_views.",
    )
    parser.add_argument(
        "--auto_test_view_candidates",
        nargs="*",
        default=[],
        help="Optional candidate pool for auto test-view selection. Empty means all views.",
    )
    parser.add_argument(
        "--auto_test_view_downsample_wh",
        nargs=2,
        type=int,
        default=[128, 128],
        help="Downsample size used when scoring temporal variation.",
    )
    parser.add_argument("--test_stride", type=int, default=5, help="Evaluate test views every N frames")
    parser.add_argument("--test_first_frame", type=int, default=1, help="First frame included in test split")
    parser.add_argument(
        "--test_frame_start",
        type=int,
        default=None,
        help="Optional inclusive start of an explicit test window inside the source frame range.",
    )
    parser.add_argument(
        "--test_frame_end",
        type=int,
        default=None,
        help="Optional inclusive end of an explicit test window inside the source frame range.",
    )
    parser.add_argument(
        "--drop_test_frames_from_train_except_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, remove test frames from train split except test_first_frame",
    )
    parser.add_argument(
        "--drop_all_test_frames_from_train",
        action="store_true",
        help="If true, remove every test frame from train split. Useful for temporal-offset held-out NVS.",
    )
    parser.add_argument(
        "--resize_wh",
        nargs=2,
        type=int,
        default=[512, 512],
        help="Output image width/height. Set <=0 to keep source size and symlink.",
    )
    parser.add_argument("--overwrite_images", action="store_true", help="Overwrite existing ims/*.png")
    parser.add_argument(
        "--allow_missing_calib_cameras",
        action="store_true",
        help="Skip frame/view samples whose calibrated camera is missing instead of failing the whole build.",
    )
    parser.add_argument("--source_images_from_per_frame", action="store_true", help="Read per-frame images from calib root (prefer undistort/images, fallback hloc_sfm images) instead of render_scene_root/images/<view>/<frame>.")
    parser.add_argument("--center_crop_to_camera_wh", action="store_true", help="Center-crop source images to each camera's calibrated W/H before resize. Useful when using raw images with opencv-normalized/cropped intrinsics.")
    parser.add_argument("--point_cloud_ply", type=str, default="", help="Source sparse ply for init point cloud")
    parser.add_argument("--point_cloud_mode", type=str, default="auto", choices=["auto", "ply", "synthetic"], help="How to initialize the 4DGS point cloud.")
    parser.add_argument("--num_init_points", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    render_scene_root = Path(args.render_scene_root).resolve()
    calib_run_root = Path(args.calib_run_root).resolve()
    test_calib_run_root = Path(args.test_calib_run_root).resolve() if str(args.test_calib_run_root).strip() else None
    out_dir = Path(args.output_dir).resolve()
    ims_dir = out_dir / "ims"
    ims_dir.mkdir(parents=True, exist_ok=True)

    frame_start = int(args.frame_start)
    frame_end = int(args.frame_end)
    if frame_start > frame_end:
        raise ValueError("frame_start must be <= frame_end")
    frames = list(range(frame_start, frame_end + 1))
    has_explicit_test_start = args.test_frame_start is not None
    has_explicit_test_end = args.test_frame_end is not None
    if has_explicit_test_start != has_explicit_test_end:
        raise ValueError("--test_frame_start and --test_frame_end must be provided together")
    explicit_test_range = None
    if has_explicit_test_start:
        test_frame_start = int(args.test_frame_start)
        test_frame_end = int(args.test_frame_end)
        if not (frame_start <= test_frame_start <= test_frame_end <= frame_end):
            raise ValueError("test frame range must be inside source frame range")
        explicit_test_range = (test_frame_start, test_frame_end)

    if args.views:
        all_views = [_canon_view_id(v) for v in args.views]
    else:
        all_views = _discover_views_from_extri(render_scene_root / "cameras")
    if not all_views:
        raise RuntimeError(f"No views found in {render_scene_root / 'cameras'}")

    requested_test_views = [_canon_view_id(v) for v in args.test_views]
    auto_test_view_strategy = str(args.auto_test_view_strategy)
    auto_motion_scores: Dict[str, float] = {}
    auto_test_view_candidates: List[str] = []
    if auto_test_view_strategy == "temporal_variation":
        auto_test_view_candidates = [_canon_view_id(v) for v in args.auto_test_view_candidates] if args.auto_test_view_candidates else list(all_views)
        auto_test_view_candidates = [v for v in auto_test_view_candidates if v in set(all_views)]
        if not auto_test_view_candidates:
            raise RuntimeError("auto_test_view_candidates is empty after intersecting with available views")
        auto_count = int(args.auto_test_view_count) if int(args.auto_test_view_count) > 0 else int(len(requested_test_views))
        auto_motion_scores = _score_temporal_variation_by_view(
            render_scene_root=render_scene_root,
            view_ids=auto_test_view_candidates,
            frames=frames,
            source_frame_index_offset=int(args.source_frame_index_offset),
            downsample_wh=(int(args.auto_test_view_downsample_wh[0]), int(args.auto_test_view_downsample_wh[1])),
        )
        test_views = _select_test_views_by_temporal_variation(
            view_ids=auto_test_view_candidates,
            motion_scores=auto_motion_scores,
            count=auto_count,
        )
    else:
        test_views = list(requested_test_views)
    train_views = [v for v in all_views if v not in set(test_views)]
    if not train_views:
        raise RuntimeError("Empty train_views after excluding test_views")

    # Build the held-out temporal window while retaining all source frames as
    # train-only context for non-test views.
    if int(args.test_stride) <= 0:
        raise ValueError("test_stride must be > 0")
    if explicit_test_range is None:
        test_first = int(args.test_first_frame)
        test_last = frame_end
    else:
        test_first, test_last = explicit_test_range
    test_frames = [
        f
        for f in frames
        if test_first <= f <= test_last
        and ((f - test_first) % int(args.test_stride) == 0)
    ]
    if bool(args.drop_all_test_frames_from_train):
        train_drop_frames = list(test_frames)
    elif bool(args.drop_test_frames_from_train_except_first):
        train_drop_frames = [f for f in test_frames if f != test_first]
    else:
        train_drop_frames = []

    out_w = int(args.resize_wh[0])
    out_h = int(args.resize_wh[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError("resize_wh must be positive")

    # Cache calibrated cameras and materialize images.
    per_frame_root = _resolve_per_frame_root(calib_run_root)
    test_per_frame_root = _resolve_per_frame_root(test_calib_run_root) if test_calib_run_root is not None else None
    test_view_set = set(test_views)
    test_static_gauge_views = [_canon_view_id(view) for view in args.test_calib_static_gauge_views]
    if test_static_gauge_views and test_per_frame_root is None:
        raise ValueError("--test_calib_static_gauge_views requires --test_calib_run_root")
    test_gauge_alignment_by_frame: Dict[str, Dict[str, object]] = {}
    cams_by_frame_view: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
    image_names_by_frame_view: Dict[Tuple[int, str], str] = {}
    missing_images: List[str] = []
    missing_cameras: List[str] = []
    resize_count = 0
    symlink_count = 0
    copy_count = 0

    req_views = sorted(set(train_views + test_views), key=_natural_key)
    for frame in frames:
        frame_dir = _resolve_frame_cameras_dir(per_frame_root, frame)
        loader = CameraLoader(frame_dir)
        test_loader = None
        if test_per_frame_root is not None:
            test_frame_dir = _resolve_frame_cameras_dir(test_per_frame_root, frame)
            test_loader = CameraLoader(test_frame_dir)
        aligned_test_cameras = {}
        if test_loader is not None and test_static_gauge_views:
            target_static = {
                view: camera
                for view in test_static_gauge_views
                if (camera := _load_camera_by_alias(loader, view)) is not None
            }
            candidate_views = sorted(set(test_static_gauge_views + test_views), key=_natural_key)
            candidate_gt = {
                view: camera
                for view in candidate_views
                if (camera := _load_camera_by_alias(test_loader, view)) is not None
            }
            missing_target = sorted(set(test_static_gauge_views) - set(target_static), key=_natural_key)
            missing_candidate = sorted(set(test_static_gauge_views) - set(candidate_gt), key=_natural_key)
            if missing_target or missing_candidate:
                raise RuntimeError(
                    "Could not align separate test cameras to train gauge: "
                    f"frame={frame}, missing_train_static={missing_target}, missing_test_static={missing_candidate}"
                )
            aligned_test_cameras, alignment = align_candidate_cameras_to_target_static_gauge(
                candidate_cams=candidate_gt,
                target_cams=target_static,
                static_views=test_static_gauge_views,
                views=test_views,
            )
            test_gauge_alignment_by_frame[str(int(frame))] = alignment

        for v in req_views:
            v_key = _canon_view_id(v)
            if v_key in aligned_test_cameras:
                cam = aligned_test_cameras[v_key]
            else:
                camera_loader = test_loader if test_loader is not None and v_key in test_view_set else loader
                cam = _load_camera_by_alias(camera_loader, v_key)
            if cam is None:
                missing_cameras.append(f"frame={frame} view={v_key}")
                continue
            src_view_dir, source_view_name = _resolve_source_view_dir(render_scene_root, v_key)
            src_frame = int(frame) + int(args.source_frame_index_offset)
            if src_frame < 0:
                missing_images.append(f"{render_scene_root / 'images' / source_view_name}/<{src_frame:06d}.png>")
                continue
            if bool(args.source_images_from_per_frame):
                image_per_frame_root = _per_frame_image_root_for_view(
                    view=v_key,
                    test_views=test_view_set,
                    train_root=per_frame_root,
                    test_root=test_per_frame_root,
                )
                src_img = _resolve_per_frame_source_image(
                    image_per_frame_root,
                    frame,
                    v_key,
                )
                if src_img is None:
                    missing_images.append(
                        f"{image_per_frame_root}/f{int(frame):06d}/"
                        f"(undistort|hloc_sfm)/... view={v_key}"
                    )
                    continue
            else:
                if src_view_dir is None:
                    missing_images.append(
                        f"{render_scene_root / 'images'}/{{{','.join(_view_id_aliases(v_key))}}}/{src_frame:06d}.(png|webp|jpg|jpeg)"
                    )
                    continue
                src_img = _resolve_source_image(src_view_dir, src_frame)
                if src_img is None:
                    missing_images.append(f"{src_view_dir}/{src_frame:06d}.(png|webp|jpg|jpeg)")
                    continue

            out_name = f"v{source_view_name}_f{frame:06d}.png"
            dst_img = ims_dir / out_name
            if args.overwrite_images and (dst_img.exists() or dst_img.is_symlink()):
                dst_img.unlink()
            if (not dst_img.exists()) or args.overwrite_images:
                if out_w <= 0 or out_h <= 0:
                    if _symlink_or_copy(src_img, dst_img) == "symlink":
                        symlink_count += 1
                    else:
                        copy_count += 1
                else:
                    with Image.open(src_img) as im:
                        if bool(args.center_crop_to_camera_wh):
                            crop_w = int(round(float(cam.W)))
                            crop_h = int(round(float(cam.H)))
                            if 0 < crop_w <= im.width and 0 < crop_h <= im.height and (crop_w, crop_h) != tuple(im.size):
                                left = int((im.width - crop_w) // 2)
                                top = int((im.height - crop_h) // 2)
                                im = im.crop((left, top, left + crop_w, top + crop_h))
                        if tuple(im.size) == (out_w, out_h):
                            if _symlink_or_copy(src_img, dst_img) == "symlink":
                                symlink_count += 1
                            else:
                                copy_count += 1
                        else:
                            im = im.convert("RGB").resize((out_w, out_h), resample=Image.BILINEAR)
                            im.save(dst_img, format="PNG")
                            resize_count += 1

            sx = float(out_w) / float(cam.W)
            sy = float(out_h) / float(cam.H)
            K = np.asarray(cam.K, dtype=np.float64).copy()
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy

            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = np.asarray(cam.R, dtype=np.float64).reshape(3, 3)
            w2c[:3, 3] = np.asarray(cam.T, dtype=np.float64).reshape(3)

            key = (int(frame), v_key)
            cams_by_frame_view[key] = {"K": K, "w2c": w2c}
            image_names_by_frame_view[key] = out_name

    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} source images, e.g. {missing_images[:3]}")
    if missing_cameras and not bool(args.allow_missing_calib_cameras):
        raise RuntimeError(f"Missing {len(missing_cameras)} calibrated cameras, e.g. {missing_cameras[:3]}")
    if missing_cameras:
        print(
            "[WARN] Missing calibrated cameras will be skipped: "
            f"{len(missing_cameras)} examples={missing_cameras[:8]}"
        )

    # Frame-wise split assignment.
    def available_views(frame: int, views: List[str]) -> List[str]:
        out: List[str] = []
        for view in views:
            key = (int(frame), _canon_view_id(view))
            if key in cams_by_frame_view:
                out.append(_canon_view_id(view))
        return out

    train_views_by_frame: Dict[int, List[str]] = {}
    test_views_by_frame: Dict[int, List[str]] = {}
    for f in frames:
        if f in set(train_drop_frames):
            train_views_by_frame[f] = []
        else:
            train_views_by_frame[f] = available_views(f, train_views)
        if f in set(test_frames):
            test_views_by_frame[f] = available_views(f, test_views)
        else:
            test_views_by_frame[f] = []

    train_meta = _build_split_meta(
        frames=frames,
        views_by_frame=train_views_by_frame,
        cams_by_frame_view=cams_by_frame_view,
        image_names_by_frame_view=image_names_by_frame_view,
        out_h=out_h,
        out_w=out_w,
    )
    test_meta = _build_split_meta(
        frames=frames,
        views_by_frame=test_views_by_frame,
        cams_by_frame_view=cams_by_frame_view,
        image_names_by_frame_view=image_names_by_frame_view,
        out_h=out_h,
        out_w=out_w,
    )

    (out_dir / "train_meta.json").write_text(json.dumps(train_meta), encoding="utf-8")
    (out_dir / "test_meta.json").write_text(json.dumps(test_meta), encoding="utf-8")

    # Build init point cloud.
    point_cloud_mode = str(args.point_cloud_mode)
    point_cloud_ply = None
    if str(args.point_cloud_ply).strip():
        point_cloud_ply = Path(args.point_cloud_ply).resolve()
    else:
        scene_name = render_scene_root.name
        fallback_ply = (Path(__file__).resolve().parents[1] / "data" / "neu3d" / scene_name / "points3D_downsample2.ply").resolve()
        if fallback_ply.is_file():
            point_cloud_ply = fallback_ply

    n = int(args.num_init_points)
    if n <= 0:
        raise ValueError("num_init_points must be > 0")

    if point_cloud_mode == 'ply':
        if point_cloud_ply is None or not point_cloud_ply.is_file():
            raise FileNotFoundError('point_cloud_mode=ply but point_cloud_ply was not found')
        xyz, rgb = _read_ply_xyz_rgb(point_cloud_ply)
        rng = np.random.default_rng(int(args.seed))
        if xyz.shape[0] >= n:
            idx = rng.choice(xyz.shape[0], size=n, replace=False)
        else:
            idx = rng.choice(xyz.shape[0], size=n, replace=True)
        xyz_s = xyz[idx].astype(np.float32)
        rgb_s = rgb[idx].astype(np.float32)
    elif point_cloud_mode == 'synthetic' or point_cloud_ply is None or not point_cloud_ply.is_file():
        point_cloud_mode = 'synthetic'
        xyz_s, rgb_s = _generate_synthetic_point_cloud(
            cams_by_frame_view=cams_by_frame_view,
            train_views_by_frame=train_views_by_frame,
            num_points=n,
            seed=int(args.seed),
        )
    else:
        xyz, rgb = _read_ply_xyz_rgb(point_cloud_ply)
        rng = np.random.default_rng(int(args.seed))
        if xyz.shape[0] >= n:
            idx = rng.choice(xyz.shape[0], size=n, replace=False)
        else:
            idx = rng.choice(xyz.shape[0], size=n, replace=True)
        xyz_s = xyz[idx].astype(np.float32)
        rgb_s = rgb[idx].astype(np.float32)
        point_cloud_mode = 'ply'

    data = np.concatenate([xyz_s, rgb_s], axis=1).astype(np.float32)
    np.savez_compressed(out_dir / "init_pt_cld.npz", data=data)
    _write_point_ply(out_dir / "pointd3D.ply", xyz_s, rgb_s)

    train_samples = int(sum(len(x) for x in train_meta["cam_id"]))
    test_samples = int(sum(len(x) for x in test_meta["cam_id"]))
    summary = {
        "render_scene_root": str(render_scene_root),
        "calib_run_root": str(calib_run_root),
        "dataset_root": str(out_dir),
        "num_views_all": int(len(all_views)),
        "num_train_views": int(len(train_views)),
        "num_test_views": int(len(test_views)),
        "test_view_selection_strategy": str(auto_test_view_strategy),
        "requested_test_views": requested_test_views,
        "train_views": train_views,
        "test_views": test_views,
        "auto_test_view_candidates": auto_test_view_candidates,
        "auto_test_view_scores": {k: float(auto_motion_scores[k]) for k in sorted(auto_motion_scores.keys(), key=_natural_key)},
        "num_frames_total": int(len(frames)),
        "frame_range": [int(frame_start), int(frame_end)],
        "source_frames": [int(x) for x in frames],
        "test_frame_range": (
            [int(explicit_test_range[0]), int(explicit_test_range[1])]
            if explicit_test_range is not None
            else None
        ),
        "test_frames": [int(x) for x in test_frames],
        "train_drop_frames": [int(x) for x in train_drop_frames],
        "drop_all_test_frames_from_train": bool(args.drop_all_test_frames_from_train),
        "train_samples": int(train_samples),
        "test_samples": int(test_samples),
        "allow_missing_calib_cameras": bool(args.allow_missing_calib_cameras),
        "missing_calib_cameras": int(len(missing_cameras)),
        "missing_calib_camera_examples": missing_cameras[:20],
        "output_wh": [int(out_w), int(out_h)],
        "resized_images_written": int(resize_count),
        "symlink_images_created": int(symlink_count),
        "copied_images_written": int(copy_count),
        "calib_per_frame_dir": str(per_frame_root),
        "test_calib_per_frame_dir": str(test_per_frame_root) if test_per_frame_root is not None else str(per_frame_root),
        "test_camera_source": (
            "separate_test_calib_static_gauge_aligned"
            if test_per_frame_root is not None and test_static_gauge_views
            else "separate_test_calib"
            if test_per_frame_root is not None
            else "shared_train_calib"
        ),
        "test_calib_static_gauge_views": test_static_gauge_views,
        "test_gauge_alignment_by_frame": test_gauge_alignment_by_frame,
        "source_image_mode": "per_frame" if bool(args.source_images_from_per_frame) else "render_scene",
        "center_crop_to_camera_wh": bool(args.center_crop_to_camera_wh),
        "point_cloud_mode": str(point_cloud_mode),
        "point_cloud_ply": str(point_cloud_ply) if point_cloud_ply is not None else "",
        "num_init_points": int(data.shape[0]),
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
