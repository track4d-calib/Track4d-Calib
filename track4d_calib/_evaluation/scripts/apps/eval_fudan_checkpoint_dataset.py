from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track4d_calib.fudan_eval_utils import (
    build_eval_report,
    parse_image_name_metadata,
    validate_min_frames_by_view,
)
from track4d_calib.fudan_eval_degree_override import apply_active_degree_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Fudan 4DGS checkpoint on its test split.")
    parser.add_argument("--fudan_root", type=str, default=str(PROJECT_ROOT / "third_party" / "fudan_4dgs"))
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max_frames_per_view", type=int, default=120)
    parser.add_argument("--limit_test_frames", type=int, default=0)
    parser.add_argument(
        "--time_duration",
        nargs=2,
        type=float,
        default=None,
        help="Override the 4DGS evaluation time window, e.g. 0 1 for normalized 30-frame DynDNA subsets.",
    )
    parser.add_argument("--skip_side_images", action="store_true", help="Do not write per-frame side-by-side PNGs.")
    parser.add_argument("--skip_videos", action="store_true", help="Do not encode eval MP4 videos.")
    parser.add_argument(
        "--override_active_sh_degree",
        type=int,
        default=-1,
        help="Diagnostic render-only spatial SH degree; -1 preserves the checkpoint value.",
    )
    parser.add_argument(
        "--override_active_sh_degree_t",
        type=int,
        default=-1,
        help="Diagnostic render-only temporal SH degree; -1 preserves the checkpoint value.",
    )
    return parser.parse_args()


CLI = parse_args()
FUDAN_ROOT = Path(CLI.fudan_root).resolve()
if str(FUDAN_ROOT) not in sys.path:
    sys.path.insert(0, str(FUDAN_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import ssim


def make_4dgs_args(config_path: Path, source_path: Path):
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    args = parser.parse_args(["--config", str(config_path)])
    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            setattr(args, key, host[key])

    for key in cfg.keys():
        recursive_merge(key, cfg)
    args.source_path = str(source_path)
    args.model_path = str(Path(args.model_path))
    if CLI.time_duration is not None:
        args.time_duration = list(CLI.time_duration)
    args.dataloader = True
    args.eval = True
    return args, lp, op, pp


def open_video_writer(path: Path, first_frame: np.ndarray, fps: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def read_bgr(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image for video encoding: {path}")
    return frame


def write_video_from_image_paths(path: Path, image_paths: Sequence[Path], fps: int) -> None:
    if not image_paths:
        return
    first = read_bgr(image_paths[0])
    writer = open_video_writer(path, first, fps)
    writer.write(first)
    for image_path in image_paths[1:]:
        frame = read_bgr(image_path)
        writer.write(frame)
    writer.release()


def pad_bgr_to_shape(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pad a BGR frame so quad videos can mix crop views with odd sizes."""
    h, w = frame.shape[:2]
    if h == height and w == width:
        return frame
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    canvas[:h, :w] = frame
    return canvas


def write_quad_video_from_side_paths(path: Path, per_view_paths: Dict[str, Sequence[Path]], fps: int) -> None:
    if not per_view_paths:
        return
    n = min(len(paths) for paths in per_view_paths.values())
    if n == 0:
        return
    writer = None
    try:
        for frame_idx in range(n):
            imgs = [read_bgr(paths[frame_idx]) for paths in per_view_paths.values()]
            max_h = max(img.shape[0] for img in imgs)
            max_w = max(img.shape[1] for img in imgs)
            imgs = [pad_bgr_to_shape(img, max_h, max_w) for img in imgs]
            if len(imgs) == 2:
                canvas = np.concatenate(imgs, axis=1)
            elif len(imgs) == 3:
                top = np.concatenate(imgs[:2], axis=1)
                bottom = np.concatenate([imgs[2], np.zeros_like(imgs[2])], axis=1)
                canvas = np.concatenate([top, bottom], axis=0)
            else:
                top = np.concatenate(imgs[:2], axis=1)
                bottom = np.concatenate(imgs[2:4], axis=1)
                canvas = np.concatenate([top, bottom], axis=0)
            if writer is None:
                writer = open_video_writer(path, canvas, fps)
            writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()


def make_frame_stem(meta: Dict[str, object], fallback_idx: int) -> str:
    view = meta.get("view")
    frame = meta.get("frame")
    if view is None or frame is None:
        return f"frame_{fallback_idx:06d}"
    return f"view_{view}_frame_{int(frame):06d}"


def main() -> None:
    out_dir = Path(CLI.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args, lp, op, pp = make_4dgs_args(Path(CLI.config), Path(CLI.source_path))
    config_model_path = str(args.model_path)
    eval_scene_model_path = out_dir / "_scene_model_cache"
    eval_scene_model_path.mkdir(parents=True, exist_ok=True)
    args.model_path = str(eval_scene_model_path)

    gaussians = GaussianModel(
        args.sh_degree,
        gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration,
        rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d,
        sh_degree_t=2 if args.eval_shfs_4d else 0,
        prefilter_var=args.prefilter_var,
    )
    scene = Scene(lp.extract(args), gaussians, num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)
    gaussians.training_setup(op.extract(args))
    model_params, _ = torch.load(str(CLI.checkpoint), weights_only=False)
    gaussians.restore(model_params, op.extract(args))
    active_degree_readout = apply_active_degree_overrides(
        gaussians,
        spatial=CLI.override_active_sh_degree,
        temporal=CLI.override_active_sh_degree_t,
    )
    pipe = pp.extract(args)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    lpips_alex = LPIPS(net_type="alex", version="0.1").cuda().eval()

    render_dir = out_dir / "renders"
    gt_dir = out_dir / "gt"
    side_dir = out_dir / "side_by_side"
    render_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    side_dir.mkdir(exist_ok=True)

    rows = []
    image_paths_by_view = defaultdict(list)
    test_dataset = scene.getTestCameras()
    total = len(test_dataset)
    if CLI.limit_test_frames > 0:
        total = min(total, CLI.limit_test_frames)

    with torch.no_grad():
        for idx in tqdm(range(total), desc="render/eval test"):
            batch_item = test_dataset[idx]
            if len(batch_item) == 3:
                gt_image, viewpoint, _gt_alpha_mask = batch_item
            else:
                gt_image, viewpoint = batch_item
            gt_image = gt_image.cuda()
            viewpoint = viewpoint.cuda()
            image = torch.clamp(render(viewpoint, gaussians, pipe, background)["render"], 0.0, 1.0)
            meta = parse_image_name_metadata(str(viewpoint.image_name))
            row = {
                **meta,
                "idx": idx,
                "psnr": float(psnr(image, gt_image).mean().detach().cpu()),
                "ssim": float(ssim(image, gt_image).mean().detach().cpu()),
                "lpips_alex": float(
                    lpips_alex(
                        image.unsqueeze(0),
                        gt_image.unsqueeze(0),
                    )
                    .mean()
                    .detach()
                    .cpu()
                ),
            }
            rows.append(row)
            stem = make_frame_stem(meta, idx)
            pred_path = render_dir / f"{stem}_pred.png"
            gt_path = gt_dir / f"{stem}_gt.png"
            side_path = side_dir / f"{stem}_side.png"
            save_image(image.cpu(), pred_path)
            save_image(gt_image.cpu(), gt_path)
            if CLI.skip_side_images:
                side_path_for_entry = None
            else:
                side = torch.cat([gt_image.cpu(), image.cpu()], dim=2)
                save_image(side, side_path)
                side_path_for_entry = side_path
            if meta.get("view") is not None and meta.get("frame") is not None:
                view_key = str(meta["view"])
                frame_id = int(meta["frame"])
                image_paths_by_view[view_key].append(
                    {
                        "frame": frame_id,
                        "pred": pred_path,
                        "gt": gt_path,
                        "side": side_path_for_entry,
                    }
                )

    report = build_eval_report(rows)
    by_view = report["by_view"]
    overall = report["overall"]
    validate_min_frames_by_view(by_view, expected_frames_per_view=CLI.max_frames_per_view)

    quad_path = None
    if not CLI.skip_videos:
        for view, stats in by_view.items():
            entries = sorted(image_paths_by_view[view], key=lambda x: x["frame"])[: CLI.max_frames_per_view]
            pred_video = out_dir / f"view{view}_pred.mp4"
            gt_video = out_dir / f"view{view}_gt.mp4"
            side_video = out_dir / f"view{view}_gt_pred_side.mp4"
            write_video_from_image_paths(pred_video, [x["pred"] for x in entries], CLI.fps)
            write_video_from_image_paths(gt_video, [x["gt"] for x in entries], CLI.fps)
            side_entries = [x["side"] for x in entries if x["side"] is not None]
            if len(side_entries) == len(entries):
                write_video_from_image_paths(side_video, side_entries, CLI.fps)
                side_video_value = str(side_video)
            else:
                side_video_value = None
            stats.update(
                {
                    "pred_video": str(pred_video),
                    "gt_video": str(gt_video),
                    "side_video": side_video_value,
                }
            )

        views = list(by_view.keys())
        if len(views) >= 2:
            selected = views[: min(4, len(views))]
            per = {}
            for view in selected:
                entries = sorted(image_paths_by_view[view], key=lambda x: x["frame"])[: CLI.max_frames_per_view]
                side_entries = [x["side"] for x in entries if x["side"] is not None]
                if len(side_entries) == len(entries):
                    per[view] = side_entries
            if len(per) >= 2:
                quad_path = out_dir / "quad_gt_pred_side.mp4"
                write_quad_video_from_side_paths(quad_path, per, CLI.fps)

    summary = {
        "config": str(CLI.config),
        "checkpoint": str(CLI.checkpoint),
        "source_path": str(CLI.source_path),
        "out_dir": str(out_dir),
        "config_model_path": config_model_path,
        "scene_model_path": str(eval_scene_model_path),
        "overall": overall,
        "by_view": by_view,
        "rows": report["rows"],
        "num_rendered": len(rows),
        "limit_test_frames": CLI.limit_test_frames,
        "time_duration": [float(value) for value in args.time_duration],
        "active_degree_readout": active_degree_readout,
        "degree_override_is_render_only_diagnostic": bool(
            CLI.override_active_sh_degree >= 0
            or CLI.override_active_sh_degree_t >= 0
        ),
        "quad_side_video": None if quad_path is None else str(quad_path),
        "note": "Robust Fudan checkpoint evaluation wrapper that handles both camXX_YYYY and train/test_fXXXXXX_vVIEW naming.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
