from __future__ import annotations

import hashlib, json, os, statistics, subprocess, time

from pathlib import Path

from typing import Any, Sequence

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

def run(command: Sequence[str], *, cwd: Path, env: dict[str, str], log: Path) -> dict[str, Any]:
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command), cwd=cwd, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    row = {
        "argv": list(command), "cwd": str(cwd),
        "exit_code": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - started),
        "log": str(log), "log_sha256": sha256(log),
    }
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); log={log}")
    return row

def config(source: Path, model: Path) -> dict[str, Any]:
    return {
        "gaussian_dim": 4,
        "time_duration": [0.0, 1.0],
        "num_pts": 300000,
        "num_pts_ratio": 1.0,
        "rot_4d": True,
        "force_sh_3d": False,
        "batch_size": 4,
        "exhaust_test": False,
        "ModelParams": {
            "sh_degree": 3, "source_path": str(source), "model_path": str(model),
            "images": "images", "resolution": 1, "white_background": False,
            "data_device": "cuda", "eval": True, "extension": ".png",
            "num_extra_pts": 0, "loaded_pth": "", "frame_ratio": 1,
            "dataloader": True, "prefilter_var": 0.0,
        },
        "PipelineParams": {
            "convert_SHs_python": False, "compute_cov3D_python": False,
            "debug": False, "env_map_res": 0, "env_optimize_until": 1000000000,
            "env_optimize_from": 0, "eval_shfs_4d": True,
        },
        "OptimizationParams": {
            "iterations": 10000, "position_lr_init": 0.00016,
            "position_t_lr_init": -1.0, "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
            "feature_lr": 0.0025, "opacity_lr": 0.05, "scaling_lr": 0.005,
            "rotation_lr": 0.001, "percent_dense": 0.01, "lambda_dssim": 0.2,
            "thresh_opa_prune": 0.005, "densification_interval": 200,
            "opacity_reset_interval": 3000, "densify_from_iter": 500,
            "densify_until_iter": 10000, "densify_grad_threshold": 0.0002,
            "densify_grad_t_threshold": 0.000005,
            "densify_until_num_points": 1200000, "final_prune_from_iter": -1,
            "sh_increase_interval": 1000, "lambda_opa_mask": 0.0,
            "lambda_rigid": 0.0, "lambda_motion": 0.0,
        },
    }
