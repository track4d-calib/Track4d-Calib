from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


_TRAIN_TEST_VIEW_RE = re.compile(r"(?P<split>train|test)_f(?P<frame>\d+)_v(?P<view>.+)$")
_DYNSTATIC_BUILDER_RE = re.compile(r"^(?P<split>train|test)_[sd](?P<view>.+?)_f(?P<frame>\d+)$")
_INDEXED_TRAIN_TEST_VIEW_RE = re.compile(r"^(?P<split>train|test)_\d+_\d+_v(?P<view>.+)_f(?P<frame>\d+)$")
_CAM_VIEW_RE = re.compile(r"^cam(?P<view>\d+)_(?P<frame>\d+)$")
_GENERIC_VF_RE = re.compile(r"^v(?P<view>.+?)_f(?P<frame>\d+)(?:_.+)?$")


def _strip_ext(image_name: str) -> str:
    if "." in image_name:
        return image_name.rsplit(".", 1)[0]
    return image_name


def parse_image_name_metadata(image_name: str) -> Dict[str, Any]:
    base = Path(_strip_ext(str(image_name))).name
    match = _TRAIN_TEST_VIEW_RE.search(base)
    if match:
        view = match.group("view")
        return {
            "image_name": base,
            "split": match.group("split"),
            "view": str(int(view)) if view.isdigit() else view,
            "frame": int(match.group("frame")),
        }

    match = _DYNSTATIC_BUILDER_RE.match(base)
    if match:
        view = match.group("view")
        return {
            "image_name": base,
            "split": match.group("split"),
            "view": str(int(view)) if view.isdigit() else view,
            "frame": int(match.group("frame")),
        }

    match = _INDEXED_TRAIN_TEST_VIEW_RE.match(base)
    if match:
        return {
            "image_name": base,
            "split": match.group("split"),
            "view": match.group("view"),
            "frame": int(match.group("frame")),
        }

    match = _CAM_VIEW_RE.match(base)
    if match:
        return {
            "image_name": base,
            "split": None,
            "view": match.group("view"),
            "frame": int(match.group("frame")),
        }

    match = _GENERIC_VF_RE.match(base)
    if match:
        return {
            "image_name": base,
            "split": None,
            "view": match.group("view"),
            "frame": int(match.group("frame")),
        }

    return {
        "image_name": base,
        "split": None,
        "view": None,
        "frame": None,
    }


def _safe_float_list(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        vals.append(float(value))
    return vals


def build_global_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    psnrs = _safe_float_list(rows, "psnr")
    ssims = _safe_float_list(rows, "ssim")
    lpips_values = _safe_float_list(rows, "lpips_alex")

    def _round_or_none(values: List[float], reducer) -> float | None:
        if not values:
            return None
        return round(float(reducer(values)), 10)

    return {
        "frames": len(rows),
        "mean_psnr": _round_or_none(psnrs, np.mean),
        "median_psnr": _round_or_none(psnrs, np.median),
        "std_psnr": _round_or_none(psnrs, np.std),
        "min_psnr": _round_or_none(psnrs, np.min),
        "max_psnr": _round_or_none(psnrs, np.max),
        "mean_ssim": _round_or_none(ssims, np.mean),
        "median_ssim": _round_or_none(ssims, np.median),
        "std_ssim": _round_or_none(ssims, np.std),
        "mean_lpips_alex": _round_or_none(lpips_values, np.mean),
        "median_lpips_alex": _round_or_none(lpips_values, np.median),
        "std_lpips_alex": _round_or_none(lpips_values, np.std),
    }


def group_rows_by_view(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        view = row.get("view")
        if view is None:
            continue
        grouped[str(view)].append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    sort_key = lambda value: int(value) if str(value).isdigit() else str(value)
    for view in sorted(grouped.keys(), key=sort_key):
        view_rows = grouped[view]
        stats = build_global_summary(view_rows)
        summary[view] = {
            **stats,
            "frames": len(view_rows),
        }
    return summary


def build_eval_report(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    metric_rows = [dict(row) for row in rows]
    return {
        "overall": build_global_summary(metric_rows),
        "by_view": group_rows_by_view(metric_rows),
        "rows": metric_rows,
    }


def validate_min_frames_by_view(by_view: Dict[str, Dict[str, Any]], *, expected_frames_per_view: int) -> None:
    if int(expected_frames_per_view) <= 0:
        return
    too_short = {
        str(view): int(stats.get("frames", 0))
        for view, stats in by_view.items()
        if int(stats.get("frames", 0)) < int(expected_frames_per_view)
    }
    if too_short:
        detail = ", ".join(
            f"view {view}: {frames}/{int(expected_frames_per_view)}"
            for view, frames in sorted(too_short.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0])
        )
        raise ValueError(
            "Evaluation rendered fewer frames than requested. "
            f"This usually means time_duration filtered the dataset: {detail}"
        )
