"""Scene-independent rig scope for the common track-driven calibration path.

The contract contains camera IDs and a motion bound, never camera poses. Its
reference centers are estimated from this run's fit tracks inside the solver.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


ENVIRONMENT_KEY = "TRACK4D_CALIB_RIG_CONTRACT"
PROTOCOL = "TRACK4D_CALIB-declared-rig-track-closure-v1"


def parse_center_radius(value):
    """JSON null / CLI 'none' means independent centers without a motion cap."""
    if value is None or value == "none":
        return None
    radius = float(value)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("center motion radius must be finite and nonnegative, or none")
    return radius


def make_rig_contract(
    static_views: Sequence[str], dynamic_views: Sequence[str], *,
    center_motion_radius: float | None = 0.0,
    static_support_count: int = 2,
) -> dict[str, Any]:
    statics = list(map(str, static_views))
    dynamic = list(map(str, dynamic_views))
    if len(set(statics)) != len(statics) or len(set(dynamic)) != len(dynamic):
        raise ValueError("rig camera IDs must be unique")
    if set(statics) & set(dynamic):
        raise ValueError("static and dynamic camera rosters overlap")
    if len(statics) < 16 or len(dynamic) < 3:
        raise ValueError("track closure requires at least 16 static and 3 dynamic views")
    if any(not view or "/" in view or view in (".", "..") for view in statics + dynamic):
        raise ValueError("invalid camera ID")
    radius = parse_center_radius(center_motion_radius)
    if type(static_support_count) is not int or not 2 <= static_support_count <= 8:
        raise ValueError("static support count must be an integer from 2 to 8")
    groups = max(1, len(dynamic) // 3)
    base, remainder = divmod(len(dynamic), groups)
    bands = []
    offset = 0
    for index in range(groups):
        size = base + int(index < remainder)
        bands.append(dynamic[offset:offset + size])
        offset += size
    result = {
        "protocol": PROTOCOL,
        "static_views": statics,
        "dynamic_views": dynamic,
        "relay_bands": bands,
        # Shard labels are not geometric anchors. The latter are selected from
        # fresh track confidence by the conditioned relay builder.
        "relay_shard_labels": [band[0] for band in bands],
        "center_motion_radius": radius,
        "center_radius_units": "current_run_static_SfM_coordinate_units",
        "center_reference": ("current_run_fit_only_global_center_per_camera" if radius is not None
                             else "current_run_per_node_fit_pose"),
        "center_scope": ("per_frame_offset_about_one_physical_camera_reference" if radius is not None
                         else "independent_per_frame_center_without_motion_bound"),
        "static_pair_selection": "fresh_fit_track_triangulation",
        "previous_pose_or_track_inputs": False,
    }
    if static_support_count != 2:
        result["static_support_count"] = static_support_count
    return result


def load_rig_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = make_rig_contract(
        payload["static_views"], payload["dynamic_views"],
        center_motion_radius=payload["center_motion_radius"],
        static_support_count=payload.get("static_support_count", 2),
    )
    if payload != expected:
        raise ValueError("rig contract differs from its declared camera/motion policy")
    return expected


def center_solver_kwargs(
    nodes: Sequence[tuple[int, str]], radius: float | None,
) -> dict[str, Any]:
    """Zero fixes centers; positive bounds offsets; None frees translation."""
    limit = parse_center_radius(radius)
    if limit == 0:
        return {}
    groups = {
        (int(frame), str(view)): json.dumps([int(frame), str(view)], separators=(",", ":"))
        for frame, view in nodes
    }
    if len(groups) != len(nodes):
        raise ValueError("center optimizer nodes must be unique")
    return {
        "center_group_by_node": groups,
        "center_trust_radius_by_group": {group: limit for group in groups.values()},
    }
