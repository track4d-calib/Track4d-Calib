"""Shared relay validation and local-fallback planning for track_closure."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np


BANDS = (
    ("81", "82", "83"),
    ("84", "86", "87"),
    ("88", "89", "91"),
    ("92", "93", "94"),
)
RUNTIME_TARGETS = ("82", "87", "88", "92")
STATIC_VIEWS = tuple(
    "00 02 03 05 06 08 09 11 12 14 15 17 18 20 21 23 "
    "24 26 27 29 30 32 33 35 36 38 39 41 42 44 45 47".split()
)
TARGET_VIEWS = tuple(view for band in BANDS for view in band)
RIG_CONTRACT = None
if os.environ.get("TRACK4D_CALIB_RIG_CONTRACT"):


    _source = Path(__file__).resolve().parents[2] / "scripts/rig_contract.py"
    _spec = importlib.util.spec_from_file_location("calibration_declared_rig_contract", _source)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load rig contract implementation: {_source}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    RIG_CONTRACT = _module.load_rig_contract(Path(os.environ["TRACK4D_CALIB_RIG_CONTRACT"]))
    STATIC_VIEWS = tuple(RIG_CONTRACT["static_views"])
    TARGET_VIEWS = tuple(RIG_CONTRACT["dynamic_views"])
    BANDS = tuple(tuple(band) for band in RIG_CONTRACT["relay_bands"])
    RUNTIME_TARGETS = tuple(RIG_CONTRACT["relay_shard_labels"])
ROLES = ("fit", "confirm", "final")
FRAMES = tuple(range(30))
MINIMUM_POINTS_PER_ROLE = 12
WINDOW_SIZE = 5


def validate_relay_component(component: Sequence[str], source_view: str) -> None:
    """Check a relay job against its declared camera groups."""

    views = tuple(str(view) for view in component)
    if len(set(views)) != len(views):
        raise ValueError("relay component contains duplicate camera IDs")
    if str(source_view) not in views:
        raise ValueError("relay representative source is outside its component")
    if RIG_CONTRACT is None:
        if len(views) != 3:
            raise ValueError("legacy relay requires one three-PTZ band")
    elif views not in BANDS:
        raise ValueError("relay component must match a complete declared rig band")


def configured_frames() -> tuple[int, ...]:
    """Use the declared sequence, including in import-time runtime checks.

    Read at call time: the transfer launcher may set the scope after importing
    this module. Never infer the expected grid from the cache being validated.
    """
    count = int(os.environ.get("TRACK4D_CALIB_NUM_FRAMES", "30"))
    if count <= 0 or count % WINDOW_SIZE:
        raise ValueError("TRACK4D_CALIB_NUM_FRAMES must be a positive multiple of five")
    return tuple(range(count))


def blocked_component_windows(
    insufficient: Sequence[tuple[str, int, str]],
    *,
    bands: Sequence[Sequence[str]] = BANDS,
    frames: Sequence[int] = FRAMES,
    window_size: int = WINDOW_SIZE,
) -> list[dict[str, Any]]:
    """Map recoverable weak node-roles to atomic component windows."""

    weak = {
        (str(view), int(frame), str(role)) for view, frame, role in insufficient
    }
    ordered_frames = [int(frame) for frame in frames]
    blocked: list[dict[str, Any]] = []
    for component in bands:
        views = tuple(str(view) for view in component)
        for offset in range(0, len(ordered_frames), int(window_size)):
            window = ordered_frames[offset : offset + int(window_size)]
            keys = sorted(
                key for key in weak if key[0] in views and key[1] in window
            )
            if keys:
                blocked.append(
                    {
                        "component": list(views),
                        "window": window,
                        "blocked_node_roles": [list(key) for key in keys],
                        "policy": "exact_StaticPnP_component_fallback",
                    }
                )
    return blocked


def _relay_sources(
    payload: Mapping[str, Any], bands: Sequence[tuple[str, ...]],
) -> tuple[dict[tuple[str, ...], str], bool]:
    """Read the builder's source identity; a shard label is not its source."""

    sources = {band: band[0] for band in bands}
    if "source_view" in payload:
        shards = [payload]
    else:
        shards = payload.get("shards", [])
    declared = isinstance(shards, list) and any(
        isinstance(shard, Mapping) and ("anchor" in shard or "source_view" in shard)
        for shard in shards
    )
    if not declared:

        return sources, False
    if len(shards) != len(bands):
        raise ValueError("declared relay source shard coverage mismatch")
    observed: dict[tuple[str, ...], str] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("declared relay source shard is not a mapping")
        component = tuple(str(view) for view in shard.get("component", shard.get("band", [])))
        if "band" in shard and tuple(map(str, shard["band"])) != component:
            raise ValueError("relay component identities disagree")
        if component not in sources or component in observed:
            raise ValueError("declared relay source shard coverage mismatch")
        identities = {str(shard[key]) for key in ("anchor", "source_view") if key in shard}
        if len(identities) != 1:
            raise ValueError("missing or conflicting relay source identity")
        source = identities.pop()
        if source not in component:
            raise ValueError("relay representative source is outside its component")
        observed[component] = source
    return observed, True


def validate_relay_payload(
    payload: Mapping[str, Any],
    *,
    bands: Sequence[Sequence[str]] = BANDS,
    frames: Sequence[int] = FRAMES,
    roles: Sequence[str] = ROLES,
    minimum_points: int = MINIMUM_POINTS_PER_ROLE,
) -> dict[str, Any]:
    """Validate hard integrity while classifying low support as recoverable."""

    observations = payload.get("observations")
    if not isinstance(observations, dict):
        raise ValueError("relay cache has no observation dictionary")
    normalized_bands = tuple(tuple(str(view) for view in band) for band in bands)
    normalized_frames = tuple(int(frame) for frame in frames)
    normalized_roles = tuple(str(role) for role in roles)
    source_by_band, sources_declared = _relay_sources(payload, normalized_bands)
    expected = {
        (view, frame, role)
        for band in normalized_bands
        for view in band
        for frame in normalized_frames
        for role in normalized_roles
    }
    if set(observations) != expected:
        missing = sorted(expected - set(observations))[:10]
        extra = sorted(set(observations) - expected)[:10]
        raise ValueError(f"relay cache coverage mismatch: missing={missing}, extra={extra}")

    counts: list[int] = []
    insufficient: list[tuple[str, int, str]] = []
    for band in normalized_bands:
        for frame in normalized_frames:
            ids_by_view_and_role: dict[tuple[str, str], set[int]] = {}
            xyz_by_role_and_id: dict[tuple[str, int], np.ndarray] = {}
            for view in band:
                role_sets: list[set[int]] = []
                for role in normalized_roles:
                    row = observations[(view, frame, role)]
                    point_ids = np.asarray(row["point_ids"], dtype=np.int64).reshape(-1)
                    xyz = np.asarray(row["object_points"], dtype=np.float64).reshape(-1, 3)
                    image = np.asarray(row["image_points"], dtype=np.float64).reshape(-1, 2)
                    if len(point_ids) != len(xyz) or len(point_ids) != len(image):
                        raise ValueError(f"relay arrays disagree at {view}:{frame}:{role}")
                    if len(np.unique(point_ids)) != len(point_ids):
                        raise ValueError(f"duplicate relay IDs at {view}:{frame}:{role}")
                    if np.any(point_ids < 0):
                        raise ValueError(f"negative relay ID at {view}:{frame}:{role}")
                    if not np.isfinite(xyz).all() or not np.isfinite(image).all():
                        raise ValueError(f"nonfinite relay values at {view}:{frame}:{role}")
                    count = int(len(point_ids))
                    counts.append(count)
                    if count < int(minimum_points):
                        insufficient.append((view, frame, role))
                    ids = set(map(int, point_ids))
                    ids_by_view_and_role[(view, role)] = ids
                    role_sets.append(ids)
                    for point_id, point_xyz in zip(point_ids, xyz):
                        key = (role, int(point_id))
                        previous = xyz_by_role_and_id.get(key)
                        if previous is not None and not np.array_equal(
                            previous, point_xyz
                        ):
                            raise ValueError(
                                f"shared point ID changed XYZ at {band}:{frame}:{key}"
                            )
                        xyz_by_role_and_id[key] = point_xyz
                if any(
                    role_sets[left] & role_sets[right]
                    for left, right in ((0, 1), (0, 2), (1, 2))
                ):
                    raise ValueError(f"relay fit/confirm/final IDs overlap at {view}:{frame}")

            frame_is_supported = not any(
                (view, frame, role) in insufficient
                for view in band
                for role in normalized_roles
            )
            source_view = source_by_band[band]
            for role in normalized_roles:
                source = ids_by_view_and_role[(source_view, role)]
                if sources_declared:
                    for view in band:
                        if not ids_by_view_and_role[(view, role)].issubset(source):
                            raise ValueError(
                                f"relay IDs absent from declared source {source_view} at {view}:{frame}:{role}"
                            )
                if frame_is_supported:
                    if not all(
                        source & ids_by_view_and_role[(view, role)]
                        for view in band if view != source_view
                    ):
                        raise ValueError(
                            f"supported band has no shared relayed IDs at {band}:{frame}:{role}"
                        )

    blocked = blocked_component_windows(
        insufficient,
        bands=normalized_bands,
        frames=normalized_frames,
        window_size=WINDOW_SIZE,
    )
    return {
        "bands": [list(band) for band in normalized_bands],
        "source_by_component": {",".join(band): source for band, source in source_by_band.items()},
        "relay_source_identity_declared": sources_declared,
        "node_role_count": len(expected),
        "minimum_points_per_node_role": int(min(counts)),
        "minimum_required_points_per_node_role": int(minimum_points),
        "node_roles_at_least_minimum": int(
            np.count_nonzero(np.asarray(counts) >= int(minimum_points))
        ),
        "node_roles_below_minimum": len(insufficient),
        "below_minimum_keys": [list(key) for key in sorted(insufficient)],
        "blocked_component_windows": blocked,
        "blocked_component_window_count": len(blocked),
        "all_node_roles_have_minimum_support": not insufficient,
        "below_minimum_policy": "whole_5frame_3PTZ_component_exact_Static_fallback",
        "same_point_ID_has_identical_frame_local_XYZ_across_band": True,
        "fit_confirm_final_IDs_disjoint": True,
        "numeric_thresholds_changed": False,
        "hard_integrity_status": "pass",
        "support_status": (
            "pass_all_supported"
            if not insufficient
            else "pass_with_recoverable_component_fallback"
        ),
    }


def validate_relay_file(path: Path, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one relay cache and return its payload and shared validation audit."""

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("relay cache payload must be a dictionary")
    return payload, validate_relay_payload(payload, **kwargs)


def validate_runtime_xyz_cache_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate runtime shard identity without turning weak support into corruption."""

    frames = configured_frames()
    if str(payload.get("protocol", "")).startswith(
        "PAIR_FALLBACK-frozen-conditioned-dual-static-band-relay-merged"
    ):
        return {
            "runtime_merged_relay_adapter": True,
            "cache_is_revalidated_by_complete_transfer": True,
        }
    shards = payload.get("shards")
    if not isinstance(shards, list) or len(shards) != len(BANDS):
        raise ValueError("dual-static XYZ cache has the wrong shard count")
    expected = {
        frozenset(band): target
        for band, target in zip(BANDS, RUNTIME_TARGETS, strict=True)
    }
    observed: dict[frozenset[str], str] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("dual-static XYZ shard is not a mapping")
        component = frozenset(str(view) for view in shard.get("component", []))
        target = str(shard.get("target_view", ""))
        if component in observed:
            raise ValueError("dual-static XYZ cache repeats a PTZ band")
        observed[component] = target
    if observed != expected:
        raise ValueError("dual-static XYZ cache has the wrong PTZ bands")

    audit = validate_relay_payload(payload, frames=frames)
    return {
        **audit,
        "target_by_component": {
            ",".join(sorted(component)): target
            for component, target in expected.items()
        },
        "recoverable_low_support_is_not_import_time_corruption": True,
    }
