"""A scene-independent public configuration, using SfM coordinate units."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re


from ._engine.runtime.scripts.spsg_settings import resolve_spsg_settings, spsg_arguments

REQUIRED = {"scene", "static_views", "dynamic_views", "resolution", "center_radius"}
OPTIONAL = {
    "track_validation",
    "center_motion",
    "spsg",
    "spsg_aug",
    "dynamic_overlap_crop",
    "level",
    "dataset",
    "calibration_test_views",
    "evaluation",
    "description",
    "motion_model",
    "frames",
    "static_support_count",
    "static_refinement",
    "static_refinement_fit_frames",
    "closure_loss",
    "relay_source_quality",
    "temporal_anchor_count",
    "temporal_anchor_spacing",
    "pnp_max_points",
    "anchor_minimum_coverage",
    "anchor_points_per_band",
    "nonanchor_static_supports",
}


LEVELS = ("fast", "balanced", "quality")
LEVEL_SETTINGS = {
    "center_radius", "center_motion",
    "temporal_anchor_count", "temporal_anchor_spacing", "static_support_count", "pnp_max_points",
    "anchor_minimum_coverage", "nonanchor_static_supports",
    "anchor_points_per_band",
    "relay_source_quality", "static_refinement", "static_refinement_fit_frames",
}


def load_config(path: Path, level: str | None = None, frames: int | None = None) -> dict:
    """Resolve calibration settings without changing the scene or evaluation."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    if frames is not None:
        raw["frames"] = frames
    presets = raw.pop("levels", None)
    if presets is None:
        if level is not None:
            raise ValueError("this configuration does not declare calibration levels")
        return validate_config(raw)
    if (
        not isinstance(presets, dict)
        or set(presets) != set(LEVELS)
    ):
        raise ValueError("levels must declare fast, balanced and quality")
    resolved = {}
    for name, settings in presets.items():
        if not isinstance(settings, dict) or settings.keys() - LEVEL_SETTINGS:
            raise ValueError(f"level {name} may only override calibration settings")
        resolved[name] = validate_config(dict(raw, **settings, level=name))
    selected = level or "balanced"
    if selected not in resolved:
        raise ValueError(f"calibration level {selected!r} is not declared in this configuration")
    return resolved[selected]


def _validate_views(values: object, name: str, minimum: int = 1) -> None:
    if not isinstance(values, list) or any(
        not isinstance(view, str) or not view.isdecimal() for view in values
    ):
        raise ValueError(f"{name} must contain numeric camera IDs encoded as strings")
    if len(values) < minimum or len(set(map(int, values))) != len(values):
        raise ValueError(f"{name} needs at least {minimum} distinct camera IDs")


def _validate_resolution(value: object, name: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise ValueError(f"{name} must be [width, height], both positive integers")


def validate_config(config: dict) -> dict:
    """Return a validated copy with defaults; dataset names do not select solvers."""
    if not isinstance(config, dict) or not REQUIRED <= config.keys():
        raise ValueError(f"configuration requires {sorted(REQUIRED)}")
    if config.keys() - REQUIRED - OPTIONAL:
        raise ValueError(
            f"unknown configuration fields: {sorted(config.keys() - REQUIRED - OPTIONAL)}"
        )
    config = dict(config)
    if "track_validation" in config and config["track_validation"] is not None:
        partition = config["track_validation"]
        if not isinstance(partition, dict) or partition.keys() - {"exclusion_radius_px"}:
            raise ValueError("track_validation accepts exclusion_radius_px")
        partition = {"exclusion_radius_px": 2., **partition}
        value = partition["exclusion_radius_px"]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("track_validation.exclusion_radius_px must be positive")
        config["track_validation"] = partition
    spsg = config.get("spsg", {})
    if "spsg_aug" in config:
        augmentation = config.pop("spsg_aug")
        if type(augmentation) is not bool:
            raise ValueError("spsg_aug must be a boolean")
        if not isinstance(spsg, dict):
            raise ValueError("spsg must be an object")
        if "overlap_crop" in spsg and spsg["overlap_crop"] is not augmentation:
            raise ValueError("spsg_aug conflicts with spsg.overlap_crop")
        spsg = {**spsg, "overlap_crop": augmentation}
        if not augmentation:
            for key in ("crop_resize_max", "crop_geometry_threshold_px"):
                spsg.pop(key, None)
    config["spsg"] = resolve_spsg_settings(spsg)
    crop = config.setdefault("dynamic_overlap_crop", True)
    if type(crop) is not bool:
        raise ValueError("dynamic_overlap_crop must be a boolean")
    if "level" in config and config["level"] not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    for key, minimum in (("static_views", 16), ("dynamic_views", 3)):
        _validate_views(config[key], key, minimum)
    if set(map(int, config["static_views"])) & set(map(int, config["dynamic_views"])):
        raise ValueError("static and dynamic views overlap")
    tests = config.get("calibration_test_views", [])
    _validate_views(tests, "calibration_test_views", minimum=0)
    if not set(tests) <= set(config["dynamic_views"]):
        raise ValueError("calibration_test_views must be a unique dynamic-view subset")
    _validate_resolution(config["resolution"], "resolution")
    radius = config["center_radius"]
    if radius is not None and (
        type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0
    ):
        raise ValueError(
            "center_radius must be null (free translation) or finite and nonnegative in SfM units"
        )
    inferred = (
        "free-center" if radius is None else "fixed-axis" if radius == 0 else "bounded-center"
    )
    model = config.get("motion_model", inferred)
    if model != inferred:
        raise ValueError(
            "motion_model must agree with center_radius: null=free-center, zero=fixed-axis, positive=bounded-center"
        )
    config["motion_model"] = model
    if config.get("center_motion") is None:
        config["center_motion"] = None
    else:
        motion = config.get("center_motion", {})
        if not isinstance(motion, dict):
            raise ValueError("center_motion must be an object")
        if motion.get("model") == "rgb_window":
            if motion.keys() - {"model", "radius_fraction", "rotation_model"} or radius is not None:
                raise ValueError("rgb_window accepts radius_fraction and rotation_model, with center_radius=null")
            motion = {"radius_fraction": 0., "rotation_model": "framewise", **motion}
            value = motion["radius_fraction"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("center_motion.radius_fraction must be finite and nonnegative")
            if motion["rotation_model"] != "framewise":
                raise ValueError("center_motion.rotation_model must be framewise")
            if "track_validation" in config and config["track_validation"] is None:
                raise ValueError("rgb_window requires track validation before PnP")
            config.setdefault("track_validation", {"exclusion_radius_px": 2.0})
        else:
            raise ValueError("center_motion.model must be rgb_window")
        config["center_motion"] = motion
    frames = config.get("frames", 30)
    if type(frames) is not int or frames < 2:
        raise ValueError("frames must be an integer of at least 2")
    config["frames"] = frames
    if (config.get("center_motion") or {}).get("model") == "rgb_window" and frames < 4:
        raise ValueError("Window motion refinement needs at least four frames")
    for key, default, minimum, maximum in (
        ("pnp_max_points", 150, 12, 100000),
        ("anchor_points_per_band", 600, 24, 100000),
        ("anchor_minimum_coverage", 24, 0, 100000),
        ("nonanchor_static_supports", 0, 0, 8),
    ):
        value = config.get(key, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be an integer from {minimum} to {maximum}")
        config[key] = value

    spacing = config.get("temporal_anchor_spacing")
    if spacing is not None:
        if type(spacing) is not int or spacing < 1:
            raise ValueError("temporal_anchor_spacing must be a positive integer")
        count = 1 + math.ceil((frames - 1) / spacing)
        if "temporal_anchor_count" in config and config["temporal_anchor_count"] != count:
            raise ValueError("temporal_anchor_count disagrees with temporal_anchor_spacing")
        config["temporal_anchor_count"] = count
    anchor_count = config.get("temporal_anchor_count")
    if "temporal_anchor_count" in config and (
        type(anchor_count) is not int or not 2 <= anchor_count <= frames
    ):
        raise ValueError("temporal_anchor_count must be an integer from 2 to frames")
    supports = config.get("static_support_count", 2)
    if type(supports) is not int or not 2 <= supports <= 8:
        raise ValueError("static_support_count must be an integer from 2 to 8")
    config["static_support_count"] = supports
    source_quality = config.get("relay_source_quality", "fast")
    if source_quality != "fast":
        raise ValueError("relay_source_quality must be fast for sparse anchor matching")
    config["relay_source_quality"] = source_quality
    refinement = config.get("static_refinement", "off")
    if refinement not in ("off", "fast", "precise"):
        raise ValueError("static_refinement must be off, fast or precise")
    config["static_refinement"] = refinement
    fit_frames = config.get("static_refinement_fit_frames", [0])
    heldout = {(frames - 1) // 2, frames - 1}
    if (
        not isinstance(fit_frames, list)
        or not fit_frames
        or any(type(f) is not int or not 0 <= f < frames for f in fit_frames)
        or fit_frames != sorted(set(fit_frames))
        or (refinement != "off" and heldout.intersection(fit_frames))
    ):
        raise ValueError(
            "static_refinement_fit_frames must be ordered, unique frame IDs excluding the middle and final verification frames"
        )
    config["static_refinement_fit_frames"] = fit_frames
    closure_loss = config.pop("closure_loss", "huber")
    if closure_loss != "huber":
        raise ValueError("closure_loss must be huber for the geometric repair solver")
    if (
        not isinstance(config["scene"], str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["scene"]) is None
    ):
        raise ValueError("scene must be a simple label, not a path")
    if "evaluation" in config:
        evaluation_settings(config)
    return config


def evaluation_settings(config: dict) -> dict:
    """Resolve the common 30-frame readout from declared view and rig settings."""
    options = config.get("evaluation")
    if not isinstance(options, dict):
        raise ValueError("evaluation settings are required for the NVS readout")
    fixed = {"frames": 30, "iters": 10000, "seed": 0, "batch_size": 4, "checkpoint": "none"}
    allowed = set(fixed) | {
        "test_views",
        "camera_source",
        "resolution",
        "init_mode",
        "roma_ref_views",
        "time_denominator",
        "temporal_sigma",
        "source_resolution",
        "render_resolution_factor",
        "time_duration_end",
        "init_points",
        "roma_target_views",
        "roma_matches_per_pair",
    }
    if set(options) - allowed:
        raise ValueError(f"unknown evaluation settings: {sorted(set(options) - allowed)}")
    for key, value in fixed.items():
        if key in options and (type(options[key]) is not type(value) or options[key] != value):
            raise ValueError(f"the common readout requires {key}={value!r}")
    result = {
        **fixed,
        "resolution": config["resolution"],
        "init_mode": "roma_ref_pairs",
        "time_denominator": 29,
        "temporal_sigma": 0.1,
        "init_points": 300000,
        "render_resolution_factor": 1,
        "time_duration_end": 1.0,
        "roma_matches_per_pair": 2048,
        **options,
    }
    tests = result.get("test_views")
    _validate_views(tests, "evaluation.test_views")
    statics, dynamics = config["static_views"], config["dynamic_views"]
    if set(map(int, tests)) & set(map(int, statics)):
        raise ValueError("evaluation views must not overlap static initialization cameras")
    source = result.get("camera_source")
    if source == "calibrated":
        if not set(tests) <= set(dynamics) or set(tests) != set(
            config.get("calibration_test_views", [])
        ):
            raise ValueError("calibrated test cameras must match calibration_test_views")
    elif source == "external":
        overlap = set(tests) & set(dynamics)
        if overlap != set(config.get("calibration_test_views", [])):
            raise ValueError(
                "reconstruction test cameras participating in calibration must be declared "
                "in calibration_test_views; their reference poses remain evaluation-only"
            )
    else:
        raise ValueError("evaluation.camera_source must be external or calibrated")
    result["dynamic_train_views"] = [v for v in dynamics if v not in tests]
    if not result["dynamic_train_views"]:
        raise ValueError("NVS needs at least one calibrated dynamic training view")
    resolution = result["resolution"]
    _validate_resolution(resolution, "evaluation.resolution")
    result.setdefault("source_resolution", resolution)
    source_resolution = result["source_resolution"]
    _validate_resolution(source_resolution, "evaluation.source_resolution")
    for key in ("render_resolution_factor", "init_points", "roma_matches_per_pair"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"evaluation.{key} must be a positive integer")
    if result["render_resolution_factor"] not in (1, 2, 3, 4, 8):
        raise ValueError(
            "evaluation.render_resolution_factor must be 1, 2, 3, 4 or 8 for this runtime"
        )
    if source_resolution != [v * result["render_resolution_factor"] for v in resolution]:
        raise ValueError("source_resolution must equal resolution times render_resolution_factor")
    duration = result["time_duration_end"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("evaluation.time_duration_end must be finite and positive")
    if result["init_mode"] not in ("roma_ref_pairs", "sparse_sfm"):
        raise ValueError("evaluation.init_mode must be roma_ref_pairs or sparse_sfm")
    refs = result.get("roma_ref_views", [statics[len(statics) * i // 4] for i in range(4)])
    _validate_views(refs, "evaluation.roma_ref_views")
    if not set(refs) <= set(statics):
        raise ValueError("evaluation.roma_ref_views must be unique static camera IDs")
    result["roma_ref_views"] = refs if result["init_mode"] == "roma_ref_pairs" else []
    targets = result.get("roma_target_views", statics)
    train_views = set(statics + result["dynamic_train_views"])
    _validate_views(targets, "evaluation.roma_target_views")
    if not set(targets) <= train_views:
        raise ValueError("evaluation.roma_target_views must be unique training camera IDs")
    result["roma_target_views"] = targets if result["init_mode"] == "roma_ref_pairs" else []
    if type(result["time_denominator"]) is not int or result["time_denominator"] not in (29, 30):
        raise ValueError("evaluation.time_denominator must be 29 or 30")
    if duration < 29 / result["time_denominator"]:
        raise ValueError("evaluation.time_duration_end must include all 30 frame timestamps")
    sigma = result["temporal_sigma"]
    if type(sigma) not in (float, int) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError("evaluation.temporal_sigma must be finite and nonnegative")
    return result


def load_dependencies(path: Path) -> dict[str, Path]:
    """Resolve dependency paths relative to the file that declares them."""
    data = json.loads(path.read_text())
    keys = {"hloc", "cotracker", "cotracker_checkpoint", "romav2_src", "romav2_weights"}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError(f"dependencies must declare exactly {sorted(keys)}")
    if any(not isinstance(value, str) or not value.strip() for value in data.values()):
        raise ValueError("dependency paths must be nonempty strings")
    return {key: (path.parent / Path(value).expanduser()).resolve() for key, value in data.items()}


def check_dependencies(deps: dict[str, Path]) -> list[str]:
    required = {
        "hloc": deps["hloc"] / "hloc/__init__.py",
        "SuperPoint weights": deps["hloc"]
        / "third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth",
        "SuperGlue weights": deps["hloc"]
        / "third_party/SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth",
        "cotracker": deps["cotracker"] / "cotracker/__init__.py",
        "cotracker_checkpoint": deps["cotracker_checkpoint"],
        "romav2_src": deps["romav2_src"] / "romav2/__init__.py",
        "romav2_weights": deps["romav2_weights"],
    }
    return [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
