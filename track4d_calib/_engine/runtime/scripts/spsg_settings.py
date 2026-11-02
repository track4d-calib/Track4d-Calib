"""Shared SuperPoint/SuperGlue settings for static and framewise SfM."""
import math


DEFAULT_SPSG = {
    "resize_max": 1024,
    "resize_force": False,
    "max_keypoints": 2048,
    "sinkhorn_iterations": 50,
    "match_threshold": 0.2,
    "overlap_crop": True,
    "crop_resize_max": 1024,
    "crop_geometry_threshold_px": 2.0,
}


def resolve_spsg_settings(settings):
    """Resolve shared defaults and validate explicit overrides."""
    if not isinstance(settings, dict):
        raise ValueError("spsg must be an object")
    resolved = {**DEFAULT_SPSG, **settings}
    if resolved["overlap_crop"] is False:
        for key in ("crop_resize_max", "crop_geometry_threshold_px"):
            if key not in settings:
                resolved.pop(key)
    spsg_arguments(resolved)
    return resolved


def spsg_arguments(settings):
    if not isinstance(settings, dict):
        raise ValueError("spsg must be an object")
    fields = {
        "resize_max": ("--feature_resize_max", 64, 4096),
        "max_keypoints": ("--max_keypoints", 128, 16384),
        "sinkhorn_iterations": ("--superglue_sinkhorn_iterations", 1, 1000),
    }
    if set(settings) - set(fields) - {"resize_force", "match_threshold", "overlap_crop", "crop_resize_max", "crop_geometry_threshold_px"}:
        raise ValueError("Unknown spsg setting")
    args = []
    for key, (flag, low, high) in fields.items():
        if key in settings:
            value = settings[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer from {low} to {high}")
            args.extend([flag, str(value)])
    if "resize_force" in settings:
        if type(settings["resize_force"]) is not bool:
            raise ValueError("resize_force must be boolean")
        if settings["resize_force"]:
            args.append("--feature_resize_force")
    if "match_threshold" in settings:
        value = settings["match_threshold"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("match_threshold must be between zero and one")
        args.extend(["--superglue_match_threshold", str(value)])
    if "overlap_crop" in settings:
        if type(settings["overlap_crop"]) is not bool:
            raise ValueError("overlap_crop must be boolean")
        if settings["overlap_crop"]:
            args.append("--overlap_crop")
    if "crop_geometry_threshold_px" in settings:
        value = settings["crop_geometry_threshold_px"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 10:
            raise ValueError("crop_geometry_threshold_px must be in (0, 10]")
        if not settings.get("overlap_crop", False):
            raise ValueError("crop_geometry_threshold_px requires overlap_crop")
        args.extend(["--crop_geometry_threshold_px", str(value)])
    if "crop_resize_max" in settings:
        value = settings["crop_resize_max"]
        if type(value) is not int or not 64 <= value <= 4096:
            raise ValueError("crop_resize_max must be an integer from 64 to 4096")
        if not settings.get("overlap_crop", False):
            raise ValueError("crop_resize_max requires overlap_crop")
        args.extend(["--crop_resize_max", str(value)])
    return args
