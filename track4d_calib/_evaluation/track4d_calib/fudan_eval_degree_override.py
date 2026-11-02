from __future__ import annotations

from typing import Any


def _resolve_degree(
    requested: int,
    *,
    checkpoint: int,
    maximum: int,
    label: str,
) -> int:
    requested = int(requested)
    if requested == -1:
        return int(checkpoint)
    if requested < 0 or requested > int(maximum):
        raise ValueError(
            f"{label} degree override must be -1 or within [0, {maximum}], "
            f"got {requested}"
        )
    return requested


def apply_active_degree_overrides(
    model: Any,
    *,
    spatial: int = -1,
    temporal: int = -1,
) -> dict[str, int]:
    checkpoint_spatial = int(model.active_sh_degree)
    checkpoint_temporal = int(model.active_sh_degree_t)
    render_spatial = _resolve_degree(
        spatial,
        checkpoint=checkpoint_spatial,
        maximum=int(model.max_sh_degree),
        label="spatial SH",
    )
    render_temporal = _resolve_degree(
        temporal,
        checkpoint=checkpoint_temporal,
        maximum=int(model.max_sh_degree_t),
        label="temporal SH",
    )
    model.active_sh_degree = render_spatial
    model.active_sh_degree_t = render_temporal
    return {
        "checkpoint_spatial": checkpoint_spatial,
        "checkpoint_temporal": checkpoint_temporal,
        "render_spatial": render_spatial,
        "render_temporal": render_temporal,
    }
