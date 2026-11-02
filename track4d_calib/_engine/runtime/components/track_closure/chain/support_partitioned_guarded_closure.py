#!/usr/bin/env python3
"""Partition closure components by shared track support."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


CROSSFIT_WRAPPER = Path(__file__).resolve().with_name("roma_crossfit_guard_closure.py")
MAX_COMPONENT_VIEWS = 3


def _load_crossfit():
    spec = importlib.util.spec_from_file_location("crossfit_closure", CROSSFIT_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {CROSSFIT_WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CROSSFIT = _load_crossfit()
BASE = CROSSFIT.BASE
ORIGINAL_COMPONENTS = BASE.target_components_from_window_visibility


def support_partition(
    component: Sequence[str],
    *,
    shared_count_by_edge: Mapping[str, Sequence[int]],
    target_order: Sequence[str],
    max_views: int,
) -> list[list[str]]:
    """Kruskal-style deterministic grouping by summed public track support."""
    views = [str(view) for view in component]
    order = {str(view): index for index, view in enumerate(target_order)}
    maximum = int(max_views)
    if maximum < 2:
        raise ValueError("max_views must be at least two")
    if len(views) <= maximum:
        return [views]
    clusters = {view: {view} for view in views}

    def root(view: str) -> str:
        for key, members in clusters.items():
            if view in members:
                return key
        raise KeyError(view)

    edges = []
    for left_index, left in enumerate(views):
        for right in views[left_index + 1 :]:
            key = f"{left}:{right}"
            reverse = f"{right}:{left}"
            counts = shared_count_by_edge.get(key, shared_count_by_edge.get(reverse, ()))
            support = int(sum(int(value) for value in counts))
            edges.append((-support, order[left], order[right], left, right))
    for _negative_support, _left_order, _right_order, left, right in sorted(edges):
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            continue
        merged = clusters[left_root] | clusters[right_root]
        if len(merged) > maximum:
            continue
        clusters[left_root] = merged
        del clusters[right_root]
    result = [sorted(members, key=order.get) for members in clusters.values()]
    result.sort(key=lambda members: min(order[view] for view in members))
    if sorted(view for group in result for view in group) != sorted(views):
        raise RuntimeError("support partition lost or duplicated a target view")
    if any(len(group) > maximum for group in result):
        raise RuntimeError("support partition exceeded the frozen component limit")
    return result


def partitioned_components(*args, **kwargs):
    graph = ORIGINAL_COMPONENTS(*args, **kwargs)
    partitioned = []
    source_components = []
    for component in graph["components"]:
        groups = support_partition(
            component,
            shared_count_by_edge=graph["shared_count_by_edge"],
            target_order=kwargs["target_views"],
            max_views=MAX_COMPONENT_VIEWS,
        )
        partitioned.extend(groups)
        source_components.append({"source": list(component), "partitions": groups})
    result = dict(graph)
    result["components"] = partitioned
    result["support_partition"] = {
        "max_component_views": MAX_COMPONENT_VIEWS,
        "policy": "descending_summed_shared_track_support_with_size_cap",
        "sources": source_components,
    }
    return result


partitioned_components.__module__ = BASE.__name__
partitioned_components.__name__ = "target_components_from_window_visibility"
partitioned_components.__qualname__ = "target_components_from_window_visibility"
BASE.target_components_from_window_visibility = partitioned_components


def annotate_audit() -> None:
    CROSSFIT.annotate_audit()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1])
    path = output_root / "PUBLIC_GEOMETRY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = "PARTITION-support-partitioned-plus-RoMa-crossfit-guard-v1"
    payload["method_override"]["support_partitioned_components"] = {
        "max_component_views": MAX_COMPONENT_VIEWS,
        "selection_evidence": "summed public same-time shared-track counts",
        "candidate_solver_changed": False,
        "numeric_geometry_thresholds_changed": False,
    }
    payload["attestation"]["component_partition_uses_no_GT_test_or_NVS"] = True
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    exit_code = BASE.main()
    annotate_audit()
    raise SystemExit(exit_code)
