#!/usr/bin/env python3
"""Run track closure with a deterministic PnP seed."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "components/anchor_tracking/stage_code/scripts/geometry/run_track_closure.py"


def _load_source() -> Any:
    source_root = SOURCE.parents[2]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("seed3_frozen_temporal_pnp", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--min-inliers-per-seed", type=int, default=3)
    parser.add_argument("--allow-sparse-static-roster", action="store_true")
    wrapper_args, runner_args = parser.parse_known_args()
    if int(wrapper_args.min_inliers_per_seed) < 1:
        raise ValueError("--min-inliers-per-seed must be positive")
    sys.argv = [sys.argv[0], *runner_args]
    module = _load_source()
    if wrapper_args.allow_sparse_static_roster:
        def validate_sparse_runtime_track_views(
            *,
            bundle_views: Any,
            static_views: Any,
            target_views: Any,
        ) -> None:
            views = [str(view) for view in bundle_views]
            if len(views) != len(set(views)):
                raise ValueError("track bundle view IDs must be unique")
            static = {str(view) for view in static_views}
            targets = {str(view) for view in target_views}
            actual = set(views)
            undeclared = sorted(actual - static - targets)
            missing_targets = sorted(targets - actual)
            if undeclared:
                raise ValueError(
                    f"track bundle contains undeclared runtime view: {undeclared}"
                )
            if missing_targets:
                raise ValueError(
                    f"track bundle is missing target view: {missing_targets}"
                )
            if len(actual & static) < 2:
                raise ValueError("sparse track bundle needs at least two static views")

        module.validate_runtime_track_views = validate_sparse_runtime_track_views
    original = module.verify_multiseed_inlier_support

    def verify_seed3(
        *,
        inlier_point_ids: np.ndarray,
        point_ranges: Any,
        min_seed_banks: int = 2,
        min_inliers_per_seed: int = 4,
    ) -> dict[str, Any]:
        if int(min_seed_banks) != 2 or int(min_inliers_per_seed) != 4:
            raise RuntimeError("upstream independent-seed defaults changed")
        return original(
            inlier_point_ids=inlier_point_ids,
            point_ranges=point_ranges,
            min_seed_banks=2,
            min_inliers_per_seed=int(wrapper_args.min_inliers_per_seed),
        )

    module.verify_multiseed_inlier_support = verify_seed3
    result = int(module.main())
    if result == 0:
        from track4d_calib.same_time_multiview_pnp import _triangulate_robust
        output = Path(runner_args[runner_args.index("--output-root") + 1])
        _triangulate_robust.export_cache(output / "triangulation_cache.json.gz")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
