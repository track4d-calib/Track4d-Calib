"""Explicit COLMAP seeds for a reproducible current-run static map."""

from __future__ import annotations


def seeded_colmap_options(seed: int) -> tuple[dict, dict]:
    """Keep HLOC's verification thresholds, and seed every SfM RANSAC stage.

    Python/NumPy seeds do not control COLMAP's C++ RNG. The caller must also
    use one mapper thread when deterministic image registration is required.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("SfM seed must be a nonnegative integer")
    verification = {"ransac": {"max_num_trials": 20000, "min_inlier_ratio": 0.1,
                                "random_seed": seed}}
    mapping = {"random_seed": seed, "mapper": {"random_seed": seed},
               "triangulation": {"random_seed": seed}}
    return verification, mapping
