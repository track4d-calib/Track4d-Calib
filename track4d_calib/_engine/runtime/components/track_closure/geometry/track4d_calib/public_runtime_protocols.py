"""Protocol identities for public runtimes that hide dynamic-camera poses."""

from __future__ import annotations


BALANCED_RING_CENTER_HELDOUT_PROTOCOL = (
    "DynDNA-balanced-ring-train-center-heldout-v1"
)
NEU3D_TRACKSTRESS_PROTOCOL = "dyn-neu3d-trackstress-calibration-runtime-v1"
HIDDEN_GENERATOR_PROTOCOL = "hidden-generator-non-common-center-v1"
SPARSE_ANCHOR_PROTOCOL = "sparse-anchor-partially-observed-parent-v1"
ALL_PARENT_OBSERVED_PROTOCOL = "all-parent-observed-dynamic-pose-hidden-v1"
INDEPENDENT_PTZ_RECOVERABLE_PROTOCOL = (
    "DynDNA-independent-PTZ-recoverable-benchmark-v1"
)

PUBLIC_NON_COMMON_CENTER_RUNTIME_PROTOCOLS = frozenset(
    {
        BALANCED_RING_CENTER_HELDOUT_PROTOCOL,
        NEU3D_TRACKSTRESS_PROTOCOL,
        HIDDEN_GENERATOR_PROTOCOL,
        SPARSE_ANCHOR_PROTOCOL,
        ALL_PARENT_OBSERVED_PROTOCOL,
        INDEPENDENT_PTZ_RECOVERABLE_PROTOCOL,
    }
)


def is_public_non_common_center_runtime_protocol(protocol: object) -> bool:
    return str(protocol) in PUBLIC_NON_COMMON_CENTER_RUNTIME_PROTOCOLS
