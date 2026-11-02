# Architecture

Track4D-Calib uses one pipeline for every declared camera array:

```text
RGB + intrinsics
    → static reconstruction
    → sparse cross-view matching and temporal tracking
    → static-supported triangulation and PnP
    → detect → repair → verify → commit
    → camera export
```

## Track-driven geometric closure

Tracks connect six parts of calibration:

1. Shared identities unify correspondences across views and time.
2. Anchor observations support map construction and triangulation.
3. Track-indexed 2D–3D observations supply dynamic-camera PnP.
4. Inlier support, reprojection error and confidence identify poses requiring repair.
5. Local bundle adjustment refines poses against fitting observations.
6. Independent observations determine whether a proposed update improves geometry.

Static-camera poses define the common coordinate system. The scene configuration
controls the observation budget and optional refinement stages.

## Modules

| Module | Responsibility |
| --- | --- |
| `track4d_calib/cli.py` | Public `plan`, `preflight` and `run` commands |
| `track4d_calib/config.py` | Configuration loading and validation |
| `track4d_calib/dataset.py` | Image and intrinsic-file validation |
| `track4d_calib/runner.py` | Pipeline execution and camera export |
| `track4d_calib/_engine/runtime/sparse/` | Matching, tracking, registration and refinement |
| `track4d_calib/_engine/runtime/components/` | Shared geometry components |
| `track4d_calib/evaluation.py`, `track4d_calib/_evaluation/` | Optional reconstruction adapters |

## Extending the method

New captures use the existing data format and a scene configuration.
Matcher and tracker changes preserve camera, frame and point identities.
Refinement modules consume shared observations and provide candidates for the
existing geometric checks. Keep numerical estimation separate from orchestration.

Reconstruction backends consume exported cameras and images; they do not change
the calibration pipeline. See [camera output](DATA.md#output-cameras).
