# Configuration

Start from [configs/scene.example.json](../configs/scene.example.json) for a new
capture, or use a [dataset preset](DATASETS.md). All presets use the same schema.

## Scene settings

| Field | Meaning |
| --- | --- |
| `scene` | Capture name used in output metadata |
| `static_views` | Ordered list of static camera IDs, encoded as numeric strings |
| `dynamic_views` | Ordered list of moving camera IDs, disjoint from `static_views` |
| `resolution` | Prepared image dimensions as `[width, height]` |
| `frames` | Number of synchronized frames to process |
| `levels` | Settings for `fast`, `balanced` and `quality` |

Keep the remaining template settings when preparing a new capture. Camera IDs,
image geometry and intrinsics must follow the [data contract](DATA.md).

## Calibration levels

| Level | Use |
| --- | --- |
| `fast` | Smaller matching and tracking budget |
| `balanced` | Default configuration for routine calibration |
| `quality` | Additional refinement for demanding captures and demos |

Choose a level with `--level`. Settings are declared in each preset; all levels
use the same entry point and preserve camera rosters and image resolution.
`track4d-calib plan` accepts the same arguments as `run` and prints the resolved
settings without starting calibration.

## Local dependencies

Copy [dependencies.example.json](../configs/dependencies.example.json) to
`configs/dependencies.local.json` and set:

| Field | Path |
| --- | --- |
| `hloc` | HLoc checkout |
| `cotracker` | CoTracker checkout |
| `romav2_src` | RoMaV2 `src/` directory |
| `cotracker_checkpoint` | CoTracker checkpoint |
| `romav2_weights` | RoMaV2 checkpoint |

Relative paths resolve from the dependency file's directory.

## Reconstruction settings

The optional `evaluation` object defines the held-out views, image resolution
and reconstruction settings. Keep a dataset's supplied evaluation configuration
when reproducing its results. See [reconstruction](EVALUATION.md) for usage.
