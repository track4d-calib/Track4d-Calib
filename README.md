# Track4D-Calib

Track4D-Calib is a fast calibration method for synchronized camera arrays containing
static and moving cameras. It combines sparse cross-view matching with temporal
tracking to estimate dynamic-camera poses in a shared coordinate system.

**Track-driven geometric closure** connects registration, local refinement and
independent geometric verification. All datasets use the same calibration pipeline.

[Project page](https://track4d-calib.github.io/Track4d-Calib/).
Further documentation, examples and related materials are expected by November 30, 2026.

## Datasets

Prepared inputs are available on [Hugging Face](https://huggingface.co/datasets/dlt7/dyn-calib):
22 scenes from Dyn-DNA-Rendering, Dyn-Neu3D and Dyn-4DV (56.38 GB).
Download an individual scene with the Hugging Face CLI:

```bash
python -m pip install huggingface_hub
hf download dlt7/dyn-calib dyn-dna-rendering-0009_12-inputs.tar --repo-type dataset --local-dir downloads
```

See [data preparation](docs/DATASETS.md) for unpacking and conversion to calibration inputs.

## Quick start

Follow the [installation guide](docs/INSTALL.md), then select a
[dataset configuration](docs/DATASETS.md) or prepare your own capture using the
[data format](docs/DATA.md).

```bash
SCENE=data/capture_01
track4d-calib preflight --source "$SCENE" --config "$SCENE/config.json" --dependencies configs/dependencies.local.json --output outputs/check
track4d-calib run --source "$SCENE" --config "$SCENE/config.json" --dependencies configs/dependencies.local.json --output outputs/capture_01
```

The default level is `balanced`. Use `--level fast` for a smaller compute budget
or `--level quality` for additional refinement. Use a new output directory for each run.

## Prepare your own data

Export synchronized RGB images and calibrated intrinsics:

```text
capture_01/
├── config.json
├── cameras/intri.yml
└── images/<camera_id>/<frame_id>.png
```

Copy [the scene template](configs/scene.example.json) to `config.json` and set
`scene`, `static_views`, `dynamic_views` and `resolution`. Camera IDs must match
the image directories and intrinsic entries. A camera that only rotates belongs
in `dynamic_views`.

The [data guide](docs/DATA.md) covers capture requirements, naming and intrinsics.
The [configuration guide](docs/CONFIGURATION.md) describes the shared settings.

## Camera output

Dynamic camera poses are exported to
`outputs/capture_01/poses/f<frame_id>/cameras_pred/`. Intrinsics and extrinsics use
OpenCV YAML; all cameras share the static-array coordinate system. See
[camera conventions](docs/DATA.md#output-cameras) and
[trajectory visualization](docs/VISUALIZATION.md).

## Example results

The balanced presets were validated on frames 0–29 of each scene below.
Reconstruction uses the optional Fudan backend with 10,000 iterations and seed 0.

| Dataset | Scene | PSNR / dB | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| Dyn-DNA-Rendering | 0009_12 | 31.12 | 0.9650 | 0.0383 |
| Dyn-Neu3D | coffee_martini | 30.11 | 0.9457 | 0.0300 |
| Dyn-4DV | Seq5 | 27.52 | 0.8393 | 0.1073 |

[Evaluation settings](docs/benchmarks/default_30f.json) specify the images,
view splits and reconstruction budget. See [benchmarking](docs/BENCHMARK.md)
for runtime comparison.

## Optional reconstruction

Use the calibrated cameras with a reconstruction backend of your choice.
The [reconstruction guide](docs/EVALUATION.md) provides an optional 4DGS adapter
and camera export. Reconstruction software is installed separately.

## Repository layout

| Path | Contents |
| --- | --- |
| `track4d_calib/` | Calibration library and command-line entry point |
| `configs/` | Scene presets and local dependency paths |
| [`tools/`](tools/README.md) | Setup, calibration, benchmarking, reconstruction, visualization and checks |
| `dependencies/`, `third_party/` | Dependency versions, notices and optional checkouts |
| `docs/` | Usage and implementation guides |

See [architecture](docs/ARCHITECTURE.md), [contributing](CONTRIBUTING.md) and
[third-party notices](THIRD_PARTY.md).
