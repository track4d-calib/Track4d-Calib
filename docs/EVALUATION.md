# Optional reconstruction

Calibration exports cameras independently of reconstruction software. Use the
YAML cameras directly, or export NeRF/Blender transforms for another backend:

```bash
python tools/reconstruction/export_reconstruction.py --run outputs/capture_01 --source data/capture_01 --output outputs/capture_01_transforms
```

The export contains images, per-camera intrinsics, camera-to-world transforms
and timestamps. Add `--test-views <ids>` to reserve calibrated views for
reconstruction evaluation. Coordinate conventions are specified in
[data format](DATA.md#output-cameras).

## Included 4DGS adapter

An optional adapter supports the public
[Fudan 4DGS implementation](https://github.com/fudan-zvg/4d-gaussian-splatting).
Fetch it and follow its installation instructions in a separate training environment:

```bash
python tools/setup/fetch_dependencies.py --component 4dgs --output third_party
python -m pip install -e '.[evaluation]'
```

Build the backend's CUDA extensions with the training environment. From the
backend directory, obtain the rasterizer library path with:

```bash
python -c 'from gaussian_renderer.diff_gaussian_rasterization import _C; print(_C.__file__)'
```

Use that path for `--renderer` and the training environment's Python for
`--train-python`. Evaluate a calibrated sequence with:

```bash
python tools/reconstruction/run_experiment.py --calibration-run outputs/scene --config configs/dyn-dna-rendering.json --output outputs/reconstruction \
  --dependencies configs/dependencies.local.json --train-python /path/to/training_env/bin/python \
  --renderer /path/to/diff_gaussian_rasterization.so --canonical-scene /path/to/evaluation_scene
```

Use the matching scene preset containing an `evaluation` section and match
`--frames` to the calibration run. Backend and DINOv3 paths default to
`third_party/4dgs` and `third_party/dinov3`.

For `evaluation.camera_source: external`, `--canonical-scene` contains the
[evaluation cameras and images](DATA.md#external-evaluation-cameras).
For `calibrated`, omit that argument; test views belong to the calibrated
roster and are excluded from reconstruction training.

`result.json` contains PSNR, SSIM, LPIPS and per-view results. Renders and logs
are retained; use `--keep-checkpoints` to retain training checkpoints.
Keep the same reconstruction setup across methods when comparing calibration quality.
