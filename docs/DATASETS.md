# Data preparation

Download the prepared datasets from
[Hugging Face](https://huggingface.co/datasets/dlt7/dyn-calib).
Each scene package contains one RGB video per calibration camera, camera
intrinsics and a scene configuration.

| Dataset | Scenes | Static / moving cameras per scene | Resolution |
| --- | ---: | ---: | --- |
| Dyn-DNA-Rendering | 6 | 32 / 12 | 512 × 512 |
| Dyn-Neu3D | 6 | 20 / 16 | 1352 × 1014 |
| Dyn-4DV | 10 (`Seq0`–`Seq9`) | 57 / 10 | 1280 × 720 |

The [inventory](datasets.json) lists the 22 scene archives and their sizes.
The complete collection occupies 56.38 GB. Follow the
[download example](../README.md#datasets) for a single scene, or omit the archive
filename to download the complete collection.

## Prepare and run

Extract the scene archive under `data/`, then decode it with FFmpeg installed:

```bash
mkdir -p data && tar -xf downloads/dyn-dna-rendering-0009_12-inputs.tar -C data
python tools/data/extract_videos.py --package data/dyn-dna-rendering-0009_12-inputs --output data/0009_12
track4d-calib run --source data/0009_12 --config data/0009_12/config.json --dependencies configs/dependencies.local.json --output outputs/0009_12
```

The converter writes synchronized PNG images, intrinsics and `config.json`
in the standard [data format](DATA.md). Point `--source` at this output directory.

Use `--level fast`, `balanced` or `quality` to select a calibration budget.

## Package layout

```text
<scene_package>/
├── config.json
├── manifest.json
├── cameras/intri.yml
└── videos/<camera_id>.mkv
```

Videos use lossless H.264 RGB encoding at the original image resolution.
The manifest records camera IDs, frame count and image dimensions.
Calibration uses synchronized frame indices.

See [optional reconstruction](EVALUATION.md) and [trajectory visualization](VISUALIZATION.md).

Re-encoding the data with lossy compression or changing the decoding and
preprocessing workflow can cause a small performance gap. COLMAP's randomized
estimation may also produce slight run-to-run differences.
