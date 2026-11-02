# Calibration benchmark

Compare complete calibration commands on the same prepared data, GPU and CPU
allocation. Timing includes static reconstruction, every anchor, tracking,
registration, refinement and camera export.

Copy [the benchmark plan](benchmarks/plan.example.json) to
`configs/benchmark.local.json` and set its input paths and scene configuration:

```bash
python tools/benchmark/benchmark_calibration.py --plan configs/benchmark.local.json --output outputs/benchmark
python tools/benchmark/plot_speed_accuracy.py --summary outputs/benchmark/summary.json --output outputs/benchmark/curve
```

The plan compares framewise COLMAP + SuperPoint/SuperGlue with the three Track4D-Calib
levels. The baseline reconstructs every frame. Results are written to
`summary.json` and `summary.csv`.

```text
speedup = framewise calibration time / Track4D-Calib calibration time
```

Use identical matcher settings for the two methods. For quality comparisons,
keep images, view split, preprocessing, initialization, reconstruction resolution,
iterations and seed fixed. Reconstruction training is timed separately.
Report incomplete pose grids alongside their elapsed time.

Identify hardware and configuration with each result. Matching-cost estimates
must be labeled separately from measured end-to-end speedup.
