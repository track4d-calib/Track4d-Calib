# Tools

Run commands from the repository root.

| Directory | Purpose | Main entry points |
| --- | --- | --- |
| [setup](setup/) | Install pinned external dependencies | `fetch_dependencies.py` |
| [data](data/) | Decode camera videos into calibration images | `extract_videos.py` |
| [calibration](calibration/) | Calibrate the HUAHUA example | `demo_huahua.sh` |
| [benchmark](benchmark/) | Compare runtime and reconstruction quality | `benchmark_calibration.py`, `benchmark_framewise.py`, `plot_speed_accuracy.py` |
| [reconstruction](reconstruction/) | Export cameras and evaluate reconstruction | `export_reconstruction.py`, `run_experiment.py` |
| [visualization](visualization/) | Render and view camera trajectories | `visualize_camera_array.py`, `serve_visualizations.py` |
| [checks](checks/) | Validate source, configuration and distributions | `check.py`, `check_distribution.py` |
