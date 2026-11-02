# Camera-array visualization

The viewer combines synchronized image panels with a 3D camera array and
motion trails. Static and dynamic cameras use distinct colors and matching IDs.

## Render a capture

Install Blender 4.5 and FFmpeg with libx264, then run:

```bash
python tools/visualization/visualize_camera_array.py --config configs/dyn-dna-rendering.json --images /path/to/scene/images --poses outputs/scene/poses --static-cameras outputs/scene/calibration/runtime_scene/cameras/static --output outputs/scene_visualization
python tools/visualization/serve_visualizations.py --directory outputs/scene_visualization --bind 0.0.0.0 --port 8080
```

Open `http://<server-ip>:8080/`. Match `--frames` to the available images and
calibrated poses. Use `--preview-only` to check the composition, `--azimuth` and
`--elevation` to select a viewpoint, or `--focus-views <ids> --global-context`
for a camera-group close-up with a full-array locator.

Images and cameras follow the [data format](DATA.md). All input cameras must
share one coordinate system. The default display uses recorded camera motion.
The manifest identifies reference and estimated cameras and any display transforms.
