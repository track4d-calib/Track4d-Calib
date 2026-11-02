# Data format

All calibration presets consume the same scene layout. The scene name and dataset label do
not change file parsing or solver selection.
For dataset presets, archive checks and a first calibration run, follow the
[dataset guide](DATASETS.md).

## Calibration scene

```text
scene/
├── cameras/intri.yml
└── images/
    ├── 00/
    │   ├── 000000.png
    │   └── ...
    └── 80/
        ├── 000000.png
        └── ...
```

| Input | Contract |
| --- | --- |
| Camera IDs | Numeric strings, identical in configuration, directory names and YAML suffixes |
| Camera roster | At least 16 static and 3 dynamic cameras, disjoint and ordered |
| Frames | Contiguous, zero-based, six-digit filenames; the same declared frame range in every camera |
| Synchronization | The same frame ID denotes the same capture time in every camera |
| Images | 8-bit RGB PNG; common width and height matching `resolution` |
| Camera model | Undistorted pinhole; one fixed intrinsic matrix per camera for the sequence |
| Extrinsics | Estimated by calibration; no input pose file is required |

Keep the reference cameras rigidly mounted and synchronize all streams to a
common timeline. Each moving view should overlap multiple static views with
sufficient texture and triangulation baseline. Use exposure times that preserve
trackable features. Keep lens settings fixed throughout the capture, then
undistort and export all views at the configured resolution.

Preserve zero padding in camera IDs. For example, `00` requires `images/00/`, `K_00`,
`D_00`, `H_00` and `W_00`. IDs `0` and `00` cannot be assigned to different cameras. Roster
order affects deterministic camera grouping.

Select the desired synchronized segment before export, then renumber it from zero. Missing
images, asynchronous frame extraction and time-varying zoom are not represented by this
format. Extra cameras outside the configured roster do not participate in calibration.

`track4d-calib preflight` validates image files, dimensions, intrinsic entries and the complete
declared camera/frame grid. It cannot determine temporal synchronization or camera overlap
from filenames.

## Intrinsics

`cameras/intri.yml` uses OpenCV FileStorage YAML. Each declared camera has:

| Entry | Shape | Meaning |
| --- | --- | --- |
| `K_<id>` | 3 × 3 | Finite intrinsic matrix with positive focal lengths and last row `[0, 0, 1]` |
| `D_<id>` | 5 × 1 | Zero distortion coefficients for the undistorted images |
| `W_<id>` | Scalar | Image width in pixels |
| `H_<id>` | Scalar | Image height in pixels |

Image coordinates have their origin at the top left, x to the right and y down. For zero
skew:

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Undistort the images with the calibrated lens model and retain the intrinsic matrix of the
resulting images. Setting distortion coefficients to zero without undistorting the image
does not produce valid input.

Write the file with OpenCV rather than a general YAML serializer, which may omit the OpenCV
matrix tags:

```python
from pathlib import Path
import cv2
import numpy as np

path = Path("capture_01/cameras/intri.yml")
path.parent.mkdir(parents=True, exist_ok=True)
storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
try:
    # intrinsics maps each declared camera ID to its prepared-image K.
    for view, K in intrinsics.items():
        storage.write(f"K_{view}", np.asarray(K, dtype=np.float64))
        storage.write(f"D_{view}", np.zeros((5, 1), dtype=np.float64))
        storage.write(f"W_{view}", int(width))
        storage.write(f"H_{view}", int(height))
finally:
    storage.release()
```

`intrinsics`, `width` and `height` come from your own intrinsic calibration and image
preparation. For example, one camera entry at 1280 × 720 is:

```yaml
%YAML:1.0
---
K_00: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [950., 0., 640., 0., 950., 360., 0., 0., 1.]
D_00: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [0., 0., 0., 0., 0.]
W_00: 1280
H_00: 720
```

Replace the illustrative values with your prepared-image intrinsics and add
every configured camera.

### Crop and resize

For a crop with top-left corner `(x0, y0)`, followed by image scaling `(sx, sy)`, update the
matrix by:

```text
A = [[sx,  0, -sx*x0],
     [ 0, sy, -sy*y0],
     [ 0,  0,      1]]
K_prepared = A @ K_undistorted
```

Update `W`, `H` and the configuration's `resolution` to the final image dimensions. Apply
the same per-camera image transform throughout the sequence. This format stores one `K` per
camera, so frame-dependent crop or zoom must be converted to a consistent image coordinate
system before calibration.

## Output cameras

Dynamic cameras are written to `poses/f<frame:06d>/cameras_pred/{intri,extri}.yml`. The
static rig is at `calibration/runtime_scene/cameras/static/{intri,extri}.yml` in the run
directory.

Both use the OpenCV world-to-camera convention:

```text
X_camera = R @ X_world + T
C_world  = -R.T @ T
```

Camera axes are +x right, +y down and +z forward. `Rot_<id>` is a 3 × 3 rotation matrix;
`T_<id>` is a 3 × 1 translation vector. The intrinsic file follows the input convention.

To obtain a camera-to-world homogeneous transform:

```text
T_world_from_camera = [[R.T, C_world],
                       [0, 0, 0, 1]]
```

The reconstructed world coordinate system has arbitrary SfM origin, orientation and scale. A
positive `center_radius` uses these coordinate units. External metric alignment requires
known geometry or reference cameras.

## External evaluation cameras

For `evaluation.camera_source: external`, pass `--canonical-scene` to the reconstruction
evaluator. That directory contains:

```text
canonical_scene/
└── dataset/
    ├── public/<scene>/cameras/static/
    │   ├── intri.yml
    │   └── extri.yml
    └── private/
        ├── gt_long_poses.npz
        └── evaluation/<scene>/heldout/
            ├── cameras/intri.yml
            └── images/<view>/<frame>.png
```

`gt_long_poses.npz` contains:

| Array | Shape and convention |
| --- | --- |
| `long_views` | `[views]`, Unicode camera IDs in camera-axis order |
| `R` | `[frames, views, 3, 3]`, world-to-camera rotations |
| `T` | `[frames, views, 3]`, world-to-camera translations |

These arrays must load with `allow_pickle=False`. Shared static cameras provide the
similarity alignment between reference cameras and the reconstructed rig. External test
poses are evaluation-only. Test cameras are normally disjoint from calibration cameras;
if their RGB images also participate in calibration, declare them explicitly in
`calibration_test_views`. Their images remain excluded from reconstruction training.

For `evaluation.camera_source: calibrated`, test views are members of the dynamic roster and
must match `calibration_test_views`. Their images participate in calibration and are
excluded from reconstruction training. Their committed poses supply evaluation cameras; no
external camera tree is required.
