# Installation

Use Linux, Python 3.10+, an NVIDIA GPU and a compatible CUDA toolkit/compiler.
The tested environment uses Python 3.10, PyTorch 2.11 and CUDA 12.8.
Install `strace`, FFmpeg and the Python virtual-environment package through your system
package manager.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r dependencies/calibration.txt
python -m pip install -e .
python tools/setup/fetch_dependencies.py --output third_party
```

Build the tracking extension with your CUDA toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
MAX_JOBS=4 FUSED_LOCAL_CORR_TORCH_VERSION=2.11.0 python -m pip install --no-deps --no-build-isolation ./third_party/fused-local-corr
```

Download the model weights from their upstream distributions:

| Model | Location |
| --- | --- |
| [SuperPoint / SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) | `third_party/hloc/third_party/SuperGluePretrainedNetwork/models/weights/` |
| [CoTracker3 offline](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth) | `checkpoints/cotracker3/scaled_offline.pth` |
| [RoMaV2](https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt) | `checkpoints/romav2.pt` |

Copy `configs/dependencies.example.json` to `configs/dependencies.local.json`
and set its five source/weight paths. Relative paths resolve from the JSON
file's directory. Dependencies and checkpoints may live outside this repository.

Run `python -m pip check`, then follow the [quick start](../README.md#quick-start).
Optional reconstruction setup is described in [reconstruction](EVALUATION.md).
