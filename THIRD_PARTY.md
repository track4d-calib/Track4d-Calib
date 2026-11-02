# Third-party components

Dependencies and model weights retain their upstream terms.

| Component | Source and license |
| --- | --- |
| HLoc | [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization), Apache-2.0; included [notice](dependencies/HLOC_LICENSE) |
| RoMaV2 | [RoMaV2](https://github.com/Parskatt/RoMaV2), MIT |
| Fused Local Correlation | [fused-local-corr](https://github.com/Parskatt/fused-local-corr), MIT; built against the installed PyTorch and CUDA toolkit |
| CoTracker | [CoTracker](https://github.com/facebookresearch/co-tracker), CC BY-NC 4.0 |
| DINOv3 | [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| COLMAP / PyCOLMAP | [COLMAP](https://github.com/colmap/colmap), BSD-3-Clause |
| 4DGS (optional) | [Fudan 4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting); see its upstream license and Gaussian Splatting notices |
| SuperGlue | [SuperGlue license](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE) |

`tools/setup/patches/hloc.patch` modifies HLoc's feature and matcher interfaces for
persistent workers. The accompanying HLoc notice applies to this patch.
Pinned source URLs, revisions and roles are declared in
[dependencies/third-party.json](dependencies/third-party.json). The fetcher stores checkouts under `third_party/`.

The optional 4DGS evaluation backend and its compiled rasterizer are installed
separately; see [evaluation setup](docs/EVALUATION.md). Python package dependencies
are declared in `pyproject.toml`; the tested calibration environment is pinned in
`dependencies/calibration.txt`. PyTorch and torchvision are installed for the
chosen CUDA runtime. Upstream code and weights retain their
respective licenses.

`tools/setup/patches/fudan.patch` adapts the pinned Fudan backend's timestamp and
intrinsic loading and exposes temporal initialization as a setting. Its upstream
MIT and Gaussian Splatting notices remain in the external checkout. The patch
does not vendor the backend or its compiled extensions into Track4D-Calib.
