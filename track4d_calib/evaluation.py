"""Run the common configured 30-frame readout after geometry and input audits pass."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .config import (
    check_dependencies,
    evaluation_settings,
    load_config,
    load_dependencies,
    validate_config,
)
from .runner import digest, write_json
from .verification import verify_calibration


def snapshot_verification(destination: Path) -> None:
    """Keep the evaluator's geometry checks self-contained and versioned."""
    package = Path(__file__).parent
    modules = ["verification.py", "trajectory_verification.py"]
    modules += [f"_engine/runtime/sparse/{name}.py" for name in (
        "camera_motion", "center_trajectory", "motion_verification", "track_partition")]
    destination.mkdir(parents=True)
    (destination / "__init__.py").write_text("from .verification import verify_calibration\n")
    for relative in modules:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / relative, target)


def verify_renderer(python: Path, code: Path, renderer: Path, gpu: int) -> dict:
    """Check the extension actually loaded by the training Python before work."""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(code / "scripts/geometry/nvs_4dgs/runtime_compat"), str(code)]
            ),
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TRACK4D_CALIB_PREBUILT_DGR_SO": str(renderer.resolve()),
            "TRACK4D_CALIB_PREBUILT_DGR_SHA256": digest(renderer),
        }
    )
    script = """import json, os
from pathlib import Path
import torch.utils.cpp_extension as cpp
identity = getattr(cpp.load, '_track4d_calib_prebuilt', None)
assert identity is not None, 'prebuilt renderer startup hook did not load'
extension = cpp.load('diff_gaussian_rasterization', sources=[])
assert Path(extension.__file__).resolve() == Path(os.environ['TRACK4D_CALIB_PREBUILT_DGR_SO']).resolve()
print(json.dumps({'path': str(Path(extension.__file__).resolve()), 'sha256': identity[2], 'passed': True}))
"""
    result = subprocess.run(
        [str(python), "-c", script],
        cwd=code,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Settings override; otherwise use the calibration's saved config",
    )
    parser.add_argument(
        "--canonical-scene", type=Path, help="Required only for external evaluation cameras"
    )
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--fudan-runtime", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--dinov3-src", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=0,
                        help="First source frame in the reconstruction window")
    args = parser.parse_args(argv)
    root = args.run.resolve()
    calibrated = verify_calibration(root)  # Must precede access to any evaluation-only camera.
    if args.frame_start < 0 or args.frame_start + 30 > calibrated["frames"]:
        raise ValueError("reconstruction window lies outside the calibrated sequence")
    saved = json.loads((root / "protocol.json").read_text())
    config = (
        load_config(args.config.resolve())
        if args.config
        else validate_config(saved.get("config", {}))
    )
    settings = evaluation_settings(config)
    if settings["camera_source"] == "external" and args.canonical_scene is None:
        raise ValueError("external evaluation cameras require --canonical-scene")
    settings["calibration"] = {
        key: config[key] for key in ("scene", "static_views", "dynamic_views")
    }
    deps = load_dependencies(args.dependencies.resolve())
    if missing := check_dependencies(deps):
        raise ValueError(f"missing dependencies: {missing}")
    dinov3 = args.dinov3_src.resolve()
    if not (dinov3 / "hubconf.py").is_file():
        raise ValueError("--dinov3-src must contain hubconf.py for the frozen point initializer")
    output = args.output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    try:
        code = output / "code"
        shutil.copytree(
            Path(__file__).parent / "_evaluation",
            code,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        snapshot_verification(code / "runtime/scripts/calibration_verification")
        (code / "third_party").mkdir()
        (code / "third_party/hloc").symlink_to(deps["hloc"], target_is_directory=True)
        (code / "third_party/RoMaV2").symlink_to(
            deps["romav2_src"].parent, target_is_directory=True
        )
        (code / "third_party/dinov3").symlink_to(dinov3, target_is_directory=True)
        # The point initializer resolves this path relative to its code.
        (code / "checkpoints").mkdir()
        (code / "checkpoints/romav2.pt").symlink_to(deps["romav2_weights"])
        write_json(output / "settings.json", settings)
        renderer_sha256 = digest(args.renderer)
        command = [
            sys.executable,
            str(code / "runtime/scripts/entrypoints/evaluate_calibration.py"),
            "--calibration-root",
            str(root / "calibration"),
            "--root",
            str(output / "nvs"),
            "--settings",
            str(output / "settings.json"),
            "--project-root",
            str(code),
            "--backend-dir",
            str(code / "runtime/scripts"),
            "--calib-python",
            sys.executable,
            "--train-python",
            str(args.train_python.absolute()),
            "--fudan-runtime",
            str(args.fudan_runtime.resolve()),
            "--evaluator",
            str(code / "scripts/apps/eval_fudan_checkpoint_dataset.py"),
            "--prebuilt-dgr",
            str(args.renderer.resolve()),
            "--prebuilt-dgr-sha256",
            renderer_sha256,
            "--gpu",
            str(args.gpu),
            "--frame-start",
            str(args.frame_start),
        ]
        if args.canonical_scene is not None:
            command.extend(["--canonical-scene", str(args.canonical_scene.resolve())])
        write_json(
            output / "protocol.json",
            {
                "command": command,
                "status": "frozen_before_evaluation_inputs_and_training",
                "calibration_result_sha256": digest(root / "result.json"),
                "settings": settings,
                "frame_ids": list(range(args.frame_start, args.frame_start + 30)),
                "settings_sha256": digest(output / "settings.json"),
                "code_sha256": {
                    str(p.relative_to(code)): digest(p)
                    for p in sorted(code.rglob("*.py"))
                    if "third_party" not in p.relative_to(code).parts
                },
                "renderer_sha256": renderer_sha256,
                "runtime_train_sha256": digest(args.fudan_runtime / "train.py"),
                "initializer_weights_sha256": digest(deps["romav2_weights"]),
                "dinov3_source_sha256": {
                    str(p.relative_to(dinov3)): digest(p) for p in sorted(dinov3.rglob("*.py"))
                },
            },
        )
        write_json(
            output / "renderer_loaded.json",
            verify_renderer(args.train_python, code, args.renderer, args.gpu),
        )
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": os.pathsep.join([str(code), str(deps["romav2_src"])]),
                "ROMAV2_WEIGHTS": str(deps["romav2_weights"]),
                "PYTHONNOUSERSITE": "1",
            }
        )
        write_json(output / "status.json", {"status": "running"})
        with (output / "evaluation.log").open("x") as log:
            subprocess.run(
                command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        verify_calibration(root)
        result = json.loads((output / "nvs/SUMMARY.json").read_text())
        write_json(output / "result.json", result)
        write_json(output / "status.json", {"status": "complete"})
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
