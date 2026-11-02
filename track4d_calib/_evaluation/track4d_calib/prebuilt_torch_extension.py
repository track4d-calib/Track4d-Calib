from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_extension_module(name: str, path: Path) -> Any:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prebuilt extension {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def install_prebuilt_extension_override(
    *,
    extension_name: str,
    binary_path: Path,
    expected_sha256: str,
    cpp_extension_module: Any | None = None,
    module_loader: Callable[[str, Path], Any] | None = None,
) -> dict[str, Any]:
    binary = Path(binary_path).resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"prebuilt torch extension not found: {binary}")
    observed_sha256 = _sha256_file(binary)
    if observed_sha256 != str(expected_sha256).lower():
        raise ValueError(
            "prebuilt torch extension SHA256 mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}, path={binary}"
        )

    if cpp_extension_module is None:
        import torch.utils.cpp_extension as cpp_extension_module

    loader = module_loader or _load_extension_module
    original_load = cpp_extension_module.load
    installed = getattr(original_load, "_track4d_calib_prebuilt", None)
    identity = (str(extension_name), str(binary), observed_sha256)
    if installed is not None:
        if installed != identity:
            raise RuntimeError(
                "a different prebuilt torch extension override is already installed"
            )
    else:
        def scoped_load(name: str, *args, **kwargs):
            if str(name) == str(extension_name):
                return loader(str(extension_name), binary)
            return original_load(name, *args, **kwargs)

        scoped_load._track4d_calib_prebuilt = identity
        cpp_extension_module.load = scoped_load

    return {
        "extension_name": str(extension_name),
        "path": str(binary),
        "sha256": observed_sha256,
        "size_bytes": binary.stat().st_size,
    }


def install_prebuilt_extension_from_environment() -> dict[str, Any] | None:
    path = os.environ.get("TRACK4D_CALIB_PREBUILT_DGR_SO", "").strip()
    sha256 = os.environ.get("TRACK4D_CALIB_PREBUILT_DGR_SHA256", "").strip()
    if not path and not sha256:
        return None
    if not path or not sha256:
        raise RuntimeError(
            "TRACK4D_CALIB_PREBUILT_DGR_SO and TRACK4D_CALIB_PREBUILT_DGR_SHA256 "
            "must be set together"
        )
    return install_prebuilt_extension_override(
        extension_name="diff_gaussian_rasterization",
        binary_path=Path(path),
        expected_sha256=sha256,
    )
