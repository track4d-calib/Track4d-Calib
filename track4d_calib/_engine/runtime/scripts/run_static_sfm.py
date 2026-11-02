#!/usr/bin/env python3
"""Instrument the declared static SfM entry without changing its arguments."""

import argparse
from pathlib import Path
import runpy
import sys


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sfm-runner", type=Path, required=True)
    parser.add_argument("--hloc-root", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    for source in (options.sfm_runner.resolve().parents[1], options.hloc_root.resolve()):
        sys.path.insert(0, str(source))
    from gpu_timing import install_hloc
    install_hloc()
    sys.argv = [str(options.sfm_runner.resolve()), *remaining]
    runpy.run_path(str(options.sfm_runner.resolve()), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
