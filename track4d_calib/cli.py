"""One public entry point for all declared camera rigs."""

import argparse
import json
from pathlib import Path
import sys

from .config import LEVELS, check_dependencies, load_config, load_dependencies
from .runner import command, run


def build_parser() -> argparse.ArgumentParser:
    """Describe the shared plan, validation and calibration interface."""
    parser = argparse.ArgumentParser(prog="track4d-calib", description="Track4D-Calib: track-driven geometric closure")
    parser.add_argument("action", choices=("plan", "preflight", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--level", choices=LEVELS,
        help="Calibration budget declared in the scene config; defaults to balanced",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Scene directory containing images/ and cameras/intri.yml",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; never overwritten"
    )
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--frames",
        type=int,
        choices=(30, 120),
        help="Override the frame count declared in settings",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config.resolve(), level=args.level, frames=args.frames)
        frames = config["frames"]
        deps = load_dependencies(args.dependencies.resolve())
        source, output = args.source.expanduser().resolve(), args.output.expanduser().absolute()
        if args.gpu < 0:
            raise ValueError("gpu must be nonnegative")
        if args.action == "plan":
            print(
                json.dumps(
                    {
                        "config": config,
                        "command": command(
                            config,
                            source=source,
                            output=output / "calibration",
                            code=output / "code",
                            dependencies=deps,
                            gpu=args.gpu,
                            frames=frames,
                            pace=0.0,
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        missing = check_dependencies(deps)
        if missing:
            raise ValueError("missing dependencies:\n" + "\n".join(missing))
        result = run(
            config,
            source=source,
            output=output,
            dependencies=deps,
            gpu=args.gpu,
            frames=frames,
            pace=0.0,
            preflight_only=args.action == "preflight",
        )
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"track4d-calib: {error}", file=sys.stderr)
        return 1
