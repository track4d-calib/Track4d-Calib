#!/usr/bin/env python3
"""Fetch optional or calibration dependencies without replacing existing clones."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def verify_checkout(destination, source, root):
    revision = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if revision != source["revision"]:
        raise ValueError(f"{destination}: expected revision {source['revision']}, got {revision}")
    expected = json.loads((root / "dependencies" / source["source_manifest"]).read_text())["files"]
    mismatches = [rel for rel, digest in expected.items()
                  if not (destination / rel).is_file()
                  or hashlib.sha256((destination / rel).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise ValueError(f"{destination}: source differs from dependency lock: {mismatches}")
    return len(expected)


def fetch_dependency(name, source, output, root):
    destination = output / name
    if destination.is_symlink():
        raise FileExistsError(f"dependency destination is a symlink: {destination}")
    if destination.exists():
        count = verify_checkout(destination, source, root)
        print(f"Verified existing {name}: {source['revision']}, {count} source files")
        return
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=output) as temporary:
        checkout = Path(temporary) / name
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        source["url"], str(checkout)], check=True, timeout=600)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", source["revision"]],
                       check=True, timeout=600)
        if name == "hloc":
            subprocess.run(["git", "-C", str(checkout), "submodule", "update", "--init", "--depth", "1",
                            "third_party/SuperGluePretrainedNetwork"], check=True, timeout=600)
            subprocess.run(["git", "-C", str(checkout), "apply", str(root / "tools/setup/patches/hloc.patch")],
                           check=True, timeout=60)
        for patch in source.get("patches", []):
            subprocess.run(["git", "-C", str(checkout), "apply", str(root / patch)],
                           check=True, timeout=60)
        count = verify_checkout(checkout, source, root)
        checkout.rename(destination)
    print(f"Verified {name}: {source['revision']}, {count} source files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("third_party"))
    parser.add_argument("--component", choices=("calibration", "4dgs"), default="calibration",
                        help="Fetch calibration dependencies or the optional external renderer")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().absolute()
    registry = json.loads((root / "dependencies/third-party.json").read_text())
    sources = (registry["sources"] if args.component == "calibration"
               else {"4dgs": registry["optional"]["4dgs"]})
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        fetch_dependency(name, source, output, root)
    print("Sources ready. Install their dependencies and obtain model weights as described in README.md.")


if __name__ == "__main__":
    main()
