#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec "${PYTHON:-python}" -m track4d_calib run --config configs/huahua.json --level "${LEVEL:-quality}" "$@"
