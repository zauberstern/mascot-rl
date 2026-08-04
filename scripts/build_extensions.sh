#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/env_polaris.sh"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
python setup_extensions.py build_ext --inplace -j"${CMAKE_BUILD_PARALLEL_LEVEL}"
echo "[mascotrl] Extensions built."
