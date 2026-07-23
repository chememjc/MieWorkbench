#!/usr/bin/env bash
# One-shot build for the MieWorkbench C engine.
#   ./build.sh            release build into build/
#   ./build.sh test       build + run the C unit tests (ctest)
#   ./build.sh clean      remove the build directory
set -euo pipefail
cd "$(dirname "$0")"

# Machine paths (MIEWB_NVCC, MIEWB_CUDA_ARCH, ...) from <repo>/miewb.env —
# only-if-unset, so an already-exported env var still wins. Tolerate a
# missing/absent file (older checkouts, or a machine with no miewb.env yet)
# and don't let it trip set -e.
MIEWB_ENV_QUIET=1
source ../scripts/miewb_env.sh || true

if [[ "${1:-}" == "clean" ]]; then
    rm -rf build
    echo "cleaned."
    exit 0
fi

cmake -G Ninja -B build \
    ${MIEWB_NVCC:+-DCMAKE_CUDA_COMPILER="$MIEWB_NVCC"} \
    ${MIEWB_CUDA_ARCH:+-DMIEWB_CUDA_ARCH="$MIEWB_CUDA_ARCH"} \
    >/dev/null
ninja -C build

if [[ "${1:-}" == "test" ]]; then
    ctest --test-dir build --output-on-failure
fi

echo "built: $(realpath build/miewb-trace)"
