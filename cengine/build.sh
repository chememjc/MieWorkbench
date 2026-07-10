#!/usr/bin/env bash
# One-shot build for the MieWorkbench C engine.
#   ./build.sh            release build into build/
#   ./build.sh test       build + run the C unit tests (ctest)
#   ./build.sh clean      remove the build directory
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
    rm -rf build
    echo "cleaned."
    exit 0
fi

cmake -G Ninja -B build >/dev/null
ninja -C build

if [[ "${1:-}" == "test" ]]; then
    ctest --test-dir build --output-on-failure
fi

echo "built: $(realpath build/miewb-trace)"
