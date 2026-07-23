#!/usr/bin/env bash
# ===========================================================================
# fetch_finufft.sh — build-time fetch + build of cuFINUFFT (Apache-2.0),
# the optional NUFFT angular-spectrum gather fast path (P1).
#
# cuFINUFFT is NOT vendored in-repo (a whole CUDA library, unlike the single
# yyjson.c). This script clones the pinned tag and builds the CUDA-only
# static library into cengine/vendor/finufft/build-cuda. The C engine's
# CMake auto-detects that location; MIEWB_CUFINUFFT overrides it with a
# prebuilt root (containing include/ and a build*/libcufinufft.a).
#
# The gather NUFFT route is OPTIONAL: miewb-trace builds and passes every
# test WITHOUT this — the route is simply disabled (falls through to the
# tiled/exact gather).
#
# Requirements: git, CUDA 13 (/usr/local/cuda-13), ninja, and CMake >= 3.24
# (cuFINUFFT's own floor; the system cmake here is 3.22 — pip install cmake
# into a venv, or point $CMAKE at a newer one).
# ===========================================================================
set -euo pipefail

TAG=v2.5.1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HERE/finufft"

# Machine paths (MIEWB_NVCC, MIEWB_CUDA_ARCH, ...) from <repo>/miewb.env —
# only-if-unset, so an already-exported env var still wins. Tolerate a
# missing/absent file and don't let it trip set -e. Path is relative to
# this script's own directory (cengine/vendor/), not the caller's CWD.
MIEWB_ENV_QUIET=1
source "$HERE/../../scripts/miewb_env.sh" || true

CMAKE="${CMAKE:-cmake}"
NVCC="${MIEWB_NVCC:-/usr/local/cuda-13/bin/nvcc}"   # legacy fallback
ARCH="${MIEWB_CUDA_ARCH:-89}"   # legacy fallback: RTX 4090 = SM 8.9

if [ ! -d "$DST/.git" ]; then
  echo "[fetch_finufft] cloning cuFINUFFT $TAG -> $DST"
  git clone --depth 1 --branch "$TAG" \
    https://github.com/flatironinstitute/finufft.git "$DST"
fi

# SHARED (STATIC_LINKING=OFF): a self-contained libcufinufft.so device-links
# its own CUDA code, so it drops cleanly into the C miewb-trace executable
# (a static .a would leak __cudaRegisterLinkedBinary_* device symbols).
echo "[fetch_finufft] configuring CUDA-only shared libcufinufft ($CMAKE)"
"$CMAKE" -G Ninja -B "$DST/build-cuda" -S "$DST" \
  -DFINUFFT_USE_CUDA=ON -DFINUFFT_USE_CPU=OFF \
  -DFINUFFT_STATIC_LINKING=OFF -DFINUFFT_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release

echo "[fetch_finufft] building libcufinufft.so"
ninja -C "$DST/build-cuda" cufinufft

echo "[fetch_finufft] done: $DST/build-cuda/libcufinufft.so"
