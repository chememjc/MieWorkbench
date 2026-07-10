# ============================================================================
# MieWorkbench C engine — miewb-trace
#
# Build:  ./build.sh        (or: cmake -G Ninja -B build && ninja -C build)
# Output: build/miewb-trace
#
# Floating-point policy (plan D6): STRICT. -O3 but no -ffast-math and
# -ffp-contract=off — the physics kernels are gated against the Python
# engine at 1e-12 and FMA contraction changes results. Relaxation happens
# per-file, later, only where parity-proven safe. -g is always on
# (negligible runtime cost; makes every crash backtrace resolvable).
#
# CUDA joins in phase D: enable_language(CUDA), arch 89 (RTX 4090),
# CUDAToolkit_ROOT defaulting to /usr/local/cuda-13 (the system nvcc at
# /usr/bin is CUDA 11.5 — do NOT let CMake pick it up).
# ============================================================================
cmake_minimum_required(VERSION 3.22)
project(miewb-trace LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED COMPONENTS C)

# strict-fp optimization flags (see header comment)
set(MIEWB_C_FLAGS
  -O3
  -march=native
  -g
  -fno-math-errno
  -ffp-contract=off
  -Wall
  -Wextra
  -Wno-unused-parameter
)

add_executable(miewb-trace
  src/main.c
  src/log.c
  src/npyio.c
  src/raybuf.c
  src/request.c
  src/bvh.c
  src/mesh.c
  src/ledger.c
  src/detector.c
  src/trace.c
  vendor/yyjson/yyjson.c
)

target_compile_options(miewb-trace PRIVATE ${MIEWB_C_FLAGS})
target_include_directories(miewb-trace PRIVATE src vendor)
target_link_libraries(miewb-trace PRIVATE OpenMP::OpenMP_C m)

# -rdynamic so backtrace_symbols_fd() resolves our own function names in
# crash dumps (log.c crash_handler)
target_link_options(miewb-trace PRIVATE -rdynamic)

enable_testing()
add_subdirectory(tests)
