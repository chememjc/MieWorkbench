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
  src/gather.c
  src/gather_nufft.c
  src/trace.c
  vendor/yyjson/yyjson.c
)

target_compile_options(miewb-trace PRIVATE ${MIEWB_C_FLAGS})
target_include_directories(miewb-trace PRIVATE src vendor)
target_link_libraries(miewb-trace PRIVATE OpenMP::OpenMP_C m)

# -rdynamic so backtrace_symbols_fd() resolves our own function names in
# crash dumps (log.c crash_handler)
target_link_options(miewb-trace PRIVATE -rdynamic)

# ---- CUDA (phase D gather) — optional; engine falls back to the CPU
# gather kernel when absent. The system nvcc (/usr/bin, CUDA 11.5) must
# NOT be picked up: default to the CUDA 13 toolkit unless CUDACXX is set.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX}
   AND EXISTS /usr/local/cuda-13/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-13/bin/nvcc)
endif()
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  target_sources(miewb-trace PRIVATE src/cuda/gather.cu)
  # the target predates enable_language, so set its arch property directly
  set_target_properties(miewb-trace PROPERTIES
    CUDA_ARCHITECTURES 89)              # RTX 4090 (SM 8.9)
  target_compile_definitions(miewb-trace PRIVATE MIEWB_HAS_CUDA)
  target_compile_options(miewb-trace PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(miewb-trace PRIVATE CUDA::cudart)
  message(STATUS "CUDA gather enabled (${CMAKE_CUDA_COMPILER})")

  # ---- cuFINUFFT (P1 NUFFT angular-spectrum gather fast path) — OPTIONAL.
  # Auto-detect the build-time-fetched checkout (cengine/vendor/finufft,
  # built by vendor/fetch_finufft.sh); MIEWB_CUFINUFFT overrides with a
  # prebuilt root. Absent => the NUFFT route compiles out and the gather
  # falls through to the tiled/exact kernels (all tests still pass).
  set(_finufft_root "$ENV{MIEWB_CUFINUFFT}")
  if(NOT _finufft_root)
    set(_finufft_root "${CMAKE_CURRENT_SOURCE_DIR}/vendor/finufft")
  endif()
  find_path(CUFINUFFT_INCLUDE_DIR cufinufft.h
    HINTS "${_finufft_root}/include" "${_finufft_root}" NO_DEFAULT_PATH)
  # A self-contained SHARED libcufinufft.so (built by fetch_finufft.sh with
  # FINUFFT_STATIC_LINKING=OFF) sidesteps CUDA relocatable-device-code
  # linking into this C executable — the .so device-links internally.
  find_library(CUFINUFFT_LIB cufinufft
    HINTS "${_finufft_root}/build-cuda" "${_finufft_root}/build"
          "${_finufft_root}/lib" "${_finufft_root}" NO_DEFAULT_PATH)
  if(CUFINUFFT_INCLUDE_DIR AND CUFINUFFT_LIB)
    get_filename_component(_cufinufft_dir "${CUFINUFFT_LIB}" DIRECTORY)
    target_compile_definitions(miewb-trace PRIVATE MIEWB_HAS_CUFINUFFT)
    target_include_directories(miewb-trace PRIVATE ${CUFINUFFT_INCLUDE_DIR})
    target_link_libraries(miewb-trace PRIVATE ${CUFINUFFT_LIB} CUDA::cufft)
    # find the .so at runtime without LD_LIBRARY_PATH
    set_property(TARGET miewb-trace APPEND PROPERTY
      BUILD_RPATH "${_cufinufft_dir}")
    set_property(TARGET miewb-trace APPEND PROPERTY
      INSTALL_RPATH "${_cufinufft_dir}")
    message(STATUS "NUFFT gather enabled (cuFINUFFT: ${CUFINUFFT_LIB})")
  else()
    message(STATUS "cuFINUFFT not found — NUFFT gather route disabled "
                   "(run cengine/vendor/fetch_finufft.sh or set "
                   "MIEWB_CUFINUFFT)")
  endif()
else()
  message(STATUS "CUDA not found — CPU gather only")
endif()

enable_testing()
add_subdirectory(tests)
