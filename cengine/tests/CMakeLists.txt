# C-side unit tests (ctest): golden-value checks for the kernel headers.
# These mirror the Python invariants in scripts/raytracer/tests/
# test_kernels.py — same physics, same tolerances.
add_executable(test_fresnel_gold test_fresnel_gold.c)
target_include_directories(test_fresnel_gold PRIVATE ../src)
target_compile_options(test_fresnel_gold PRIVATE ${MIEWB_C_FLAGS})
target_link_libraries(test_fresnel_gold PRIVATE m)
add_test(NAME fresnel_gold COMMAND test_fresnel_gold)

add_executable(tool_surf_roots tool_surf_roots.c)
target_include_directories(tool_surf_roots PRIVATE ../src)
target_compile_options(tool_surf_roots PRIVATE ${MIEWB_C_FLAGS})
target_link_libraries(tool_surf_roots PRIVATE m)

add_executable(test_rng test_rng.c)
target_include_directories(test_rng PRIVATE ../src)
target_compile_options(test_rng PRIVATE ${MIEWB_C_FLAGS})
target_link_libraries(test_rng PRIVATE m)
add_test(NAME rng COMMAND test_rng)
