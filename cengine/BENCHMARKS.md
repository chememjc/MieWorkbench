# C engine benchmark (bench_engines.py)

- git: `293ceb1`  rays=1e+06 resolution=2048 nlambda=9
- host: RTX 4090 Laptop (CUDA 13), 32-core CPU
- python engine: --workers auto (process-sharded trace, torch-CUDA gather)

| scene | py wall | py trace | py gather | C wall | C trace | C gather | wall speedup | stage speedup |
|---|---|---|---|---|---|---|---|---|
| beam_expander | 39.5s | 36.2s | 0.0s | 7.0s | 4.83s | 0.06s | **5.6x** | 7.4x |
| camera_triplet | 237.9s | 235.1s | 0.0s | 25.1s | 23.25s | 0.01s | **9.5x** | 10.1x |
| czerny_turner | 4.3s | 3.6s | 0.0s | 1.5s | 0.90s | 0.02s | **2.8x** | 3.9x |
| dobsonian | 24.5s | 23.7s | 0.0s | 2.9s | 1.86s | 0.01s | **8.6x** | 12.7x |
| fiber_coupler | 37.9s | 34.6s | 0.0s | 6.8s | 4.18s | 0.01s | **5.6x** | 8.3x |
| ghost_doublet | 77.5s | 74.1s | 0.0s | 10.2s | 7.98s | 0.01s | **7.6x** | 9.3x |
| microscope_objective | 1128.1s | 1123.9s | 0.0s | 45.5s | 41.16s | 1.91s | **24.8x** | 26.1x |
| newtonian | 36.3s | 35.6s | 0.0s | 4.4s | 1.39s | 1.94s | **8.2x** | 10.7x |
| prism_spectrometer | 72.4s | 68.9s | 0.0s | 9.4s | 5.29s | 1.91s | **7.7x** | 9.6x |
| scatter_plate | 49.7s | 46.5s | 0.0s | 7.2s | 3.28s | 1.92s | **6.9x** | 8.9x |
| schmidt_cassegrain | 227.4s | 223.6s | 0.0s | 10.2s | 5.96s | 1.91s | **22.3x** | 28.4x |

Skipped (explicit, no silent caps):

- curved_focal — auto-routes to python
- gaussian_bench — auto-routes to python
- ktp_walkoff — auto-routes to python
- michelson — C run: timeout
- michelson_folded — python baseline: timeout (C wall 3579.6s — speedup > 1.5x)

geometric mean: **8.3x wall**, **10.6x trace+gather** over 11 scenes

Gate (plan): >= 1.5x geometric-mean stage speedup — PASS.

## Michelson supplemental (@2e5 rays)

The two michelson-family scenes are coherent-gather-dominated and exceed
the 5400 s per-engine budget at 1e6 rays (the "timeout" rows above), so
they were re-measured at 2e5 rays, resolution/nlambda unchanged (git
`e14125f`, GPU verified healthy for both engines' runs):

| scene | py wall | py trace | py gather | C wall | C trace | C gather | wall speedup |
|---|---|---|---|---|---|---|---|
| michelson_folded | 4662.4s | 24.4s | 4635.1s | 716.8s | 1.60s | 713.42s | **6.5x** |
| michelson | timeout (>5400s) | — | — | 1122.7s | — | — | **>4.8x** |

michelson's Python baseline exceeds the budget even at 2e5 rays; its C
wall of 1122.7 s bounds the speedup below at >4.8x. Both scenes clear
the >=1.5x gate. (michelson routes to Python under `auto` today —
`extra_detector_faces` is unported — so these rows are `--engine`-forced
measurements of the same geometry.)
