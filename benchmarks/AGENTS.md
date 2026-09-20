# Benchmark guidance

Inherit `../AGENTS.md`. Benchmarks measure performance and resource behaviour;
they supplement, but never replace, correctness tests.

## Current structure

```text
benchmarks/
|-- CMakeLists.txt
|-- benchmark_p2p.cpp                 P2P representations/backends
|-- benchmark_uniform_fmm.cpp         setup and repeated FMM evaluation
|-- benchmark_cache_initialisation.cpp cache construction/reuse
|-- benchmark_dense_direct_construction.cpp
|                                     dense all-to-all construction phases
|-- benchmark_operator_representation.cpp
|                                     precomputed versus procedural exact
|                                     operators (P2P/P2M/L2P, hot and
|                                     streaming, construction separately)
|-- benchmark_operator_representation_cuda.{hpp,cu}
|                                     its procedural prism P2P device kernel
|-- run_benchmarks.py                 profile runner and result handling
|-- run_construction_matrix.py        cold static-plan construction matrix
|                                     (per-phase timings, plan bytes, peak
|                                     resident set; resumable)
|-- run_dense_construction_matrix.py  dense construction matrix, one process
|                                     per row
|-- run_p2p_sweep.py                  representation sweep
|-- run_p2p_packing_matrix.py         geometry/backend/packing matrix
|-- run_operator_representation.py    representation matrix (resumable)
|-- analyse_operator_representation.py
|                                     ratios, retained bytes and the
|                                     amortisation break-even
|-- run_high_occupancy_p2p.py         dense-leaf CUDA study
|-- run_phase3d_regression.py,        internal cross-backend regression and
|   run_phase3d_startup.py,           policy matrix, startup decomposition,
|   analyse_phase3d_regression.py     and their analyser
|-- baselines/phase3d/                retained engineering baseline of the
|                                     three Phase-3D drivers (not an Article1
|                                     benchmark; see its README)
`-- external/fmm3d/                   FMM3D adapters, pinned installer and the
                                      exploratory comparison notebook
```

Every driver, runner and forceable alternative here is retained for the
Article1 benchmark campaign that follows the implementation freeze; prune
benchmark code only after that campaign. Future benchmark directories are
`direct/`, `p2p/`, `far_field/`, and `fmm/` once there are enough cohesive
drivers to justify them.

## Rules

- Keep setup, one-time packing/upload, repeated evaluation, and transfer timing
  distinct. Static-plan reuse is part of the measured contract. A construction
  measurement disables the geometry cache, so every row is a cold build; a
  cache measurement says so explicitly.
- Finite bodies default to point far-field models so that a backend or packing
  comparison isolates the near field. Use `--far-field-model exact` when the
  finite P2M/L2P operators are what is being measured; without it they are
  never built.
- Record precision, geometry, expansion, backend, packing, thread/device, and
  problem-size configuration needed to reproduce a result.
- Preserve established CSV/result schemas unless a migration is explicit.
- Do not make benchmark-only operator mathematics. Exercise production plans
  or clearly labelled reference/experimental representations. A representation
  study that needs a formula in another precision or on the device shares the
  production kernel (as `benchmark_operator_representation_cuda.cu` shares
  `src/geometry/primitives/rectangular_prism_point_kernel.hpp`); it does not
  copy the mathematics.
- A benchmark may retain a measured-losing representation when it is useful
  evidence for a paper and is isolated from production execution. Say which
  one it is and where the measurement is recorded.
- CUDA decomposition comparisons must check launch/transfer/allocation counts,
  synchronisation and near/far overlap, not elapsed time alone.

## Build and run

```bash
cmake --fresh --preset benchmark
cmake --build --preset benchmark -j

# oneMKL or combined CPU/CUDA/oneMKL when available
cmake --fresh --preset benchmark-mkl
cmake --build --preset benchmark-mkl -j
cmake --fresh --preset benchmark-all
cmake --build --preset benchmark-all -j

python benchmarks/run_benchmarks.py --help
python benchmarks/run_p2p_sweep.py --help
python benchmarks/run_operator_representation.py --help
```

Use `profile-all` for NVTX-enabled profiling. See `docs/benchmarks.md` and
the profiling section of `docs/benchmarks.md` before producing or interpreting result artefacts.
