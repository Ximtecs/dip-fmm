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
|-- run_benchmarks.py                 profile runner and result handling
|-- run_p2p_sweep.py                  representation sweep
`-- run_high_occupancy_p2p.py         dense-leaf CUDA study
```

Future benchmark directories are `direct/`, `p2p/`, `far_field/`, and `fmm/`
once there are enough cohesive drivers to justify them.

## Rules

- Keep setup, one-time packing/upload, repeated evaluation, and transfer timing
  distinct. Static-plan reuse is part of the measured contract.
- Record precision, geometry, expansion, backend, packing, thread/device, and
  problem-size configuration needed to reproduce a result.
- Preserve established CSV/result schemas unless a migration is explicit.
- Do not make benchmark-only operator mathematics. Exercise production plans
  or clearly labelled reference/experimental representations.
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
```

Use `profile-all` for NVTX-enabled profiling. See `docs/benchmarks.md` and
`docs/profiling.md` before producing or interpreting result artefacts.
