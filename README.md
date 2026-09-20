# dip-fmm

`dip-fmm` (`cdfmm`) is a C++20 fast multipole method for repeated
magnetic-field evaluation on fixed geometry. The tree, the interaction
topology and every operator are built once; each new magnetisation state is
then evaluated with the retained plan. The kernel is
`G(r) = 1/(4*pi*|r|)` and the primary result is `H = -grad(phi)`, with the
scalar potential available on supported paths.

```cpp
cdfmm::UniformFmm fmm(source_positions, target_positions, options);
for (const auto& moments : moment_states) {
    const auto values = fmm.evaluate(moments);
}
```

- **Geometry**: point dipoles, axis-aligned rectangular prisms and
  tetrahedra as sources and targets in every combination, with exact
  analytical near-field tensors (including each finite body's own
  demagnetising field) and finite or point far-field operators.
- **Expansions**: real spherical harmonics (default) and an independent
  Cartesian Taylor formulation.
- **Trees**: a complete uniform octree or a geometry-only adaptive octree
  feeding the same static plan.
- **Precision**: true FP32 (default) or FP64 execution throughout.
- **Backends**: portable CPU, oneMKL M2L, hybrid CUDA (`CudaPartial`) and
  fully device-resident CUDA (`CudaFull`); exact direct CPU and CUDA plans as
  references. `Auto` resolves to the static CPU plan and never substitutes an
  O(N^2) calculation.
- **Periodicity**: fully periodic cubic cells with the zero-`k` convention.
- **Reuse**: a validated persistent cache of the operator bank and geometry
  plans (`CDFMM_CACHE_DIR`, `CDFMM_DISABLE_CACHE`, `cdfmm-precompute`).
- **Interfaces**: C++, Python, an unconditional C ABI and an optional Fortran
  wrapper.

Self exclusion uses an explicit target-to-source identity map, never
coordinate equality. Partial or rectangular periodicity and a runtime
MagTense backend are future work; the capability boundary is stated in the
[documentation index](docs/index.md).

## Build and test

After activating the Conda environment, every checked-in preset uses the same
two commands:

```console
cmake --fresh --preset <preset>
cmake --build --preset <preset> -j
ctest --preset <preset>
```

Presets are `release` (CUDA + oneMKL, installs into the environment), `dev`
(portable CPU), `cuda`, `notebooks`, `magtense`, `benchmark`, `benchmark-mkl`,
`benchmark-all` and `profile-all`. The
[installation guide](docs/installation.md) starts from the complete release
installation and covers every variant, clean rebuilds and environment
recovery.

## Python

```console
python -m pip install .
python -m pytest python_tests -v
```

```python
import numpy as np, cdfmm

options = cdfmm.UniformFmmOptions()
options.expansion_order = 6
options.tree.max_level = 3
plan = cdfmm.UniformFmm(positions, positions, options)
H = plan.evaluate(moments, target_source_indices=np.arange(len(positions)))["H"]
```

[Getting started](docs/getting-started.md) explains the units, shapes and the
plan lifecycle; the six tutorials under `examples/tutorials/` walk through the
whole Python API (`jupyter lab examples/tutorials`, see
[Examples and tutorials](docs/examples.md)).

## Documentation

| | |
|---|---|
| [Index and capability boundary](docs/index.md) | [Installation](docs/installation.md) |
| [Getting started](docs/getting-started.md) | [Geometry](docs/geometry.md) |
| [Execution backends and precision](docs/backends.md) | [Caching and periodicity](docs/caching-and-periodicity.md) |
| [Trees and parameter selection](docs/trees-and-parameter-selection.md) | [Examples and tutorials](docs/examples.md) |
| [C and Fortran](docs/c-and-fortran.md) | [Benchmarks and profiling](docs/benchmarks.md) |
| [Architecture](docs/architecture.md) | [Mathematical conventions](docs/math/conventions.md) |
| [Spherical expansions](docs/math/spherical-expansions.md) | [Finite geometry](docs/math/finite-geometry.md) |

The Sphinx/MyST site integrates the Doxygen API reference through Breathe:
`sphinx-build -W -b html docs docs/_build/html`.

## Benchmarks

```console
cmake --fresh --preset benchmark-all
cmake --build --preset benchmark-all -j
python benchmarks/run_benchmarks.py --profile standard --max-threads 8 \
  --executable build-bench-all/benchmarks/benchmark_uniform_fmm
```

The [benchmark guide](docs/benchmarks.md) describes every driver, the timing
categories and how to read them.

## Licence

Apache-2.0.
