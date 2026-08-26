# dip-fmm

`dip-fmm` is a C++20 fast multipole method for repeated magnetic-field
evaluation on fixed geometry. It supports real spherical-harmonic (default)
and Cartesian expansions, point dipoles, uniform-cuboid sources and targets,
free-space or fully periodic cubic cells, FP32 and FP64 execution, portable
CPU, oneMKL, hybrid CUDA, full CUDA, and Python bindings. The implementation
uses `G(r) = 1/(4*pi*|r|)` and treats `H = -grad(phi)` as the primary result,
with optional scalar potential output on supported paths.

`UniformFmm` constructs the uniform tree and reusable P2M, M2M, M2L, L2L,
L2P, and exact `list1` P2P operators once. Repeated calls then replace only the
dipole moments. The independent Cartesian reference traversal remains
selectable for validation.

```cpp
cdfmm::UniformFmm fmm(source_positions, target_positions, options);
for (const auto& moments : moment_states) {
    const auto values = fmm.evaluate(moments);
}
```

`options.precision` selects true FP32 or FP64 execution; FP32 is the default.
In FP32 mode all static operators, expansion state, arithmetic, scratch, and
CUDA buffers use FP32, while input positions and moments may still be supplied
as FP64. Python results and coefficient arrays use the selected NumPy dtype.

Python exposes the same spherical default. Assign `"cartesian"` or
`"spherical"` to `options.expansion_basis`, or use the corresponding enum.

Execution can be selected with `options.backend`: `CpuReference`, `CpuStatic`,
`CudaPartial`, or `CudaFull`. `Auto` deliberately selects `CpuStatic`; it can
never substitute an O(N^2) direct calculation for an FMM traversal.
`CudaPartial` (with `CudaM2LP2P` retained as an alias) runs P2M, M2M, L2L, and
L2P on the CPU, applies cached dense M2L
matrices with cuBLAS, and applies the cached sparse list-1 P2P tensor on an
independent CUDA stream. `CudaM2L` remains a compatibility alias. The
separately exposed `cuda_direct_p2p_reference` is the O(N^2) GPU numerical
reference. For changing moments on fixed geometry, `CudaDirectPlan` retains
positions and the self-identity map on the device across evaluations.

## Status

The production plan is static and non-adaptive. Both expansion bases support
point-dipole to point-target FMM on CPU static, oneMKL, CUDA partial, and CUDA
full backends. Both also support uniform-cuboid sources and analytically
volume-averaged cuboid targets. Fully periodic evaluation requires an explicit
cubic cell and currently implements the zero-`k=0` convention on static CPU
and CUDA plans.

Self exclusion uses an explicit target-to-source identity map rather than
coordinate equality. CUDA-full keeps all static operators and coefficient
state resident and, for repeated field evaluation, transfers only changing
moments to the device and the final user-ordered field back. Adaptive trees,
partial/rectangular periodicity, a stable C/Fortran interface, and MagTense
integration remain future work. See the [periodic-boundary
documentation](docs/periodic-boundaries.md) and [roadmap](docs/roadmap.md) for
the precise capability boundary.

## Build and test

After activating the required Conda environment, compiling with any checked-in
preset always uses the same two-command pattern:

```console
cmake --fresh --preset <preset>
cmake --build --preset <preset> -j
```

Available presets are `release`, `dev`, `cuda`, `notebooks`, `magtense`,
`benchmark`, `benchmark-mkl`, `benchmark-all`, and `profile-all`. The default
`release` preset enables CUDA and oneMKL and installs the native library and
Python extension into the active Conda environment. The `cuda`, `notebooks`,
and `magtense` presets also install into that environment. The
[installation guide](docs/installation.md#install-from-scratch)
starts with the complete release installation and then describes feature
variants, every preset, clean rebuilds, and environment recovery.

For example, build and optionally test the installed release configuration
after creating the Conda environment as described in the installation guide:

```console
cmake --fresh --preset release
cmake --build --preset release -j
ctest --preset release
```

## Python

```console
python -m pip install .
python -m pytest python_tests -v
```

```python
import cdfmm

result = cdfmm.p2p_dipole_pair(
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    output="both",
)
print(result)
```

The experimental Python interface exposes direct evaluation, Cartesian
reference operators, both `UniformFmm` expansion bases, uniform-tree
inspection, complete evaluation, plan statistics, and per-node multipole and
local inspection.

## Interactive examples

The development environment includes Matplotlib, JupyterLab, ipykernel, and
ipywidgets for the structured examples under `examples/notebooks/`. Update an
existing environment and start Jupyter from the repository root with:

```console
conda env update -n cdfmm -f environment.yml
conda activate cdfmm
jupyter lab
```

VSCode users can open a notebook and select the `Python (cdfmm)` kernel. The
numbered notebooks progress from exact P2P through individual operators, the
complete tree traversal, CUDA memory, parameter selection, and a spherical
CUDA-full comparison with FMM3D. Focused notebooks separately compare
Cartesian with spherical, FP32 with FP64, and cuboid FMM with direct cuboid
physics. See the [notebook catalogue](examples/notebooks/README.md) and
[examples guide](docs/examples.md).

These examples are interactive learning and validation tools, not replacements
for the automated C++ and Python tests. Matplotlib and Jupyter remain optional
example dependencies and are not required by the core Python package.

## Documentation

- [Overview and current limitations](docs/overview.md)
- [Installation and building](docs/installation.md)
- [Getting started](docs/getting-started.md)
- [Mathematical formulation](docs/math.md)
- [Real spherical-harmonic expansions](docs/spherical-expansions.md)
- [Static-geometry architecture](docs/static-architecture.md)
- [Execution backends and data transfers](docs/backends.md)
- [Uniform tree](docs/uniform-tree.md)
- [Operator reference](docs/operators.md)
- [Validation and testing](docs/validation.md)
- [Benchmarks and output schema](docs/benchmarks.md)
- [Roadmap](docs/roadmap.md)

The Sphinx/MyST site integrates Doxygen API output through Breathe and is ready
for Read the Docs.  Local documentation build instructions are in the
[installation guide](docs/installation.md).

## Combined performance benchmarks

OpenMP parallel execution, built-in setup/evaluation phase timings, CUDA direct
P2P, and runtime-selectable portable/oneMKL static M2L are available in one
benchmark executable:

```console
cmake --fresh --preset benchmark-all
cmake --build --preset benchmark-all
python benchmarks/run_benchmarks.py --profile standard --max-threads 8 \
  --executable build-bench-all/benchmarks/benchmark_uniform_fmm
```

See the [benchmark guide](docs/benchmarks.md) for profiles, CSV fields, figures,
setup/evaluation separation, and thread-safety constraints.
