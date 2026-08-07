# dip-fmm

`dip-fmm` is a C++20 Cartesian-coordinate fast multipole method project
specialised for point-dipole interactions.  It uses
`G(r) = 1/(4*pi*|r|)` and treats the magnetic field `H = -grad(phi)` as the
primary result, with optional scalar potential output.

## Status

The CPU reference operator layer is implemented: P2M, M2M, M2L, L2L, L2P,
M2P, and direct P2P, together with Taylor-jet Laplace derivatives and validation
helpers.  The repository also contains a complete, non-adaptive,
Morton-sorted uniform octree with source/target permutations, hierarchical
ranges, and `list1`/`list2` interactions.

The operators and tree are **not yet connected into an end-to-end FMM
evaluation**.  Adaptive trees, persistent/static geometry optimisation, CUDA,
and MagTense/Fortran integration are not implemented.  See the
[roadmap](docs/roadmap.md) for the next milestone and later research.

## Build and test

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCDFMM_BUILD_TESTS=ON -DCDFMM_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Examples build by default.  Enable the current P2P benchmark with
`-DCDFMM_BUILD_BENCHMARKS=ON`.

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

The experimental Python interface exposes direct pair/sum evaluation and
uniform-tree inspection.

## Documentation

- [Overview and current limitations](docs/overview.md)
- [Installation and building](docs/installation.md)
- [Getting started](docs/getting-started.md)
- [Mathematical formulation](docs/math.md)
- [Uniform tree](docs/uniform-tree.md)
- [Operator reference](docs/operators.md)
- [Validation and testing](docs/validation.md)
- [Roadmap](docs/roadmap.md)

The Sphinx/MyST site integrates Doxygen API output through Breathe and is ready
for Read the Docs.  Local documentation build instructions are in the
[installation guide](docs/installation.md).
