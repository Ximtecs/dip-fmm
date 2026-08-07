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

The experimental Python interface exposes direct evaluation, every current
expansion operator, Cartesian coefficient ordering, and uniform-tree
inspection.

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
nine notebooks progress from exact P2P through P2M, M2M, M2P, M2L, L2L, and
L2P, then compare complete operator chains and visualise the uniform tree,
including Morton ordering, leaf occupancy, `list1`, and `list2`. See the
[notebook catalogue](examples/notebooks/README.md) for the full sequence.

These examples are interactive learning and validation tools, not replacements
for the automated C++ and Python tests. Matplotlib and Jupyter remain optional
example dependencies and are not required by the core Python package.

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
