# dip-fmm

`dip-fmm` (`cdfmm`) is a C++20 fast multipole method for the magnetic field
of many dipoles on **fixed geometry**: the tree, the interaction topology and
every operator are built once, and each new magnetisation state is then
evaluated with the retained plan. Its primary result is the magnetic field
$H = -\nabla\phi$ of the Laplace kernel $G(r) = 1/(4\pi|r|)$; the scalar
potential is available on supported paths.

## What it does

- **Sources and targets**: point dipoles, axis-aligned rectangular prisms and
  tetrahedra, in any of the nine combinations, with the **exact** analytical
  interaction tensors of the finite bodies in the near field (including each
  body's own demagnetising field) and finite or point far-field operators.
- **Expansions**: real spherical harmonics by default ($(p+1)^2$
  coefficients) and a complete independent Cartesian Taylor formulation.
- **Trees**: a complete Morton-sorted uniform octree, or a geometry-only
  adaptive octree feeding the same static plan.
- **Precision**: true FP32 or FP64 execution, operators, state and device
  buffers included.
- **Backends**: portable CPU, oneMKL for the M2L stage, a hybrid CUDA backend
  and a fully device-resident CUDA backend, plus exact direct CPU and CUDA
  plans as references and for small problems.
- **Periodicity**: fully periodic cubic cells with the zero-$k$ convention.
- **Reuse**: a validated persistent cache of the operator bank and geometry
  plans across processes.
- **Interfaces**: C++, Python (pybind11), an unconditional C ABI and an
  optional Fortran wrapper; C++ and Python tests, six tutorials, and
  benchmark drivers.

## Where to start

| You want to | Read |
|---|---|
| build and install | [Installation](installation.md) |
| run a first evaluation and understand the plan lifecycle | [Getting started](getting-started.md), then tutorial 1 |
| use prisms or tetrahedra | [Geometry](geometry.md), tutorial 2 |
| choose a backend, precision or execution option | [Execution backends](backends.md), tutorial 3 |
| reuse plans across processes or use periodic cells | [Caching and periodicity](caching-and-periodicity.md), tutorial 4 |
| pick the order and depth, or use an adaptive tree | [Trees and parameter selection](trees-and-parameter-selection.md), tutorial 5 |
| call the library from C or Fortran | [C and Fortran](c-and-fortran.md) |
| see the notebooks and C++ examples | [Examples and tutorials](examples.md) |
| measure or profile | [Benchmarks and profiling](benchmarks.md) |
| understand or extend the code | [Architecture](architecture.md), [API reference](api.rst) |
| check a sign, a normalisation or a formula | [Mathematical conventions](math/conventions.md), [Spherical expansions](math/spherical-expansions.md), [Finite geometry](math/finite-geometry.md) |

## Capability boundary

`UniformTree` is complete and uniform; `AdaptiveTree` provides the compact
non-uniform topology path for the same static plan. `CudaFull` is the
field-only device-resident path; the CPU and hybrid backends also return the
scalar potential. Periodicity means an explicit cubic cell with the zero-$k$
convention on the static CPU and CUDA plans; partial periodicity and
rectangular cells are not implemented. Prisms are axis aligned and bodies are
uniformly magnetised. There is no runtime MagTense dependency: the prism
formulas are analytical adaptations of the MagTense conventions, validated
against MagTense in `examples/validation/`. The C ABI is unconditional; the
Fortran wrapper and its smoke test are optional
(`CDFMM_BUILD_FORTRAN_INTERFACE`). Grain generation, mesh refinement, a
MagTense runtime backend, and Windows/MSVC or CUDA hosted CI are future work,
not current promises.

```{toctree}
:maxdepth: 1
:caption: User guide

installation
getting-started
geometry
backends
caching-and-periodicity
trees-and-parameter-selection
examples
c-and-fortran
benchmarks
```

```{toctree}
:maxdepth: 1
:caption: Mathematics

math/conventions
math/spherical-expansions
math/finite-geometry
```

```{toctree}
:maxdepth: 1
:caption: Developer

architecture
api
```
