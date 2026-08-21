# dip-fmm documentation

`dip-fmm` is a C++20 implementation of Cartesian fast-multipole
operators for point-dipole interactions.  The magnetic field $H$ is its primary
output; scalar-potential evaluation is optional.

The mathematical operators, complete uniform-tree geometry, reusable static
plans, and portable CPU traversal are implemented. Optional oneMKL, partial
CUDA, and full CUDA execution consume the same canonical operators. The tree
is currently non-adaptive.

```{toctree}
:maxdepth: 2
:caption: User guide

overview
static-architecture
backends
installation
precision
getting-started
fmm-overview
math
cartesian-expansions
laplace-derivatives
uniform-tree
operators
validation
examples
benchmarks
profiling
parameter-selection
roadmap
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
