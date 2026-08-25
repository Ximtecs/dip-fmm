# dip-fmm documentation

`dip-fmm` is a C++20 fixed-geometry fast multipole method for magnetic dipole
fields. It provides real spherical-harmonic and Cartesian expansions, reusable
static operators, FP32/FP64 execution, and optional CPU, oneMKL, and CUDA
backends. The magnetic field $H$ is its primary output; scalar potential is
optional on supported paths.

Point dipoles are supported by both bases. Cartesian plans also support
uniform-cuboid sources to point targets. The tree is complete and non-adaptive.

```{toctree}
:maxdepth: 2
:caption: User guide

overview
static-architecture
backends
installation
fortran-interface
precision
getting-started
fmm-overview
math
cartesian-expansions
spherical-expansions
laplace-derivatives
uniform-tree
operators
validation
examples
benchmarks
profiling
parameter-selection
static-p2p
cuda-m2l-performance
roadmap
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
