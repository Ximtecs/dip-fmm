# dip-fmm documentation

`dip-fmm` is a C++20 reference implementation of Cartesian fast-multipole
operators for point-dipole interactions.  The magnetic field $H$ is its primary
output; scalar-potential evaluation is optional.

The CPU mathematical operators, complete uniform tree geometry, and functional
reference uniform FMM are implemented. The traversal is a correctness baseline
and deliberately has no adaptive-tree or fixed-geometry optimisation.

```{toctree}
:maxdepth: 2
:caption: User guide

overview
installation
getting-started
fmm-overview
math
cartesian-expansions
laplace-derivatives
uniform-tree
operators
validation
examples
roadmap
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
