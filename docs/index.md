# dip-fmm documentation

`dip-fmm` is a C++20 reference implementation of Cartesian fast-multipole
operators for point-dipole interactions.  The magnetic field $H$ is its primary
output; scalar-potential evaluation is optional.

```{warning}
The CPU mathematical operators and complete uniform tree geometry are
implemented, but the tree is not yet connected into an end-to-end FMM solver.
Use direct P2P for complete evaluations and the expansion operators for
experimentation and validation.
```

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
