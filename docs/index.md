# dip-fmm documentation

`dip-fmm` is a C++20 reference implementation of Cartesian fast-multipole
operators for point-dipole interactions.  The magnetic field $H$ is its primary
output; scalar-potential evaluation is optional.

```{warning}
The CPU mathematical operators, complete uniform tree geometry, and reference
P2M/M2M upward pass are implemented, but there is not yet an end-to-end FMM
solver. Use direct P2P for complete field evaluations and the upward pass for
multipole inspection and validation.
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
