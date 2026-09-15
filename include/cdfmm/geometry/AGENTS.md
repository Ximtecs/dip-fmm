# Geometry public interfaces

Inherit `../../../AGENTS.md` and `../AGENTS.md`. `models.hpp` owns source and
target evaluation choices; `primitives/` owns physical point, rectangular-prism,
and tetrahedron integration geometry. Geometry may depend on `math` and `core`,
never plans, FMM orchestration, CUDA, or vendor execution libraries.

Point geometry is represented by its `Vec3` representative and the model enums;
do not add a marker type solely for directory symmetry. Keep exact analytical
prism and tetrahedron declarations with their physical primitives.

`primitives/rectangular_prism.hpp` owns the `RectangularPrism` record, its
legacy `CuboidSize` spelling, and the prism-averaged monomial. `cdfmm/cuboid.hpp`
and `cdfmm/rectangular_prism.hpp` are public compatibility façades over it and
declare nothing of their own.
