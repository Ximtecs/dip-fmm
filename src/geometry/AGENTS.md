# Geometry implementation

Inherit `../../AGENTS.md` and the public geometry guidance. `primitives/` owns
the exact rectangular-prism and tetrahedron analytical implementations. Preserve
their formulae, singular limits, normalisation, exception behaviour, and source
provenance during structural work.

Dense-direct pair mathematics remains in the transitional `cuboid.cpp`, while
the persistent `DenseDirectPlan` now belongs to `src/plan/direct/dense.cpp`.
Its portable and oneMKL execution mechanics belong to the internal direct
backends under `src/backend/cpu/direct` and `src/backend/mkl/direct`. Do not
pull either execution concern into this geometry subsystem.
