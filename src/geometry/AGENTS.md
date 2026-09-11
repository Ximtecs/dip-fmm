# Geometry implementation

Inherit `../../AGENTS.md` and the public geometry guidance. `primitives/` owns
the exact rectangular-prism and tetrahedron analytical implementations. Preserve
their formulae, singular limits, normalisation, exception behaviour, and source
provenance during structural work.

Dense-direct pair mathematics remains in the transitional `cuboid.cpp`, while
the persistent `DenseDirectPlan` now belongs to `src/plan/direct/dense.cpp`.
Its portable and oneMKL execution mechanics remain temporarily co-located with
the plan; extracting those backends is a later step. Do not pull either concern
into this geometry subsystem.
