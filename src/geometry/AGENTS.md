# Geometry implementation

Inherit `../../AGENTS.md` and the public geometry guidance. `primitives/` owns
the exact rectangular-prism and tetrahedron analytical implementations. Preserve
their formulae, singular limits, normalisation, exception behaviour, and source
provenance during structural work.

Dense-direct plans and oneMKL policy remain in the transitional `cuboid.cpp` and
are deferred to the operators/plan/backend refactor. Do not pull them into this
subsystem.
