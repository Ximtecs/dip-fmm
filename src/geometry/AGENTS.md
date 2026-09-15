# Geometry implementation

Inherit `../../AGENTS.md` and the public geometry guidance. `primitives/` owns
the exact rectangular-prism and tetrahedron analytical implementations. Preserve
their formulae, singular limits, normalisation, exception behaviour, and source
provenance during structural work.

`primitives/rectangular_prism.cpp` is also the authoritative home for the
prism-averaged monomial `J_beta(d,h)`, exposed as
`rectangular_prism_averaged_monomial` with `cuboid_averaged_monomial` as its
retained flat spelling. That path deliberately keeps its own side-length check
rather than the stricter volume validation used by the exact pair tensors;
preserve that difference and the historical "cuboid" exception wording.

The geometry-pair tensor dispatch belongs to `src/operators/p2p.cpp`, and the
persistent `DenseDirectPlan` belongs to `src/plan/direct/dense.cpp`.
Its portable and oneMKL execution mechanics belong to the internal direct
backends under `src/backend/cpu/direct` and `src/backend/mkl/direct`. Do not
pull either execution concern into this geometry subsystem.
