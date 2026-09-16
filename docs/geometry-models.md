# Geometry and evaluation models

`UniformFmmOptions` separates physical geometry from the approximation used in
each FMM stage. Sources use `SourceGeometry` (`POINT_DIPOLE`,
`RECTANGULAR_PRISM`, or `TETRAHEDRON`); targets use `TargetGeometry` (`POINT`,
`RECTANGULAR_PRISM`, or `TETRAHEDRON`). A `RectangularPrism` is axis-aligned
and stores positive full side lengths `hx`, `hy`, and `hz` (the physical
extents are `hx/2`, `hy/2`, and `hz/2`). A `Tetrahedron` stores
four representative-relative vertices; the representative is normally the
centroid. Moments are always total moments, `m = V M`.

The exact rectangular-prism P2P tensors are direct adaptations of MagTense's
`getN_prism_3D` and `getAvgN_prism_3D` formulas (including their `F1`/`F2`
analytic primitives). With MagTense's demagnetization tensor notation `N`,
the runtime total-moment operator is `K = N / Vs`, where `Vs` is the source
prism volume, so the field contraction is applied to `m = Vs M`. CUDA backends
consume these precomputed tensors; prism integration is not performed during
evaluation.

The four model controls are independent:

```python
options.near_field_source_model  # SourceModel.POINT_DIPOLE or EXACT_GEOMETRY
options.near_field_target_model  # TargetModel.POINT or EXACT_GEOMETRY
options.far_field_source_model   # P2M source treatment
options.far_field_target_model   # L2P target treatment
```

Exact near-field P2P supports all nine source/target combinations of point,
rectangular prism, and tetrahedron. The tetrahedron↔tetrahedron,
prism↔tetrahedron, and tetrahedron↔prism tensors are evaluated during plan
construction from the shared polyhedron surface formulation (see
[math](math.md)) and cached for repeated application; finite self
interactions are retained for every finite source.
For tetrahedron point evaluations on a face, the analytical boundary value uses
the MagTense-compatible one-sided limiting convention; edge and vertex
coincidences remain singular.
Far-field P2M and L2P support all source/target records independently, subject
to the selected static backend. Geometry records may be supplied once for all
objects or once per object in original user order; permutations are applied by
the plan.
