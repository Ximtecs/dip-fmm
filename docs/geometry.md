# Geometry: points, prisms and tetrahedra

`UniformFmm` separates the **physical geometry** of every source and target
from the **model** used at each stage of the algorithm. Point dipoles,
axis-aligned rectangular prisms and tetrahedra are supported as sources and as
targets; the exact near field supports all nine combinations, and the far
field supports every finite body independently of the other side. The
formulas are in [Finite geometry](math/finite-geometry.md); tutorial 2 walks
through the API.

## Records

| Type | Fields | Meaning |
|---|---|---|
| position (`Vec3`, or a row of an `(N, 3)` array) | representative position | a point, a prism centre, or a tetrahedron representative (normally the centroid) |
| `RectangularPrism(hx, hy, hz)` | **full** side lengths | axis-aligned prism; the physical extents are `±hx/2`, `±hy/2`, `±hz/2` |
| `Tetrahedron(vertices)` | four vertices **relative to the representative** | one record can be shared by a regular mesh; `volume`, `signed_volume`, `centroid_offset` are available |

Records are passed through the options once for all bodies (a one-element
list) or once per body in original user order; the plan applies its own
permutations. Runtime input is always the total moment $m = V M$ of each body,
in A m$^2$; magnetisation is never inferred and no volume scaling is
performed. The flat spelling `CuboidSize` is a C++ compatibility alias of
`RectangularPrism`.

## Geometry and model selectors

```python
options.source_geometry            # SourceGeometry.POINT_DIPOLE | RECTANGULAR_PRISM | TETRAHEDRON
options.target_geometry            # TargetGeometry.POINT | RECTANGULAR_PRISM | TETRAHEDRON
options.source_sizes, options.target_sizes             # [RectangularPrism, ...]
options.source_tetrahedra, options.target_tetrahedra   # [Tetrahedron, ...]

options.near_field_source_model    # SourceModel.EXACT_GEOMETRY (default) | POINT_DIPOLE
options.near_field_target_model    # TargetModel.EXACT_GEOMETRY (default) | POINT
options.far_field_source_model     # P2M: EXACT_GEOMETRY (default) | POINT_DIPOLE
options.far_field_target_model     # L2P: EXACT_GEOMETRY (default) | POINT
```

The four model selectors are independent. With the defaults, every `list1`
pair uses the exact body-to-body tensor and the far field uses the exact
finite P2M and volume-averaged L2P operators. Setting a far-field model to its
point variant substitutes the point-dipole P2M or the centre-sampled L2P while
the near field stays exact; this is the controlled way to measure what the
finite far-field operators contribute for a given mesh (for centred cubes they
cannot help below order five, see [Finite geometry](math/finite-geometry.md)).
Setting a near-field model to its point variant makes the near field treat the
bodies as points too.

| Source | Target | Near-field pair tensor | Far-field endpoint operators |
|---|---|---|---|
| point | point | point dipole formula (recomputed from positions or stored) | point P2M, point L2P |
| prism | point | exact prism point field (MagTense `getN_prism_3D` adaptation) | prism P2M, point L2P |
| point | prism | exact volume-averaged point field | point P2M, prism L2P |
| prism | prism | exact prism-to-prism average (`getAvgN_prism_3D` adaptation) | prism P2M, prism L2P |
| tetrahedron | point | analytical polyhedron point field | tetrahedron P2M, point L2P |
| point | tetrahedron | exact volume average over the tetrahedron | point P2M, tetrahedron L2P |
| tetrahedron | tetrahedron | analytical surface (Galerkin) integral | tetrahedron P2M, tetrahedron L2P |
| prism | tetrahedron, tetrahedron | prism | analytical surface integral (twelve prism triangles, four tetrahedron faces) | the corresponding finite operators |

Every pair executes on the portable CPU, oneMKL, `CudaPartial` and `CudaFull`
backends in FP32 and FP64 through every stored-tensor packing
([Execution backends](backends.md)); the enforcing test is
`tests/test_p2p_geometry_matrix.cpp`.

## Self interaction and identity

| Source | Target coincides with the source | Identity map | Near-field self pair |
|---|---|---|---|
| point dipole | yes | supplied | omitted (singular) |
| point dipole | yes | absent | evaluated and singular: supply the map |
| prism or tetrahedron | yes | either | kept: the finite demagnetising self field, $-M/3$ for a cube |

Identity is a matter of indices, never of coordinate equality. The map can be
fixed in the options (`fixed_target_source_indices`, required by some CUDA
packings) or passed to each `evaluate` call; entry `i` is the original source
index of target `i`, or `-1`.

## Root box and finite bodies

The automatic root cube encloses the **bodies**, not only their centres: for
prisms and tetrahedra the bounds include each body's extent. An explicit root
must likewise contain every body, and a periodic cell must contain every
representative position ([Caching and periodicity](caching-and-periodicity.md)).
A regular lattice whose spacing divides the root box puts bodies exactly on
octree boundaries, which is the slowest-converging position for the far-field
expansion; offsetting the root or the lattice restores ordinary convergence
(tutorials 2 and 5).

## The dense reference

`DenseDirectPlan` (and `CudaDenseDirectPlan`) accepts the same geometry
records, model selectors and identity map and evaluates every pair exactly,
storing the six symmetric tensor components as dense $N_t \times N_s$
matrices. It is the reference that every finite-geometry claim in the tests
and tutorials is checked against, and an exact $O(N^2)$ evaluator in its own
right for small problems. `direct_p2p_reference` and
`cuda_direct_p2p_reference` are the corresponding point-dipole references
without a retained plan.

## Boundaries

- Prisms are axis aligned; rotated prisms are not represented.
- Magnetisation is uniform within a body.
- The `CpuReference` backend accepts point geometry only.
- Grain generation and mesh refinement are outside the library: it consumes
  the integration elements a mesh produces.
