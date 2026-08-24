# FMM overview

`UniformFmm` is a fixed-geometry evaluator for many magnetic-moment states.
Its central design rule is to move geometry-dependent work into construction
whenever practical and retain only typed linear operator application in the
repeated path.

```text
fixed source and target geometry
    -> uniform tree, Morton permutations, list1, and list2
    -> reusable P2M
    -> reusable M2M child classes
    -> reusable M2L transfer classes
    -> reusable L2L child classes
    -> reusable L2P rows
    +  reusable exact list1 P2P tensor

changing moments
    -> P2M -> M2M -> M2L -> L2L -> L2P --+
    +-------------------------> exact P2P --+-> field
```

Construction fixes independent source and target geometry, the complete
uniform tree, expansion order and basis, scalar precision, execution backend,
and any immutable self-identity map. An `evaluate` call accepts a new moment
for every source in original user order. Source moments are Morton-permuted,
the expansion and result state is cleared, and results are returned in original
target order. Empty subtrees remain zero, so one state never contaminates the
next.

## Hierarchical evaluation

1. **P2M:** each occupied source leaf applies its fixed source-to-multipole map.
2. **M2M:** child multipoles are shifted and added from the leaves towards the
   root using one of eight child-offset classes per used level.
3. **M2L:** every target box accumulates its well-separated `list2` source
   boxes. Integer displacement classes share level-independent dense matrices;
   precomputed degree factors account for box width.
4. **L2L:** parent locals are shifted and added towards the leaves using the
   corresponding eight child classes.
5. **L2P:** fixed target rows evaluate leaf locals as potential and/or field.
6. **P2P:** the exact cached tensor applies every `list1` near interaction.
7. **Assembly:** far and near contributions are added and unpermuted.

The far and near partitions do not overlap. Singular self-pairs are excluded
only through an explicit target-to-source identity map; equal coordinates do
not imply particle identity.

## Shared architecture, two expansion bases

Cartesian Taylor and real spherical-harmonic expansions use the same tree,
interaction plans, static-operator lifecycle, precision model, and execution
placement. They differ in coefficient ordering, coefficient count, and the
construction of the five far-field operators. Cartesian order `p` stores
`(p+1)(p+2)(p+3)/6` coefficients; spherical order `p` stores `(p+1)^2`.
The exact near-field P2P tensor is basis-independent.

Real spherical harmonics are the default for point sources. Cartesian remains
a complete independent formulation and is also the implemented basis for
uniform-cuboid sources. The CPU reference traversal is Cartesian-only; both
bases use static plans on the production CPU and CUDA backends. See
[Mathematical formulation](math.md) for conventions and
[Execution backends](backends.md) for stage placement.

## Geometry and output boundary

`UniformFmm` supports point targets. Point-dipole sources work with Cartesian
or spherical expansions; axis-aligned uniform-cuboid sources use Cartesian
expansions and exact cuboid-to-point near fields. Lower-level direct and
operator APIs additionally expose volume-averaged cuboid targets, but that
target model is not an end-to-end `UniformFmm` option.

CPU static and reference-capable paths support the applicable field,
potential, or combined output modes. CUDA-full is the device-resident repeated
field path. Backend requests are explicit: unavailable or incompatible
combinations raise an error rather than silently switching algorithms.
