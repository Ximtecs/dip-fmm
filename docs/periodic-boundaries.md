# Fully periodic boundary conditions

The periodic API requires an explicit fundamental cell.  The first supported
mode is deliberately narrow: all three axes are periodic and the cell is a
cube.  Enabling periodicity makes the cell centre and side length the FMM root
centre and width; the period is never inferred from particle bounds.  Partial
periodicity and rectangular cells are rejected so that later extensions can be
added without giving existing inputs ambiguous meanings.

A three-dimensional dipolar lattice sum is conditionally convergent, so the
word “periodic” is not by itself a physical specification.  The
`PeriodicConvention::ZeroK0` convention omits reciprocal `k=0` and adds no
macroscopic surface or demagnetising term.  Consequently, an exactly uniform
magnetisation has `H_demag = 0`.

## Wrapped topology

Periodic boxes are represented by a central-tree node and a three-component
integer image shift.  An unwrapped coordinate `q` at level `l` is converted by
mathematical floor division into a coordinate modulo `2^l` and an image shift.
The complete pair is the identity: node indices alone must not be used for
deduplication, particularly in shallow trees.

The topology helpers construct list1 as the periodic 3-by-3-by-3 neighbourhood
and list2 as `children(parent list1) - node list1`.  They do not replicate the
tree or particle arrays.  Child displacement classes remain in the ordinary
uniform-tree `[-3,3]^3` list2 class set.

> **Implementation status:** this change establishes and validates the
> explicit cell and wrapped interaction identities.  The production evaluator
> does not yet apply the root Ewald periodiser or wrapped P2P/M2L plans; enabling
> the cell currently changes root geometry only.  It must therefore not yet be
> used as a periodic solver.
