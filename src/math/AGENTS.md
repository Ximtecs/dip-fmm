# Mathematical implementation

Inherit `../../AGENTS.md` and the public math guidance. This directory owns
Taylor jets, Laplace derivatives, and real spherical-harmonic implementation,
including `solid_harmonic_recurrence.hpp`, the allocation-free host/device
recurrence for the real regular solid harmonics and their gradients that the
procedural point P2M/L2P executors use; it must keep reproducing
`regular_solid_harmonics` (checked by `tests/test_spherical_harmonics.cpp`).
Keep these routines independent of geometry, tree construction, plans,
backends, and FMM orchestration, and preserve all mathematical conventions.
