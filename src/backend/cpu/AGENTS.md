# Portable static-plan execution

Inherit `../../AGENTS.md` and the public CPU-backend guidance. This directory
owns application loops for immutable static plans. `direct/` contains
dense-direct execution, `p2p/` contains canonical and packed P2P plus list-1
execution and the position-based point executor (`geometry.{hpp,cpp}`, which
computes every pair through `operators/p2p_point_kernel.hpp` and never
redefines the formula), `m2l/` contains prepared M2L execution and its
transfer-class block schedule (`schedule.{hpp,cpp}`), and `far_field/` contains
the public entry-map reference kernels plus the derived execution packing
(`packing.{hpp,cpp}`: dense P2M rows, level-scaled M2M/L2L column banks and
flat L2P rows) that the hierarchy executes; the packing is built once at
construction from the canonical operators and must not change their
mathematics, identity, or per-target accumulation order. Keep plan
construction and precision conversion in `src/plan`, mathematical construction
in `src/operators`, and vendor/device resources elsewhere. SIMD specialisation
must preserve the portable loop's arithmetic, identity, and accumulation
semantics and introduce no evaluation-time allocation.
