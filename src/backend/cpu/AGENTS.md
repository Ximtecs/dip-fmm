# Portable static-plan execution

Inherit `../../AGENTS.md` and the public CPU-backend guidance. This directory
owns application loops for immutable static plans. `direct/` contains
dense-direct execution, `p2p/` contains canonical and packed P2P plus list-1
execution, `m2l/` contains prepared M2L execution, and `far_field/` contains
prepared P2M/L2P entries and level-scaled M2M/L2L translation. Keep plan
construction and precision conversion in `src/plan`, mathematical construction
in `src/operators`, and vendor/device resources elsewhere. SIMD specialisation
must preserve the portable loop's arithmetic, identity, and accumulation
semantics and introduce no evaluation-time allocation.
