# Portable static-plan execution

Inherit `../../AGENTS.md` and the public CPU-backend guidance. This directory
owns application loops for immutable static plans. Keep plan construction and
precision conversion in `src/plan`, mathematical construction in
`src/operators`, and vendor/device resources elsewhere. SIMD specialisation
must preserve the portable loop's arithmetic, identity, and accumulation
semantics and introduce no evaluation-time allocation.
