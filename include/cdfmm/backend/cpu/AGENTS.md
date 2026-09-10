# CPU static-plan execution interface

Inherit `../../AGENTS.md`. This directory exposes portable CPU application
of already-prepared immutable plan data. It owns no operator construction,
geometry preparation, plan packing, backend selection, or persistent device
resources. Preserve accumulation order, explicit identity exclusion, SIMD
semantics, and allocation-free repeated evaluation.
