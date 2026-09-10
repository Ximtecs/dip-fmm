# Static-plan implementation guidance

Inherit `../AGENTS.md`, the repository root guidance, and the interface rules
in `include/cdfmm/plan/AGENTS.md`.

```text
plan/
|-- precision.cpp              explicit plan-wide FP64-to-FP32 conversion
`-- p2p/
    |-- packing_memory.cpp     representation memory accounting
    |-- compact.cpp            canonical rows to SoA
    |-- leaf.cpp               canonical rows to dense leaf rectangles
    |-- dictionary.cpp         leaf data to magnitude/sign dictionary
    |-- signed_dictionary.cpp  leaf data to signed reduced dictionary
    |-- bsr.cpp                canonical rows to plain BSR(3)
    `-- dictionary_detail.hpp  exact-bit dictionary ordering mechanics
```

Builders are deterministic, backend-independent construction work. They may
validate that a packing exactly covers canonical P2P rows, but must not alter
the interaction set, accumulation order, identity semantics, or precision
conversion. Keep apply loops and vendor/device resources in backend code. In
particular BSR construction must not include CUDA or cuSPARSE facilities.
