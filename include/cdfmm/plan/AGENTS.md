# Static-plan interface guidance

Inherit `../AGENTS.md` and the repository root guidance. This directory owns
immutable, backend-independent descriptions prepared once for fixed geometry.

```text
plan/
|-- static_coefficient.hpp       sparse static coefficient maps
|-- l2p.hpp                      fixed local-evaluation rows
|-- m2l.hpp                      translation classes and schedules
|-- precision.hpp                explicit FP64-to-FP32 conversion
|-- static_plan.hpp              plan-data umbrella
`-- p2p/
    |-- canonical.hpp            authoritative target-row tensor data
    |-- memory.hpp               representation memory accounting
    |-- compact.hpp              particle-row SoA packing
    |-- leaf.hpp                 dense leaf-pair packing
    |-- dictionary.hpp           magnitude/sign dictionary packing
    |-- signed_dictionary.hpp    signed reduced dictionary packing
    |-- bsr.hpp                  backend-neutral BSR(3) packing
    `-- tensor_dictionary.hpp    Tensor6 canonicalisation and token packing
```

`p2p/tensor_dictionary.hpp` is dependency-free (no other `cdfmm` include) and
is consumed only by dictionary/signed-dictionary plan construction and their
CPU/CUDA execution. It moved here from the flat `cdfmm/tensor_dictionary.hpp`,
which is now a forwarding compatibility façade; the token encoding itself did
not change.

The canonical P2P rows are the sole mathematical truth. Every compact, leaf,
dictionary, signed-dictionary, and BSR object is a deterministic derivative of
those rows and must preserve their interaction set. Plans contain no mutable
coefficient state, scratch, device pointers, vendor descriptors, or execution
policy. Preserve field order, scalar precision, coefficient ordering, identity
markers, and cache-visible members during compatibility refactors.
