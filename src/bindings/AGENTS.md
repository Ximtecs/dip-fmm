# Language-adapter implementation guidance

Inherit `../../AGENTS.md` and `../AGENTS.md`.

This directory adapts supported C++ interfaces to the stable C ABI. It does
not own solver algorithms or backend resources. The C ABI is a compatibility
contract: exported symbols, opaque-plan semantics, and public struct/enum
layouts require explicit ABI review before changing. The Fortran wrapper
consumes this C ABI; it must not call C++ implementation details directly.

Keep language adapters independent of implementation-only FMM, backend, cache,
and CUDA headers. New adapter code should use the supported public interfaces
under `include/cdfmm/`.
