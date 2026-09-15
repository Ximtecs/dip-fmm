# Python binding guidance

Inherit `../AGENTS.md`. This directory adapts the supported C++ interface to
Python; it does not own solver algorithms.

## Current structure

```text
python/
|-- module.cpp      single pybind11 entrypoint and registration order
|-- internal.hpp    shared conversion and registration declarations
|-- core.cpp        NumPy/Python conversion helpers and basic types
|-- geometry.cpp    physical geometry records and geometry enums
|-- tree.cpp        uniform/adaptive trees and static topology exposure
|-- operators.cpp   expansion operators and mathematical helpers
|-- direct.cpp      direct plans, references, and capability queries
`-- fmm.cpp         UniformFmm, options, suggestions, and diagnostics
```

Binding files are grouped by exposed subsystem, not arbitrary line count, and
should continue to call C++ library logic.

## Rules and validation

- Match C++ defaults, enum meanings, precision/dtype, ownership, exceptions,
  and self-identity semantics exactly.
- Do not reimplement geometry, operators, packing, parameter selection, or FMM
  orchestration in binding code.
- Preserve established Python names during the behaviour-preserving refactor.
- Add focused tests under `python_tests/` for binding-visible changes.

```bash
cmake --fresh --preset dev
cmake --build --preset dev -j
PYTHONPATH=build python -m pytest python_tests -v
```

Using `PYTHONPATH=build` selects the just-built extension without requiring an
install into the active environment.
