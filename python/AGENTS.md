# Python binding guidance

Inherit `../AGENTS.md`. This directory adapts the supported C++ interface to
Python; it does not own solver algorithms.

## Current structure

```text
python/
`-- bindings.cpp    pybind11 module: types, options, plans, execution, diagnostics
```

`bindings.cpp` is currently a large mixed binding unit and a later
decomposition candidate. Future files should be grouped by exposed subsystem,
not by arbitrary line count, and should continue to call C++ library logic.

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
