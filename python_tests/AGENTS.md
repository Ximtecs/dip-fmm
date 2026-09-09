# Python test guidance

Inherit `../AGENTS.md`. This suite verifies the pybind11 interface, end-to-end
FMM behaviour, benchmark runners, and the executable/structural contracts of
selected notebooks and helper scripts.

## Current structure

```text
python_tests/
|-- test_{python_bindings,operator_bindings,geometry_bindings}.py
|-- test_{uniform_fmm_python,adaptive_tree,spherical_fmm,
|   fmm_precision,periodic_fmm,fmm_memory}.py
|-- test_{benchmark_runner,p2p_sweep,high_occupancy_p2p_runner}.py
|-- test_{adaptive_notebook,precision_notebook,
|   geometry_magtense_all_to_all_notebook}.py
|-- test_{fmm3d_comparison,magtense_cuboid_comparison}.py
`-- test_{profiling_setup,parameter_selection}.py
```

## Rules and validation

- Test the just-built extension, not an unrelated installed `cdfmm`.
- Keep external MagTense, FMM3D, oneMKL, and CUDA cases explicitly optional;
  skip only when the dependency/runtime is genuinely absent.
- Notebook contract tests should avoid executing heavyweight interactive work
  while still checking valid JSON, stable cell IDs, compilable code, and
  reusable helper behaviour.
- Binding-visible changes need focused Python assertions as well as the
  relevant lower-level C++ test.

```bash
cmake --fresh --preset dev
cmake --build --preset dev -j
PYTHONPATH=build python -m pytest python_tests -v

# Focused file or test
PYTHONPATH=build python -m pytest python_tests/test_python_bindings.py -v
```
