# Python test guidance

Inherit `../AGENTS.md`. This suite verifies the pybind11 interface, end-to-end
FMM behaviour, the benchmark runners, and the executability of the tutorials.

## Current structure

```text
python_tests/
|-- test_{python_bindings,operator_bindings,geometry_bindings}.py
|                                       binding smoke and negative API contracts
|-- test_{uniform_fmm_python,uniform_tree_python,adaptive_tree,spherical_fmm,
|   fmm_precision,periodic_fmm,parameter_selection}.py
|                                       end-to-end FMM behaviour
|-- test_fmm_memory.py, memory_model.py plan byte accounting against an
|                                       independent storage model
|-- test_adaptive_showcase.py           the tutorial-5 geometry/topology helpers
|-- test_tutorial_notebooks.py          structure and headless execution of
|                                       every notebook in examples/tutorials
|-- test_{benchmark_runner,p2p_sweep,high_occupancy_p2p_runner,
|   phase3d_regression_matrix}.py       benchmark drivers and analysers
|-- test_fmm3d_comparison.py            FMM3D adapter unit tests, installer
|                                       guards, notebook structure
|-- test_magtense_{cuboid,geometry}_comparison.py
|                                       MagTense environment pins, notebook
|                                       structure, and the numerical
|                                       comparison when MagTense is importable
`-- test_profiling_setup.py             preset/CMake/profiling pins
```

## Rules and validation

- Test the just-built extension, not an unrelated installed `cdfmm`. The
  tutorial test resolves `PYTHONPATH` entries to absolute paths before it
  starts a kernel in the tutorial directory for exactly this reason.
- Keep external MagTense, FMM3D, oneMKL, and CUDA cases explicitly optional;
  skip only when the dependency/runtime is genuinely absent.
- A retained notebook is either executed (tutorials) or structurally checked
  (external comparisons that need another environment). Do not keep a test
  for a notebook that no longer exists.
- Binding-visible changes need focused Python assertions as well as the
  relevant lower-level C++ test.

```bash
cmake --fresh --preset dev
cmake --build --preset dev -j
PYTHONPATH=build python -m pytest python_tests -v

# Focused file or test
PYTHONPATH=build python -m pytest python_tests/test_python_bindings.py -v
```
