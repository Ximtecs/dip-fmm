# Example guidance

Inherit `../AGENTS.md`. Examples are readable, user-facing demonstrations and
research validation aids. They are not substitutes for automated tests.

## Current structure

```text
examples/
|-- CMakeLists.txt
|-- single_box_demo.cpp
|-- operator_convergence_demo.cpp
|-- plot_uniform_tree.py
|-- fortran/                 optional C/Fortran integration example
|-- notebooks/               numbered operator-to-FMM learning sequence
|-- simple_notebooks/        focused comparison/reuse notebooks
`-- matrix_notebooks/        interaction-matrix visualisation
```

`examples/notebooks/` also contains shared Python helpers and optional FMM3D
installation/comparison support. MagTense and FMM3D comparisons are external
validation paths, not runtime dip-fmm backends.

## Rules

- Optimise for clarity: descriptive names, deterministic small setups, explicit
  units/conventions, and no dense one-line code.
- Keep heavyweight output, local kernels, caches, and generated datasets out of
  version control unless deliberately curated.
- A notebook change should preserve reproducibility and avoid unrelated output
  churn. Do not rewrite user-modified notebooks while changing architecture.
- New behaviour demonstrated here also needs an automated C++ or Python test.

## Validation

The `dev` preset builds both C++ examples. Shared notebook scripts and selected
notebooks are exercised by `python_tests/`. The full notebook environment uses:

```bash
cmake --fresh --preset notebooks
cmake --build --preset notebooks -j
ctest --preset notebooks
PYTHONPATH=build-notebooks python -m pytest python_tests -v
```

Run MagTense/FMM3D-specific checks only when their separate environments are
available, and report otherwise rather than weakening the comparison.
