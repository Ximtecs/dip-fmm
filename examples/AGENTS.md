# Example guidance

Inherit `../AGENTS.md`. Examples are readable, user-facing demonstrations and
validation aids. They are not substitutes for automated tests.

## Current structure

```text
examples/
|-- CMakeLists.txt
|-- single_box_demo.cpp              P2M + M2L + L2P against direct P2P (C++)
|-- operator_convergence_demo.cpp    P2M + M2P convergence over orders (C++)
|-- fortran/                         optional C/Fortran integration example
|-- tutorials/                       the six canonical Python tutorials,
|                                    tutorial_utils.py, adaptive_showcase.py
`-- validation/                      MagTense comparison notebooks (separate
                                     environment; not tutorials)
```

`examples/tutorials/README.md` lists the tutorials and what each covers. The
FMM3D comparison material is benchmark raw material under
`benchmarks/external/fmm3d/`, not an example. MagTense and FMM3D are external
validation paths, not runtime dip-fmm backends.

## Rules

- One tutorial per supported user workflow; extend an existing tutorial
  before adding a notebook, and do not add exploratory or development
  notebooks here (use a scratch location that is not committed).
- Every tutorial is CPU-executable on the portable build in well under a
  minute, guards CUDA and oneMKL cells with the availability queries, uses only
  the public `cdfmm` API, states units and array shapes, explains options
  before non-trivial code, and is committed without outputs.
- Optimise for clarity: descriptive names, deterministic small setups, explicit
  units/conventions, and no dense one-line code.
- New behaviour demonstrated here also needs an automated C++ or Python test.

## Validation

The `dev` preset builds both C++ examples. Every tutorial is executed by
`python_tests/test_tutorial_notebooks.py` (structure and headless execution on
a generic `python3` kernel):

```bash
cmake --fresh --preset dev
cmake --build --preset dev -j
PYTHONPATH=build python -m pytest python_tests/test_tutorial_notebooks.py -v
```

Run the MagTense validation notebooks only when their environment is available
(`examples/validation/README.md`), and report otherwise rather than weakening
the comparison.
