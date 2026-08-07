# Installation and building

## CMake build

A C++20 compiler and CMake 3.20 or newer are required.  Tests and Python
bindings fetch Catch2 and pybind11 during configuration.

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCDFMM_BUILD_TESTS=ON -DCDFMM_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Important options are:

| Option | Default | Purpose |
|---|---:|---|
| `CDFMM_BUILD_TESTS` | `ON` | Build the Catch2 C++ test suite. |
| `CDFMM_BUILD_EXAMPLES` | `ON` | Build the C++ demonstration programs. |
| `CDFMM_BUILD_BENCHMARKS` | `OFF` | Build the current standalone P2P benchmark. |
| `CDFMM_BUILD_PYTHON` | `ON` | Build the pybind11 extension module. |

Enable benchmarks with `-DCDFMM_BUILD_BENCHMARKS=ON`; the executable is
`build/benchmarks/benchmark_p2p`.  Examples are placed under `build/examples`.

## Python extension

The package requires Python 3.9 or newer and NumPy:

```console
python -m pip install .
python -m pytest python_tests -v
```

This builds and installs the native `cdfmm` module through scikit-build-core.

## Documentation

Install Sphinx dependencies and Doxygen, then build locally:

```console
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Sphinx invokes Doxygen automatically.  XML is generated beneath
`docs/_build/doxygen` and is not committed.  Open
`docs/_build/html/index.html` after a successful build.  The checked-in
`.readthedocs.yaml` performs the equivalent build on Read the Docs.
