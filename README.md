# cartesian-dipole-fmm

Cartesian-coordinate fast multipole method specialised for dipole interactions.

Current status: CPU operator layer only; no tree yet. Includes direct M2P evaluation for validating multipole expansions against direct P2P.

## Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCDFMM_BUILD_TESTS=ON -DCDFMM_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

## Python
python -m pip install .
python -m pytest python_tests -v

## C++ examples
See `examples/single_box_demo.cpp` and `examples/operator_convergence_demo.cpp`.

The validation helpers in `include/cdfmm/validation.hpp` provide reusable
error metrics and direct-P2P reference evaluation utilities for tests and
operator diagnostics.

## Python example
```python
import cdfmm
r=cdfmm.p2p_dipole_pair([1,0,0],[0,0,0],[1,0,0],output="both")
print(r)
```

## Mathematical convention
G(r)=1/(4*pi*|r|), H=-grad(phi). See `docs/math.md`.

## Roadmap
1. CPU operator layer
2. Uniform non-adaptive tree
3. Persistent geometry plan
4. Adaptive tree
5. CUDA P2P/M2L
6. MagTense/Fortran interface

## Uniform tree status

The repository now includes an initial complete uniform Morton-sorted tree structure for geometry organisation and neighbour-list inspection. Upward and downward FMM passes are intentionally not implemented yet.
