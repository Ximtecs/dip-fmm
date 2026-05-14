# cartesian-dipole-fmm

Cartesian-coordinate fast multipole method specialised for dipole interactions.

Current status: CPU operator layer only; no tree yet.

## Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCDFMM_BUILD_TESTS=ON -DCDFMM_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

## Python
python -m pip install .
python -m pytest python_tests -v

## C++ example
See `examples/single_box_demo.cpp`.

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
