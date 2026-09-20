# Examples and tutorials

## Tutorials

Six notebooks under `examples/tutorials/` cover the supported Python API in
the order a new user needs it. Each runs on the portable CPU build in well
under a minute, guards its CUDA and oneMKL cells with the availability queries,
uses only the public API, and is executed by the Python test suite on every CI
run.

| Notebook | Covers |
|---|---|
| `01_getting_started.ipynb` | the plan lifecycle, repeated evaluation with changing moments, units and shapes, the exact direct reference, what triggers a rebuild |
| `02_finite_geometry.ipynb` | prism and tetrahedron records, `m = V M`, the four model selectors, finite self fields versus point self exclusion, all nine source/target pairs against the dense direct plan |
| `03_backends_and_execution.ipynb` | precision, every backend, forced P2P packings, precomputed versus procedural point expansions, expansion basis, diagnostics, what to leave on `AUTO` |
| `04_cache_and_periodicity.ipynb` | cache keys and cold/warm reuse, which geometries share a plan, periodic cubic cells and the zero-$k$ convention (Lorentz field $+M/3$ for point dipoles, $H = 0$ for cubes filling the cell), a 27-image direct check |
| `05_trees_and_parameters.ipynb` | the uniform tree, depth against near/far work, the two advisers, `AdaptiveTree` on the same static plan |
| `06_operator_chain.ipynb` | P2P, P2M, M2M, M2P, M2L, L2L and L2P with the public operator functions, every route against the direct sum, the coefficients of a real plan |

```console
conda activate cdfmm
jupyter lab examples/tutorials
```

Select the kernel of the environment that can import `cdfmm`. The notebooks
are committed without outputs; `tutorial_utils.py` and `adaptive_showcase.py`
are their plotting and geometry helpers and reimplement no operator.

## C++ examples

`examples/single_box_demo.cpp` places two dipoles in one source box, builds a
multipole expansion, translates it to a distant local centre, and compares
L2P output with direct P2P at one target (P2M + M2L + L2P, both outputs).
`examples/operator_convergence_demo.cpp` constructs a deterministic random
source cloud and distant targets and compares P2M + M2P fields with direct P2P
for orders two through six. Both are built by the `dev` preset:

```console
./build/examples/single_box_demo
./build/examples/operator_convergence_demo
```

`examples/fortran/magtense_style_demag.f90` is the optional Fortran
integration example ([C and Fortran](c-and-fortran.md)).

## External validation

`examples/validation/` holds two notebooks that compare the exact
finite-body direct fields with MagTense (uniformly magnetised cubes, and a
1000 × 1000 prism/tetrahedron matrix against MagTense tile types). They need
the separate `cdfmm-magtense` environment described in their README and in
[Installation](installation.md); `python_tests/test_magtense_cuboid_comparison.py`
runs the numerical comparison when MagTense is importable. The FMM3D
comparison material lives with the benchmarks
([Benchmarks and profiling](benchmarks.md)).
