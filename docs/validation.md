# Validation and testing

## Manual CUDA validation

CUDA tests are intentionally not executed by GitHub Actions. They must be run
manually on an NVIDIA CUDA-capable system:

```console
cmake --preset cuda
cmake --build --preset cuda
ctest --preset cuda
python -m pytest python_tests -v
```

For field-only repeated evaluation, inspect `cuda_plan_statistics`: after an
unchanged identity map is resident, H2D traffic is exactly one user-order
moment array and D2H traffic is exactly one user-order field array. Potential
traffic is added only when requested. Plan generation and static upload counters
make accidental setup reconstruction visible.

The project validates small mathematical pieces analytically before composing
them.  Unit tests cover multi-index counts and ordering, Taylor-jet algebra,
Laplace derivatives, output modes, axial and transverse direct dipoles, tree
topology, sorting, ranges, and interaction lists.

The default exact static M2L backend is checked against the independently
retained `m2l_add()` traversal. Select that validation path explicitly with
`UniformFmmOptions::m2l_backend = M2LBackend::Reference`; static-plan matrices,
interaction maps, and scratch storage are constructed only once.

Operator accuracy tests compare these paths with direct P2P:

- **P2M + M2P** checks source coefficients and kernel differentiation;
- **P2M + M2L + L2P** checks conversion to and evaluation of local expansions;
- **P2M + M2M + M2P** checks child-to-parent translation;
- direct pair tests compare against analytic dipole configurations.

The reusable `direct_p2p_reference` helper evaluates multiple targets, while
`compute_error_metrics` reports mean, RMS, and maximum relative errors plus
mean and maximum absolute errors.  The relative denominator is floored for
near-zero reference fields.

Run the C++ suite after configuring with tests enabled:

```console
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run installed Python bindings against their tests with:

```console
python -m pytest python_tests -v
```

Complete-traversal tests compare `UniformFmm` with direct P2P across expansion
orders and check the convergence trend. They also cover depth-zero direct
evaluation, target unsorting, repeated magnetic states, empty complete-tree
boxes, output modes, and explicit index-based source self exclusion. Python
tests exercise the same compiled evaluator, array shapes, ordering, and reset
behaviour.

## Parallel correctness and benchmark accuracy

When OpenMP is enabled, automated tests compare deterministic complete-FMM
results obtained with one and multiple threads and verify repeated-evaluation
state reset. The benchmark optionally evaluates the same state with parallel
direct P2P and writes mean, RMS, and maximum relative field errors beside the
phase timings. Large automated cases disable the quadratic direct reference
explicitly rather than reporting a misleading zero error.
