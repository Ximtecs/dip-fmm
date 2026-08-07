# Validation and testing

The project validates small mathematical pieces analytically before composing
them.  Unit tests cover multi-index counts and ordering, Taylor-jet algebra,
Laplace derivatives, output modes, axial and transverse direct dipoles, tree
topology, sorting, ranges, and interaction lists.

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

There is not yet an end-to-end FMM traversal to test.  Direct comparison of the
complete uniform FMM, including result unsorting and `list1` near fields, is a
required part of the next milestone.
