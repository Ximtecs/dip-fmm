# Examples

## `single_box_demo`

`examples/single_box_demo.cpp` places two dipoles in one source box, builds a
multipole expansion, translates it to a distant local centre, and compares
L2P output with direct P2P at one target.  It demonstrates the
P2M + M2L + L2P composition and requests both potential and field.

## `operator_convergence_demo`

`examples/operator_convergence_demo.cpp` constructs a deterministic random
source cloud and distant targets.  For orders two through six it compares
P2M + M2P fields with direct P2P and prints aggregate relative errors.  Separate
random seeds make the diagnostic reproducible.

After a default build, run:

```console
./build/examples/single_box_demo
./build/examples/operator_convergence_demo
```

## Uniform-tree plot

`examples/plot_uniform_tree.py` builds a depth-two Python `UniformTree`, plots
Morton-sorted sources, and labels occupied leaf centres by Morton index.  It
requires Matplotlib in addition to the package's NumPy dependency:

```console
python examples/plot_uniform_tree.py
```

The plot is a geometry inspection aid, not an FMM field evaluation.

## Benchmark

`benchmarks/benchmark_p2p.cpp` is a minimal standalone timing of one direct sum
over 10,000 sources.  Build it with `CDFMM_BUILD_BENCHMARKS=ON`.  It is useful
as a smoke benchmark, but it is not a statistically rigorous suite; expanded
kernel and end-to-end coverage is tracked in the [roadmap](roadmap.md).
