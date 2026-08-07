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

## Interactive operator and tree notebooks

The comprehensive examples under `examples/notebooks/` form a guided sequence
through direct P2P, every current expansion operator, composed operator paths,
and the uniform Morton-sorted tree. They include mathematical definitions,
deterministic problem setups, direct-reference comparisons, convergence plots,
spatial error maps, and optional ipywidgets controls.

The final tree notebook draws actual cube boundaries, Morton-coloured sources,
leaf occupancy, and distinct `list1`/`list2` boxes around a selected node. It
does not implement an upward pass, downward pass, or complete tree FMM.

Start Jupyter from the repository root after activating the development
environment:

```console
conda activate cdfmm
jupyter lab
```

See `examples/notebooks/README.md` for the ordered list and select
`Python (cdfmm)` as the kernel.

## Benchmark

`benchmarks/benchmark_p2p.cpp` is a minimal standalone timing of one direct sum
over 10,000 sources.  Build it with `CDFMM_BUILD_BENCHMARKS=ON`.  It is useful
as a smoke benchmark, but it is not a statistically rigorous suite; expanded
kernel and end-to-end coverage is tracked in the [roadmap](roadmap.md).
