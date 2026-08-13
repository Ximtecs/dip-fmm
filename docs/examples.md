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

The tree notebook draws actual cube boundaries, Morton-coloured sources,
leaf occupancy, and distinct `list1`/`list2` boxes around a selected node. It
is followed by an upward-pass notebook that draws populated nodes and
child-to-parent translation arrows, then compares the hierarchical root with
direct root P2M coefficient by coefficient. Neither notebook implements M2L,
a downward pass, near-field traversal, or a complete tree FMM.

Start Jupyter from the repository root after activating the development
environment:

```console
conda activate cdfmm
jupyter lab
```

See `examples/notebooks/README.md` for the ordered list and select
`Python (cdfmm)` as the kernel.

Notebook 11 compares CUDA-full, CUDA-partial, and oneMKL CPU-static with the
pinned FMM3D 2.1.0 Laplace dipole implementation on the same deterministic
source-point problem. It verifies coincident source/target geometry and reports
sampled exact-reference accuracy together with one-run and persistent
changing-moment performance. From the repository root, install and launch it
with:

```console
conda env update -n cdfmm -f environment.yml
conda env update -n cdfmm -f environment-cuda.yml
conda env update -n cdfmm -f environment-fmm3d.yml
conda activate cdfmm
cmake --fresh --preset notebooks
cmake --build --preset notebooks -j
ctest --preset notebooks
./examples/notebooks/install_fmm3d.sh
jupyter lab examples/notebooks/11_fmm3d_comparison.ipynb
```

This installed Python build contains CUDA-full, CUDA-partial, oneMKL, and the
static-M2L inspection binding required by notebook 12. Restart an already open
notebook kernel after rebuilding so it does not retain the previous extension.

Notebook 12 constructs real CPU-static plans across particle-count, expansion-
order, and tree-depth sweeps, then compares their total and intermediate
storage counters with an independent estimate derived from the uniform-tree
interaction lists.  When CUDA is available, it also constructs CUDA-partial
and CUDA-full plans and verifies that both reuse the same bounded set of
normalised M2L transfer-class matrices.

## Benchmark

`benchmarks/benchmark_p2p.cpp` is a minimal standalone timing of one direct sum
over 10,000 sources.  Build it with `CDFMM_BUILD_BENCHMARKS=ON`.  It is useful
as a smoke benchmark, but it is not a statistically rigorous suite; expanded
kernel and end-to-end coverage is tracked in the [roadmap](roadmap.md).
