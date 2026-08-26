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

## Focused comparison notebooks

Each focused notebook answers a different question:

| Notebook | Purpose |
|---|---|
| `examples/notebooks/10_uniform_downward_pass.ipynb` | Basic complete FMM decomposition and direct-reference accuracy |
| `examples/simple_notebooks/simple_cuboid_fmm_direct_compare.ipynb` | Cartesian uniform-cuboid-to-point FMM versus exact dense cuboid direct |
| `examples/simple_notebooks/simple_cuboid_p2m_l2p_direct_compare.ipynb` | Spherical point/cuboid P2M and L2P comparison with fixed exact cuboid P2P |
| `examples/simple_notebooks/simple_cuboid_magtense_compare.ipynb` | Direct cuboid field comparison with MagTense conventions |
| `examples/simple_notebooks/simple_dense_direct_precision_compare.ipynb` | FP32 versus FP64 across available static FMM backends |
| `examples/simple_notebooks/simple_cartesian_spherical_fmm_compare.ipynb` | Cartesian versus spherical finite-cuboid CUDA-full FMM |
| `examples/notebooks/14_periodic_fmm_direct_compare.ipynb` | Selectable point-dipole or cuboid periodic/non-periodic CUDA direct and spherical FMM sweep, with exact cuboid near fields and point far-field expansions |
| `examples/notebooks/11_fmm3d_comparison.ipynb` | Spherical CUDA-full FMM versus FMM3D 2.1.0 |

The Cartesian/spherical notebook uses one 512-cuboid grid, volume-averaged
cuboid targets, and a common exact cuboid-to-cuboid field. It records
coefficient count, error, initialisation, first and median repeated evaluation
time, stage timings, and retained host-plus-device memory for orders 1--6, 8,
and 10 in the original point comparison and orders 1--6 in the current finite-
cuboid comparison. Both plans use FP64 CUDA-full; it is a controlled basis
comparison, not a claim about every geometry or backend. The measured findings
and their limitations are recorded in [Performance benchmarks](benchmarks.md).

## Interactive operator and tree notebooks

The comprehensive examples under `examples/notebooks/` form a guided sequence
through direct P2P, every current expansion operator, composed operator paths,
and the uniform Morton-sorted tree. They include mathematical definitions,
deterministic problem setups, direct-reference comparisons, convergence plots,
spatial error maps, and optional ipywidgets controls.

The tree notebook draws actual cube boundaries, Morton-coloured sources, leaf
occupancy, and distinct `list1`/`list2` boxes around a selected node. The next
notebooks inspect the upward pass and the complete downward/near-field
decomposition. They visualise state produced by the compiled implementation;
they do not reimplement traversal in Python.

Start Jupyter from the repository root after activating the development
environment:

```console
conda activate cdfmm
jupyter lab
```

See `examples/notebooks/README.md` for the ordered list and select
`Python (cdfmm)` as the kernel.

Notebook 11 compares the real spherical CUDA-full backend with the pinned
FMM3D 2.1.0 Laplace dipole implementation. Its main sweep covers 20,000--60,000
coincident source/target points, cdfmm orders 4, 6, and 8, depths 3--5, and
FMM3D tolerances $10^{-3}$ and $10^{-4}$. It reports sampled exact-reference
accuracy and steady-state changing-moment throughput. From the repository root,
install and launch it with:

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

The installed Python build also contains the static-M2L inspection binding
required by notebook 12. Restart an already open notebook kernel after
rebuilding so it does not retain the previous extension.

Notebook 12 constructs real CPU-static plans across particle-count, expansion-
order, and tree-depth sweeps, then compares their total and intermediate
storage counters with an independent estimate derived from the uniform-tree
interaction lists.  When CUDA is available, it also constructs CUDA-partial
and CUDA-full plans and verifies that both reuse the same bounded set of
normalised M2L transfer-class matrices.

## Benchmark

`benchmarks/benchmark_p2p.cpp` is a minimal standalone timing of one direct sum
over 10,000 sources. Build it with `CDFMM_BUILD_BENCHMARKS=ON`. It is a smoke
benchmark; use `benchmark_uniform_fmm` and the Python benchmark driver for
controlled multi-backend setup, accuracy, repeated-runtime, and memory output.
