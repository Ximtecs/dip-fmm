# Interactive notebooks

These notebooks demonstrate the current C++ operator layer and complete
uniform-tree geometry through the thin `cdfmm` Python interface. They are
interactive scientific examples and validation aids; the C++ and Python test
suites remain the automated correctness checks.

Run Jupyter from the repository root so the shared plotting helpers are
available:

```console
conda activate cdfmm
jupyter lab
```

The sequence follows the data flow through the implemented operators:

| Notebook | Topic |
|---|---|
| `00_direct_p2p.ipynb` | Exact direct dipole interaction and field plot |
| `01_p2m_multipole_expansion.ipynb` | Cartesian multi-indices and P2M coefficients |
| `02_m2m_translation.ipynb` | Child-to-parent multipole translation |
| `03_m2p_evaluation.ipynb` | Far-field convergence with order and distance |
| `04_m2l_local_expansion.ipynb` | Conversion to a target-centred local expansion |
| `05_l2l_translation.ipynb` | Parent-to-child local translation |
| `06_l2p_evaluation.ipynb` | Spatial local-expansion error near a target centre |
| `07_operator_chain.ipynb` | Comparison of all implemented operator paths |
| `08_uniform_tree.ipynb` | Morton order, occupancy, boxes, list1, and list2 |
| `09_uniform_upward_pass.ipynb` | Leaf P2M, hierarchical M2M, and direct-root equivalence |
| `10_uniform_downward_pass.ipynb` | M2L/L2L routes, list1 near field, and complete-FMM accuracy |
| `11_fmm3d_comparison.ipynb` | CUDA-full, CUDA-partial, and oneMKL performance versus FMM3D 2.1.0 |
| `12_cuda_memory_usage.ipynb` | Estimated-versus-measured total and intermediate CPU/CUDA plan storage sweeps |
| `13_parameter_selection.ipynb` | Empirical depth and order selection for performance and accuracy |

Select the `Python (cdfmm)` kernel in JupyterLab or VSCode. Important
parameters are collected near the top of each notebook, and all random
experiments use deterministic seeds. The notebooks call the compiled C++
operators; `example_utils.py` contains only plotting, batched calls, and error
diagnostics.

The uniform tree provides geometry and interaction lists, while `UniformFmm`
executes the complete reference traversal in compiled C++. Notebook 10 only
visualises that state and compares it with compiled direct P2P; it does not
reimplement traversal logic in Python.

## FMM3D comparison setup

Notebook 11 has one additional pinned dependency. Install it once from the
repository root, then restart the notebook kernel:

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

The installer builds the optimised FMM3D 2.1.0 Python wrapper in the active
`cdfmm` environment and is safe to rerun. Its source checkout is kept below
the ignored `.external/` directory. The notebook defaults to 10,000 coincident
sources and targets at order 6 and depth 2. It compares all three cdfmm paths
with an exact target sample and times 1, 10, 100, and 1,000 changing-moment
evaluations. Plans are released between backends so CUDA allocations do not
accumulate.

FMM3D defaults to `eps=1e-3`, for which the pinned v2.1.0 Laplace
implementation selects spherical expansion order `nterms=12`.

Notebook 12 requires the current compiled `cdfmm` module, including the
`static_m2l_matrix` inspection binding installed by the `notebooks` preset. It
reconstructs the
current host and CUDA storage layouts from tree interactions without allocating
CUDA plans, so oversized configurations can be inspected safely.

## MagTense cuboid comparison setup

The direct uniformly magnetised cube comparison uses an isolated environment
because MagTense 2.2.0 pins older NumPy and Intel runtime packages than the main
development environment. Create the complete environment from the repository
root, build the combined CUDA/oneMKL/portable-CPU Python module, and launch the
notebook with:

```console
conda env create -f environment-magtense.yml
conda activate cdfmm-magtense
module load cuda
cmake --preset magtense --fresh
cmake --build --preset magtense -j
python -m pytest python_tests/test_magtense_cuboid_comparison.py -v
jupyter lab examples/simple_notebooks/simple_cuboid_magtense_compare.ipynb
```

Set `CDFMM_BACKEND` in the physical-problem cell to `"normal-cpu"`,
`"mkl-cpu"`, or `"cuda"`. All three modes use the same physical geometry and
error analysis; unavailable compiled backends are reported explicitly.

Load the CUDA module before configuring. A failed CUDA configuration leaves an
incomplete Ninja directory, so run `cmake --preset magtense --fresh` again
after loading the module; do not invoke the build preset until configuration
has completed successfully. This preset uses MKL's `intel_thread` backend and
disables cdfmm's separate GNU OpenMP runtime to avoid loading two incompatible
OpenMP runtimes into the comparison process.

This initial comparison evaluates each finite source cube at every cube centre,
including the finite self-field. MagTense's public Python API provides point
evaluation for rectangular prisms, so the exact receiving-volume-averaged
cuboid tensor remains outside this comparison.
