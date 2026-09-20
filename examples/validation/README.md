# External validation notebooks

These notebooks compare cdfmm's exact finite-body direct fields with the
independent MagTense implementation of the same rectangular-prism and
tetrahedron formulas. They are validation aids for the analytical tensors,
not tutorials: they need the separate `cdfmm-magtense` environment and are
not executed by CI (`python_tests/test_magtense_cuboid_comparison.py` runs
the numerical comparison itself when MagTense is importable and checks the
notebooks' structure otherwise).

| Notebook | Comparison |
|---|---|
| `magtense_cuboid_compare.ipynb` | uniformly magnetised cubes: cdfmm `DenseDirectPlan`/`CudaDenseDirectPlan` against `magtense.magstatics.get_H_field`, portable, oneMKL and CUDA backends |
| `magtense_geometry_compare.ipynb` | 1000 x 1000 all-to-all prism and tetrahedron sources against point targets and volume-averaged targets, against MagTense tile types 2 and 5 |

```console
conda env create -f environment-magtense.yml
conda activate cdfmm-magtense
cmake --preset magtense --fresh
cmake --build --preset magtense -j
python -m pytest python_tests/test_magtense_cuboid_comparison.py -v
jupyter lab examples/validation
```

MagTense 2.2.0 pins older NumPy and Intel runtime packages than the main
environment, which is why it has its own environment and preset; the
`magtense` preset uses MKL's Intel threading layer and disables cdfmm's GNU
OpenMP runtime so that one process does not load two OpenMP runtimes.
