# FMM3D comparison material

Raw material for the external comparison against FMM3D 2.1.0, kept with the
benchmarks rather than the tutorials: the final publication comparison is
designed after the implementation freeze and reuses these utilities, not the
exploratory notebook as it stands.

| File | Role |
|---|---|
| `fmm3d_comparison.py` | adapters and metrics: the FMM3D `nterms` table, gradient sign/shape conversion, RMS/maximum relative error, timed calls, out-of-memory detection, source-point validation (unit-tested by `python_tests/test_fmm3d_comparison.py`) |
| `install_fmm3d.sh` | pinned installer for the FMM3D 2.1.0 Python wrapper in the `cdfmm` environment with one GNU toolchain (guard logic unit-tested) |
| `fmm3d_comparison.ipynb` | exploratory sweep: spherical `CUDA_FULL` at orders 4-10 and depths 3-5 against FMM3D tolerances 1e-3 and 1e-4 on 20k-60k coincident points |

```console
conda env update -n cdfmm -f environment.yml
conda env update -n cdfmm -f environment-cuda.yml
conda env update -n cdfmm -f environment-fmm3d.yml
conda activate cdfmm
cmake --fresh --preset notebooks
cmake --build --preset notebooks -j
./benchmarks/external/fmm3d/install_fmm3d.sh
jupyter lab benchmarks/external/fmm3d/fmm3d_comparison.ipynb
```

The installer keeps its source checkout below the ignored `.external/`
directory and is safe to rerun.
