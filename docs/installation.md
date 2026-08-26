# Installation and building

This guide begins with the recommended complete installation. It then covers
CPU-only and oneMKL-free variants, specialised presets, routine updates, clean
resets, and troubleshooting.

The commands assume Linux, Git, internet access, and Conda. Miniforge is the
recommended Conda distribution. The default release build also requires an
NVIDIA GPU with a driver compatible with the CUDA toolkit declared in
`environment-cuda.yml`.

## Install from scratch

The recommended installation is an optimised release build with both CUDA and
oneMKL enabled. It installs the native library, headers, and Python extension
into the `cdfmm` Conda environment.

### 1. Clone the repository

```console
git clone https://github.com/Ximtecs/dip-fmm.git
cd dip-fmm
```

All remaining commands in this guide are run from this repository root: the
directory containing `CMakeLists.txt`, `CMakePresets.json`, and
`environment.yml`.

### 2. Create the Conda environment

Create the base environment, add the CUDA development toolkit, and activate
the result:

```console
conda env create --file environment.yml
conda env update --name cdfmm --file environment-cuda.yml
conda activate cdfmm
```

The base file supplies CMake, Ninja, C++ and Python development tools, oneMKL,
the test stack, documentation tools, and Jupyter. The supplemental CUDA file
adds a consistent NVIDIA compiler, headers, runtime, cuBLAS, and NVTX toolkit.
`mamba` may be substituted for `conda` in the create and update commands.

### 3. Configure CMake

```console
cmake --fresh --preset release
```

This creates `build-release/` and configures an optimised build with CUDA,
oneMKL, the Python extension, examples, and tests. `--fresh` discards any old
CMake cache for this build directory, without touching source files.

### 4. Compile and install

```console
cmake --build --preset release -j
```

This is the only compilation command required. The `release` build preset
targets `install`, so it also places the C library, public headers, and Python
extension in the active Conda environment. Restart any existing Python,
Jupyter, or VSCode kernel after rebuilding so it loads the new extension.

Confirm that the installed module exposes both accelerated backends:

```console
python -c "import cdfmm; print('CUDA:', cdfmm.cuda_full_available()); print('oneMKL:', cdfmm.one_mkl_available())"
```

Both values should be `True`. If CUDA is unavailable, first check that the
machine has a supported NVIDIA GPU and that `nvidia-smi` can see it.

### 5. Optionally run the tests

Testing is recommended after the first installation, but is not required to
compile or install the project:

```console
ctest --preset release
python -m pytest python_tests -v
```

The first command runs the compiled C++ suite. The second tests the installed
Python module from the active environment.

The complete default workflow is therefore:

```console
git clone https://github.com/Ximtecs/dip-fmm.git
cd dip-fmm
conda env create --file environment.yml
conda env update --name cdfmm --file environment-cuda.yml
conda activate cdfmm
cmake --fresh --preset release
cmake --build --preset release -j

# Optional validation
ctest --preset release
python -m pytest python_tests -v
```

## Choose CUDA and oneMKL features

The default `release` preset enables both acceleration systems. CMake options
can disable either one while retaining the same optimised, installed release
layout.

### Release build without CUDA

CUDA is not required for a portable CPU build. Skip `environment-cuda.yml`
when creating the environment, then configure the release preset with CUDA
disabled:

```console
conda env create --file environment.yml
conda activate cdfmm
cmake --fresh --preset release -DCDFMM_ENABLE_CUDA=OFF
cmake --build --preset release -j
ctest --preset release  # Optional
```

This retains oneMKL and installs the Python extension, library, and headers.

### Release build without oneMKL

```console
cmake --fresh --preset release -DCDFMM_ENABLE_MKL=OFF
cmake --build --preset release -j
ctest --preset release  # Optional
```

CUDA remains enabled. CPU static-matrix work uses the portable internal
backend instead of oneMKL.

### Release build without CUDA or oneMKL

```console
cmake --fresh --preset release \
  -DCDFMM_ENABLE_CUDA=OFF \
  -DCDFMM_ENABLE_MKL=OFF
cmake --build --preset release -j
ctest --preset release  # Optional
```

This is the smallest feature variant of the same installed release build. For
day-to-day CPU development without installation, the `dev` preset described
below is usually more convenient.

## Available CMake presets

Every checked-in preset uses the same configure/build pattern:

```console
cmake --fresh --preset <preset>
cmake --build --preset <preset> -j
```

Use the same name for both commands.

| Preset | Build directory | Intended use | CUDA | oneMKL | Installs into active environment | Test command |
| --- | --- | --- | --- | --- | --- | --- |
| `release` | `build-release/` | Recommended optimised library, Python, examples, and tests | Yes | Yes | Yes | `ctest --preset release` |
| `dev` | `build/` | Portable CPU development with library, Python, examples, and tests | No | No | No | `ctest --preset dev` |
| `cuda` | `build-cuda/` | CUDA validation and benchmarks without oneMKL | Yes | No | Yes | `ctest --preset cuda` |
| `notebooks` | `build-notebooks/` | CUDA and oneMKL notebook/benchmark work | Yes | Yes | Yes | `ctest --preset notebooks` |
| `magtense` | `build-magtense/` | MagTense comparison configuration | Yes | Yes | Yes | `ctest --preset magtense` |
| `benchmark` | `build-bench/` | Portable CPU benchmarks with `icpx`, OpenMP, and LTO | No | No | No | None |
| `benchmark-mkl` | `build-bench-mkl/` | CPU benchmarks with oneMKL | No | Yes | No | None |
| `benchmark-all` | `build-bench-all/` | Portable CPU, oneMKL, and CUDA benchmarks | Yes | Yes | No | None |
| `profile-all` | `build-profile-all/` | Combined CUDA/oneMKL build with NVTX and debug symbols | Yes | Yes | No | None |

Only presets listed with a test command compile the C++ test suite. The
`release`, `cuda`, `notebooks`, and `magtense` build presets target `install`;
the other presets leave their outputs only in the stated build directory.

## Routine updates and rebuilds

(updating-an-existing-environment)=
### Update an existing environment

After pulling repository changes, synchronise the existing environment rather
than recreating it:

```console
conda env update --name cdfmm --file environment.yml --prune
conda env update --name cdfmm --file environment-cuda.yml
conda activate cdfmm
```

Omit the CUDA update for a CPU-only installation. The base update uses
`--prune` to remove packages that are no longer declared; the CUDA update is
applied afterwards so its supplemental packages are retained.

### Recompile after source changes

An existing build is incremental. Ninja recompiles only affected files:

```console
conda activate cdfmm
cmake --build --preset release -j
```

You normally do not need to reconfigure. Use `cmake --fresh --preset release`
again after changing compilers, environments, or major CMake feature options.

### Build the documentation

```console
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The generated homepage is `docs/_build/html/index.html`.

### Start the notebooks

The notebook stack is already included in `environment.yml`:

```console
conda activate cdfmm
jupyter lab
```

Run Jupyter from the repository root and select the `Python (cdfmm)` kernel.
If the kernel is not registered automatically, run:

```console
python -m ipykernel install --user --name cdfmm --display-name "Python (cdfmm)"
```

## Other installation options

### Python-only portable installation

For a portable Python package without the preset-managed CUDA/oneMKL build:

```console
conda env create --file environment.yml
conda activate cdfmm
python -m pip install .
python -m pytest python_tests -v  # Optional
```

Pip uses scikit-build-core to compile and install the native extension. This
workflow is convenient for Python use, but the `release` preset is the
recommended route when CUDA and oneMKL are required.

### FMM3D notebook comparison

FMM3D 2.1.0 has additional compiler and NumPy constraints. Add its environment
overlay and run the checked-in installer after completing the main setup:

```console
conda env update --name cdfmm --file environment-fmm3d.yml
conda activate cdfmm
./examples/notebooks/install_fmm3d.sh
```

See [Examples and notebooks](examples.md) for the associated comparison
workflow.

### MagTense comparison environment

MagTense uses a separate, pinned environment because its Python and runtime
requirements differ from the main development environment:

```console
conda env create --file environment-magtense.yml
conda activate cdfmm-magtense
```

Use the `magtense` preset when configuring cdfmm for those comparisons. A CUDA
toolkit must also be available because that preset enables CUDA.

### C and Fortran integration

The C ABI shared library and public header are built unconditionally. Enable
the optional modern Fortran wrapper while configuring any suitable preset:

```console
cmake --fresh --preset release -DCDFMM_BUILD_FORTRAN_INTERFACE=ON
cmake --build --preset release -j
```

See [C and Fortran integration](fortran-interface.md) for compiler and linking
details.

### Manual CMake configuration

Presets are recommended because they keep compiler and feature choices
consistent. A minimal portable development build can also be configured
explicitly:

```console
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCDFMM_ENABLE_CUDA=OFF \
  -DCDFMM_ENABLE_MKL=OFF \
  -DCDFMM_BUILD_TESTS=ON \
  -DCDFMM_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

(clean-resets)=
## Clean resets

### Reset only the CMake configuration

The normal clean reconfiguration keeps compiled artefacts but discards cached
CMake choices:

```console
cmake --fresh --preset release
cmake --build --preset release -j
```

Use this first for stale compiler paths or changed build options.

### Remove and rebuild one generated build directory

For a completely clean release compilation, remove only its generated build
directory and recreate it:

```console
cmake -E remove_directory build-release
cmake --fresh --preset release
cmake --build --preset release -j
```

This does not remove source files or the Conda environment. Other preset build
directories are independent and can be removed in the same way when needed.

### Recreate the Conda environment from scratch

Use this only when updating the existing environment does not resolve a broken
or inconsistent installation:

```console
conda deactivate
conda env remove --name cdfmm
conda env create --file environment.yml
conda env update --name cdfmm --file environment-cuda.yml
conda activate cdfmm
cmake -E remove_directory build-release
cmake --fresh --preset release
cmake --build --preset release -j
```

Omit the CUDA environment update and configure with
`-DCDFMM_ENABLE_CUDA=OFF` when recreating a CPU-only installation.

## Troubleshooting

### The environment already exists

Do not run `conda env create` again. Follow
[Update an existing environment](#updating-an-existing-environment), then
reconfigure with `cmake --fresh --preset release` if build options changed.

### The wrong Python module is imported

Check that the active executables and imported extension belong to `cdfmm`:

```console
which python
python -c "import cdfmm; print(cdfmm.__file__)"
```

Reactivate the environment and rebuild the installing preset if either path
points elsewhere. Restart long-running Python and Jupyter processes after an
installation.

### The oneMKL backend is unavailable

The default release configuration must report `CDFMM_ENABLE_MKL=ON` during
CMake configuration. Check the active environment and oneMKL package metadata:

```console
test -f "$CONDA_PREFIX/lib/cmake/mkl/MKLConfig.cmake"
python -c "import cdfmm; print(cdfmm.one_mkl_available())"
```

If necessary, update `environment.yml`, run a fresh release configuration,
and rebuild. A CPU-only build and an MKL-free build are different choices:
`CDFMM_ENABLE_CUDA=OFF` does not disable oneMKL.

### CUDA configuration fails

Confirm the driver and Conda toolkit before reconfiguring:

```console
nvidia-smi
nvcc --version
cmake --fresh --preset release
```

The presets clear Conda's `NVCC_PREPEND_FLAGS`, use `g++` as the CUDA host
compiler, and compile for the local GPU with
`CMAKE_CUDA_ARCHITECTURES=native`. The installed NVIDIA driver must support the
toolkit version pinned in `environment-cuda.yml`.

### CMake remembers an old compiler

First use `cmake --fresh --preset <preset>`. If that is insufficient, remove
only the corresponding generated build directory as described in
[Clean resets](#clean-resets), then configure it again from the repository
root.
