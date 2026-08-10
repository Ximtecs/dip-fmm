# Installation and building

This guide starts from a clean source checkout.  It assumes only Git, internet
access, and Conda; [Miniforge](https://github.com/conda-forge/miniforge) is the
recommended Conda distribution because it uses conda-forge by default.

## Optional oneMKL backend

The portable build uses an exact internal grouped dense kernel. For Intel
performance builds, use oneMKL's CMake package:

```console
cmake -S . -B build-mkl -DCMAKE_CXX_COMPILER=icpx \
  -DCDFMM_ENABLE_MKL=ON -DCDFMM_ENABLE_OPENMP=ON
cmake --build build-mkl -j
```

Static M2L issues one DGEMM per transfer class. It does not place an OpenMP
region around DGEMM, preventing accidental OpenMP-by-MKL multiplication. Set
and record `OMP_NUM_THREADS` and `MKL_NUM_THREADS` explicitly when benchmarking.

## Quick start

Clone the project and enter the checkout:

```console
git clone https://github.com/Ximtecs/dip-fmm.git
cd dip-fmm
```

**Unless stated otherwise, every command in this guide is run from the
repository root:** the directory containing `CMakeLists.txt`, `pyproject.toml`,
and `environment.yml`.

Create the comprehensive development environment once, then activate it:

```console
conda env create -f environment.yml
conda activate cdfmm
```

`mamba env create -f environment.yml` is an interchangeable, often faster,
first command when Mamba is available.  Configure, build, and test from the
same repository-root directory:

```console
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
python -m pip install .
python -m pytest python_tests -v
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The initial CMake configuration may download Catch2 and pybind11.  The final
command writes the documentation homepage to
`<repo-root>/docs/_build/html/index.html`.

## Repository root and generated directories

```text
dip-fmm/                         <- <repo-root>
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
├── environment.yml
├── include/
├── src/
├── python/
├── tests/
├── python_tests/
├── examples/
├── benchmarks/
└── docs/
```

In this guide, `<repo-root>` means the `dip-fmm/` directory above.  Commands
use `-S .` because the current directory is `<repo-root>`, not its parent,
`<repo-root>/build/`, or `<repo-root>/docs/`.  Generated files belong in such
directories as `<repo-root>/build/` and `<repo-root>/docs/_build/`; do not run
the commands by changing into those directories.

## Development environment

The intended workflow is:

```text
clone repository -> create environment once -> activate cdfmm -> configure/build
                 -> develop -> run tests -> build documentation
```

For later sessions, environment creation is unnecessary.  Run:

```console
cd <path-to>/dip-fmm
conda activate cdfmm
```

(updating-an-existing-environment)=
### Updating an existing environment

After pulling repository changes, synchronise the existing `cdfmm` environment
with the current `environment.yml` from `<repo-root>`:

```console
conda env update --name cdfmm --file environment.yml --prune
conda activate cdfmm
```

This installs the newest compatible versions allowed by `environment.yml`, adds
new dependencies, and, because of `--prune`, removes dependencies that are no
longer required by the file.  The environment does not need to be deleted and
created again.  If Mamba is available, `mamba env update --name cdfmm --file
environment.yml --prune` is an interchangeable, often faster command.

The readable `environment.yml` specifies the direct tools, rather than pinning
every machine-specific transitive package.  It uses Python 3.11 as the
reproducible development version; the package metadata supports Python 3.9 and
newer.

| Component | Purpose |
|---|---|
| Intel oneAPI `icpx` | Compile the high-performance C++20 library, benchmarks, and Python extension |
| CMake | Configure the C++ project |
| Ninja | Provide one cross-platform CMake build backend |
| Python | Run the bindings and development tools |
| NumPy | Supply Python numerical arrays |
| Matplotlib | Plot operator convergence and uniform-tree geometry |
| JupyterLab | Run the interactive scientific examples |
| ipykernel | Expose the `cdfmm` environment as a notebook kernel |
| ipywidgets | Provide optional interactive notebook controls |
| pybind11 | Implement the C++/Python binding layer |
| scikit-build-core | Drive CMake when building the Python package |
| pytest | Run Python tests |
| Doxygen | Extract the public C++ API as XML |
| Sphinx | Generate the documentation site |
| Breathe | Integrate Doxygen XML into Sphinx |
| MyST parser | Add Markdown support to Sphinx |
| sphinx-rtd-theme | Style the generated documentation |

`docs/requirements.txt` is deliberately retained for Read the Docs and for
documentation-only installations outside Conda.  Its four Python constraints
match `environment.yml`; normal Conda development does **not** require
`python -m pip install -r docs/requirements.txt`.

The notebook stack is part of the development environment, not the core
package dependency set. Outside Conda, it can be installed explicitly with
`python -m pip install ".[examples]"`.

### Interactive examples

After updating or creating the development environment, launch Jupyter from
`<repo-root>`:

```console
conda env update -n cdfmm -f environment.yml
conda activate cdfmm
jupyter lab
```

Open `examples/notebooks/00_direct_p2p.ipynb` to begin the operator sequence.
JupyterLab and VSCode should offer `Python (cdfmm)` as the kernel. If a local
Conda installation does not register it automatically, run:

```console
python -m ipykernel install --user --name cdfmm --display-name "Python (cdfmm)"
```

The notebooks use Matplotlib and ipywidgets for optional visual controls while
calling the compiled C++ operator implementations for all numerical work.

### Compiler strategy and platform status

The environment requests Intel oneAPI's C++ compiler package. On Linux this
provides `icpx`, the supported compiler for performance development and
benchmark reproduction. Portable standards-compliant source remains buildable
with other C++20 compilers, but benchmark comparisons must record the compiler
metadata emitted by the executable. The oneAPI Conda path in this repository
has not been validated on native Windows; Windows toolchain validation remains
tracked in the [roadmap](roadmap.md).

## CMake builds

### Normal development build

From `<repo-root>`, with `cdfmm` activated, the quick-start configuration
builds tests, examples, and Python bindings (their project defaults are `ON`):

```console
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Here `-S .` selects the current repository as the source tree; `-B build`
writes generated files to `<repo-root>/build/`; `-G Ninja` selects Ninja from
the Conda environment; and `-DCMAKE_BUILD_TYPE=Release` enables optimised
compilation.  `cmake --build build -j 4`, for example, permits four concurrent
build jobs; omitting `-j` is simpler and lets the backend choose its default.
Examples are written beneath `<repo-root>/build/examples/`.

The equivalent convenience interface is:

```console
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The checked-in presets use Ninja and separate binary directories.  They are
shortcuts, not a separate build system; the explicit commands remain useful
for understanding or customising a build.

### Recompiling after source changes

After changing a C++ source or header file, rebuild the existing development
tree from `<repo-root>`:

```console
conda activate cdfmm
cmake --build build
ctest --test-dir build --output-on-failure
```

The build is incremental: Ninja recompiles only the affected files and then
relinks the necessary targets.  There is normally no need to delete `build/` or
repeat the initial CMake configuration.  The preset equivalents are:

```console
cmake --build --preset dev
ctest --preset dev
```

CMake normally detects changes to `CMakeLists.txt` and regenerates the build
files automatically.  When changing CMake options, or if automatic regeneration
does not occur, configure again before rebuilding:

```console
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The manually built Python extension remains under `build/`; it does not replace
a copy previously installed into the Conda environment.  To run Python code or
Python tests against the latest C++ changes using the documented installed
package workflow, rebuild and reinstall the extension before testing:

```console
python -m pip install . --no-deps --force-reinstall
python -m pytest python_tests -v
```

For a change confined to Python test files, no compilation or reinstallation is
needed; rerun `python -m pytest python_tests -v` directly.

### C++-only build

This separate build disables Python and tests while retaining examples:

```console
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCDFMM_BUILD_PYTHON=OFF -DCDFMM_BUILD_TESTS=OFF
cmake --build build-release
```

The equivalent commands are `cmake --preset release` and
`cmake --build --preset release`.

### Python-focused build and installation

From `<repo-root>`, the following command asks pip and scikit-build-core to
configure CMake, compile the native `cdfmm` extension, and install it into the
currently active `cdfmm` environment:

```console
python -m pip install .
```

Pip normally creates an isolated build environment using the requirements in
`pyproject.toml`; scikit-build-core and pybind11 are also present in Conda for
interactive development and offline diagnosis.  The `[tool.scikit-build]`
arguments enable Python while disabling C++ tests and examples, so this
package build is independent of the manual `<repo-root>/build/` tree.  NumPy is
installed as the package's runtime dependency.

Run the Python tests after installation:

```console
python -m pytest python_tests -v
```

An editable scikit-build-core installation may be useful in future, but it is
not the documented default until rebuild behaviour for this native extension
has been validated on every development platform.

### Benchmark build

Keep benchmark configuration isolated from normal development:

```console
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DCDFMM_BUILD_BENCHMARKS=ON -DCDFMM_BUILD_TESTS=OFF -DCDFMM_BUILD_EXAMPLES=OFF -DCDFMM_BUILD_PYTHON=OFF
cmake --build build-bench
```

The executable is `<repo-root>/build-bench/benchmarks/benchmark_p2p` on Linux
and normally has an `.exe` suffix on Windows.  The equivalent preset commands
are `cmake --preset benchmark` and `cmake --build --preset benchmark`.

## Building the documentation

All documentation dependencies, including Doxygen, are already in the Conda
environment.  From `<repo-root>`, run:

```console
conda activate cdfmm
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Sphinx invokes Doxygen automatically.  Open
`<repo-root>/docs/_build/html/index.html` locally after a successful build.
Doxygen XML is generated under `<repo-root>/docs/_build/doxygen/`; both output
directories are generated artefacts and intentionally are not committed.  Read
the Docs uses `.readthedocs.yaml` and the separate `docs/requirements.txt`.

## Troubleshooting

### The environment already exists

Do not run `conda env create` again.  Follow
[Updating an existing environment](#updating-an-existing-environment) to
synchronise the existing `cdfmm` environment with `environment.yml`.

### The wrong Python is active

On Linux/macOS run `which python`; in Windows PowerShell run
`where.exe python`.  The first result should be inside the `cdfmm` environment.
Confirm the main tools with:

```console
python --version
cmake --version
ninja --version
doxygen --version
clang++ --version
```

On Windows the Clang driver may be named `clang-cl`; use
`where.exe clang++` and `where.exe clang-cl` to inspect both possibilities.

### CMake remembers an old compiler

CMake caches absolute compiler paths.  After changing environments or
compilers, safely remove the generated tree from `<repo-root>` and configure it
again:

```console
cmake -E remove_directory build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

### No compiler is found

First reactivate `cdfmm`, check the compiler commands above, and remove the old
build directory.  On Linux, updating the Conda environment should restore the
Conda-supplied Clang.  On Windows, Conda does not promise a complete MSVC/SDK
installation: install the Visual Studio C++ workload described above if the
diagnostic concerns SDK headers, libraries, `link.exe`, or runtime components.
When reporting a problem, include `conda list clangxx`, the compiler version,
and the complete CMake diagnostic.

## Intel oneAPI performance build and OpenMP

The reference performance compiler is Intel oneAPI `icpx` (the C++ driver, not
the `ifx` Fortran compiler). `environment.yml` supplies the oneAPI DPC++/C++
compiler package on the supported Linux development path. Build the optimised
benchmark configuration with:

```console
conda activate cdfmm
cmake --preset benchmark
cmake --build --preset benchmark
OMP_NUM_THREADS=16 ./build-bench/benchmarks/benchmark_uniform_fmm --help
```

The benchmark preset selects `icpx`, OpenMP, Release `-O3`, native CPU tuning,
and IPO when supported. For explicit configuration, the equivalent core
settings are `CXX=icpx cmake -S . -B build-bench -G Ninja
-DCMAKE_BUILD_TYPE=Release -DCDFMM_BUILD_BENCHMARKS=ON
-DCDFMM_ENABLE_OPENMP=ON`. Disable internal parallelism with
`-DCDFMM_ENABLE_OPENMP=OFF`; otherwise use `OMP_NUM_THREADS` or the benchmark's
`--threads` option. See [Performance benchmarks](benchmarks.md) for the complete
runner workflow.
