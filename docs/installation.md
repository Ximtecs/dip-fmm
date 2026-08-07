# Installation and building

This guide starts from a clean source checkout.  It assumes only Git, internet
access, and Conda; [Miniforge](https://github.com/conda-forge/miniforge) is the
recommended Conda distribution because it uses conda-forge by default.

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

The readable `environment.yml` specifies the direct tools, rather than pinning
every machine-specific transitive package.  It uses Python 3.11 as the
reproducible development version; the package metadata supports Python 3.9 and
newer.

| Component | Purpose |
|---|---|
| Clang/Clang++ | Compile the C++20 `cdfmm_core` library and Python extension |
| CMake | Configure the C++ project |
| Ninja | Provide one cross-platform CMake build backend |
| Python | Run the bindings and development tools |
| NumPy | Supply Python numerical arrays |
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

### Compiler strategy and platform status

The environment requests conda-forge's `clangxx` package rather than the
`cxx-compiler` metapackage.  On Linux this provides the Clang/Clang++ programs
inside the environment and avoids requiring a separately installed GCC.
`cxx-compiler` is useful for conda-forge package recipes, but its native
Windows toolchain normally selects MSVC and does not itself make Visual Studio,
the MSVC libraries, and the Windows SDK disappear as external prerequisites.

Conda-forge also publishes Clang packages for Windows, but a native Windows
link still needs compatible Windows SDK and C/C++ runtime components.  This
repository has not yet verified `environment.yml` on a clean Windows machine;
therefore it does **not** claim a fully self-contained or CI-supported Windows
toolchain.  Install Visual Studio Build Tools with the **Desktop development
with C++** workload if Clang reports missing Windows headers, libraries, or a
linker.  Linux is the current development path; Windows validation is tracked
in the [roadmap](roadmap.md).

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

Update it from `<repo-root>` after `environment.yml` changes:

```console
conda env update -f environment.yml --prune
conda activate cdfmm
```

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
