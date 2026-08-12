#!/usr/bin/env bash
set -euo pipefail

readonly FMM3D_VERSION="2.1.0"
readonly FMM3D_TAG="v${FMM3D_VERSION}"
readonly FMM3D_URL="https://github.com/flatironinstitute/FMM3D.git"

if [[ "${CONDA_DEFAULT_ENV:-}" != "cdfmm" || -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the cdfmm Conda environment before installing FMM3D." >&2
    exit 2
fi

case "$(command -v python)" in
    "${CONDA_PREFIX}"/*) ;;
    *)
        echo "The active Python executable is not from ${CONDA_PREFIX}." >&2
        echo "Reactivate the cdfmm Conda environment and rerun this installer." >&2
        exit 2
        ;;
esac

readonly INSTALL_MARKER="${CONDA_PREFIX}/.cdfmm-fmm3d-${FMM3D_VERSION}"

if [[ -f "${INSTALL_MARKER}" ]] && python -c "import fmm3dpy"
then
    echo "FMM3D ${FMM3D_VERSION} is already installed in ${CONDA_PREFIX}."
    exit 0
fi

if ! python <<'PY'
import numpy
import setuptools

raise SystemExit(
    0
    if int(numpy.__version__.split(".", 1)[0]) < 2
    and int(setuptools.__version__.split(".", 1)[0]) < 60
    else 1
)
PY
then
    echo "FMM3D 2.1.0 requires numpy<2 and setuptools<60 to build." >&2
    echo "Run: conda env update -n cdfmm -f environment-fmm3d.yml" >&2
    exit 4
fi

readonly REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
readonly DEPENDENCY_ROOT="${REPOSITORY_ROOT}/.external"
readonly SOURCE_DIR="${DEPENDENCY_ROOT}/fmm3d-${FMM3D_VERSION}"
readonly GNU_CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
readonly GNU_CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
readonly GNU_FC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gfortran"

for compiler in "${GNU_CC}" "${GNU_CXX}" "${GNU_FC}"; do
    if [[ ! -x "${compiler}" ]]; then
        echo "Required GNU compiler is missing: ${compiler}" >&2
        echo "Rerun: conda env update -n cdfmm -f environment-fmm3d.yml" >&2
        exit 4
    fi
done

mkdir -p "${DEPENDENCY_ROOT}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git clone --depth 1 --branch "${FMM3D_TAG}" \
        "${FMM3D_URL}" "${SOURCE_DIR}"
fi

if [[ "$(git -C "${SOURCE_DIR}" describe --tags --exact-match 2>/dev/null)" \
      != "${FMM3D_TAG}" ]]; then
    echo "${SOURCE_DIR} is not the pinned ${FMM3D_TAG} checkout." >&2
    echo "Remove that directory and rerun this installer." >&2
    exit 3
fi

# The cdfmm environment also contains Intel's C++ compiler. NumPy distutils
# otherwise selects icx for generated f2py C code while GNU Fortran performs
# the final link, leaving unresolved Intel runtime symbols. Rebuild every
# native FMM3D object with one explicit Conda GNU toolchain.
make -C "${SOURCE_DIR}" objclean \
    CC="${GNU_CC}" CXX="${GNU_CXX}" FC="${GNU_FC}"
rm -f -- "${SOURCE_DIR}/lib-static/libfmm3d.a"
make -C "${SOURCE_DIR}" libfmm3d.a FAST_KER=ON \
    CC="${GNU_CC}" CXX="${GNU_CXX}" FC="${GNU_FC}"
(
    cd "${SOURCE_DIR}/python"
    export CC="${GNU_CC}"
    export CXX="${GNU_CXX}"
    export FC="${GNU_FC}"
    export F77="${GNU_FC}"
    export FMM_FLIBS="-lm -lstdc++ -lgomp -fopenmp"
    # Discard generated f2py sources from any interrupted build. In particular,
    # they must not be reused if pip previously selected a different NumPy API.
    python setup.py clean --all
    rm -rf -- "${SOURCE_DIR}/python/build"
    python -m pip install --no-build-isolation . --verbose
)

python <<'PY'
import importlib.metadata

import fmm3dpy

installed = importlib.metadata.version("fmm3dpy")
print(f"Imported fmm3dpy distribution {installed} successfully.")
PY

# Upstream v2.1.0 retains distribution metadata version 1.0.0. The marker
# records the source release installed into this exact Conda environment.
printf '%s\n' "${FMM3D_TAG}" > "${INSTALL_MARKER}"
echo "Installed FMM3D source release ${FMM3D_VERSION} successfully."
