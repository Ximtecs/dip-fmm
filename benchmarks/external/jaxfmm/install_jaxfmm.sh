#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Pinned jaxFMM environment for the GPU comparison campaign.
#
# Creates the Conda prefix ~/.conda/envs/jaxfmm (Python 3.12) with JAX and its
# CUDA 13 wheels, clones jaxFMM at the pinned release tag below the ignored
# .external/ directory and installs it editable so the benchmarked source is
# exactly the inspected checkout. Safe to rerun. Runs on the efficiency cores.
set -euo pipefail

readonly JAXFMM_TAG="v0.3.3"
readonly JAXFMM_URL="https://gitlab.com/jaxfmm/jaxfmm.git"
readonly JAX_PIN="0.11.2"
readonly PYTHON_VERSION="3.12"
readonly CONDA_BIN="${CONDA_EXE:-/opt/software/miniforge/latest/bin/conda}"
readonly PREFIX="${JAXFMM_ENV_PREFIX:-$HOME/.conda/envs/jaxfmm}"
readonly EFFICIENCY_CORES="16-31"

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
readonly REPOSITORY_ROOT
readonly SOURCE_DIR="${REPOSITORY_ROOT}/.external/jaxfmm"

mkdir -p "${REPOSITORY_ROOT}/.external"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git clone -q "${JAXFMM_URL}" "${SOURCE_DIR}"
fi
git -C "${SOURCE_DIR}" checkout -q "${JAXFMM_TAG}"
if [[ "$(git -C "${SOURCE_DIR}" describe --tags --exact-match 2>/dev/null)" != "${JAXFMM_TAG}" ]]; then
    echo "${SOURCE_DIR} is not at ${JAXFMM_TAG}" >&2
    exit 3
fi

if [[ ! -x "${PREFIX}/bin/python" ]]; then
    taskset -c "${EFFICIENCY_CORES}" "${CONDA_BIN}" create -y -p "${PREFIX}" -c conda-forge \
        "python=${PYTHON_VERSION}" pip
fi
taskset -c "${EFFICIENCY_CORES}" "${PREFIX}/bin/python" -m pip install --upgrade pip
taskset -c "${EFFICIENCY_CORES}" "${PREFIX}/bin/python" -m pip install \
    "jax[cuda13]==${JAX_PIN}" numpy scipy pytest matplotlib
taskset -c "${EFFICIENCY_CORES}" "${PREFIX}/bin/python" -m pip install --no-deps -e "${SOURCE_DIR}"

JAX_PLATFORMS=cpu "${PREFIX}/bin/python" - <<'PY'
import importlib.metadata as md
import jax, jaxlib, jaxfmm
print(f"jax {jax.__version__}, jaxlib {jaxlib.__version__}, jaxFMM {md.version('jaxFMM')} from {jaxfmm.__file__}")
PY
echo "jaxFMM ${JAXFMM_TAG} environment ready at ${PREFIX}"
