#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Optional validation dependency; normal dip-fmm builds never import MagTense.
python -m pip install "magtense==2.2.1"
python -c "import magtense; print('MagTense import succeeded')"
