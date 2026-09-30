# SPDX-License-Identifier: Apache-2.0
"""jaxFMM GPU comparison campaign for dip-fmm.

The package is deliberately split by the Python environment that may import
each module:

* ``geometry``, ``reference``, ``environment``, ``timing``, ``locking`` and
  ``results`` depend on NumPy only and are imported from every process;
* ``jaxfmm_dipole`` imports JAX and jaxFMM and runs only in the ``jaxfmm``
  Conda environment;
* ``dipfmm_runner`` imports the ``cdfmm`` extension and runs only in the
  ``cdfmm`` environment with ``PYTHONPATH`` pointing at a frozen build.

Nothing here modifies the solver; the campaign is a consumer of both codes.
"""

CAMPAIGN_NAME = "jaxfmm_gpu_comparison"
CAMPAIGN_REVISION = "jaxfmm_fp32_order6_v1"
