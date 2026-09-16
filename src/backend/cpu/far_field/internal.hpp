// SPDX-License-Identifier: Apache-2.0
#pragma once

// Internal CPU far-field execution: the hierarchy runs on the derived
// level-scaled/dense packing declared in packing.hpp.  The public
// apply_static_operator / apply_static_l2p_evaluator entry-map functions in
// cdfmm/backend/cpu/far_field.hpp remain the canonical reference kernels.
#include "backend/cpu/far_field/packing.hpp"
