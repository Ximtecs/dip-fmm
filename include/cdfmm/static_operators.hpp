// SPDX-License-Identifier: Apache-2.0
#pragma once

// Compatibility umbrella. New code should include the responsibility-specific
// operator, plan, or CPU execution header directly.  The flat includes below
// reproduce the pre-v0.2 transitive surface of this path, which carried the
// geometry, pair-tensor, dense-direct, and flat dynamic-operator declarations.
#include "cdfmm/backend/cpu/static_plan_apply.hpp"
#include "cdfmm/cuboid.hpp"
#include "cdfmm/geometry.hpp"
#include "cdfmm/operators.hpp"
#include "cdfmm/operators/operators.hpp"
#include "cdfmm/periodic.hpp"
#include "cdfmm/plan/static_plan.hpp"
#include "cdfmm/precision.hpp"
#include "cdfmm/spherical_harmonics.hpp"
#include "cdfmm/tensor_dictionary.hpp"
#include "cdfmm/tetrahedron.hpp"
