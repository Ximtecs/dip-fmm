// SPDX-License-Identifier: Apache-2.0
#pragma once

// Compatibility umbrella for the original flat cuboid include.  It owns no
// declarations of its own: the prism record, the legacy `CuboidSize` spelling,
// and the averaged-monomial mathematics are canonical under
// cdfmm/geometry/primitives/, the pair-tensor construction is canonical under
// cdfmm/operators/, and `DenseDirectPlan` is canonical under cdfmm/plan/.
// The include set reproduces the pre-v0.2 transitive surface so that this path
// remains source compatible.
#include "cdfmm/core/precision.hpp"
#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/coefficients.hpp"
#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/pair_tensor.hpp"
#include "cdfmm/math/vec3.hpp"
#include "cdfmm/operators/p2p.hpp"
#include "cdfmm/plan/direct/dense.hpp"
