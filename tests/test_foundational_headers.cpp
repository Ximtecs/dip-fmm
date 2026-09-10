// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

// Canonical foundational interfaces.
#include "cdfmm/core/output_flags.hpp"
#include "cdfmm/core/precision.hpp"
#include "cdfmm/core/timing.hpp"
#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/coefficients.hpp"
#include "cdfmm/math/laplace_derivatives.hpp"
#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/pair_tensor.hpp"
#include "cdfmm/math/potential_field.hpp"
#include "cdfmm/math/spherical_harmonics.hpp"
#include "cdfmm/math/taylor_jet.hpp"
#include "cdfmm/math/vec3.hpp"
#include "cdfmm/tree/adaptive_tree.hpp"
#include "cdfmm/tree/indexing.hpp"
#include "cdfmm/tree/morton.hpp"
#include "cdfmm/tree/node.hpp"
#include "cdfmm/tree/statistics.hpp"
#include "cdfmm/tree/uniform_tree.hpp"

// Pre-v0.2 compatibility include paths.
#include "cdfmm/adaptive_tree.hpp"
#include "cdfmm/coefficients.hpp"
#include "cdfmm/geometry.hpp"
#include "cdfmm/laplace_derivatives.hpp"
#include "cdfmm/multi_index.hpp"
#include "cdfmm/output_flags.hpp"
#include "cdfmm/precision.hpp"
#include "cdfmm/rectangular_prism.hpp"
#include "cdfmm/spherical_harmonics.hpp"
#include "cdfmm/taylor_jet.hpp"
#include "cdfmm/tetrahedron.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/tree_node.hpp"
#include "cdfmm/uniform_tree.hpp"
#include "cdfmm/vec3.hpp"

TEST_CASE("Canonical foundational headers coexist with compatibility shims")
{
    cdfmm::PhaseTiming timing;
    timing.add(0.25);

    const cdfmm::Vec3 displacement{1.0, 2.0, 3.0};
    const cdfmm::RectangularPrism prism{2.0, 3.0, 4.0};
    const cdfmm::PairTensor tensor{};

    CHECK(timing.calls == 1);
    CHECK(prism.volume() == 24.0);
    CHECK(cdfmm::dot(displacement, displacement) == 14.0);
    CHECK(tensor.xx == 0.0);
    CHECK(cdfmm::has_flag(cdfmm::OutputFlags::Both,
                          cdfmm::OutputFlags::Field));
}
