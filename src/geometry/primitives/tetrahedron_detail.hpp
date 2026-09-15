// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm::detail {

struct PreparedTetrahedron {
    double volume{0.0};

    // Representative-relative vertices.
    std::array<Vec3, 4> vertices{};

    // Representative-relative triangular faces.
    std::array<std::array<Vec3, 3>, 4> faces{};

    // Unit outward normal corresponding to each face.
    std::array<Vec3, 4> outward_normals{};
};

[[nodiscard]] PreparedTetrahedron prepare_tetrahedron(
    const Tetrahedron& tetrahedron);

[[nodiscard]] PairTensor tetrahedron_tetrahedron_tensor_prepared(
    const Vec3& target_minus_source_representative,
    const PreparedTetrahedron& source,
    const PreparedTetrahedron& target,
    bool coincident_same_geometry);

/**
 * @brief Evaluates the unnormalised constant-density 1/R triangle pair
 * integral.
 *
 * This header is intentionally private to analytical-kernel tests.  The
 * production geometry API exposes only the resulting tetrahedron tensor.
 */
[[nodiscard]] double triangle_triangle_laplace_integral(
    const std::array<Vec3, 3>& first,
    const std::array<Vec3, 3>& second);

} // namespace cdfmm::detail
