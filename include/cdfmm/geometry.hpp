// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/vec3.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Generic operator data
//------------------------------------------------------------------------------

/** @brief Six independent components of a symmetric Cartesian pair tensor. */
struct PairTensor {
    double xx{0.0};
    double xy{0.0};
    double xz{0.0};
    double yy{0.0};
    double yz{0.0};
    double zz{0.0};
};

/** @brief Full side lengths of an axis-aligned rectangular prism. */
struct RectangularPrism {
    double hx{0.0};
    double hy{0.0};
    double hz{0.0};

    /// @brief Returns the prism volume.
    [[nodiscard]] double volume() const noexcept { return hx * hy * hz; }
};

//------------------------------------------------------------------------------
// Physical geometry
//------------------------------------------------------------------------------

/** @brief Physical geometry occupied by each magnetic source. */
enum class SourceGeometry {
    /// A singular point dipole carrying the supplied total moment.
    PointDipole,
    /// An axis-aligned rectangular prism carrying the supplied total moment.
    RectangularPrism,
    /// An arbitrary non-degenerate tetrahedron centred on the representative point.
    Tetrahedron,

};

/** @brief Physical geometry over which a target field may be averaged. */
enum class TargetGeometry {
    /// Evaluate the field at the representative target point.
    Point,
    /// Average the field over an axis-aligned rectangular prism.
    RectangularPrism,
    /// Average the field over an arbitrary non-degenerate tetrahedron.
    Tetrahedron,

};

//------------------------------------------------------------------------------
// Approximation/evaluation models
//------------------------------------------------------------------------------

/** @brief Source approximation selected independently for each FMM regime. */
enum class SourceModel {
    /// Use the ordinary point-dipole approximation at the representative point.
    PointDipole,
    /// Use the complete finite source geometry.
    ExactGeometry,
};

/** @brief Target evaluation selected independently for each FMM regime. */
enum class TargetModel {
    /// Evaluate at the representative target point.
    Point,
    /// Evaluate the exact volume average over the target geometry.
    ExactGeometry,
};

} // namespace cdfmm
