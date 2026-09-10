// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

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
