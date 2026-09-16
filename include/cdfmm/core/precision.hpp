// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/** @brief Selects operator, expansion-state, scratch, and execution precision. */
enum class StaticPrecision {
    /// Retain and execute completed operators and dynamic state in FP32.
    Float32,
    /// Retain and execute completed operators and dynamic state in FP64.
    Float64
};

/** @brief Expansion basis used by the reusable far-field hierarchy. */
enum class ExpansionBasis {
    /// Factorial-normalised total-degree Cartesian Taylor coefficients.
    Cartesian,
    /// Minimal real tesseral solid harmonics ordered by degree then m.
    Spherical
};

} // namespace cdfmm
