// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <vector>
namespace cdfmm {
/**
 * @brief Linear storage for multipole or local expansion coefficients.
 *
 * Coefficients are indexed by a companion MultiIndexSet:
 *
 *   coeff[i] <-> multi-index basis[i]
 *
 * The same convention is used for M (multipole) and L (local) vectors.
 */
using CoeffVector = std::vector<double>;
}
