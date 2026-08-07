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
 * The same container is used for M (multipole) and L (local) vectors, but the
 * mathematical normalisation depends on the operator: multipole entries
 * multiply raw kernel derivatives, whereas local entries are raw derivatives
 * of the incoming potential and multiply factorial-normalised monomials.
 */
using CoeffVector = std::vector<double>;
}
