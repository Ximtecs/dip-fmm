// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/coefficients.hpp"
#include "cdfmm/multi_index.hpp"
#include "cdfmm/output_flags.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Multipole and local expansion operators
//------------------------------------------------------------------------------

/**
 * @brief Builds dipole multipole coefficients about a chosen expansion centre.
 *
 * The coefficients follow the repository sign convention for dipole potential,
 * where phi(x) = sum_j m_j.(x-x_j)/(4*pi*|x-x_j|^3) and H = -grad(phi).
 *
 * Coefficients are stored in the same linear order as @p basis. The
 * multi-index alpha denotes derivative order with respect to Cartesian axes.
 *
 * NOTE: The monopole term M_(0,0,0) is identically zero for pure dipole
 * sources because each coefficient contribution depends on alpha-e_k.
 */
CoeffVector p2m_dipole(const MultiIndexSet &basis, const Vec3 &centre,
                       std::span<const Vec3> source_positions,
                       std::span<const Vec3> dipole_moments);

/**
 * @brief Adds one child multipole expansion into a parent expansion centre.
 *
 * The translation vector @p d is parent-centre minus child-centre, and the
 * translation is a multi-index Taylor shift in that displacement.
 */
void m2m_add(const MultiIndexSet &basis, const Vec3 &d,
             std::span<const double> child, std::span<double> parent);

/**
 * @brief Adds a source multipole expansion into a target local expansion.
 *
 * This computes L_beta += sum_alpha M_alpha D_(alpha+beta) G(R), where
 * R = c_target - c_source.
 */
void m2l_add(const MultiIndexSet &basis, const Vec3 &R,
             std::span<const double> M, std::span<double> L);

/**
 * @brief Adds a parent local expansion shifted to a child expansion centre.
 *
 * The translation vector @p d is child-centre minus parent-centre.
 */
void l2l_add(const MultiIndexSet &basis, const Vec3 &d,
             std::span<const double> parent, std::span<double> child);

/**
 * @brief Evaluates a local expansion at one target position.
 *
 * Potential output is optional. Field output is computed as H = -grad(phi).
 * Field-only is the default because H is the main quantity of interest.
 */
PotentialField l2p_eval(const MultiIndexSet &basis, const Vec3 &centre,
                        const Vec3 &target, std::span<const double> L,
                        OutputFlags output = OutputFlags::Field);

/**
 * @brief Evaluates a multipole expansion directly at one target position.
 *
 * This M2P evaluation is useful for validating P2M and M2M independently of
 * local-expansion operators. Field-only output is the default because H is the
 * primary quantity of interest in this repository.
 */
PotentialField m2p_eval(const MultiIndexSet &basis_p, const CoeffVector &M,
                        const Vec3 &source_centre,
                        const Vec3 &target_position,
                        OutputFlags output = OutputFlags::Field);

//------------------------------------------------------------------------------
// Direct near-field operators
//------------------------------------------------------------------------------

/**
 * @brief Evaluates one dipole source contribution at one target position.
 *
 * Uses the direct dipole formula with the repository sign convention
 * consistent with H = -grad(phi).
 */
PotentialField p2p_dipole_pair(const Vec3 &target, const Vec3 &source,
                               const Vec3 &moment,
                               OutputFlags output = OutputFlags::Field);

/**
 * @brief Sums direct dipole contributions from all sources at one target.
 *
 * @param self_index Optional source index to skip for source-point evaluation
 * to avoid singular self-interaction.
 */
PotentialField p2p_dipole_sum(const Vec3 &target, std::span<const Vec3> sources,
                              std::span<const Vec3> moments,
                              OutputFlags output = OutputFlags::Field,
                              int self_index = -1);

} // namespace cdfmm
