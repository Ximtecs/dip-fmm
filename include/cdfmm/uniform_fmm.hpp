// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/coefficients.hpp"
#include "cdfmm/multi_index.hpp"
#include "cdfmm/uniform_tree.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/**
 * @brief Options for the reference uniform-FMM upward pass.
 *
 * Geometry options are kept alongside the expansion order at setup, while
 * dipole moments are supplied separately for each upward evaluation.
 */
struct UniformFmmOptions {
  /// @brief Maximum total degree of the Cartesian multipole expansion.
  int expansion_order{4};
  /// @brief Complete uniform-tree geometry options.
  UniformTreeOptions tree{};
};

/**
 * @brief Reference leaf-P2M and hierarchical-M2M evaluator.
 *
 * Construction fixes and Morton-sorts the source geometry. Calling
 * `upward_pass` accepts moments in the original user source order and replaces
 * all stored node multipoles. For every node, the resulting expansion
 * represents all source dipoles in that node's subtree about that node's
 * centre.
 *
 * This class intentionally implements only the upward pass; it does not yet
 * evaluate fields or perform M2L, downward, or near-field traversals.
 */
class UniformFmm {
public:
  /**
   * @brief Constructs fixed source geometry for repeated upward passes.
   *
   * @param source_positions Source positions in user order.
   * @param options Expansion and uniform-tree configuration.
   */
  UniformFmm(const std::vector<Vec3> &source_positions,
             const UniformFmmOptions &options = {});

  /**
   * @brief Recomputes all node multipoles for a new dipole state.
   *
   * Input moments correspond one-to-one with the original source-position
   * ordering supplied to the constructor. They are permuted internally to
   * the tree's Morton order before leaf P2M.
   *
   * @param dipole_moments Dipole moments in original user source order.
   * @throws std::invalid_argument if the number of moments is incorrect.
   */
  void upward_pass(std::span<const Vec3> dipole_moments);

  /// @brief Returns the fixed complete uniform-tree geometry.
  [[nodiscard]] const UniformTree &tree() const;
  /// @brief Returns the Cartesian coefficient basis used by every node.
  [[nodiscard]] const MultiIndexSet &basis() const;

  /**
   * @brief Returns a read-only multipole for a flat tree-node index.
   *
   * Coefficients follow `basis()` ordering. Before the first upward pass and
   * for empty subtrees, every coefficient is zero.
   */
  [[nodiscard]] std::span<const double> multipole(int node_index) const;

  /// @brief Returns the root multipole, whose flat node index is zero.
  [[nodiscard]] std::span<const double> root_multipole() const;

private:
  UniformTree tree_;
  MultiIndexSet basis_;
  std::vector<CoeffVector> multipoles_{};
  std::vector<Vec3> sorted_dipole_moments_{};
};

} // namespace cdfmm
