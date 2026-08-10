// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/coefficients.hpp"
#include "cdfmm/multi_index.hpp"
#include "cdfmm/output_flags.hpp"
#include "cdfmm/uniform_tree.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/**
 * @brief Options for the reference uniform FMM traversal.
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
 * @brief Functional reference evaluator for a complete uniform FMM tree.
 *
 * Construction fixes and Morton-sorts the source geometry. Calling
 * `upward_pass` accepts moments in the original user source order and replaces
 * all stored node multipoles. For every node, the resulting expansion
 * represents all source dipoles in that node's subtree about that node's
 * centre.
 *
 * When target geometry is supplied, `evaluate` completes P2M, M2M, M2L, L2L,
 * L2P, and direct list1 P2P. Results are returned in original user target
 * order. The independently callable passes and node coefficients are exposed
 * for validation and education rather than as an optimised execution plan.
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
   * @brief Constructs fixed source and target geometry for complete evaluation.
   *
   * Source and target populations are sorted independently by `UniformTree`.
   * Supplying equal coordinate arrays does not implicitly identify particles;
   * self identities are passed explicitly to `evaluate`.
   *
   * @param source_positions Source positions in user order.
   * @param target_positions Target positions in user order.
   * @param options Expansion and uniform-tree configuration.
   */
  UniformFmm(const std::vector<Vec3> &source_positions,
             const std::vector<Vec3> &target_positions,
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

  /**
   * @brief Rebuilds all local expansions by M2L and downward L2L translation.
   *
   * Existing locals are cleared first. At each level, a target node inherits
   * its parent local before accumulating its own list2 interactions.
   */
  void downward_pass();

  /**
   * @brief Evaluates the complete far-field and near-field traversal.
   *
   * The optional identity map has one entry per target in original user order.
   * Entry `i` is the original source index representing the same particle, or
   * -1 for no self identity. Only that precise source is skipped during list1
   * P2P; coordinate equality is deliberately not used as an identity test.
   *
   * @param dipole_moments Moments in original user source order.
   * @param output Requested potential and/or magnetic field components.
   * @param target_source_indices Explicit target-to-source identity map.
   * @return Values in the original user target ordering.
   */
  [[nodiscard]] std::vector<PotentialField>
  evaluate(std::span<const Vec3> dipole_moments,
           OutputFlags output = OutputFlags::Field,
           std::span<const int> target_source_indices = {});

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

  /** @brief Returns a node local expansion in `basis()` ordering. */
  [[nodiscard]] std::span<const double> local(int node_index) const;

  /// @brief Returns the root multipole, whose flat node index is zero.
  [[nodiscard]] std::span<const double> root_multipole() const;

private:
  UniformTree tree_;
  MultiIndexSet basis_;
  std::vector<CoeffVector> multipoles_{};
  std::vector<CoeffVector> locals_{};
  std::vector<Vec3> sorted_dipole_moments_{};
};

} // namespace cdfmm
