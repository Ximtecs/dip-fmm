// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <stdexcept>

#include "cdfmm/operators.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------

UniformFmm::UniformFmm(const std::vector<Vec3> &source_positions,
                       const UniformFmmOptions &options)
    : tree_(source_positions, options.tree), basis_(options.expansion_order) {
  if (options.expansion_order < 0) {
    throw std::invalid_argument(
        "UniformFmmOptions.expansion_order must be >= 0");
  }

  multipoles_.assign(tree_.nodes().size(),
                     CoeffVector(static_cast<std::size_t>(basis_.size()), 0.0));
  sorted_dipole_moments_.resize(source_positions.size());
}

//------------------------------------------------------------------------------
// Upward pass
//------------------------------------------------------------------------------

void UniformFmm::upward_pass(std::span<const Vec3> dipole_moments) {
  if (dipole_moments.size() != tree_.sorted_source_positions().size()) {
    throw std::invalid_argument("UniformFmm::upward_pass requires one dipole "
                                "moment per source position");
  }

  for (CoeffVector &M : multipoles_) {
    std::fill(M.begin(), M.end(), 0.0);
  }

  // User moments arrive in original source order. The tree convention is
  // permutation[sorted] = original, matching its Morton-sorted positions.
  const auto permutation = tree_.source_permutation();
  for (std::size_t sorted_index = 0; sorted_index < permutation.size();
       ++sorted_index) {
    const int original_index = permutation[sorted_index];
    sorted_dipole_moments_[sorted_index] =
        dipole_moments[static_cast<std::size_t>(original_index)];
  }

  const auto nodes = tree_.nodes();
  const auto sorted_positions = tree_.sorted_source_positions();

  // Node source ranges are half-open intervals into both Morton-sorted
  // source arrays. Empty leaves retain the zero expansion initialised above.
  for (const int leaf_index : tree_.leaf_indices()) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    if (leaf.source_count() == 0) {
      continue;
    }

    const auto positions =
        sorted_positions.subspan(leaf.source_begin, leaf.source_count());
    const std::span<const Vec3> moments(sorted_dipole_moments_);
    multipoles_[static_cast<std::size_t>(leaf_index)] =
        p2m_dipole(basis_, leaf.centre, positions,
                   moments.subspan(leaf.source_begin, leaf.source_count()));
  }

  // Each translation uses d = parent centre - child centre, as required by
  // m2m_add. A parent accumulates all child subtrees and therefore represents
  // the union of their sources about the parent centre.
  for (int level = tree_.leaf_level() - 1; level >= 0; --level) {
    const int begin = level_offset(level);
    const int end = level_offset(level + 1);
    for (int parent_index = begin; parent_index < end; ++parent_index) {
      const TreeNode &parent = nodes[static_cast<std::size_t>(parent_index)];
      CoeffVector &parent_M =
          multipoles_[static_cast<std::size_t>(parent_index)];
      for (const int child_index : parent.children) {
        const TreeNode &child = nodes[static_cast<std::size_t>(child_index)];
        if (child.source_count() == 0) {
          continue;
        }

        const Vec3 d = parent.centre - child.centre;
        m2m_add(basis_, d, multipoles_[static_cast<std::size_t>(child_index)],
                parent_M);
      }
    }
  }
}

//------------------------------------------------------------------------------
// Public inspection
//------------------------------------------------------------------------------

const UniformTree &UniformFmm::tree() const { return tree_; }

const MultiIndexSet &UniformFmm::basis() const { return basis_; }

std::span<const double> UniformFmm::multipole(const int node_index) const {
  return multipoles_.at(static_cast<std::size_t>(node_index));
}

std::span<const double> UniformFmm::root_multipole() const {
  return multipoles_.front();
}

} // namespace cdfmm
