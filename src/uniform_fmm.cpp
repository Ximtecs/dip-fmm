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
  locals_.assign(tree_.nodes().size(),
                 CoeffVector(static_cast<std::size_t>(basis_.size()), 0.0));
  sorted_dipole_moments_.resize(source_positions.size());
}

UniformFmm::UniformFmm(const std::vector<Vec3> &source_positions,
                       const std::vector<Vec3> &target_positions,
                       const UniformFmmOptions &options)
    : tree_(source_positions, target_positions, options.tree),
      basis_(options.expansion_order) {
  if (options.expansion_order < 0) {
    throw std::invalid_argument(
        "UniformFmmOptions.expansion_order must be >= 0");
  }

  multipoles_.assign(tree_.nodes().size(),
                     CoeffVector(static_cast<std::size_t>(basis_.size()), 0.0));
  locals_.assign(tree_.nodes().size(),
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
// Downward pass
//------------------------------------------------------------------------------

void UniformFmm::downward_pass() {
  for (CoeffVector &L : locals_) {
    std::fill(L.begin(), L.end(), 0.0);
  }

  const auto nodes = tree_.nodes();
  for (int level = 1; level <= tree_.leaf_level(); ++level) {
    const int begin = level_offset(level);
    const int end = level_offset(level + 1);
    for (int target_index = begin; target_index < end; ++target_index) {
      const TreeNode &target = nodes[static_cast<std::size_t>(target_index)];
      if (target.target_count() == 0) {
        continue;
      }

      // Inherit coarser far-field information before adding interactions
      // introduced at this level. The operator convention is
      // d = child centre - parent centre.
      const TreeNode &parent = nodes[static_cast<std::size_t>(target.parent)];
      const Vec3 d = target.centre - parent.centre;
      l2l_add(basis_, d, locals_[static_cast<std::size_t>(target.parent)],
              locals_[static_cast<std::size_t>(target_index)]);

      for (const int source_index : target.list2) {
        const TreeNode &source = nodes[static_cast<std::size_t>(source_index)];
        if (source.source_count() == 0) {
          continue;
        }

        // M2L requires R = target centre - source centre.
        const Vec3 R = target.centre - source.centre;
        m2l_add(basis_, R, multipoles_[static_cast<std::size_t>(source_index)],
                locals_[static_cast<std::size_t>(target_index)]);
      }
    }
  }
}

//------------------------------------------------------------------------------
// Complete reference evaluation
//------------------------------------------------------------------------------

std::vector<PotentialField>
UniformFmm::evaluate(std::span<const Vec3> dipole_moments,
                     const OutputFlags output,
                     std::span<const int> target_source_indices) {
  const std::size_t target_count = tree_.sorted_target_positions().size();
  if (!target_source_indices.empty() &&
      target_source_indices.size() != target_count) {
    throw std::invalid_argument("UniformFmm::evaluate identity map must be "
                                "empty or contain one entry per target");
  }
  for (const int source_index : target_source_indices) {
    if (source_index < -1 ||
        source_index >= static_cast<int>(dipole_moments.size())) {
      throw std::invalid_argument(
          "UniformFmm::evaluate identity map contains an invalid source index");
    }
  }

  upward_pass(dipole_moments);
  downward_pass();

  std::vector<PotentialField> sorted_results(target_count);
  const auto nodes = tree_.nodes();
  const auto targets = tree_.sorted_target_positions();
  const auto sources = tree_.sorted_source_positions();
  const auto target_permutation = tree_.target_permutation();
  const auto source_inverse = tree_.source_inverse_permutation();

  for (const int leaf_index : tree_.leaf_indices()) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    if (leaf.target_count() == 0) {
      continue;
    }

    for (std::size_t target_index = leaf.target_begin;
         target_index < leaf.target_end; ++target_index) {
      PotentialField result =
          l2p_eval(basis_, leaf.centre, targets[target_index],
                   locals_[static_cast<std::size_t>(leaf_index)], output);

      int self_sorted_index = -1;
      if (!target_source_indices.empty()) {
        const int original_target = target_permutation[target_index];
        const int original_source =
            target_source_indices[static_cast<std::size_t>(original_target)];
        if (original_source >= 0) {
          self_sorted_index =
              source_inverse[static_cast<std::size_t>(original_source)];
        }
      }

      // list1 is the complete near-field partition. Each source box is
      // evaluated once, while list2 contributions are represented by locals.
      for (const int neighbour_index : leaf.list1) {
        const TreeNode &neighbour =
            nodes[static_cast<std::size_t>(neighbour_index)];
        if (neighbour.source_count() == 0) {
          continue;
        }

        int local_self_index = -1;
        if (self_sorted_index >= static_cast<int>(neighbour.source_begin) &&
            self_sorted_index < static_cast<int>(neighbour.source_end)) {
          local_self_index =
              self_sorted_index - static_cast<int>(neighbour.source_begin);
        }
        const PotentialField near = p2p_dipole_sum(
            targets[target_index],
            sources.subspan(neighbour.source_begin, neighbour.source_count()),
            std::span<const Vec3>(sorted_dipole_moments_)
                .subspan(neighbour.source_begin, neighbour.source_count()),
            output, local_self_index);
        result.phi += near.phi;
        result.H += near.H;
      }
      sorted_results[target_index] = result;
    }
  }

  std::vector<PotentialField> user_results(target_count);
  for (std::size_t sorted_index = 0; sorted_index < target_count;
       ++sorted_index) {
    const int original_index = target_permutation[sorted_index];
    user_results[static_cast<std::size_t>(original_index)] =
        sorted_results[sorted_index];
  }
  return user_results;
}

//------------------------------------------------------------------------------
// Public inspection
//------------------------------------------------------------------------------

const UniformTree &UniformFmm::tree() const { return tree_; }

const MultiIndexSet &UniformFmm::basis() const { return basis_; }

std::span<const double> UniformFmm::multipole(const int node_index) const {
  return multipoles_.at(static_cast<std::size_t>(node_index));
}

std::span<const double> UniformFmm::local(const int node_index) const {
  return locals_.at(static_cast<std::size_t>(node_index));
}

std::span<const double> UniformFmm::root_multipole() const {
  return multipoles_.front();
}

} // namespace cdfmm
