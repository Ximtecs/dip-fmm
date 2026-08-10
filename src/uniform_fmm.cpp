// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "cdfmm/operators.hpp"

namespace cdfmm {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void accumulate_phase(PhaseTiming& aggregate, const PhaseTiming& value)
{
    aggregate.total_seconds += value.total_seconds;
    aggregate.calls += value.calls;
}

void accumulate_timings(EvaluationTimings& aggregate,
                        const EvaluationTimings& value)
{
    accumulate_phase(aggregate.moment_permutation, value.moment_permutation);
    accumulate_phase(aggregate.multipole_reset, value.multipole_reset);
    accumulate_phase(aggregate.p2m, value.p2m);
    accumulate_phase(aggregate.m2m, value.m2m);
    accumulate_phase(aggregate.local_reset, value.local_reset);
    accumulate_phase(aggregate.l2l, value.l2l);
    accumulate_phase(aggregate.m2l, value.m2l);
    accumulate_phase(aggregate.l2p, value.l2p);
    accumulate_phase(aggregate.p2p, value.p2p);
    accumulate_phase(aggregate.result_unpermutation,
                     value.result_unpermutation);
    accumulate_phase(aggregate.total, value.total);
    aggregate.evaluations += value.evaluations;
}

} // namespace

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------

UniformFmm::UniformFmm(const std::vector<Vec3>& source_positions,
                       const UniformFmmOptions& options)
    : tree_(source_positions, options.tree), basis_(options.expansion_order)
{
    if (options.expansion_order < 0) {
        throw std::invalid_argument(
            "UniformFmmOptions.expansion_order must be >= 0"
        );
    }

    multipoles_.assign(
        tree_.nodes().size(),
        CoeffVector(static_cast<std::size_t>(basis_.size()), 0.0)
    );
    locals_.assign(
        tree_.nodes().size(),
        CoeffVector(static_cast<std::size_t>(basis_.size()), 0.0)
    );
    sorted_dipole_moments_.resize(source_positions.size());
}

UniformFmm::UniformFmm(const std::vector<Vec3>& source_positions,
                       const std::vector<Vec3>& target_positions,
                       const UniformFmmOptions& options)
    : tree_(source_positions, target_positions, options.tree),
      basis_(options.expansion_order)
{
    if (options.expansion_order < 0) {
        throw std::invalid_argument(
            "UniformFmmOptions.expansion_order must be >= 0"
        );
    }

    multipoles_.assign(
        tree_.nodes().size(),
        CoeffVector(static_cast<std::size_t>(basis_.size()), 0.0)
    );
    locals_.assign(
        tree_.nodes().size(),
        CoeffVector(static_cast<std::size_t>(basis_.size()), 0.0)
    );
    sorted_dipole_moments_.resize(source_positions.size());
    sorted_results_.resize(target_positions.size());
}

//------------------------------------------------------------------------------
// Upward pass
//------------------------------------------------------------------------------

void UniformFmm::upward_pass(std::span<const Vec3> dipole_moments)
{
    if (dipole_moments.size() != tree_.sorted_source_positions().size()) {
        throw std::invalid_argument(
            "UniformFmm::upward_pass requires one moment per source position"
        );
    }

    auto phase_start = Clock::now();
    #pragma omp parallel for schedule(static) if(multipoles_.size() >= 64)
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(multipoles_.size()); ++index) {
        CoeffVector& M = multipoles_[static_cast<std::size_t>(index)];
        std::fill(M.begin(), M.end(), 0.0);
    }
    last_timings_.multipole_reset.add(elapsed_seconds(phase_start));

    phase_start = Clock::now();
    const auto permutation = tree_.source_permutation();
    #pragma omp parallel for schedule(static) if(permutation.size() >= 256)
    for (std::ptrdiff_t sorted_index = 0;
         sorted_index < static_cast<std::ptrdiff_t>(permutation.size());
         ++sorted_index) {
        const int original_index =
            permutation[static_cast<std::size_t>(sorted_index)];
        sorted_dipole_moments_[static_cast<std::size_t>(sorted_index)] =
            dipole_moments[static_cast<std::size_t>(original_index)];
    }
    last_timings_.moment_permutation.add(elapsed_seconds(phase_start));

    const auto nodes = tree_.nodes();
    const auto sorted_positions = tree_.sorted_source_positions();
    const auto occupied_leaves = tree_.occupied_source_leaves();
    phase_start = Clock::now();
    #pragma omp parallel for schedule(static) if(occupied_leaves.size() >= 8)
    for (std::ptrdiff_t occupied_index = 0;
         occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
         ++occupied_index) {
        const int leaf_index =
            occupied_leaves[static_cast<std::size_t>(occupied_index)];
        const TreeNode& leaf = nodes[static_cast<std::size_t>(leaf_index)];
        multipoles_[static_cast<std::size_t>(leaf_index)] = p2m_dipole(
            basis_, leaf.centre,
            sorted_positions.subspan(leaf.source_begin, leaf.source_count()),
            std::span<const Vec3>(sorted_dipole_moments_)
                .subspan(leaf.source_begin, leaf.source_count())
        );
    }
    last_timings_.p2m.add(elapsed_seconds(phase_start));

    phase_start = Clock::now();
    for (int level = tree_.leaf_level() - 1; level >= 0; --level) {
        const int begin = level_offset(level);
        const int end = level_offset(level + 1);
        #pragma omp parallel for schedule(static) if(end - begin >= 8)
        for (int parent_index = begin; parent_index < end; ++parent_index) {
            const TreeNode& parent = nodes[static_cast<std::size_t>(parent_index)];
            if (parent.source_count() == 0) {
                continue;
            }
            CoeffVector& parent_M =
                multipoles_[static_cast<std::size_t>(parent_index)];
            for (const int child_index : parent.children) {
                const TreeNode& child = nodes[static_cast<std::size_t>(child_index)];
                if (child.source_count() == 0) {
                    continue;
                }
                const Vec3 d = parent.centre - child.centre;
                m2m_add(basis_, d,
                        multipoles_[static_cast<std::size_t>(child_index)],
                        parent_M);
            }
        }
    }
    last_timings_.m2m.add(elapsed_seconds(phase_start));
}

//------------------------------------------------------------------------------
// Downward pass
//------------------------------------------------------------------------------

void UniformFmm::downward_pass()
{
    auto phase_start = Clock::now();
    #pragma omp parallel for schedule(static) if(locals_.size() >= 64)
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(locals_.size()); ++index) {
        CoeffVector& L = locals_[static_cast<std::size_t>(index)];
        std::fill(L.begin(), L.end(), 0.0);
    }
    last_timings_.local_reset.add(elapsed_seconds(phase_start));

    const auto nodes = tree_.nodes();
    for (int level = 1; level <= tree_.leaf_level(); ++level) {
        const int begin = level_offset(level);
        const int end = level_offset(level + 1);

        phase_start = Clock::now();
        #pragma omp parallel for schedule(static) if(end - begin >= 8)
        for (int target_index = begin; target_index < end; ++target_index) {
            const TreeNode& target = nodes[static_cast<std::size_t>(target_index)];
            if (target.target_count() == 0) {
                continue;
            }
            const TreeNode& parent = nodes[static_cast<std::size_t>(target.parent)];
            const Vec3 d = target.centre - parent.centre;
            l2l_add(basis_, d, locals_[static_cast<std::size_t>(target.parent)],
                    locals_[static_cast<std::size_t>(target_index)]);
        }
        last_timings_.l2l.add(elapsed_seconds(phase_start));

        phase_start = Clock::now();
        #pragma omp parallel for schedule(static) if(end - begin >= 8)
        for (int target_index = begin; target_index < end; ++target_index) {
            const TreeNode& target = nodes[static_cast<std::size_t>(target_index)];
            if (target.target_count() == 0) {
                continue;
            }
            CoeffVector& target_L = locals_[static_cast<std::size_t>(target_index)];
            for (const int source_index : target.list2) {
                const TreeNode& source = nodes[static_cast<std::size_t>(source_index)];
                if (source.source_count() == 0) {
                    continue;
                }
                const Vec3 R = target.centre - source.centre;
                m2l_add(basis_, R,
                        multipoles_[static_cast<std::size_t>(source_index)],
                        target_L);
            }
        }
        last_timings_.m2l.add(elapsed_seconds(phase_start));
    }
}

//------------------------------------------------------------------------------
// Complete evaluation
//------------------------------------------------------------------------------

std::vector<PotentialField> UniformFmm::evaluate(
    std::span<const Vec3> dipole_moments,
    const OutputFlags output,
    std::span<const int> target_source_indices)
{
    std::vector<PotentialField> results(
        tree_.sorted_target_positions().size()
    );
    evaluate_into(dipole_moments, results, output, target_source_indices);
    return results;
}

void UniformFmm::evaluate_into(
    std::span<const Vec3> dipole_moments,
    std::span<PotentialField> results,
    const OutputFlags output,
    std::span<const int> target_source_indices)
{
    const std::size_t target_count = tree_.sorted_target_positions().size();
    if (results.size() != target_count) {
        throw std::invalid_argument(
            "UniformFmm::evaluate_into requires one result per target"
        );
    }
    if (!target_source_indices.empty() &&
        target_source_indices.size() != target_count) {
        throw std::invalid_argument(
            "UniformFmm::evaluate identity map has incorrect length"
        );
    }
    for (const int source_index : target_source_indices) {
        if (source_index < -1 ||
            source_index >= static_cast<int>(dipole_moments.size())) {
            throw std::invalid_argument(
                "UniformFmm::evaluate identity map contains an invalid index"
            );
        }
    }

    last_timings_ = {};
    const auto evaluation_start = Clock::now();
    upward_pass(dipole_moments);
    downward_pass();

    const auto nodes = tree_.nodes();
    const auto targets = tree_.sorted_target_positions();
    const auto sources = tree_.sorted_source_positions();
    const auto target_permutation = tree_.target_permutation();
    const auto source_inverse = tree_.source_inverse_permutation();
    const auto occupied_leaves = tree_.occupied_target_leaves();

    auto phase_start = Clock::now();
    #pragma omp parallel for schedule(static) if(occupied_leaves.size() >= 8)
    for (std::ptrdiff_t occupied_index = 0;
         occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
         ++occupied_index) {
        const int leaf_index =
            occupied_leaves[static_cast<std::size_t>(occupied_index)];
        const TreeNode& leaf = nodes[static_cast<std::size_t>(leaf_index)];
        for (std::size_t target_index = leaf.target_begin;
             target_index < leaf.target_end; ++target_index) {
            sorted_results_[target_index] = l2p_eval(
                basis_, leaf.centre, targets[target_index],
                locals_[static_cast<std::size_t>(leaf_index)], output
            );
        }
    }
    last_timings_.l2p.add(elapsed_seconds(phase_start));

    phase_start = Clock::now();
    #pragma omp parallel for schedule(static) if(occupied_leaves.size() >= 8)
    for (std::ptrdiff_t occupied_index = 0;
         occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
         ++occupied_index) {
        const int leaf_index =
            occupied_leaves[static_cast<std::size_t>(occupied_index)];
        const TreeNode& leaf = nodes[static_cast<std::size_t>(leaf_index)];
        for (std::size_t target_index = leaf.target_begin;
             target_index < leaf.target_end; ++target_index) {
            int self_sorted_index = -1;
            if (!target_source_indices.empty()) {
                const int original_target = target_permutation[target_index];
                const int original_source = target_source_indices[
                    static_cast<std::size_t>(original_target)
                ];
                if (original_source >= 0) {
                    self_sorted_index = source_inverse[
                        static_cast<std::size_t>(original_source)
                    ];
                }
            }

            PotentialField& result = sorted_results_[target_index];
            for (const int neighbour_index : leaf.list1) {
                const TreeNode& neighbour =
                    nodes[static_cast<std::size_t>(neighbour_index)];
                if (neighbour.source_count() == 0) {
                    continue;
                }
                int local_self_index = -1;
                if (self_sorted_index >=
                        static_cast<int>(neighbour.source_begin) &&
                    self_sorted_index < static_cast<int>(neighbour.source_end)) {
                    local_self_index = self_sorted_index -
                        static_cast<int>(neighbour.source_begin);
                }
                const PotentialField near = p2p_dipole_sum(
                    targets[target_index],
                    sources.subspan(
                        neighbour.source_begin, neighbour.source_count()
                    ),
                    std::span<const Vec3>(sorted_dipole_moments_).subspan(
                        neighbour.source_begin, neighbour.source_count()
                    ),
                    output, local_self_index
                );
                result.phi += near.phi;
                result.H += near.H;
            }
        }
    }
    last_timings_.p2p.add(elapsed_seconds(phase_start));

    phase_start = Clock::now();
    #pragma omp parallel for schedule(static) if(target_count >= 256)
    for (std::ptrdiff_t sorted_index = 0;
         sorted_index < static_cast<std::ptrdiff_t>(target_count);
         ++sorted_index) {
        const int original_index =
            target_permutation[static_cast<std::size_t>(sorted_index)];
        results[static_cast<std::size_t>(original_index)] =
            sorted_results_[static_cast<std::size_t>(sorted_index)];
    }
    last_timings_.result_unpermutation.add(elapsed_seconds(phase_start));
    last_timings_.total.add(elapsed_seconds(evaluation_start));
    last_timings_.evaluations = 1;
    accumulate_timings(aggregate_timings_, last_timings_);
}

//------------------------------------------------------------------------------
// Public inspection
//------------------------------------------------------------------------------

const UniformTree& UniformFmm::tree() const { return tree_; }
const MultiIndexSet& UniformFmm::basis() const { return basis_; }

std::span<const double> UniformFmm::multipole(const int node_index) const
{
    return multipoles_.at(static_cast<std::size_t>(node_index));
}

std::span<const double> UniformFmm::local(const int node_index) const
{
    return locals_.at(static_cast<std::size_t>(node_index));
}

std::span<const double> UniformFmm::root_multipole() const
{
    return multipoles_.front();
}

const EvaluationTimings& UniformFmm::last_timings() const
{
    return last_timings_;
}

const EvaluationTimings& UniformFmm::aggregate_timings() const
{
    return aggregate_timings_;
}

void UniformFmm::reset_timings()
{
    aggregate_timings_ = {};
}

} // namespace cdfmm
