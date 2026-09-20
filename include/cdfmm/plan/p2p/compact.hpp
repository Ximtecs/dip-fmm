// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

#include "cdfmm/plan/p2p/canonical.hpp"
#include "cdfmm/plan/p2p/memory.hpp"

namespace cdfmm {

/**
 * @brief Source-index and SoA packing of canonical FP64 particle rows.
 *
 * The same rows as StaticP2POperator with every field of the block record
 * split into its own array, so the row loop of a target streams six
 * contiguous component arrays.  Entry `i` of every array describes the same
 * canonical pair.
 */
struct StaticP2PCompactPlan {
    int source_count{0};                            ///< Number of sorted sources.
    int target_count{0};                            ///< Number of sorted targets.
    std::vector<int> row_offsets{};                 ///< Range of pairs per target (CSR offsets, `target_count + 1` entries).
    std::vector<int> source_indices{};              ///< Sorted source index of each pair.
    std::vector<unsigned char> skip_for_identity{}; ///< Identity marker of each pair (see StaticDipoleBlock).
    std::array<std::vector<double>, 3> potential{}; ///< Potential row coefficients px, py, pz per pair.
    std::array<std::vector<double>, 6> tensors{};   ///< Field tensor components xx, xy, xz, yy, yz, zz per pair.

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Source-index and SoA packing of canonical FP32 particle rows. */
struct FloatStaticP2PCompactPlan {
    int source_count{0};                            ///< Number of sorted sources.
    int target_count{0};                            ///< Number of sorted targets.
    std::vector<int> row_offsets{};                 ///< Range of pairs per target (CSR offsets, `target_count + 1` entries).
    std::vector<int> source_indices{};              ///< Sorted source index of each pair.
    std::vector<unsigned char> skip_for_identity{}; ///< Identity marker of each pair (see StaticDipoleBlock).
    std::array<std::vector<float>, 3> potential{};  ///< Potential row coefficients px, py, pz per pair.
    std::array<std::vector<float>, 6> tensors{};    ///< Field tensor components xx, xy, xz, yy, yz, zz per pair.

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Splits the canonical FP64 rows into the SoA packing. */
[[nodiscard]] StaticP2PCompactPlan
build_static_p2p_compact_plan(const StaticP2POperator& operator_map);

/** @brief Splits the canonical FP32 rows into the SoA packing. */
[[nodiscard]] FloatStaticP2PCompactPlan
build_static_p2p_compact_plan(const FloatStaticP2POperator& operator_map);

} // namespace cdfmm
