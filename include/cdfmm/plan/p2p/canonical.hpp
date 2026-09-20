// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

/**
 * @brief One canonical FP64 P2P pair record in Morton-sorted particle order.
 *
 * The six field components store xx, xy, xz, yy, yz, and zz. Potential rows
 * store px, py, and pz. Only central-image point interactions may carry a
 * non-zero identity-exclusion marker.
 */
struct StaticDipoleBlock {
    int target{0};              ///< Sorted target index.
    int source{0};              ///< Sorted source index.
    double px{0.0};             ///< Potential row: phi += px * m.x + py * m.y + pz * m.z.
    double py{0.0};             ///< Potential row, y coefficient.
    double pz{0.0};             ///< Potential row, z coefficient.
    double xx{0.0};             ///< Field tensor component T_xx (H = T m).
    double xy{0.0};             ///< Field tensor component T_xy = T_yx.
    double xz{0.0};             ///< Field tensor component T_xz = T_zx.
    double yy{0.0};             ///< Field tensor component T_yy.
    double yz{0.0};             ///< Field tensor component T_yz = T_zy.
    double zz{0.0};             ///< Field tensor component T_zz.
    int skip_for_identity{1};   ///< Non-zero when an identity match omits this pair (point self pair).
};

/** @brief FP32 canonical pair record with the same field order and semantics. */
struct FloatStaticDipoleBlock {
    int target{0};              ///< Sorted target index.
    int source{0};              ///< Sorted source index.
    float px{0.0F};             ///< Potential row: phi += px * m.x + py * m.y + pz * m.z.
    float py{0.0F};             ///< Potential row, y coefficient.
    float pz{0.0F};             ///< Potential row, z coefficient.
    float xx{0.0F};             ///< Field tensor component T_xx (H = T m).
    float xy{0.0F};             ///< Field tensor component T_xy = T_yx.
    float xz{0.0F};             ///< Field tensor component T_xz = T_zx.
    float yy{0.0F};             ///< Field tensor component T_yy.
    float yz{0.0F};             ///< Field tensor component T_yz = T_zy.
    float zz{0.0F};             ///< Field tensor component T_zz.
    int skip_for_identity{1};   ///< Non-zero when an identity match omits this pair (point self pair).
};

#if defined(__CUDACC__)
#define CDFMM_PLAN_HOST_DEVICE __host__ __device__
#else
#define CDFMM_PLAN_HOST_DEVICE
#endif

/** @brief Accumulates one canonical FP64 symmetric tensor product. */
CDFMM_PLAN_HOST_DEVICE inline void accumulate_static_dipole_block(
    const StaticDipoleBlock& block,
    const Vec3& moment,
    Vec3& H) noexcept
{
    H.x += block.xx * moment.x + block.xy * moment.y + block.xz * moment.z;
    H.y += block.xy * moment.x + block.yy * moment.y + block.yz * moment.z;
    H.z += block.xz * moment.x + block.yz * moment.y + block.zz * moment.z;
}

/** @brief Accumulates one canonical FP32 symmetric tensor product. */
CDFMM_PLAN_HOST_DEVICE inline void accumulate_static_dipole_block(
    const FloatStaticDipoleBlock& block,
    const FloatVec3& moment,
    FloatVec3& H) noexcept
{
    H.x += block.xx * moment.x + block.xy * moment.y + block.xz * moment.z;
    H.y += block.xy * moment.x + block.yy * moment.y + block.yz * moment.z;
    H.z += block.xz * moment.x + block.yz * moment.y + block.zz * moment.z;
}

#undef CDFMM_PLAN_HOST_DEVICE

/**
 * @brief Authoritative FP64 target-row representation of exact P2P data.
 *
 * `row_offsets[t]..row_offsets[t+1]` addresses the canonical interaction
 * sequence for target `t`. All execution packings are derived from these rows.
 */
struct StaticP2POperator {
    int source_count{0};                     ///< Number of sorted sources.
    int target_count{0};                     ///< Number of sorted targets.
    std::vector<int> row_offsets{};          ///< Range of `blocks` per target (CSR offsets, `target_count + 1` entries).
    std::vector<StaticDipoleBlock> blocks{}; ///< Pair records in canonical row order.

    /// Retained bytes of the rows and blocks.
    [[nodiscard]] std::size_t memory_bytes() const noexcept;
};

/** @brief Authoritative FP32 target-row P2P representation. */
struct FloatStaticP2POperator {
    int source_count{0};                          ///< Number of sorted sources.
    int target_count{0};                          ///< Number of sorted targets.
    std::vector<int> row_offsets{};               ///< Range of `blocks` per target (CSR offsets, `target_count + 1` entries).
    std::vector<FloatStaticDipoleBlock> blocks{}; ///< Pair records in canonical row order.

    /// Retained bytes of the rows and blocks.
    [[nodiscard]] std::size_t memory_bytes() const noexcept;
};

} // namespace cdfmm
