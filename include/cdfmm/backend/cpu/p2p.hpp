// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/plan/static_plan.hpp"

namespace cdfmm {

void apply_static_p2p_operator(
    const StaticP2POperator& operator_map, std::span<const Vec3> dipole_moments,
    std::span<Vec3> H, std::span<const int> target_source_indices = {});
void apply_static_p2p_operator(
    const FloatStaticP2POperator& operator_map,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {});
void apply_static_p2p_compact_plan(
    const StaticP2PCompactPlan& plan, std::span<const Vec3> dipole_moments,
    std::span<Vec3> H, std::span<const int> target_source_indices = {});
void apply_static_p2p_compact_plan(
    const FloatStaticP2PCompactPlan& plan,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {});
void apply_static_p2p_leaf_plan(
    const StaticP2PLeafPlan& plan, std::span<const Vec3> dipole_moments,
    std::span<Vec3> H, std::span<const int> target_source_indices = {});
void apply_static_p2p_leaf_plan(
    const FloatStaticP2PLeafPlan& plan,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {});
void apply_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan& plan,
    std::span<const Vec3> dipole_moments, std::span<Vec3> H,
    std::span<const int> target_source_indices = {});
void apply_static_p2p_tensor_dictionary_plan(
    const FloatStaticP2PTensorDictionaryPlan& plan,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {});
void apply_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan& plan,
    std::span<const Vec3> dipole_moments, std::span<Vec3> H);
void apply_static_p2p_signed_tensor_dictionary_plan(
    const FloatStaticP2PSignedTensorDictionaryPlan& plan,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H);
void apply_static_p2p_signed_tensor_dictionary_plan_whole_tile(
    const StaticP2PSignedTensorDictionaryPlan& plan,
    std::span<const Vec3> dipole_moments, std::span<Vec3> H);
void apply_static_p2p_signed_tensor_dictionary_plan_whole_tile(
    const FloatStaticP2PSignedTensorDictionaryPlan& plan,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H);
void apply_static_p2p_bsr_plan(
    const StaticP2PBsrPlan& plan, std::span<const Vec3> dipole_moments,
    std::span<Vec3> H, std::span<const int> target_source_indices = {});
void apply_static_p2p_bsr_plan(
    const FloatStaticP2PBsrPlan& plan,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {});

[[nodiscard]] const char* static_p2p_signed_simd_path() noexcept;
[[nodiscard]] int static_p2p_signed_simd_width(bool single_precision) noexcept;

} // namespace cdfmm
