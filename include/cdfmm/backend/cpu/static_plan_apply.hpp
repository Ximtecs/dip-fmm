// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/plan/static_plan.hpp"

namespace cdfmm {

void apply_static_m2l_plan(
    const StaticM2LPlan& plan, int level,
    std::span<const double> multipoles, std::span<double> locals);
void apply_static_m2l_plan(
    const FloatStaticM2LPlan& plan, int level,
    std::span<const float> multipoles, std::span<float> locals);
void apply_static_m2l_plan(
    const StaticM2LPlan& plan, int level,
    std::span<const std::vector<double>> multipoles,
    std::span<std::vector<double>> locals);

void apply_static_operator(
    const StaticCoefficientOperator& operator_map,
    std::span<const double> input, std::span<double> output);
void apply_static_operator(
    const FloatStaticCoefficientOperator& operator_map,
    std::span<const float> input, std::span<float> output);
void apply_static_coefficient_matrix(
    std::span<const double> matrix, std::span<const double> input,
    std::span<double> output);

[[nodiscard]] PotentialField apply_static_l2p_evaluator(
    const StaticL2PEvaluator& evaluator, std::span<const double> local,
    OutputFlags output = OutputFlags::Field);
[[nodiscard]] FloatPotentialField apply_static_l2p_evaluator(
    const FloatStaticL2PEvaluator& evaluator, std::span<const float> local,
    OutputFlags output = OutputFlags::Field);

void apply_static_p2p_operator(
    const StaticP2POperator& operator_map,
    std::span<const Vec3> dipole_moments, std::span<Vec3> H,
    std::span<const int> target_source_indices = {});
void apply_static_p2p_operator(
    const FloatStaticP2POperator& operator_map,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {});
void apply_static_p2p_compact_plan(
    const StaticP2PCompactPlan& plan,
    std::span<const Vec3> dipole_moments, std::span<Vec3> H,
    std::span<const int> target_source_indices = {});
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
