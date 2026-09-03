// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <string>
#include <vector>

#include "cdfmm/output_flags.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/uniform_fmm.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Full static FMM plan
//------------------------------------------------------------------------------

/** @brief One level of dependency-ordered static coefficient applications. */
struct CudaStaticLevel {
  std::vector<StaticOperatorEntry> entries{};
};

/** @brief One compact node-to-node translation using a shared matrix. */
struct CudaTranslationInteraction {
  int source_node{0};
  int target_node{0};
  int matrix_id{0};
  int level{0};
};

/** @brief Shared sparse translation classes and their node relations. */
struct CudaSharedTranslationData {
  std::vector<StaticOperatorEntry> matrices{};
  std::vector<CudaTranslationInteraction> interactions{};
  int entries_per_matrix{0};
  int matrix_count{0};
};

/** @brief FP32 shared sparse translations and node relations. */
struct FloatCudaSharedTranslationData {
  std::vector<FloatStaticOperatorEntry> matrices{};
  std::vector<CudaTranslationInteraction> interactions{};
  int entries_per_matrix{0};
  int matrix_count{0};
};

/** @brief Immutable CPU-built operators consumed by the full CUDA plan. */
struct CudaFullPlanData {
  int coefficient_count{0};
    int node_count{0};
    int source_count{0};
    int target_count{0};
  std::vector<int> source_permutation{};
  std::vector<int> target_permutation{};
  std::vector<int> coefficient_degrees{};
  std::vector<StaticOperatorEntry> p2m{};
  CudaSharedTranslationData m2m{};
  StaticM2LPlan m2l{};
  CudaSharedTranslationData l2l{};
  std::vector<StaticOperatorEntry> l2p{};
  StaticP2POperator p2p{};
  StaticP2PBsrPlan p2p_bsr{};
  StaticP2PSignedTensorDictionaryPlan p2p_dictionary{};
  std::vector<int> fixed_self_indices{};
  bool use_p2p_bsr{false};
  bool use_p2p_dictionary{false};
  bool has_fixed_self_indices{false};
};

/** @brief Immutable FP32 operators consumed by the full CUDA plan. */
struct FloatCudaFullPlanData {
  int coefficient_count{0};
  int node_count{0};
  int source_count{0};
  int target_count{0};
  std::vector<int> source_permutation{};
  std::vector<int> target_permutation{};
  std::vector<int> coefficient_degrees{};
  std::vector<FloatStaticOperatorEntry> p2m{};
  FloatCudaSharedTranslationData m2m{};
  FloatStaticM2LPlan m2l{};
  FloatCudaSharedTranslationData l2l{};
  std::vector<FloatStaticOperatorEntry> l2p{};
  FloatStaticP2POperator p2p{};
  FloatStaticP2PBsrPlan p2p_bsr{};
  FloatStaticP2PSignedTensorDictionaryPlan p2p_dictionary{};
  std::vector<int> fixed_self_indices{};
  bool use_p2p_bsr{false};
  bool use_p2p_dictionary{false};
  bool has_fixed_self_indices{false};
};

/**
 * @brief Owns a complete device-resident static CUDA FMM evaluation plan.
 *
 * All geometry and operator data are uploaded during construction. The first
 * identity map fixes self interaction metadata; changing it requires rebuilding
 * the plan. A normal field evaluation transfers only moments and final fields.
 */
class CudaFullPlan {
public:
    explicit CudaFullPlan(const CudaFullPlanData& data);
    explicit CudaFullPlan(const FloatCudaFullPlanData& data);
    ~CudaFullPlan();
  CudaFullPlan(const CudaFullPlan &) = delete;
  CudaFullPlan &operator=(const CudaFullPlan &) = delete;

  void evaluate(std::span<const Vec3> moments, std::span<Vec3> fields,
                std::span<const int> sorted_self_indices);
  void evaluate(std::span<const FloatVec3> moments,
                std::span<FloatVec3> fields,
                std::span<const int> sorted_self_indices);
  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept;
  [[nodiscard]] const CudaEvaluationTimings &timings() const noexcept;

private:
    struct Implementation;
    Implementation* implementation_{nullptr};
};

[[nodiscard]] bool cuda_runtime_available() noexcept;
[[nodiscard]] std::string cuda_runtime_description();

} // namespace cdfmm
