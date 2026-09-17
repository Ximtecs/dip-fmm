// SPDX-License-Identifier: Apache-2.0
#pragma once

// CUDA procedural reconstruction of the exact prism <-> point P2P operator
// for benchmark_operator_representation.  The kernel instantiates the shared
// precision-generic prism point tensor
// (`src/geometry/primitives/rectangular_prism_point_kernel.hpp`) on the
// device, so the mathematics is the production formula; only the retained
// bodies and the list-1 work items are resident, no pair tensors.  This is an
// experimental representation for the Phase-3B.5b study, not a production
// executor.

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm_bench {

/** @brief One target and one dense source range of its list-1 neighbourhood. */
struct ProceduralWorkItem {
  int target{0};
  int source_begin{0};
  int source_count{0};
};

/** @brief Host view of a prism <-> point scene for the procedural kernel. */
struct ProceduralPrismScene {
  std::span<const cdfmm::Vec3> source_positions;
  std::span<const cdfmm::Vec3> target_positions;
  /// One common record or one per body; the prism side of the pair.
  std::span<const cdfmm::RectangularPrism> prisms;
  /// True when the source is the prism (prism -> point); false for the
  /// reciprocal point -> prism pair, whose tensor is the target prism's.
  bool prism_is_source{true};
  /// True when the source is a point, whose coincident self pair is the
  /// singular one the identity map excludes.
  bool skip_point_self_pair{false};
  std::vector<ProceduralWorkItem> items;
};

class ProceduralPrismCudaPlan {
public:
  explicit ProceduralPrismCudaPlan(const ProceduralPrismScene& scene);
  ~ProceduralPrismCudaPlan();
  ProceduralPrismCudaPlan(const ProceduralPrismCudaPlan&) = delete;
  ProceduralPrismCudaPlan& operator=(const ProceduralPrismCudaPlan&) = delete;

  /**
   * @brief One complete update with FP32 moments and fields.
   *
   * `double_math` reconstructs each tensor in FP64 before the FP32
   * accumulation; otherwise the whole reconstruction runs in FP32.  Returns
   * the device kernel time in seconds; the call itself is synchronous and
   * includes the moment upload and field download.
   */
  double evaluate(std::span<const cdfmm::FloatVec3> moments,
                  std::span<cdfmm::FloatVec3> fields, bool double_math);

  /** @brief One complete FP64 update (FP64 reconstruction and accumulation). */
  double evaluate(std::span<const cdfmm::Vec3> moments,
                  std::span<cdfmm::Vec3> fields);

  /// Persistent device bytes: positions, prism records and work items.
  [[nodiscard]] std::size_t device_bytes() const noexcept;
  [[nodiscard]] std::size_t item_count() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace cdfmm_bench
