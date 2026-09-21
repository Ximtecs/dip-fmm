// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/plan/direct/dense.hpp"

namespace cdfmm {

/** @brief Reports whether the CUDA dense cuboid executor is available. */
[[nodiscard]] bool cuda_dense_direct_available() noexcept;

/**
 * @brief Owns six device-resident cuboid pair-tensor matrices.
 *
 * Geometry tensors are constructed once on the host and uploaded in
 * target-major order. Repeated evaluations transfer only the total source
 * moments and resulting magnetic fields; nine precision-matched cuBLAS
 * GEMVs perform the direct all-to-all contraction. FP64 remains the default
 * for backward compatibility; callers may explicitly select FP32 storage.
 */
class CudaDenseDirectPlan {
public:
  /**
   * @brief Builds the six matrices on the host and uploads them.
   *
   * Arguments and semantics match DenseDirectPlan, including `timing_level`,
   * which is forwarded to the host plan and also gates this plan's own
   * internal setup record (host build, context, allocation, upload).  `Off`
   * reads no clock.  Construction throws when no CUDA device is available.
   */
  CudaDenseDirectPlan(
      std::span<const Vec3> source_positions,
      std::span<const Vec3> target_positions,
      SourceGeometry source_geometry = SourceGeometry::PointDipole,
      TargetGeometry target_geometry = TargetGeometry::Point,
      std::span<const CuboidSize> source_sizes = {},
      std::span<const CuboidSize> target_sizes = {},
      std::span<const int> target_source_indices = {},
      StaticPrecision static_precision = StaticPrecision::Float64,
      std::span<const Tetrahedron> source_tetrahedra = {},
      std::span<const Tetrahedron> target_tetrahedra = {},
      SourceModel source_model = SourceModel::ExactGeometry,
      TargetModel target_model = TargetModel::ExactGeometry,
      TimingLevel timing_level = TimingLevel::Off
  );

  ~CudaDenseDirectPlan();

  CudaDenseDirectPlan(const CudaDenseDirectPlan &) = delete;
  CudaDenseDirectPlan &operator=(const CudaDenseDirectPlan &) = delete;

  /**
   * @brief Applies the cached device tensors to one total-moment state.
   *
   * @param total_moments Total source moments in source order.
   * @return Magnetic field values in target order.
   */
  [[nodiscard]] std::vector<Vec3> evaluate(
      std::span<const Vec3> total_moments
  );

  /// @brief Returns the number of fixed source positions.
  [[nodiscard]] std::size_t source_count() const noexcept;

  /// @brief Returns the number of fixed target positions.
  [[nodiscard]] std::size_t target_count() const noexcept;

  /// @brief Returns the bytes used by immutable tensor matrices.
  [[nodiscard]] std::size_t tensor_memory_bytes() const noexcept;

  /// @brief Returns persistent device storage used by this plan.
  [[nodiscard]] std::size_t persistent_device_bytes() const noexcept;

  /// @brief Returns precision used to store and execute the static tensors.
  [[nodiscard]] StaticPrecision static_precision() const noexcept;

private:
  struct Implementation;
  Implementation *implementation_{nullptr};
};

} // namespace cdfmm
