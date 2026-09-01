// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/cuboid.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Persistent CUDA dense direct operator
//------------------------------------------------------------------------------

/** @brief Reports whether the CUDA dense cuboid executor is available. */
[[nodiscard]] bool cuda_dense_direct_available() noexcept;

/**
 * @brief Owns six device-resident cuboid pair-tensor matrices.
 *
 * Geometry tensors are constructed once on the host and uploaded in
 * target-major order. Repeated evaluations transfer only the total source
 * moments and resulting magnetic fields; nine precision-matched cuBLAS GEMVs
 * perform the direct all-to-all contraction. FP64 remains the default for
 * backward compatibility; callers may explicitly select FP32 storage.
 */
class CudaDenseDirectPlan {
public:
    CudaDenseDirectPlan(
        std::span<const Vec3> source_positions,
        std::span<const Vec3> target_positions,
        SourceGeometry source_geometry = SourceGeometry::PointDipole,
        TargetGeometry target_geometry = TargetGeometry::Point,
        std::span<const CuboidSize> source_sizes = {},
        std::span<const CuboidSize> target_sizes = {},
        std::span<const int> target_source_indices = {},
        StaticPrecision static_precision = StaticPrecision::Float64
    );

    ~CudaDenseDirectPlan();

    CudaDenseDirectPlan(const CudaDenseDirectPlan&) = delete;
    CudaDenseDirectPlan& operator=(const CudaDenseDirectPlan&) = delete;

    /**
     * @brief Applies the cached device tensors to one total-moment state.
     *
     * @param total_moments Total source moments in source order.
     * @return Magnetic field values in target order.
     */
    [[nodiscard]] std::vector<Vec3> evaluate(
        std::span<const Vec3> total_moments
    );

    [[nodiscard]] std::size_t source_count() const noexcept;
    [[nodiscard]] std::size_t target_count() const noexcept;
    [[nodiscard]] std::size_t tensor_memory_bytes() const noexcept;
    [[nodiscard]] std::size_t persistent_device_bytes() const noexcept;
    [[nodiscard]] StaticPrecision static_precision() const noexcept;

private:
    struct Implementation;
    Implementation* implementation_{nullptr};
};

} // namespace cdfmm
