// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "cdfmm/core/precision.hpp"
#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

// NOTE(cdfmm): `CuboidSize` is the legacy spelling of `RectangularPrism` and is
// declared by the canonical geometry primitive header included above.  It stays
// reachable from this path for source compatibility.

/** @brief Selects the GEMV implementation used by a dense direct plan. */
enum class DenseDirectBackend {
    Automatic,
    Portable,
    OneMkl
};

/** @brief Reports whether this build contains the oneMKL dense backend. */
[[nodiscard]] bool dense_direct_mkl_available() noexcept;

/**
 * @brief Six-matrix dense direct plan for fixed, independently selected geometry.
 *
 * Matrices are target-major with shape Nt x Ns. Construction performs all
 * geometry work; repeated evaluations reuse private execution staging.
 */
class DenseDirectPlan {
public:
    /**
     * @brief Builds the six exact pair-tensor matrices for fixed geometry.
     *
     * Finite records are one common record or one per body in user order; the
     * models select whether a finite body enters with its exact tensor or as
     * a point at its representative.  `target_source_indices` (target i ->
     * source j, or -1) omits the singular point self pairs and is fixed for
     * the plan's lifetime.
     */
    DenseDirectPlan(
        std::span<const Vec3> source_positions,
        std::span<const Vec3> target_positions,
        SourceGeometry source_geometry = SourceGeometry::PointDipole,
        TargetGeometry target_geometry = TargetGeometry::Point,
        std::span<const CuboidSize> source_sizes = {},
        std::span<const CuboidSize> target_sizes = {},
        std::span<const int> target_source_indices = {},
        StaticPrecision static_precision = StaticPrecision::Float32,
        std::span<const Tetrahedron> source_tetrahedra = {},
        std::span<const Tetrahedron> target_tetrahedra = {},
        SourceModel source_model = SourceModel::ExactGeometry,
        TargetModel target_model = TargetModel::ExactGeometry
    );

    /// @brief Deep-copies the matrices; the private staging is recreated.
    DenseDirectPlan(const DenseDirectPlan& other);
    /// @brief Deep-copies the matrices; the private staging is recreated.
    DenseDirectPlan& operator=(const DenseDirectPlan& other);
    /// @brief Transfers the matrices and staging.
    DenseDirectPlan(DenseDirectPlan&& other) noexcept;
    /// @brief Transfers the matrices and staging.
    DenseDirectPlan& operator=(DenseDirectPlan&& other) noexcept;
    ~DenseDirectPlan();

    /**
     * @brief Applies the cached tensors to one total-moment state.
     *
     * Automatic selects oneMKL when it is compiled into the library and the
     * portable implementation otherwise. Explicit selection makes comparative
     * benchmarks reproducible without rebuilding the geometry plan.
     *
     * @param total_moments Total source moments in source order.
     * @param backend Dense matrix-vector implementation to use.
     * @return Magnetic field values in target order.
     */
    [[nodiscard]] std::vector<Vec3> evaluate(
        std::span<const Vec3> total_moments,
        DenseDirectBackend backend = DenseDirectBackend::Automatic
    ) const;

    /// @brief Number of sources (matrix columns).
    [[nodiscard]] std::size_t source_count() const noexcept { return ns_; }
    /// @brief Number of targets (matrix rows).
    [[nodiscard]] std::size_t target_count() const noexcept { return nt_; }
    /// @brief Bytes retained by the six matrices at the plan's precision.
    [[nodiscard]] std::size_t tensor_memory_bytes() const noexcept;
    /// @brief Returns the scalar precision of the six immutable matrices.
    [[nodiscard]] StaticPrecision static_precision() const noexcept
    {
        return static_precision_;
    }
    /// @brief Number of stored tensor components per pair (the six of a symmetric tensor).
    [[nodiscard]] std::size_t tensor_component_count() const noexcept { return 6; }
    /** @brief Returns matrices for an explicitly selected FP64 plan. */
    [[nodiscard]] const std::array<std::vector<double>, 6>& matrices() const
    {
        return std::get<DoubleMatrices>(matrices_);
    }
    /** @brief Returns matrices for an explicitly selected FP32 plan. */
    [[nodiscard]] const std::array<std::vector<float>, 6>& float_matrices() const
    {
        return std::get<FloatMatrices>(matrices_);
    }

private:
    struct Impl;

    std::size_t ns_{0};
    std::size_t nt_{0};
    using FloatMatrices = std::array<std::vector<float>, 6>;
    using DoubleMatrices = std::array<std::vector<double>, 6>;
    StaticPrecision static_precision_{StaticPrecision::Float32};
    std::variant<FloatMatrices, DoubleMatrices> matrices_{FloatMatrices{}};
    // Mutable staging belongs to the private execution state and is reused
    // across evaluations without changing the immutable matrices above.
    std::unique_ptr<Impl> impl_;
};

} // namespace cdfmm
