// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <variant>
#include <vector>

#include "cdfmm/coefficients.hpp"
#include "cdfmm/multi_index.hpp"
#include "cdfmm/precision.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Geometry and pair tensors
//------------------------------------------------------------------------------

/** @brief Selects the physical model used for a magnetic source. */
enum class SourceGeometry {
    /// Singular point dipole carrying a total magnetic moment.
    PointDipole,
    /// Axis-aligned uniformly magnetised cuboid carrying a total moment.
    UniformCuboid
};

/** @brief Selects point or volume-averaged field evaluation. */
enum class TargetGeometry {
    /// Evaluate at one point.
    Point,
    /// Return the analytical average over an axis-aligned cuboid.
    VolumeAveragedCuboid
};

/** @brief Side lengths of an axis-aligned rectangular cuboid. */
struct CuboidSize {
    double hx{0.0};
    double hy{0.0};
    double hz{0.0};

    /// @brief Returns the cuboid volume.
    [[nodiscard]] double volume() const noexcept { return hx * hy * hz; }
};

/** @brief Six independent components of a symmetric Cartesian pair tensor. */
struct PairTensor {
    double xx{0.0};
    double xy{0.0};
    double xz{0.0};
    double yy{0.0};
    double yz{0.0};
    double zz{0.0};
};

/**
 * @brief Evaluates the factorial-normalised monomial averaged over a cuboid.
 *
 * This is J_beta(d,h) = V^-1 integral_V (d+u)^beta/beta! dV and is evaluated
 * by its finite even-power sum, without numerical quadrature.
 */
[[nodiscard]] double cuboid_averaged_monomial(
    const MultiIndex& beta,
    const Vec3& d,
    const CuboidSize& h
);

/**
 * @brief Constructs the exact moment-to-field tensor for one geometry pair.
 *
 * Runtime inputs are total moments m=V*M. Cuboid source normalisation is
 * consequently included in the returned tensor. Point-point coincidence is
 * singular unless @p omit_singular_point_pair is true.
 */
[[nodiscard]] PairTensor build_pair_tensor(
    const Vec3& target_position,
    const Vec3& source_position,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    TargetGeometry target_geometry = TargetGeometry::Point,
    const CuboidSize& source_size = {},
    const CuboidSize& target_size = {},
    bool omit_singular_point_pair = false
);

//------------------------------------------------------------------------------
// Persistent dense direct operator
//------------------------------------------------------------------------------

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
 * geometry work; repeated evaluations contain only packing and nine GEMVs.
 */
class DenseDirectPlan {
public:
    DenseDirectPlan(
        std::span<const Vec3> source_positions,
        std::span<const Vec3> target_positions,
        SourceGeometry source_geometry = SourceGeometry::PointDipole,
        TargetGeometry target_geometry = TargetGeometry::Point,
        std::span<const CuboidSize> source_sizes = {},
        std::span<const CuboidSize> target_sizes = {},
        std::span<const int> target_source_indices = {},
        StaticPrecision static_precision = StaticPrecision::Float32
    );

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

    [[nodiscard]] std::size_t source_count() const noexcept { return ns_; }
    [[nodiscard]] std::size_t target_count() const noexcept { return nt_; }
    [[nodiscard]] std::size_t tensor_memory_bytes() const noexcept;
    /// @brief Returns the scalar precision of the six immutable matrices.
    [[nodiscard]] StaticPrecision static_precision() const noexcept
    {
        return static_precision_;
    }
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
    std::size_t ns_{0};
    std::size_t nt_{0};
    using FloatMatrices = std::array<std::vector<float>, 6>;
    using DoubleMatrices = std::array<std::vector<double>, 6>;
    StaticPrecision static_precision_{StaticPrecision::Float32};
    std::variant<FloatMatrices, DoubleMatrices> matrices_{FloatMatrices{}};
    // Component staging is retained and reused across repeated evaluations.
    mutable std::array<std::vector<float>, 3> float_moments_{};
    mutable std::array<std::vector<float>, 3> float_fields_{};
    mutable std::array<std::vector<double>, 3> double_moments_{};
    mutable std::array<std::vector<double>, 3> double_fields_{};
};

} // namespace cdfmm
