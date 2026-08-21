// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/cuboid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>

#ifdef CDFMM_USE_MKL
#include <mkl.h>
#endif

namespace cdfmm {
namespace {

double factorial(const int n)
{
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

void validate_size(const CuboidSize& h, const char* name)
{
    if (!(h.hx > 0.0 && h.hy > 0.0 && h.hz > 0.0)) {
        throw std::invalid_argument(std::string(name) +
                                    " dimensions must be positive");
    }
}

double safe_asinh_ratio(const double numerator, const double a, const double b)
{
    const double denominator = std::hypot(a, b);
    if (denominator == 0.0) {
        return 0.0;
    }
    return std::asinh(numerator / denominator);
}

double safe_atan_ratio(const double numerator, const double denominator)
{
    if (denominator != 0.0) {
        return std::atan(numerator / denominator);
    }
    if (numerator == 0.0) {
        return 0.0;
    }
    return std::copysign(0.5 * std::numbers::pi, numerator);
}

// Newell's finite-volume diagonal primitive. This independently implements
// the published rectangular-cell formula; no MagTense source is incorporated.
double newell_f(const double x, const double y, const double z)
{
    const double R = std::hypot(std::hypot(x, y), z);
    if (R == 0.0) {
        return 0.0;
    }
    return 0.5 * y * (z * z - x * x) * safe_asinh_ratio(y, x, z) +
           0.5 * z * (y * y - x * x) * safe_asinh_ratio(z, x, y) -
           x * y * z * safe_atan_ratio(y * z, x * R) +
           (2.0 * x * x - y * y - z * z) * R / 6.0;
}

// Newell's off-diagonal primitive uses explicit limiting ratios at faces,
// edges and self positions rather than arbitrary coordinate perturbations.
double newell_g(const double x, const double y, const double z)
{
    const double R = std::hypot(std::hypot(x, y), z);
    if (R == 0.0) {
        return 0.0;
    }
    return x * y * z * safe_asinh_ratio(z, x, y) +
           y * (3.0 * z * z - y * y) * safe_asinh_ratio(x, y, z) / 6.0 +
           x * (3.0 * z * z - x * x) * safe_asinh_ratio(y, x, z) / 6.0 -
           z * z * z * safe_atan_ratio(x * y, z * R) / 6.0 -
           z * y * y * safe_atan_ratio(x * z, y * R) / 2.0 -
           z * x * x * safe_atan_ratio(y * z, x * R) / 2.0 -
           x * y * R / 3.0;
}

template <typename Primitive>
double finite_volume_sum(const Vec3& r, const CuboidSize& source,
                         const CuboidSize& target, Primitive primitive)
{
    constexpr int weights[4] = {1, -1, -1, 1};
    const double x_offsets[4] = {
        -0.5 * (source.hx + target.hx),
        -0.5 * source.hx + 0.5 * target.hx,
         0.5 * source.hx - 0.5 * target.hx,
         0.5 * (source.hx + target.hx)};
    const double y_offsets[4] = {
        -0.5 * (source.hy + target.hy),
        -0.5 * source.hy + 0.5 * target.hy,
         0.5 * source.hy - 0.5 * target.hy,
         0.5 * (source.hy + target.hy)};
    const double z_offsets[4] = {
        -0.5 * (source.hz + target.hz),
        -0.5 * source.hz + 0.5 * target.hz,
         0.5 * source.hz - 0.5 * target.hz,
         0.5 * (source.hz + target.hz)};
    double result = 0.0;
    for (int ix = 0; ix < 4; ++ix) {
        for (int iy = 0; iy < 4; ++iy) {
            for (int iz = 0; iz < 4; ++iz) {
                const double x = r.x + x_offsets[ix];
                const double y = r.y + y_offsets[iy];
                const double z = r.z + z_offsets[iz];
                result += weights[ix] * weights[iy] * weights[iz] *
                    primitive(x, y, z);
            }
        }
    }
    return result;
}

PairTensor cuboid_cuboid_tensor(const Vec3& r, const CuboidSize& source,
                                const CuboidSize& target)
{
    validate_size(source, "source cuboid");
    validate_size(target, "target cuboid");
    const double scale = 1.0 / (4.0 * std::numbers::pi *
                                 source.volume() * target.volume());
    PairTensor tensor;
    tensor.xx = scale * finite_volume_sum(
        r, source, target, [](double x, double y, double z) {
            return newell_f(x, y, z);
        });
    tensor.yy = scale * finite_volume_sum(
        r, source, target, [](double x, double y, double z) {
            return newell_f(y, x, z);
        });
    tensor.zz = scale * finite_volume_sum(
        r, source, target, [](double x, double y, double z) {
            return newell_f(z, y, x);
        });
    tensor.xy = scale * finite_volume_sum(
        r, source, target, [](double x, double y, double z) {
            return newell_g(x, y, z);
        });
    tensor.xz = scale * finite_volume_sum(
        r, source, target, [](double x, double y, double z) {
            return newell_g(x, z, y);
        });
    tensor.yz = scale * finite_volume_sum(
        r, source, target, [](double x, double y, double z) {
            return newell_g(y, z, x);
        });
    return tensor;
}

PairTensor cuboid_point_tensor(const Vec3& r, const CuboidSize& source)
{
    validate_size(source, "source cuboid");
    double diagonal[3]{};
    double off_diagonal[3]{};
    for (int ix = 0; ix < 2; ++ix) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int iz = 0; iz < 2; ++iz) {
                const double x = r.x + (ix == 0 ? -0.5 : 0.5) * source.hx;
                const double y = r.y + (iy == 0 ? -0.5 : 0.5) * source.hy;
                const double z = r.z + (iz == 0 ? -0.5 : 0.5) * source.hz;
                const double R = std::hypot(std::hypot(x, y), z);
                const double sign = ((ix + iy + iz) % 2 == 0) ? -1.0 : 1.0;
                diagonal[0] += sign * safe_atan_ratio(y * z, x * R);
                diagonal[1] += sign * safe_atan_ratio(x * z, y * R);
                diagonal[2] += sign * safe_atan_ratio(x * y, z * R);
                off_diagonal[0] += sign * safe_asinh_ratio(z, x, y);
                off_diagonal[1] += sign * safe_asinh_ratio(y, x, z);
                off_diagonal[2] += sign * safe_asinh_ratio(x, y, z);
            }
        }
    }
    const double scale = 1.0 / (4.0 * std::numbers::pi * source.volume());
    return {-scale * diagonal[0], scale * off_diagonal[0],
            scale * off_diagonal[1], -scale * diagonal[1],
            scale * off_diagonal[2], -scale * diagonal[2]};
}

template <typename Scalar>
void portable_gemv(const std::vector<Scalar>& matrix, const std::size_t rows,
                   const std::size_t columns,
                   const std::vector<Scalar>& input,
                   std::vector<Scalar>& output, const bool add)
{
    if (!add) {
        std::fill(output.begin(), output.end(), 0.0);
    }
    for (std::size_t row = 0; row < rows; ++row) {
        Scalar value = Scalar{0};
        for (std::size_t column = 0; column < columns; ++column) {
            value += matrix[row * columns + column] * input[column];
        }
        output[row] += value;
    }
}

template <typename Scalar>
void gemv(const std::vector<Scalar>& matrix, const std::size_t rows,
          const std::size_t columns, const std::vector<Scalar>& input,
          std::vector<Scalar>& output, const bool add,
          const DenseDirectBackend backend)
{
    if (backend == DenseDirectBackend::Portable) {
        portable_gemv(matrix, rows, columns, input, output, add);
        return;
    }

#ifdef CDFMM_USE_MKL
    if (backend == DenseDirectBackend::OneMkl) {
        if constexpr (std::is_same_v<Scalar, float>) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<MKL_INT>(rows),
                        static_cast<MKL_INT>(columns), 1.0F, matrix.data(),
                        static_cast<MKL_INT>(columns), input.data(), 1,
                        add ? 1.0F : 0.0F, output.data(), 1);
        } else {
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<MKL_INT>(rows),
                        static_cast<MKL_INT>(columns), 1.0, matrix.data(),
                        static_cast<MKL_INT>(columns), input.data(), 1,
                        add ? 1.0 : 0.0, output.data(), 1);
        }
        return;
    }
#endif

    throw std::invalid_argument("unsupported dense direct backend");
}

} // namespace

bool dense_direct_mkl_available() noexcept
{
#ifdef CDFMM_USE_MKL
    return true;
#else
    return false;
#endif
}

double cuboid_averaged_monomial(const MultiIndex& beta, const Vec3& d,
                                const CuboidSize& h)
{
    validate_size(h, "cuboid");
    const int powers[3] = {beta.ax, beta.ay, beta.az};
    const double offsets[3] = {d.x, d.y, d.z};
    const double lengths[3] = {h.hx, h.hy, h.hz};
    double result = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        double axis_sum = 0.0;
        for (int gamma = 0; gamma <= powers[axis]; gamma += 2) {
            axis_sum += std::pow(offsets[axis], powers[axis] - gamma) /
                factorial(powers[axis] - gamma) *
                std::pow(lengths[axis], gamma) /
                (std::pow(2.0, gamma) * factorial(gamma + 1));
        }
        result *= axis_sum;
    }
    return result;
}

PairTensor build_pair_tensor(const Vec3& target_position,
                             const Vec3& source_position,
                             const SourceGeometry source_geometry,
                             const TargetGeometry target_geometry,
                             const CuboidSize& source_size,
                             const CuboidSize& target_size,
                             const bool omit_singular_point_pair)
{
    const Vec3 r = target_position - source_position;
    if (source_geometry == SourceGeometry::UniformCuboid &&
        target_geometry == TargetGeometry::VolumeAveragedCuboid) {
        return cuboid_cuboid_tensor(r, source_size, target_size);
    }

    // Reciprocity converts a point-to-volume average into the corresponding
    // finite-source field, including the required total-moment normalisation.
    if (source_geometry == SourceGeometry::PointDipole &&
        target_geometry == TargetGeometry::VolumeAveragedCuboid) {
        return cuboid_point_tensor(r, target_size);
    }

    if (source_geometry == SourceGeometry::UniformCuboid) {
        return cuboid_point_tensor(r, source_size);
    }

    const double r2 = dot(r, r);
    if (r2 == 0.0) {
        if (omit_singular_point_pair) {
            return {};
        }
        throw std::domain_error("coincident point dipole and point target");
    }
    const double inverse_r = 1.0 / std::sqrt(r2);
    const double inverse_r3 = inverse_r / r2;
    const double diagonal = inverse_r3 / (4.0 * std::numbers::pi);
    const double common = 3.0 * diagonal / r2;
    return {common * r.x * r.x - diagonal,
            common * r.x * r.y, common * r.x * r.z,
            common * r.y * r.y - diagonal, common * r.y * r.z,
            common * r.z * r.z - diagonal};
}

DenseDirectPlan::DenseDirectPlan(
    const std::span<const Vec3> source_positions,
    const std::span<const Vec3> target_positions,
    const SourceGeometry source_geometry,
    const TargetGeometry target_geometry,
    const std::span<const CuboidSize> source_sizes,
    const std::span<const CuboidSize> target_sizes,
    const std::span<const int> target_source_indices,
    const StaticPrecision static_precision)
    : ns_(source_positions.size()), nt_(target_positions.size()),
      static_precision_(static_precision)
{
    if (static_precision_ == StaticPrecision::Float32) {
        matrices_.emplace<FloatMatrices>();
    }
    if ((!source_sizes.empty() && source_sizes.size() != 1 &&
         source_sizes.size() != ns_) ||
        (!target_sizes.empty() && target_sizes.size() != 1 &&
         target_sizes.size() != nt_)) {
        throw std::invalid_argument("cuboid sizes must be common or per object");
    }
    if (!target_source_indices.empty() && target_source_indices.size() != nt_) {
        throw std::invalid_argument("target-source identity map has wrong length");
    }
    if (source_geometry == SourceGeometry::UniformCuboid && source_sizes.empty()) {
        throw std::invalid_argument("cuboid sources require dimensions");
    }
    if (target_geometry == TargetGeometry::VolumeAveragedCuboid &&
        target_sizes.empty()) {
        throw std::invalid_argument("cuboid targets require dimensions");
    }
    std::visit([&](auto& matrices) {
        for (auto& matrix : matrices) {
            matrix.resize(ns_ * nt_);
        }
    }, matrices_);
    for (std::size_t target = 0; target < nt_; ++target) {
        for (std::size_t source = 0; source < ns_; ++source) {
            const bool identity = !target_source_indices.empty() &&
                target_source_indices[target] == static_cast<int>(source);
            const CuboidSize source_size = source_sizes.empty() ? CuboidSize{} :
                source_sizes[source_sizes.size() == 1 ? 0 : source];
            const CuboidSize target_size = target_sizes.empty() ? CuboidSize{} :
                target_sizes[target_sizes.size() == 1 ? 0 : target];
            const PairTensor tensor = build_pair_tensor(
                target_positions[target], source_positions[source],
                source_geometry, target_geometry, source_size, target_size,
                identity && source_geometry == SourceGeometry::PointDipole &&
                    target_geometry == TargetGeometry::Point);
            const std::size_t index = target * ns_ + source;
            std::visit([&](auto& matrices) {
                using Scalar = typename std::decay_t<decltype(matrices)>::value_type::value_type;
                matrices[0][index] = static_cast<Scalar>(tensor.xx);
                matrices[1][index] = static_cast<Scalar>(tensor.xy);
                matrices[2][index] = static_cast<Scalar>(tensor.xz);
                matrices[3][index] = static_cast<Scalar>(tensor.yy);
                matrices[4][index] = static_cast<Scalar>(tensor.yz);
                matrices[5][index] = static_cast<Scalar>(tensor.zz);
            }, matrices_);
        }
    }
}

std::vector<Vec3> DenseDirectPlan::evaluate(
    const std::span<const Vec3> total_moments,
    DenseDirectBackend backend) const
{
    if (total_moments.size() != ns_) {
        throw std::invalid_argument("dense direct plan requires one moment per source");
    }
    if (backend == DenseDirectBackend::Automatic) {
        backend = dense_direct_mkl_available() ? DenseDirectBackend::OneMkl :
            DenseDirectBackend::Portable;
    }
    if (backend == DenseDirectBackend::OneMkl &&
        !dense_direct_mkl_available()) {
        throw std::runtime_error(
            "oneMKL dense direct backend is not enabled in this build");
    }

    std::vector<Vec3> result(nt_);
    std::visit([&](const auto& matrices) {
        using Scalar = typename std::decay_t<decltype(matrices)>::value_type::value_type;
        auto& moments = [&]() -> auto& {
            if constexpr (std::is_same_v<Scalar, float>) {
                return float_moments_;
            } else {
                return double_moments_;
            }
        }();
        auto& fields = [&]() -> auto& {
            if constexpr (std::is_same_v<Scalar, float>) {
                return float_fields_;
            } else {
                return double_fields_;
            }
        }();
        for (auto& component : moments) {
            component.resize(ns_);
        }
        for (auto& component : fields) {
            component.resize(nt_);
        }
        for (std::size_t source = 0; source < ns_; ++source) {
            moments[0][source] = static_cast<Scalar>(total_moments[source].x);
            moments[1][source] = static_cast<Scalar>(total_moments[source].y);
            moments[2][source] = static_cast<Scalar>(total_moments[source].z);
        }
        gemv(matrices[0], nt_, ns_, moments[0], fields[0], false, backend);
        gemv(matrices[1], nt_, ns_, moments[1], fields[0], true, backend);
        gemv(matrices[2], nt_, ns_, moments[2], fields[0], true, backend);
        gemv(matrices[1], nt_, ns_, moments[0], fields[1], false, backend);
        gemv(matrices[3], nt_, ns_, moments[1], fields[1], true, backend);
        gemv(matrices[4], nt_, ns_, moments[2], fields[1], true, backend);
        gemv(matrices[2], nt_, ns_, moments[0], fields[2], false, backend);
        gemv(matrices[4], nt_, ns_, moments[1], fields[2], true, backend);
        gemv(matrices[5], nt_, ns_, moments[2], fields[2], true, backend);
        for (std::size_t target = 0; target < nt_; ++target) {
            result[target] = {static_cast<double>(fields[0][target]),
                              static_cast<double>(fields[1][target]),
                              static_cast<double>(fields[2][target])};
        }
    }, matrices_);
    return result;
}

std::size_t DenseDirectPlan::tensor_memory_bytes() const noexcept
{
    const std::size_t scalar_bytes = static_precision_ == StaticPrecision::Float32
        ? sizeof(float) : sizeof(double);
    return 6 * ns_ * nt_ * scalar_bytes;
}

} // namespace cdfmm
