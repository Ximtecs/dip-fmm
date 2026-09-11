// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/direct/dense.hpp"

#include "cdfmm/cuboid.hpp"
#include "backend/cpu/direct/dense.hpp"
#include "backend/direct/dense_workspace.hpp"
#include "backend/mkl/direct/dense.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cdfmm {
namespace {

void validate_size(const CuboidSize& h, const char* name)
{
    if (!(std::isfinite(h.hx) && std::isfinite(h.hy) &&
          std::isfinite(h.hz) && h.hx > 0.0 && h.hy > 0.0 && h.hz > 0.0)) {
        throw std::invalid_argument(std::string(name) +
                                    " dimensions must be finite and positive");
    }
}

} // namespace

struct DenseDirectPlan::Impl {
    detail::dense_direct::DenseDirectWorkspace workspace;
};

DenseDirectPlan::DenseDirectPlan(const DenseDirectPlan& other)
    : ns_(other.ns_), nt_(other.nt_), static_precision_(other.static_precision_),
      matrices_(other.matrices_),
      impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr)
{
}

DenseDirectPlan& DenseDirectPlan::operator=(const DenseDirectPlan& other)
{
    if (this != &other) {
        DenseDirectPlan copy(other);
        *this = std::move(copy);
    }
    return *this;
}

DenseDirectPlan::DenseDirectPlan(DenseDirectPlan&& other) noexcept
    : ns_(other.ns_), nt_(other.nt_),
      static_precision_(other.static_precision_),
      matrices_(std::move(other.matrices_)), impl_(std::move(other.impl_))
{
}

DenseDirectPlan& DenseDirectPlan::operator=(DenseDirectPlan&& other) noexcept
{
    if (this != &other) {
        ns_ = other.ns_;
        nt_ = other.nt_;
        static_precision_ = other.static_precision_;
        matrices_ = std::move(other.matrices_);
        impl_ = std::move(other.impl_);
    }
    return *this;
}

DenseDirectPlan::~DenseDirectPlan() = default;

DenseDirectPlan::DenseDirectPlan(
    const std::span<const Vec3> source_positions,
    const std::span<const Vec3> target_positions,
    const SourceGeometry source_geometry,
    const TargetGeometry target_geometry,
    const std::span<const CuboidSize> source_sizes,
    const std::span<const CuboidSize> target_sizes,
    const std::span<const int> target_source_indices,
    const StaticPrecision static_precision,
    const std::span<const Tetrahedron> source_tetrahedra,
    const std::span<const Tetrahedron> target_tetrahedra,
    const SourceModel source_model,
    const TargetModel target_model)
    : ns_(source_positions.size()), nt_(target_positions.size()),
      static_precision_(static_precision), impl_(std::make_unique<Impl>())
{
    const SourceGeometry effective_source_geometry =
        source_model == SourceModel::ExactGeometry
            ? source_geometry : SourceGeometry::PointDipole;
    const TargetGeometry effective_target_geometry =
        target_model == TargetModel::ExactGeometry
            ? target_geometry : TargetGeometry::Point;
    const bool source_is_prism =
        effective_source_geometry == SourceGeometry::RectangularPrism;
    const bool source_is_tetrahedron =
        effective_source_geometry == SourceGeometry::Tetrahedron;
    const bool target_is_prism =
        effective_target_geometry == TargetGeometry::RectangularPrism;
    const bool target_is_tetrahedron =
        effective_target_geometry == TargetGeometry::Tetrahedron;

    if ((source_is_prism && target_is_tetrahedron) ||
        (source_is_tetrahedron && target_is_prism)) {
        throw std::invalid_argument(
            "exact prism/tetrahedron DenseDirect interactions are unsupported");
    }
    if (static_precision_ == StaticPrecision::Float32) {
        matrices_.emplace<FloatMatrices>();
    } else {
        matrices_.emplace<DoubleMatrices>();
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
    // Validate physical records from the declared geometry even when a
    // point model is selected for this plan.  This keeps DenseDirectPlan's
    // contract consistent with UniformFmm and prevents silently malformed
    // geometry from being accepted merely because it is currently ignored.
    if (source_geometry == SourceGeometry::PointDipole) {
        if (!source_sizes.empty() || !source_tetrahedra.empty()) {
            throw std::invalid_argument(
                "point-dipole sources do not accept finite geometry records");
        }
    } else if (source_geometry == SourceGeometry::RectangularPrism) {
        if (!source_tetrahedra.empty()) {
            throw std::invalid_argument(
                "rectangular-prism sources do not accept tetrahedron records");
        }
        if (source_sizes.empty()) {
            throw std::invalid_argument("cuboid sources require dimensions");
        }
    } else if (source_geometry == SourceGeometry::Tetrahedron) {
        if (!source_sizes.empty()) {
            throw std::invalid_argument(
                "tetrahedron sources do not accept rectangular-prism sizes");
        }
        if (source_tetrahedra.empty()) {
            throw std::invalid_argument("tetrahedron sources require geometry");
        }
    }
    if (target_geometry == TargetGeometry::Point) {
        if (!target_sizes.empty() || !target_tetrahedra.empty()) {
            throw std::invalid_argument(
                "point targets do not accept finite geometry records");
        }
    } else if (target_geometry == TargetGeometry::RectangularPrism) {
        if (!target_tetrahedra.empty()) {
            throw std::invalid_argument(
                "rectangular-prism targets do not accept tetrahedron records");
        }
        if (target_sizes.empty()) {
            throw std::invalid_argument("cuboid targets require dimensions");
        }
    } else if (target_geometry == TargetGeometry::Tetrahedron) {
        if (!target_sizes.empty()) {
            throw std::invalid_argument(
                "tetrahedron targets do not accept rectangular-prism sizes");
        }
        if (target_tetrahedra.empty()) {
            throw std::invalid_argument("tetrahedron targets require geometry");
        }
    }
    if (source_geometry == SourceGeometry::Tetrahedron) {
        if (source_tetrahedra.size() != 1 &&
            source_tetrahedra.size() != ns_) {
            throw std::invalid_argument(
                "tetrahedron sources must be common or per object");
        }
        for (const Tetrahedron& tetrahedron : source_tetrahedra) {
            static_cast<void>(tetrahedron_volume(tetrahedron));
        }
    }
    if (target_geometry == TargetGeometry::Tetrahedron) {
        if (target_tetrahedra.size() != 1 &&
            target_tetrahedra.size() != nt_) {
            throw std::invalid_argument(
                "tetrahedron targets must be common or per object");
        }
        for (const Tetrahedron& tetrahedron : target_tetrahedra) {
            static_cast<void>(tetrahedron_volume(tetrahedron));
        }
    }
    if (source_geometry == SourceGeometry::RectangularPrism) {
        for (const CuboidSize& size : source_sizes) {
            validate_size(size, "source cuboid");
        }
    }
    if (target_geometry == TargetGeometry::RectangularPrism) {
        for (const CuboidSize& size : target_sizes) {
            validate_size(size, "target cuboid");
        }
    }
    if (ns_ != 0 && nt_ > std::numeric_limits<std::size_t>::max() / ns_) {
        throw std::overflow_error("dense direct tensor size overflow");
    }
    std::visit([&](auto& matrices) {
        for (auto& matrix : matrices) {
            matrix.resize(ns_ * nt_);
        }
    }, matrices_);

    // Every source-target tensor owns one fixed matrix entry. Constructing
    // those entries in parallel preserves the target-major storage order and
    // introduces no floating-point reductions.
    std::exception_ptr construction_error;
    std::visit([&](auto& matrices) {
        using Scalar = typename std::decay_t<
            decltype(matrices)>::value_type::value_type;
        const std::ptrdiff_t pair_count =
            static_cast<std::ptrdiff_t>(ns_ * nt_);
#pragma omp parallel for schedule(static) if (pair_count >= 256)
        for (std::ptrdiff_t pair_index = 0; pair_index < pair_count;
             ++pair_index) {
            try {
                const std::size_t index =
                    static_cast<std::size_t>(pair_index);
                const std::size_t target = index / ns_;
                const std::size_t source = index % ns_;
                const bool identity = !target_source_indices.empty() &&
                    target_source_indices[target] ==
                        static_cast<int>(source);
                const CuboidSize source_size = source_sizes.empty()
                    ? CuboidSize{}
                    : source_sizes[source_sizes.size() == 1 ? 0 : source];
                const CuboidSize target_size = target_sizes.empty()
                    ? CuboidSize{}
                    : target_sizes[target_sizes.size() == 1 ? 0 : target];
                const Tetrahedron* source_tetrahedron = source_is_tetrahedron
                    ? &source_tetrahedra[source_tetrahedra.size() == 1 ? 0 : source]
                    : nullptr;
                const Tetrahedron* target_tetrahedron = target_is_tetrahedron
                    ? &target_tetrahedra[target_tetrahedra.size() == 1 ? 0 : target]
                    : nullptr;
                const bool omit_identity =
                    identity && effective_source_geometry ==
                        SourceGeometry::PointDipole;
                PairTensor tensor;
                if (omit_identity) {
                    tensor = {};
                } else if (source_is_tetrahedron && target_is_tetrahedron) {
                    tensor = tetrahedron_tetrahedron_tensor(
                        target_positions[target] - source_positions[source],
                        *source_tetrahedron, *target_tetrahedron);
                } else if (source_is_tetrahedron) {
                    tensor = tetrahedron_point_tensor(
                        target_positions[target] - source_positions[source],
                        *source_tetrahedron);
                } else if (target_is_tetrahedron) {
                    tensor = point_tetrahedron_tensor(
                        target_positions[target] - source_positions[source],
                        *target_tetrahedron);
                } else {
                    tensor = build_pair_tensor(
                        target_positions[target], source_positions[source],
                        effective_source_geometry, effective_target_geometry,
                        source_size, target_size, omit_identity);
                }
                matrices[0][index] = static_cast<Scalar>(tensor.xx);
                matrices[1][index] = static_cast<Scalar>(tensor.xy);
                matrices[2][index] = static_cast<Scalar>(tensor.xz);
                matrices[3][index] = static_cast<Scalar>(tensor.yy);
                matrices[4][index] = static_cast<Scalar>(tensor.yz);
                matrices[5][index] = static_cast<Scalar>(tensor.zz);
            } catch (...) {
#pragma omp critical(cdfmm_dense_direct_construction_error)
                {
                    if (!construction_error) {
                        construction_error = std::current_exception();
                    }
                }
            }
        }
    }, matrices_);
    if (construction_error) {
        std::rethrow_exception(construction_error);
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
    if (backend != DenseDirectBackend::Portable &&
        backend != DenseDirectBackend::OneMkl) {
        throw std::invalid_argument("unsupported dense direct backend");
    }

    std::vector<Vec3> result;
    std::visit([&](const auto& matrices) {
        if (backend == DenseDirectBackend::Portable) {
            detail::dense_direct::cpu::apply(
                matrices, nt_, ns_, total_moments, impl_->workspace, result);
        } else {
            detail::dense_direct::mkl::apply(
                matrices, nt_, ns_, total_moments, impl_->workspace, result);
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
