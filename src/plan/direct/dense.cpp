// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/direct/dense.hpp"

#include "cdfmm/operators/p2p.hpp"
#include "backend/cpu/direct/dense.hpp"
#include "backend/direct/dense_workspace.hpp"
#include "backend/mkl/direct/dense.hpp"
#include "geometry/primitives/tetrahedron_detail.hpp"
#include "operators/exact_operator_reuse.hpp"
#include "plan/direct/construction_statistics.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cdfmm {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

/// @brief Transient bytes held by a set of prepared bodies.
[[nodiscard]] std::size_t prepared_bytes(
    const std::vector<detail::PreparedTetrahedron>& prepared) noexcept
{
    return prepared.size() * sizeof(detail::PreparedTetrahedron);
}

/// @brief Transient bytes held by a set of prepared bodies, surfaces included.
[[nodiscard]] std::size_t prepared_bytes(
    const std::vector<detail::PolyhedronBody>& prepared) noexcept
{
    std::size_t bytes = prepared.size() * sizeof(detail::PolyhedronBody);
    for (const detail::PolyhedronBody& body : prepared) {
        bytes += body.surface.faces.size() * sizeof(std::array<Vec3, 3>);
        bytes += body.surface.outward_normals.size() * sizeof(Vec3);
    }
    return bytes;
}

void validate_size(const CuboidSize& h, const char* name)
{
    if (!(std::isfinite(h.hx) && std::isfinite(h.hy) &&
          std::isfinite(h.hz) && h.hx > 0.0 && h.hy > 0.0 && h.hz > 0.0)) {
        throw std::invalid_argument(std::string(name) +
                                    " dimensions must be finite and positive");
    }
}

} // namespace

namespace detail::dense_direct {

ConstructionStatistics& construction_statistics() noexcept
{
    // One record per thread, overwritten by that thread's next construction.
    static thread_local ConstructionStatistics statistics{};
    return statistics;
}

} // namespace detail::dense_direct

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
    detail::dense_direct::ConstructionStatistics& statistics =
        detail::dense_direct::construction_statistics();
    statistics = {};
    const auto total_start = Clock::now();
    const auto validation_start = total_start;

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
    statistics.validation.add(elapsed_seconds(validation_start));

    const auto allocation_start = Clock::now();
    std::visit([&](auto& matrices) {
        for (auto& matrix : matrices) {
            matrix.resize(ns_ * nt_);
        }
    }, matrices_);
    statistics.allocation.add(elapsed_seconds(allocation_start));
    statistics.pair_count = ns_ * nt_;
    statistics.matrix_bytes = tensor_memory_bytes();

    // Prepared finite geometry.
    //
    // A prism's or tetrahedron's derived surface, volume and outward normals
    // are a pure function of its record, but the record-level entry points
    // re-derive them on every call, so an Ns x Nt plan paid for them Ns * Nt
    // times instead of once per distinct record.  Preparing them here changes
    // no value: `tetrahedron_tetrahedron_tensor` and the mixed polyhedron
    // entry points are exactly "prepare both, then call the prepared form",
    // and preparation is deterministic.
    const auto preparation_start = Clock::now();
    const bool mixed_polyhedron_pair =
        (source_is_prism && target_is_tetrahedron) ||
        (source_is_tetrahedron && target_is_prism);
    std::vector<detail::PreparedTetrahedron> prepared_source_tetrahedra;
    std::vector<detail::PreparedTetrahedron> prepared_target_tetrahedra;
    std::vector<detail::PolyhedronBody> prepared_source_bodies;
    std::vector<detail::PolyhedronBody> prepared_target_bodies;
    std::vector<detail::PreparedTetrahedronPointField> prepared_point_fields;
    if (source_is_tetrahedron && !target_is_tetrahedron && !target_is_prism) {
        // A tetrahedron source against point targets: only the point field.
        prepared_point_fields.reserve(source_tetrahedra.size());
        for (const Tetrahedron& tetrahedron : source_tetrahedra) {
            prepared_point_fields.push_back(
                detail::prepare_tetrahedron_point_field(tetrahedron));
        }
    } else if (target_is_tetrahedron && !source_is_tetrahedron &&
               !source_is_prism) {
        // A point source against tetrahedron targets is the same tensor with
        // the displacement reversed, so the target carries the point field.
        prepared_point_fields.reserve(target_tetrahedra.size());
        for (const Tetrahedron& tetrahedron : target_tetrahedra) {
            prepared_point_fields.push_back(
                detail::prepare_tetrahedron_point_field(tetrahedron));
        }
    } else if (source_is_tetrahedron && target_is_tetrahedron) {
        prepared_source_tetrahedra.reserve(source_tetrahedra.size());
        for (const Tetrahedron& tetrahedron : source_tetrahedra) {
            prepared_source_tetrahedra.push_back(
                detail::prepare_tetrahedron(tetrahedron));
        }
        prepared_target_tetrahedra.reserve(target_tetrahedra.size());
        for (const Tetrahedron& tetrahedron : target_tetrahedra) {
            prepared_target_tetrahedra.push_back(
                detail::prepare_tetrahedron(tetrahedron));
        }
    } else if (mixed_polyhedron_pair) {
        if (source_is_prism) {
            prepared_source_bodies.reserve(source_sizes.size());
            for (const CuboidSize& size : source_sizes) {
                prepared_source_bodies.push_back(
                    detail::prepare_polyhedron_body(size));
            }
        } else {
            prepared_source_bodies.reserve(source_tetrahedra.size());
            for (const Tetrahedron& tetrahedron : source_tetrahedra) {
                prepared_source_bodies.push_back(
                    detail::prepare_polyhedron_body(tetrahedron));
            }
        }
        if (target_is_prism) {
            prepared_target_bodies.reserve(target_sizes.size());
            for (const CuboidSize& size : target_sizes) {
                prepared_target_bodies.push_back(
                    detail::prepare_polyhedron_body(size));
            }
        } else {
            prepared_target_bodies.reserve(target_tetrahedra.size());
            for (const Tetrahedron& tetrahedron : target_tetrahedra) {
                prepared_target_bodies.push_back(
                    detail::prepare_polyhedron_body(tetrahedron));
            }
        }
    }
    statistics.geometry_preparation.add(elapsed_seconds(preparation_start));
    statistics.prepared_body_count =
        prepared_source_tetrahedra.size() + prepared_target_tetrahedra.size() +
        prepared_source_bodies.size() + prepared_target_bodies.size();
    statistics.prepared_body_count += prepared_point_fields.size();
    statistics.prepared_geometry_bytes =
        prepared_bytes(prepared_source_tetrahedra) +
        prepared_bytes(prepared_target_tetrahedra) +
        prepared_bytes(prepared_source_bodies) +
        prepared_bytes(prepared_target_bodies) +
        prepared_point_fields.size() *
            sizeof(detail::PreparedTetrahedronPointField);

    const std::size_t pair_count = ns_ * nt_;

    // Per-pair operator inputs.  A record index is the common record when the
    // caller supplied one and the body's own otherwise, exactly as the
    // per-pair lookups did before.
    const auto source_record = [&](const std::size_t source) {
        return source_sizes.size() == 1 ? std::size_t{0} : source;
    };
    const auto target_record = [&](const std::size_t target) {
        return target_sizes.size() == 1 ? std::size_t{0} : target;
    };
    const auto source_tetrahedron_record = [&](const std::size_t source) {
        return source_tetrahedra.size() == 1 ? std::size_t{0} : source;
    };
    const auto target_tetrahedron_record = [&](const std::size_t target) {
        return target_tetrahedra.size() == 1 ? std::size_t{0} : target;
    };
    const auto omits_identity = [&](const std::size_t target,
                                    const std::size_t source) {
        // Physical identity comes from the explicit map, never from equal
        // coordinates, and only a point source has no self field.
        const bool identity = !target_source_indices.empty() &&
            target_source_indices[target] == static_cast<int>(source);
        return identity &&
            effective_source_geometry == SourceGeometry::PointDipole;
    };

    // Builds the exact tensor of one pair; the sole tensor-valued path.
    const auto build_tensor = [&](const std::size_t index) -> PairTensor {
        const std::size_t target = index / ns_;
        const std::size_t source = index % ns_;
        const bool omit_identity = omits_identity(target, source);
        if (omit_identity) {
            return {};
        }
        const Vec3 displacement =
            target_positions[target] - source_positions[source];
        if (source_is_tetrahedron && target_is_tetrahedron) {
            const detail::PreparedTetrahedron& prepared_source =
                prepared_source_tetrahedra[
                    source_tetrahedron_record(source)];
            const detail::PreparedTetrahedron& prepared_target =
                prepared_target_tetrahedra[
                    target_tetrahedron_record(target)];
            // The coincident selector reproduces the record-level entry
            // point: a zero displacement between bitwise-identical records.
            const bool zero_displacement = displacement.x == 0.0 &&
                displacement.y == 0.0 && displacement.z == 0.0;
            bool same_geometry = true;
            for (std::size_t vertex = 0; vertex < 4; ++vertex) {
                const Vec3& first = prepared_source.vertices[vertex];
                const Vec3& second = prepared_target.vertices[vertex];
                same_geometry = same_geometry && first.x == second.x &&
                    first.y == second.y && first.z == second.z;
            }
            return detail::tetrahedron_tetrahedron_tensor_prepared(
                displacement, prepared_source, prepared_target,
                zero_displacement && same_geometry);
        }
        if (mixed_polyhedron_pair) {
            return detail::polyhedron_pair_tensor(
                displacement,
                prepared_source_bodies[source_is_prism
                    ? source_record(source)
                    : source_tetrahedron_record(source)],
                prepared_target_bodies[target_is_prism
                    ? target_record(target)
                    : target_tetrahedron_record(target)]);
        }
        if (source_is_tetrahedron) {
            return detail::tetrahedron_point_tensor_prepared(
                displacement,
                prepared_point_fields[source_tetrahedron_record(source)]);
        }
        if (target_is_tetrahedron) {
            // `point_tetrahedron_tensor` is the tetrahedron point field
            // evaluated at the reversed displacement.
            return detail::tetrahedron_point_tensor_prepared(
                Vec3{-displacement.x, -displacement.y, -displacement.z},
                prepared_point_fields[target_tetrahedron_record(target)]);
        }
        return operators::p2p::build_pair(
            target_positions[target], source_positions[source],
            effective_source_geometry, effective_target_geometry,
            source_sizes.empty() ? CuboidSize{}
                                 : source_sizes[source_record(source)],
            target_sizes.empty() ? CuboidSize{}
                                 : target_sizes[target_record(target)],
            omit_identity);
    };

    // Exact operator classification.
    //
    // A pair tensor is a pure function of the displacement, the participating
    // body records and whether the pair is an omitted point self interaction,
    // so pairs whose inputs agree bit for bit are one operator and need be
    // evaluated once.  Regular geometry sharing one body record makes this
    // decisive: an Ns x Nt lattice reaches only as many distinct
    // displacements as the lattice has, however many bodies it holds.
    //
    // It is only worth paying for where the tensor is dear.  Measured on this
    // machine, a point-to-point pair costs about 4 ns while the cheapest
    // finite pair costs about 260 ns, and a key lookup costs tens of
    // nanoseconds: classification would be most of a point plan's build and
    // is never the larger cost of a finite one.  The predicate is therefore
    // "some side is finite", which the supported geometries separate by two
    // orders of magnitude with nothing in between, rather than an arbitrary
    // rule about geometry.
    const bool finite_pair_tensor = source_is_prism || source_is_tetrahedron ||
        target_is_prism || target_is_tetrahedron;

    detail::exact_reuse::ExactOperatorClasses classes;
    if (finite_pair_tensor && pair_count != 0) {
        const auto classification_start = Clock::now();
        // The distinct tensors are held in full until they are scattered, so
        // what has to be bounded is their storage, not the reuse ratio: cap
        // it at half the matrix bytes this plan retains anyway.  Expressing
        // the cap in bytes rather than as a fixed ratio is what admits a
        // locally refined grid, whose reuse is neither the thirtyfold of a
        // uniform block nor the none of independently shaped bodies but
        // around fivefold -- enough to be worth four seconds of an exact
        // prism-tetrahedron build, and thrown away by a ratio chosen from
        // the two extremes alone.
        const std::size_t scalar_bytes =
            static_precision_ == StaticPrecision::Float32
                ? sizeof(float) : sizeof(double);
        detail::exact_reuse::ExactReuseGate gate;
        gate.sample_reuse_factor = 2;
        gate.max_classes =
            3 * pair_count * scalar_bytes / sizeof(PairTensor);
        classes = detail::exact_reuse::classify_exact_operators(
            pair_count,
            [&](const std::size_t index) {
                const std::size_t target = index / ns_;
                const std::size_t source = index % ns_;
                detail::exact_reuse::ExactOperatorKey key;
                key.push(target_positions[target] - source_positions[source]);
                if (source_is_prism) {
                    key.push(source_sizes[source_record(source)]);
                } else if (source_is_tetrahedron) {
                    key.push(source_tetrahedra[
                        source_tetrahedron_record(source)]);
                }
                if (target_is_prism) {
                    key.push(target_sizes[target_record(target)]);
                } else if (target_is_tetrahedron) {
                    key.push(target_tetrahedra[
                        target_tetrahedron_record(target)]);
                }
                key.push(omits_identity(target, source) ? 1.0 : 0.0);
                return key;
            },
            gate);
        statistics.classification.add(elapsed_seconds(classification_start));
    }
    statistics.classified = classes.classified;
    statistics.class_map_bytes = classes.transient_bytes();

    detail::exact_reuse::FirstFailure failure;
    if (classes.classified) {
        // Build each distinct operator once, then scatter.  The schedule is
        // dynamic because an exact finite tensor's cost varies by an order of
        // magnitude between a coincident, an adjacent and a far-separated
        // pair, and a classified build has few, very unequal items.
        const auto classified_build_start = Clock::now();
        const std::ptrdiff_t class_count =
            static_cast<std::ptrdiff_t>(classes.representative.size());
        std::vector<PairTensor> tensors(classes.representative.size());
#pragma omp parallel for schedule(dynamic, 1) if (class_count >= 8)
        for (std::ptrdiff_t entry = 0; entry < class_count; ++entry) {
            const std::size_t representative =
                classes.representative[static_cast<std::size_t>(entry)];
            if (failure.superseded(representative)) {
                continue;
            }
            try {
                tensors[static_cast<std::size_t>(entry)] =
                    build_tensor(representative);
            } catch (...) {
                failure.record(representative);
            }
        }
        failure.rethrow_any();
        statistics.tensor_build.add(elapsed_seconds(classified_build_start));
        statistics.built_tensor_count = classes.representative.size();
        statistics.unique_tensor_bytes = tensors.size() * sizeof(PairTensor);

        // Each pair owns one fixed entry of each matrix, so the scatter needs
        // no atomics and no reduction; a static schedule gives every thread a
        // contiguous run of all six matrices.
        const auto materialisation_start = Clock::now();
        std::visit([&](auto& matrices) {
            using Scalar = typename std::decay_t<
                decltype(matrices)>::value_type::value_type;
            const std::ptrdiff_t scatter_count =
                static_cast<std::ptrdiff_t>(pair_count);
#pragma omp parallel for schedule(static) if (scatter_count >= 256)
            for (std::ptrdiff_t pair_index = 0; pair_index < scatter_count;
                 ++pair_index) {
                const std::size_t index =
                    static_cast<std::size_t>(pair_index);
                const PairTensor& tensor =
                    tensors[classes.class_of_item[index]];
                matrices[0][index] = static_cast<Scalar>(tensor.xx);
                matrices[1][index] = static_cast<Scalar>(tensor.xy);
                matrices[2][index] = static_cast<Scalar>(tensor.xz);
                matrices[3][index] = static_cast<Scalar>(tensor.yy);
                matrices[4][index] = static_cast<Scalar>(tensor.yz);
                matrices[5][index] = static_cast<Scalar>(tensor.zz);
            }
        }, matrices_);
        statistics.materialisation.add(
            elapsed_seconds(materialisation_start));
        statistics.total.add(elapsed_seconds(total_start));
        return;
    }

    // Without classification the build and the scatter stay one fused loop.
    // Separating them would need a complete `PairTensor` array, which at
    // forty-eight bytes a pair is larger than the matrices themselves.
    //
    // Every source-target tensor owns one fixed matrix entry, so building
    // them in parallel preserves the target-major storage order and
    // introduces no floating-point reductions.
    const auto fused_build_start = Clock::now();
    std::visit([&](auto& matrices) {
        using Scalar = typename std::decay_t<
            decltype(matrices)>::value_type::value_type;
        const std::ptrdiff_t fused_count =
            static_cast<std::ptrdiff_t>(pair_count);
        // A dynamic schedule carries the finite geometries, whose per-pair
        // cost varies with separation; the larger chunk keeps the point
        // plans, which are uniform and very cheap, on long contiguous runs.
        const int chunk = finite_pair_tensor ? 256 : 4096;
#pragma omp parallel for schedule(dynamic, chunk) if (fused_count >= 256)
        for (std::ptrdiff_t pair_index = 0; pair_index < fused_count;
             ++pair_index) {
            const std::size_t index = static_cast<std::size_t>(pair_index);
            if (failure.superseded(index)) {
                continue;
            }
            try {
                const PairTensor tensor = build_tensor(index);
                matrices[0][index] = static_cast<Scalar>(tensor.xx);
                matrices[1][index] = static_cast<Scalar>(tensor.xy);
                matrices[2][index] = static_cast<Scalar>(tensor.xz);
                matrices[3][index] = static_cast<Scalar>(tensor.yy);
                matrices[4][index] = static_cast<Scalar>(tensor.yz);
                matrices[5][index] = static_cast<Scalar>(tensor.zz);
            } catch (...) {
                failure.record(index);
            }
        }
    }, matrices_);
    failure.rethrow_any();
    statistics.tensor_build.add(elapsed_seconds(fused_build_start));
    statistics.built_tensor_count = pair_count;
    statistics.total.add(elapsed_seconds(total_start));
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
