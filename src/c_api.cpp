// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/c_api.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "cdfmm/uniform_fmm.hpp"

struct cdfmm_plan {
    std::unique_ptr<cdfmm::UniformFmm> fmm;
    std::size_t source_count{0};
    std::size_t target_count{0};
    cdfmm::StaticPrecision precision{cdfmm::StaticPrecision::Float32};
    std::vector<cdfmm::Vec3> moments;
    std::vector<cdfmm::FloatVec3> moments32;
    std::vector<cdfmm::PotentialField> results64;
    std::vector<cdfmm::FloatPotentialField> results32;
};

namespace {

thread_local std::string last_error;

int fail(const int status, std::string message) {
    last_error = std::move(message);
    return status;
}

template <typename Function> int guarded(Function &&function) noexcept {
    try {
        function();
        last_error.clear();
        return CDFMM_SUCCESS;
    } catch (const std::invalid_argument& error) {
        return fail(CDFMM_ERROR_INVALID_ARGUMENT, error.what());
    } catch (const std::logic_error& error) {
        return fail(CDFMM_ERROR_UNSUPPORTED, error.what());
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        const int status = message.find("Cuda") != std::string::npos ||
                message.find("CUDA") != std::string::npos
            ? CDFMM_ERROR_CUDA_UNAVAILABLE
            : CDFMM_ERROR_RUNTIME;
        return fail(status, message);
    } catch (const std::exception& error) {
        return fail(CDFMM_ERROR_RUNTIME, error.what());
    } catch (...) {
        return fail(CDFMM_ERROR_UNKNOWN, "unknown C++ exception");
    }
}

void require_array(const std::size_t count, const void* pointer,
                   const char* name)
{
    if (count != 0 && pointer == nullptr) {
        throw std::invalid_argument(std::string(name) + " is NULL");
    }
}

cdfmm::UniformFmmOptions translate_options(const cdfmm_options* input)
{
    cdfmm_options value{};
    cdfmm_default_options(&value);
    if (input != nullptr) {
        if (input->struct_size < sizeof(cdfmm_options)) {
            throw std::invalid_argument("cdfmm_options.struct_size is too small");
        }
        value = *input;
    }
    cdfmm::UniformFmmOptions output;
    output.expansion_order = value.expansion_order;
    output.tree.max_level = value.tree_depth;
    switch (value.precision) {
    case CDFMM_PRECISION_FLOAT32:
        output.precision = cdfmm::StaticPrecision::Float32;
        break;
    case CDFMM_PRECISION_FLOAT64:
        output.precision = cdfmm::StaticPrecision::Float64;
        break;
    default:
        throw std::invalid_argument("unknown C ABI precision");
    }
    switch (value.expansion_basis) {
    case CDFMM_BASIS_SPHERICAL:
        output.expansion_basis = cdfmm::ExpansionBasis::Spherical;
        break;
    case CDFMM_BASIS_CARTESIAN:
        output.expansion_basis = cdfmm::ExpansionBasis::Cartesian;
        break;
    default:
        throw std::invalid_argument("unknown C ABI expansion basis");
    }
    switch (value.execution_backend) {
  case CDFMM_BACKEND_AUTO:
    output.backend = cdfmm::ExecutionBackend::Auto;
    break;
  case CDFMM_BACKEND_CPU_REFERENCE:
    output.backend = cdfmm::ExecutionBackend::CpuReference;
    break;
  case CDFMM_BACKEND_CPU_STATIC:
    output.backend = cdfmm::ExecutionBackend::CpuStatic;
    break;
  case CDFMM_BACKEND_CUDA_PARTIAL:
    output.backend = cdfmm::ExecutionBackend::CudaM2LP2P;
    break;
  case CDFMM_BACKEND_CUDA_FULL:
    output.backend = cdfmm::ExecutionBackend::CudaFull;
    break;
  default:
    throw std::invalid_argument("unknown C ABI execution backend");
    }
    switch (value.static_matrix_backend) {
    case CDFMM_STATIC_MATRIX_PORTABLE:
        output.static_matrix_backend = cdfmm::StaticMatrixBackend::Portable;
        break;
    case CDFMM_STATIC_MATRIX_ONE_MKL:
        output.static_matrix_backend = cdfmm::StaticMatrixBackend::OneMkl;
        break;
    default:
        throw std::invalid_argument("unknown C ABI static matrix backend");
    }
    return output;
}

std::vector<cdfmm::Vec3> pack_positions(const std::size_t count,
                                        const double *x, const double *y,
                                        const double *z) {
    require_array(count, x, "x coordinate array");
    require_array(count, y, "y coordinate array");
    require_array(count, z, "z coordinate array");
    std::vector<cdfmm::Vec3> positions(count);
    for (std::size_t index = 0; index < count; ++index) {
        positions[index] = {x[index], y[index], z[index]};
    }
    return positions;
}

void create_plan(std::size_t source_count, const double* source_x,
                 const double* source_y, const double* source_z,
                 std::size_t target_count, const double* target_x,
                 const double* target_y, const double* target_z,
                 const int32_t* identity, cdfmm::UniformFmmOptions options,
                 cdfmm_plan** output)
{
    if (output == nullptr) {
        throw std::invalid_argument("plan output pointer is NULL");
    }
    *output = nullptr;
    if (source_count == 0 || target_count == 0) {
        throw std::invalid_argument("source and target counts must be positive");
    }
    auto sources = pack_positions(source_count, source_x, source_y, source_z);
    auto targets = pack_positions(target_count, target_x, target_y, target_z);
    if (identity != nullptr) {
        std::vector<int> map(target_count);
        for (std::size_t index = 0; index < target_count; ++index) {
            map[index] = identity[index];
        }
        options.fixed_target_source_indices = std::move(map);
    }
    auto plan = std::make_unique<cdfmm_plan>();
    plan->source_count = source_count;
    plan->target_count = target_count;
    plan->precision = options.precision;
    if (options.precision == cdfmm::StaticPrecision::Float32) {
        plan->moments32.resize(source_count);
        plan->results32.resize(target_count);
    } else {
        plan->moments.resize(source_count);
        plan->results64.resize(target_count);
    }
    plan->fmm = std::make_unique<cdfmm::UniformFmm>(sources, targets, options);
    *output = plan.release();
}

} // namespace

extern "C" {

uint32_t cdfmm_abi_version(void) { return CDFMM_ABI_VERSION; }

const char* cdfmm_get_last_error(void) { return last_error.c_str(); }

void cdfmm_default_options(cdfmm_options* options)
{
    if (options == nullptr) {
        return;
    }
    *options = {};
    options->struct_size = sizeof(cdfmm_options);
    options->expansion_order = 4;
    options->tree_depth = 1;
    options->precision = CDFMM_PRECISION_FLOAT32;
    options->expansion_basis = CDFMM_BASIS_SPHERICAL;
    options->execution_backend = CDFMM_BACKEND_AUTO;
    options->static_matrix_backend = CDFMM_STATIC_MATRIX_PORTABLE;
}

int cdfmm_plan_create_points(size_t ns, const double *sx, const double *sy,
                             const double *sz, size_t nt, const double *tx,
                             const double *ty, const double *tz,
                             const int32_t *identity,
                             const cdfmm_options *options, cdfmm_plan **plan) {
    return guarded([&] {
        create_plan(ns, sx, sy, sz, nt, tx, ty, tz, identity,
                    translate_options(options), plan);
    });
}

int cdfmm_plan_create_same_points(size_t count, const double* x,
                                  const double* y, const double* z,
                                  const cdfmm_options* options,
                                  cdfmm_plan** plan)
{
    return guarded([&] {
        std::vector<int32_t> identity(count);
        for (std::size_t index = 0; index < count; ++index) {
            identity[index] = static_cast<int32_t>(index);
        }
        create_plan(count, x, y, z, count, x, y, z, identity.data(),
                    translate_options(options), plan);
    });
}

int cdfmm_plan_create_uniform_cuboid_sources(
    size_t ns, const double *sx, const double *sy, const double *sz, size_t nt,
    const double *tx, const double *ty, const double *tz, double hx, double hy,
    double hz, const int32_t *identity, const cdfmm_options *options,
    cdfmm_plan **plan) {
    return guarded([&] {
        auto translated = translate_options(options);
        if (translated.expansion_basis != cdfmm::ExpansionBasis::Cartesian) {
            throw std::logic_error(
                "uniform cuboid sources require Cartesian expansions");
        }
        if (!(hx > 0.0 && hy > 0.0 && hz > 0.0)) {
            throw std::invalid_argument("cuboid side lengths must be positive");
        }
        require_array(ns, sx, "source_x");
        require_array(ns, sy, "source_y");
        require_array(ns, sz, "source_z");
        require_array(nt, tx, "target_x");
        require_array(nt, ty, "target_y");
        require_array(nt, tz, "target_z");
        if (ns == 0 || nt == 0) {
            throw std::invalid_argument("source and target counts must be positive");
        }
        translated.source_geometry = cdfmm::SourceGeometry::UniformCuboid;
        translated.source_sizes = {{hx, hy, hz}};
        translated.use_cuboid_p2m = true;
        // Include the finite source extent, including the one-cell case where
        // point-coordinate bounds alone have zero width.
        const auto source_x_bounds = std::minmax_element(sx, sx + ns);
        const auto source_y_bounds = std::minmax_element(sy, sy + ns);
        const auto source_z_bounds = std::minmax_element(sz, sz + ns);
        const auto target_x_bounds = std::minmax_element(tx, tx + nt);
        const auto target_y_bounds = std::minmax_element(ty, ty + nt);
        const auto target_z_bounds = std::minmax_element(tz, tz + nt);
    const double minimum_x =
        std::min(*source_x_bounds.first - 0.5 * hx, *target_x_bounds.first);
    const double maximum_x =
        std::max(*source_x_bounds.second + 0.5 * hx, *target_x_bounds.second);
    const double minimum_y =
        std::min(*source_y_bounds.first - 0.5 * hy, *target_y_bounds.first);
    const double maximum_y =
        std::max(*source_y_bounds.second + 0.5 * hy, *target_y_bounds.second);
    const double minimum_z =
        std::min(*source_z_bounds.first - 0.5 * hz, *target_z_bounds.first);
    const double maximum_z =
        std::max(*source_z_bounds.second + 0.5 * hz, *target_z_bounds.second);
    translated.tree.root_centre = {0.5 * (minimum_x + maximum_x),
            0.5 * (minimum_y + maximum_y),
            0.5 * (minimum_z + maximum_z)};
    translated.tree.root_half_width =
        1.000001 *
        std::max({0.5 * (maximum_x - minimum_x), 0.5 * (maximum_y - minimum_y),
            0.5 * (maximum_z - minimum_z)});
    create_plan(ns, sx, sy, sz, nt, tx, ty, tz, identity, std::move(translated),
                plan);
  });
}

int cdfmm_plan_create_same_uniform_cuboids(size_t count, const double *x,
                                           const double *y, const double *z,
                                           double hx, double hy, double hz,
                                           const cdfmm_options *options,
                                           cdfmm_plan **plan) {
  return guarded([&] {
    auto translated = translate_options(options);
    if (translated.expansion_basis != cdfmm::ExpansionBasis::Cartesian) {
      throw std::logic_error("uniform cuboids require Cartesian expansions");
    }
    if (!(std::isfinite(hx) && std::isfinite(hy) && std::isfinite(hz) &&
          hx > 0.0 && hy > 0.0 && hz > 0.0)) {
      throw std::invalid_argument(
          "cuboid side lengths must be finite and positive");
    }
    require_array(count, x, "x");
    require_array(count, y, "y");
    require_array(count, z, "z");
    if (count == 0) {
      throw std::invalid_argument("cuboid count must be positive");
    }
    translated.source_geometry = cdfmm::SourceGeometry::UniformCuboid;
    translated.source_sizes = {{hx, hy, hz}};
    translated.target_geometry = cdfmm::TargetGeometry::VolumeAveragedCuboid;
    translated.target_sizes = {{hx, hy, hz}};
    translated.use_cuboid_p2m = true;
    const auto xb = std::minmax_element(x, x + count);
    const auto yb = std::minmax_element(y, y + count);
    const auto zb = std::minmax_element(z, z + count);
    const cdfmm::Vec3 minimum{*xb.first - 0.5 * hx, *yb.first - 0.5 * hy,
                              *zb.first - 0.5 * hz};
    const cdfmm::Vec3 maximum{*xb.second + 0.5 * hx, *yb.second + 0.5 * hy,
                              *zb.second + 0.5 * hz};
    translated.tree.root_centre = 0.5 * (minimum + maximum);
    translated.tree.root_half_width =
        1.000001 *
        std::max({0.5 * (maximum.x - minimum.x), 0.5 * (maximum.y - minimum.y),
                  0.5 * (maximum.z - minimum.z)});
    // Finite cuboid self interactions are physical and are not excluded.
    create_plan(count, x, y, z, count, x, y, z, nullptr, std::move(translated),
                plan);
    });
}

int cdfmm_plan_evaluate_f32(cdfmm_plan *plan, const float *mx, const float *my,
                            const float *mz, float *hx, float *hy, float *hz) {
    return guarded([&] {
    if (plan == nullptr)
      throw std::invalid_argument("plan is NULL");
        if (plan->precision != cdfmm::StaticPrecision::Float32)
            throw std::invalid_argument("FP32 evaluation requires an FP32 plan");
        require_array(plan->source_count, mx, "mx");
        require_array(plan->source_count, my, "my");
        require_array(plan->source_count, mz, "mz");
        require_array(plan->target_count, hx, "hx");
        require_array(plan->target_count, hy, "hy");
        require_array(plan->target_count, hz, "hz");
        for (std::size_t index = 0; index < plan->source_count; ++index) {
            plan->moments32[index] = {mx[index], my[index], mz[index]};
        }
        plan->fmm->evaluate_into_float32(plan->moments32, plan->results32,
                                         cdfmm::OutputFlags::Field);
        for (std::size_t index = 0; index < plan->target_count; ++index) {
            hx[index] = plan->results32[index].H.x;
            hy[index] = plan->results32[index].H.y;
            hz[index] = plan->results32[index].H.z;
        }
    });
}

int cdfmm_plan_evaluate_f64(cdfmm_plan* plan, const double* mx,
                            const double* my, const double* mz, double* hx,
                            double* hy, double* hz)
{
    return guarded([&] {
        if (plan == nullptr) throw std::invalid_argument("plan is NULL");
        if (plan->precision != cdfmm::StaticPrecision::Float64)
            throw std::invalid_argument("FP64 evaluation requires an FP64 plan");
        require_array(plan->source_count, mx, "mx");
        require_array(plan->source_count, my, "my");
        require_array(plan->source_count, mz, "mz");
        require_array(plan->target_count, hx, "hx");
        require_array(plan->target_count, hy, "hy");
        require_array(plan->target_count, hz, "hz");
        for (std::size_t index = 0; index < plan->source_count; ++index) {
            plan->moments[index] = {mx[index], my[index], mz[index]};
        }
        plan->fmm->evaluate_into(plan->moments, plan->results64,
                                 cdfmm::OutputFlags::Field);
        for (std::size_t index = 0; index < plan->target_count; ++index) {
            hx[index] = plan->results64[index].H.x;
            hy[index] = plan->results64[index].H.y;
            hz[index] = plan->results64[index].H.z;
        }
    });
}

int cdfmm_plan_get_stats(const cdfmm_plan *plan, cdfmm_plan_stats *stats) {
    return guarded([&] {
        if (plan == nullptr || stats == nullptr)
            throw std::invalid_argument("plan and stats must not be NULL");
        if (stats->struct_size != 0 &&
            stats->struct_size < sizeof(cdfmm_plan_stats))
            throw std::invalid_argument("cdfmm_plan_stats.struct_size is too small");
        const auto& host = plan->fmm->static_plan_statistics();
        const auto& device = plan->fmm->cuda_plan_statistics();
        *stats = {};
        stats->struct_size = sizeof(cdfmm_plan_stats);
        stats->source_count = plan->source_count;
        stats->target_count = plan->target_count;
        stats->expansion_order = plan->fmm->expansion_order();
        stats->coefficient_count = plan->fmm->coefficient_count();
    stats->host_persistent_bytes =
        host.total_persistent_bytes() +
            plan->moments.capacity() * sizeof(cdfmm::Vec3) +
            plan->moments32.capacity() * sizeof(cdfmm::FloatVec3) +
            plan->results32.capacity() * sizeof(cdfmm::FloatPotentialField) +
            plan->results64.capacity() * sizeof(cdfmm::PotentialField);
        stats->device_persistent_bytes = device.persistent_device_bytes;
    });
}

int cdfmm_plan_get_last_evaluation_seconds(const cdfmm_plan* plan,
                                           double* seconds)
{
    return guarded([&] {
        if (plan == nullptr || seconds == nullptr)
            throw std::invalid_argument("plan and seconds must not be NULL");
        *seconds = plan->fmm->last_timings().total.total_seconds;
    });
}

void cdfmm_plan_destroy(cdfmm_plan *plan) {
    try {
        delete plan;
    } catch (...) {
        // Destruction is deliberately noexcept at the ABI boundary.
    }
}

} // extern "C"
