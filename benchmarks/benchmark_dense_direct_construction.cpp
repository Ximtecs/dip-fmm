// SPDX-License-Identifier: Apache-2.0
//
// benchmark_dense_direct_construction: cold construction and repeated
// evaluation of the exact dense all-to-all plans (Phase 3C.5).
//
// `DenseDirectPlan` is the exact dense baseline: six immutable Nt x Ns
// matrices built once from the analytical pair tensors and applied by nine
// GEMVs per evaluation.  `CudaDenseDirectPlan` builds the same host plan and
// retains only its device copy.  This driver measures the two halves of that
// contract separately, because they answer different questions:
//
//   * setup      -- what it costs to make a plan, broken down by phase, and
//                   what it holds while doing so; and
//   * evaluation -- what one moment state costs once the plan exists, which
//                   no setup optimisation is allowed to regress.
//
// Setup is decomposed with the internal construction records, so the CSV
// separates input validation, matrix allocation, prepared finite geometry,
// exact classification, tensor build and materialisation, and on CUDA the
// host build, context creation, device allocation and upload.
//
// Every source-target geometry combination the dense plan supports is
// reachable, over three workloads chosen to bracket exact redundancy:
// `lattice` (regular positions, one shared body record) has a great deal,
// `lattice-irregular` (regular positions, per-body records) isolates what
// displacement repetition alone is worth, and `random` is the negative
// control where there is essentially none.  Sources and targets are generated
// independently, so `--targets` may differ from `--sources`.
//
// `--probe-redundancy` counts the true number of distinct exact operator
// inputs with the production key and no sampling gate.  That is a measurement
// of the geometry, not a production path.
//
// The benchmark exercises production plans only.  It defines no operator
// mathematics of its own.

#include "cdfmm/backend/cuda/dense_direct.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/plan/direct/dense.hpp"

#include "backend/cuda/direct/dense_construction_statistics.hpp"
#include "operators/exact_operator_reuse.hpp"
#include "plan/direct/construction_statistics.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

using cdfmm::CuboidSize;
using cdfmm::DenseDirectBackend;
using cdfmm::DenseDirectPlan;
using cdfmm::SourceGeometry;
using cdfmm::SourceModel;
using cdfmm::StaticPrecision;
using cdfmm::TargetGeometry;
using cdfmm::TargetModel;
using cdfmm::Tetrahedron;
using cdfmm::Vec3;

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

enum class Geometry { Point, Prism, Tetrahedron };

enum class Workload {
    /// Regular positions and one shared body record: maximal exact redundancy.
    Lattice,
    /// Regular positions and per-body records: displacement repetition only.
    LatticeIrregular,
    /// Random positions and per-body records: the negative control.
    Random
};

enum class Backend { CpuPortable, OneMkl, Cuda };

struct Options {
    Geometry source{Geometry::Point};
    Geometry target{Geometry::Point};
    Workload workload{Workload::Lattice};
    std::size_t sources{1024};
    std::size_t targets{0};  // zero means "same as sources"
    std::vector<StaticPrecision> precisions{StaticPrecision::Float32};
    std::vector<Backend> backends{Backend::CpuPortable};
    std::size_t evaluations{20};
    std::size_t warmups{3};
    std::size_t construction_repeats{1};
    bool identity_map{true};
    bool probe_redundancy{false};
    bool checksum{false};
    int threads{0};  // zero means "leave the runtime default"
    std::uint32_t seed{20260918U};
    std::string output{};
    std::string label{};
    std::string commit{};
};

[[nodiscard]] Geometry parse_geometry(const std::string& value)
{
    if (value == "point") {
        return Geometry::Point;
    }
    if (value == "prism") {
        return Geometry::Prism;
    }
    if (value == "tetrahedron" || value == "tetra") {
        return Geometry::Tetrahedron;
    }
    throw std::invalid_argument(
        "geometry must be point, prism or tetrahedron");
}

[[nodiscard]] const char* geometry_name(const Geometry geometry)
{
    switch (geometry) {
    case Geometry::Point:
        return "point";
    case Geometry::Prism:
        return "prism";
    case Geometry::Tetrahedron:
        return "tetrahedron";
    }
    return "unknown";
}

[[nodiscard]] Workload parse_workload(const std::string& value)
{
    if (value == "lattice") {
        return Workload::Lattice;
    }
    if (value == "lattice-irregular") {
        return Workload::LatticeIrregular;
    }
    if (value == "random") {
        return Workload::Random;
    }
    throw std::invalid_argument(
        "--workload must be lattice, lattice-irregular or random");
}

[[nodiscard]] const char* workload_name(const Workload workload)
{
    switch (workload) {
    case Workload::Lattice:
        return "lattice";
    case Workload::LatticeIrregular:
        return "lattice-irregular";
    case Workload::Random:
        return "random";
    }
    return "unknown";
}

[[nodiscard]] const char* backend_name(const Backend backend)
{
    switch (backend) {
    case Backend::CpuPortable:
        return "cpu-portable";
    case Backend::OneMkl:
        return "one-mkl";
    case Backend::Cuda:
        return "cuda";
    }
    return "unknown";
}

[[nodiscard]] const char* precision_name(const StaticPrecision precision)
{
    return precision == StaticPrecision::Float32 ? "fp32" : "fp64";
}

[[nodiscard]] std::vector<std::string> split_list(const std::string& value)
{
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            parts.push_back(item);
        }
    }
    return parts;
}

void print_usage()
{
    std::cout <<
        "benchmark_dense_direct_construction options:\n"
        "  --source point|prism|tetrahedron    source geometry (default point)\n"
        "  --target point|prism|tetrahedron    target geometry (default point)\n"
        "  --workload lattice|lattice-irregular|random\n"
        "                                      geometry regime (default lattice)\n"
        "  --sources N                         source count (default 1024)\n"
        "  --targets N                         target count (default = sources)\n"
        "  --precision fp32|fp64|both          (default fp32)\n"
        "  --backends cpu,mkl,cuda             comma list (default cpu)\n"
        "  --evaluations K --warmups W         repeated evaluation sampling\n"
        "  --construction-repeats R            report the fastest of R builds\n"
        "  --no-identity                       omit the target-source identity map\n"
        "  --probe-redundancy                  count distinct exact operator inputs\n"
        "  --checksum                          hash the six matrices bit for bit\n"
        "  --threads T                         OpenMP threads for construction\n"
        "  --seed S                            random workload seed\n"
        "  --output FILE                       append CSV (header if new)\n"
        "  --label TEXT --commit SHA           metadata columns\n";
}

[[nodiscard]] Options parse_options(const int argc, char** argv)
{
    Options options;
    const auto next = [&](int& index, const char* flag) -> std::string {
        if (index + 1 >= argc) {
            throw std::invalid_argument(std::string(flag) +
                                        " requires a value");
        }
        return argv[++index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_usage();
            std::exit(0);
        } else if (argument == "--source") {
            options.source = parse_geometry(next(index, "--source"));
        } else if (argument == "--target") {
            options.target = parse_geometry(next(index, "--target"));
        } else if (argument == "--workload") {
            options.workload = parse_workload(next(index, "--workload"));
        } else if (argument == "--sources") {
            options.sources = std::stoull(next(index, "--sources"));
        } else if (argument == "--targets") {
            options.targets = std::stoull(next(index, "--targets"));
        } else if (argument == "--precision") {
            const std::string value = next(index, "--precision");
            if (value == "fp32") {
                options.precisions = {StaticPrecision::Float32};
            } else if (value == "fp64") {
                options.precisions = {StaticPrecision::Float64};
            } else if (value == "both") {
                options.precisions = {StaticPrecision::Float32,
                                      StaticPrecision::Float64};
            } else {
                throw std::invalid_argument(
                    "--precision must be fp32, fp64 or both");
            }
        } else if (argument == "--backends") {
            options.backends.clear();
            for (const std::string& item :
                 split_list(next(index, "--backends"))) {
                if (item == "cpu" || item == "cpu-portable") {
                    options.backends.push_back(Backend::CpuPortable);
                } else if (item == "mkl" || item == "one-mkl") {
                    options.backends.push_back(Backend::OneMkl);
                } else if (item == "cuda") {
                    options.backends.push_back(Backend::Cuda);
                } else {
                    throw std::invalid_argument(
                        "--backends items must be cpu, mkl or cuda");
                }
            }
        } else if (argument == "--evaluations") {
            options.evaluations = std::stoull(next(index, "--evaluations"));
        } else if (argument == "--warmups") {
            options.warmups = std::stoull(next(index, "--warmups"));
        } else if (argument == "--construction-repeats") {
            options.construction_repeats =
                std::stoull(next(index, "--construction-repeats"));
        } else if (argument == "--no-identity") {
            options.identity_map = false;
        } else if (argument == "--probe-redundancy") {
            options.probe_redundancy = true;
        } else if (argument == "--checksum") {
            options.checksum = true;
        } else if (argument == "--threads") {
            options.threads = std::stoi(next(index, "--threads"));
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(
                std::stoul(next(index, "--seed")));
        } else if (argument == "--output") {
            options.output = next(index, "--output");
        } else if (argument == "--label") {
            options.label = next(index, "--label");
        } else if (argument == "--commit") {
            options.commit = next(index, "--commit");
        } else {
            throw std::invalid_argument("unknown option " +
                                        std::string(argument));
        }
    }
    if (options.targets == 0) {
        options.targets = options.sources;
    }
    if (options.construction_repeats == 0) {
        options.construction_repeats = 1;
    }
    return options;
}

//------------------------------------------------------------------------------
// Workload generation
//------------------------------------------------------------------------------

/// @brief One independently generated set of bodies.
struct BodySet {
    std::vector<Vec3> positions;
    std::vector<CuboidSize> sizes;
    std::vector<Tetrahedron> tetrahedra;
};

/// @brief Lattice spacing, fixed so that neighbouring bodies never overlap.
constexpr double lattice_spacing = 1.0;

/// @brief A body's half-extent as a fraction of the lattice spacing.
constexpr double body_fill = 0.35;

/// @brief Returns `count` positions on a cube lattice of unit spacing.
///
/// A lattice is the regular case the exact reuse mechanism exists for: the
/// displacement between two bodies takes only a few hundred distinct values
/// however many bodies there are, and every one of them is an exact multiple
/// of the spacing, so equal displacements agree bit for bit rather than
/// approximately.
[[nodiscard]] std::vector<Vec3> lattice_positions(
    const std::size_t count, const Vec3 origin)
{
    std::size_t side = 1;
    while (side * side * side < count) {
        ++side;
    }
    std::vector<Vec3> positions;
    positions.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t x = index % side;
        const std::size_t y = (index / side) % side;
        const std::size_t z = index / (side * side);
        positions.push_back(
            {origin.x + lattice_spacing * static_cast<double>(x),
             origin.y + lattice_spacing * static_cast<double>(y),
             origin.z + lattice_spacing * static_cast<double>(z)});
    }
    return positions;
}

[[nodiscard]] std::vector<Vec3> random_positions(
    const std::size_t count, std::mt19937& engine)
{
    // The box is sized from the count so that the mean density matches the
    // lattice, which keeps the two workloads comparable per pair.
    const double extent = lattice_spacing *
        std::cbrt(static_cast<double>(std::max<std::size_t>(count, 1)));
    std::uniform_real_distribution<double> coordinate(0.0, extent);
    std::vector<Vec3> positions;
    positions.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        positions.push_back({coordinate(engine), coordinate(engine),
                             coordinate(engine)});
    }
    return positions;
}

/// @brief Returns a tetrahedron centred on its representative point.
///
/// The dense plan takes a displacement between representative points plus a
/// body record, so a record's vertices are relative to that point.  Identical
/// records at lattice positions therefore describe genuinely identical
/// operators up to the displacement.
[[nodiscard]] Tetrahedron centred_tetrahedron(const double scale)
{
    Tetrahedron tetrahedron{};
    tetrahedron.vertices[0] = {-0.5 * scale, -0.5 * scale, -0.35 * scale};
    tetrahedron.vertices[1] = {0.6 * scale, -0.4 * scale, -0.3 * scale};
    tetrahedron.vertices[2] = {-0.1 * scale, 0.7 * scale, -0.25 * scale};
    tetrahedron.vertices[3] = {0.0, -0.05 * scale, 0.6 * scale};
    return tetrahedron;
}

[[nodiscard]] BodySet make_body_set(
    const Geometry geometry, const std::size_t count, const Workload workload,
    const Vec3 origin, std::mt19937& engine)
{
    BodySet set;
    set.positions = workload == Workload::Random
        ? random_positions(count, engine)
        : lattice_positions(count, origin);

    const bool shared_record = workload == Workload::Lattice;
    // A varying record must stay well inside the lattice cell so that the
    // irregular workload differs from the regular one only in the records.
    std::uniform_real_distribution<double> jitter(0.72, 1.0);

    if (geometry == Geometry::Prism) {
        const double half = body_fill * lattice_spacing;
        if (shared_record) {
            set.sizes.push_back({half, half, half});
        } else {
            set.sizes.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                set.sizes.push_back({half * jitter(engine),
                                     half * jitter(engine),
                                     half * jitter(engine)});
            }
        }
    } else if (geometry == Geometry::Tetrahedron) {
        const double scale = body_fill * lattice_spacing;
        if (shared_record) {
            set.tetrahedra.push_back(centred_tetrahedron(scale));
        } else {
            set.tetrahedra.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                set.tetrahedra.push_back(
                    centred_tetrahedron(scale * jitter(engine)));
            }
        }
    }
    return set;
}

[[nodiscard]] SourceGeometry to_source_geometry(const Geometry geometry)
{
    switch (geometry) {
    case Geometry::Point:
        return SourceGeometry::PointDipole;
    case Geometry::Prism:
        return SourceGeometry::RectangularPrism;
    case Geometry::Tetrahedron:
        return SourceGeometry::Tetrahedron;
    }
    return SourceGeometry::PointDipole;
}

[[nodiscard]] TargetGeometry to_target_geometry(const Geometry geometry)
{
    switch (geometry) {
    case Geometry::Point:
        return TargetGeometry::Point;
    case Geometry::Prism:
        return TargetGeometry::RectangularPrism;
    case Geometry::Tetrahedron:
        return TargetGeometry::Tetrahedron;
    }
    return TargetGeometry::Point;
}

//------------------------------------------------------------------------------
// Measurement helpers
//------------------------------------------------------------------------------

/// @brief Returns the process high-water resident set in bytes, or zero.
[[nodiscard]] std::size_t peak_resident_bytes()
{
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmHWM:") {
            std::size_t kilobytes = 0;
            status >> kilobytes;
            return kilobytes * 1024;
        }
        status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}

/// @brief FNV-1a over the raw bytes of the six matrices.
///
/// The comparison this supports is bit for bit: a construction change that
/// alters the order of the work but not the mathematics must leave this
/// value unchanged.
template <typename Scalar>
[[nodiscard]] std::uint64_t hash_matrices(
    const std::array<std::vector<Scalar>, 6>& matrices)
{
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const std::vector<Scalar>& matrix : matrices) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(
            matrix.data());
        const std::size_t byte_count = matrix.size() * sizeof(Scalar);
        for (std::size_t index = 0; index < byte_count; ++index) {
            hash ^= bytes[index];
            hash *= 0x00000100000001b3ULL;
        }
    }
    return hash;
}

/// @brief Counts distinct exact operator inputs with no sampling gate.
///
/// This uses the production key, so the count is exactly the number of
/// tensors a complete exact classification would build.  It is a property of
/// the geometry, measured here rather than assumed.
[[nodiscard]] std::size_t count_distinct_exact_inputs(
    const BodySet& sources, const BodySet& targets, const Geometry source,
    const Geometry target, const std::vector<int>& identities)
{
    using cdfmm::detail::exact_reuse::ExactOperatorKey;
    using cdfmm::detail::exact_reuse::ExactOperatorKeyHash;

    std::unordered_set<ExactOperatorKey, ExactOperatorKeyHash> distinct;
    const std::size_t source_count = sources.positions.size();
    const std::size_t target_count = targets.positions.size();
    for (std::size_t target_index = 0; target_index < target_count;
         ++target_index) {
        for (std::size_t source_index = 0; source_index < source_count;
             ++source_index) {
            const bool identity = !identities.empty() &&
                identities[target_index] == static_cast<int>(source_index);
            ExactOperatorKey key;
            key.push(targets.positions[target_index] -
                     sources.positions[source_index]);
            if (source == Geometry::Prism) {
                key.push(sources.sizes[sources.sizes.size() == 1
                                           ? 0 : source_index]);
            } else if (source == Geometry::Tetrahedron) {
                key.push(sources.tetrahedra[sources.tetrahedra.size() == 1
                                                ? 0 : source_index]);
            }
            if (target == Geometry::Prism) {
                key.push(targets.sizes[targets.sizes.size() == 1
                                           ? 0 : target_index]);
            } else if (target == Geometry::Tetrahedron) {
                key.push(targets.tetrahedra[targets.tetrahedra.size() == 1
                                                ? 0 : target_index]);
            }
            key.push(identity ? 1.0 : 0.0);
            distinct.insert(key);
        }
    }
    return distinct.size();
}

/// @brief One measured configuration.
struct Row {
    std::string backend;
    std::string precision;
    double construction_seconds{0.0};
    double validation_seconds{0.0};
    double allocation_seconds{0.0};
    double geometry_preparation_seconds{0.0};
    double classification_seconds{0.0};
    double tensor_build_seconds{0.0};
    double materialisation_seconds{0.0};
    double cuda_host_construction_seconds{0.0};
    double cuda_context_seconds{0.0};
    double cuda_allocation_seconds{0.0};
    double cuda_upload_seconds{0.0};
    bool classified{false};
    std::size_t built_tensors{0};
    std::size_t prepared_bodies{0};
    std::size_t matrix_bytes{0};
    std::size_t prepared_geometry_bytes{0};
    std::size_t class_map_bytes{0};
    std::size_t unique_tensor_bytes{0};
    std::size_t persistent_device_bytes{0};
    std::size_t pinned_host_bytes{0};
    std::size_t peak_rss_bytes{0};
    double first_evaluation_seconds{0.0};
    double evaluation_seconds{0.0};
    std::uint64_t checksum{0};
};

[[nodiscard]] std::vector<Vec3> make_moments(
    const std::size_t count, std::mt19937& engine)
{
    std::uniform_real_distribution<double> component(-1.0, 1.0);
    std::vector<Vec3> moments;
    moments.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        moments.push_back(
            {component(engine), component(engine), component(engine)});
    }
    return moments;
}

/// @brief Prevents a compiler from eliding an unused evaluation result.
double consume(const std::vector<Vec3>& fields)
{
    double sum = 0.0;
    for (const Vec3& field : fields) {
        sum += field.x + field.y + field.z;
    }
    return sum;
}

Row measure_host_backend(
    const Options& options, const BodySet& sources, const BodySet& targets,
    const std::vector<int>& identities, const StaticPrecision precision,
    const Backend backend, double& checksum_sink)
{
    Row row;
    row.backend = backend_name(backend);
    row.precision = precision_name(precision);

    const SourceGeometry source_geometry = to_source_geometry(options.source);
    const TargetGeometry target_geometry = to_target_geometry(options.target);
    const SourceModel source_model = SourceModel::ExactGeometry;
    const TargetModel target_model = TargetModel::ExactGeometry;

    const auto build = [&]() {
        return DenseDirectPlan(
            sources.positions, targets.positions, source_geometry,
            target_geometry, sources.sizes, targets.sizes, identities,
            precision, sources.tetrahedra, targets.tetrahedra, source_model,
            target_model);
    };

    // The fastest of several cold builds is reported: each one constructs a
    // complete new plan, so a slow first build measures the allocator rather
    // than the construction.
    double best = std::numeric_limits<double>::infinity();
    cdfmm::detail::dense_direct::ConstructionStatistics best_statistics{};
    for (std::size_t repeat = 0; repeat < options.construction_repeats;
         ++repeat) {
        const auto start = Clock::now();
        const DenseDirectPlan plan = build();
        const double seconds = elapsed_seconds(start);
        if (seconds < best) {
            best = seconds;
            best_statistics =
                cdfmm::detail::dense_direct::construction_statistics();
        }
        if (repeat + 1 == options.construction_repeats) {
            // Measure evaluation on the last plan, which is still alive here.
            const std::vector<Vec3> moments = [&] {
                std::mt19937 engine(options.seed + 7919U);
                return make_moments(sources.positions.size(), engine);
            }();
            const DenseDirectBackend execution =
                backend == Backend::OneMkl ? DenseDirectBackend::OneMkl
                                           : DenseDirectBackend::Portable;

            const auto first_start = Clock::now();
            checksum_sink += consume(plan.evaluate(moments, execution));
            row.first_evaluation_seconds = elapsed_seconds(first_start);

            for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
                checksum_sink += consume(plan.evaluate(moments, execution));
            }
            const auto evaluation_start = Clock::now();
            for (std::size_t evaluation = 0;
                 evaluation < options.evaluations; ++evaluation) {
                checksum_sink += consume(plan.evaluate(moments, execution));
            }
            row.evaluation_seconds = options.evaluations == 0
                ? 0.0
                : elapsed_seconds(evaluation_start) /
                      static_cast<double>(options.evaluations);

            if (options.checksum) {
                row.checksum = precision == StaticPrecision::Float32
                    ? hash_matrices(plan.float_matrices())
                    : hash_matrices(plan.matrices());
            }
        }
    }

    row.construction_seconds = best;
    row.validation_seconds = best_statistics.validation.total_seconds;
    row.allocation_seconds = best_statistics.allocation.total_seconds;
    row.geometry_preparation_seconds =
        best_statistics.geometry_preparation.total_seconds;
    row.classification_seconds = best_statistics.classification.total_seconds;
    row.tensor_build_seconds = best_statistics.tensor_build.total_seconds;
    row.materialisation_seconds =
        best_statistics.materialisation.total_seconds;
    row.classified = best_statistics.classified;
    row.built_tensors = best_statistics.built_tensor_count;
    row.prepared_bodies = best_statistics.prepared_body_count;
    row.matrix_bytes = best_statistics.matrix_bytes;
    row.prepared_geometry_bytes = best_statistics.prepared_geometry_bytes;
    row.class_map_bytes = best_statistics.class_map_bytes;
    row.unique_tensor_bytes = best_statistics.unique_tensor_bytes;
    row.peak_rss_bytes = peak_resident_bytes();
    return row;
}

Row measure_cuda_backend(
    const Options& options, const BodySet& sources, const BodySet& targets,
    const std::vector<int>& identities, const StaticPrecision precision,
    double& checksum_sink)
{
    Row row;
    row.backend = backend_name(Backend::Cuda);
    row.precision = precision_name(precision);

    const SourceGeometry source_geometry = to_source_geometry(options.source);
    const TargetGeometry target_geometry = to_target_geometry(options.target);

    double best = std::numeric_limits<double>::infinity();
    cdfmm::detail::cuda_dense_direct::CudaConstructionStatistics
        best_statistics{};
    for (std::size_t repeat = 0; repeat < options.construction_repeats;
         ++repeat) {
        const auto start = Clock::now();
        cdfmm::CudaDenseDirectPlan plan(
            sources.positions, targets.positions, source_geometry,
            target_geometry, sources.sizes, targets.sizes, identities,
            precision, sources.tetrahedra, targets.tetrahedra,
            SourceModel::ExactGeometry, TargetModel::ExactGeometry);
        const double seconds = elapsed_seconds(start);
        if (seconds < best) {
            best = seconds;
            best_statistics =
                cdfmm::detail::cuda_dense_direct::cuda_construction_statistics();
        }
        if (repeat + 1 == options.construction_repeats) {
            const std::vector<Vec3> moments = [&] {
                std::mt19937 engine(options.seed + 7919U);
                return make_moments(sources.positions.size(), engine);
            }();
            const auto first_start = Clock::now();
            checksum_sink += consume(plan.evaluate(moments));
            row.first_evaluation_seconds = elapsed_seconds(first_start);

            for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
                checksum_sink += consume(plan.evaluate(moments));
            }
            const auto evaluation_start = Clock::now();
            for (std::size_t evaluation = 0;
                 evaluation < options.evaluations; ++evaluation) {
                checksum_sink += consume(plan.evaluate(moments));
            }
            row.evaluation_seconds = options.evaluations == 0
                ? 0.0
                : elapsed_seconds(evaluation_start) /
                      static_cast<double>(options.evaluations);
            row.persistent_device_bytes = plan.persistent_device_bytes();
            row.matrix_bytes = plan.tensor_memory_bytes();
        }
    }

    row.construction_seconds = best;
    row.cuda_host_construction_seconds =
        best_statistics.host_construction.total_seconds;
    row.cuda_context_seconds = best_statistics.context_creation.total_seconds;
    row.cuda_allocation_seconds = best_statistics.allocation.total_seconds;
    row.cuda_upload_seconds = best_statistics.upload.total_seconds;
    row.pinned_host_bytes = best_statistics.pinned_host_bytes;
    if (row.persistent_device_bytes == 0) {
        row.persistent_device_bytes = best_statistics.persistent_device_bytes;
    }
    // The CUDA plan delegates its exact tensors to a host plan, so that plan's
    // own record describes the construction phases underneath the upload.
    const auto& host =
        cdfmm::detail::dense_direct::construction_statistics();
    row.validation_seconds = host.validation.total_seconds;
    row.allocation_seconds = host.allocation.total_seconds;
    row.geometry_preparation_seconds =
        host.geometry_preparation.total_seconds;
    row.classification_seconds = host.classification.total_seconds;
    row.tensor_build_seconds = host.tensor_build.total_seconds;
    row.materialisation_seconds = host.materialisation.total_seconds;
    row.classified = host.classified;
    row.built_tensors = host.built_tensor_count;
    row.prepared_bodies = host.prepared_body_count;
    row.prepared_geometry_bytes = host.prepared_geometry_bytes;
    row.class_map_bytes = host.class_map_bytes;
    row.unique_tensor_bytes = host.unique_tensor_bytes;
    row.peak_rss_bytes = peak_resident_bytes();
    return row;
}

//------------------------------------------------------------------------------
// Output
//------------------------------------------------------------------------------

constexpr const char* csv_header =
    "label,commit,source,target,workload,sources,targets,pairs,identity_map,"
    "precision,backend,threads,"
    "construction_seconds,validation_seconds,allocation_seconds,"
    "geometry_preparation_seconds,classification_seconds,"
    "tensor_build_seconds,materialisation_seconds,"
    "precision_conversion_seconds,"
    "cuda_host_construction_seconds,cuda_context_seconds,"
    "cuda_allocation_seconds,cuda_upload_seconds,"
    "classified,built_tensors,distinct_exact_inputs,prepared_bodies,"
    "matrix_bytes,prepared_geometry_bytes,class_map_bytes,"
    "unique_tensor_bytes,persistent_device_bytes,pinned_host_bytes,"
    "peak_rss_bytes,"
    "first_evaluation_seconds,evaluation_seconds,evaluations,"
    "construction_ns_per_pair,evaluation_ns_per_pair,checksum";

void write_row(std::ostream& stream, const Options& options, const Row& row,
               const std::size_t pairs, const std::size_t distinct_inputs,
               const int threads)
{
    const double ns_per_pair = pairs == 0
        ? 0.0 : row.construction_seconds * 1.0e9 /
                    static_cast<double>(pairs);
    const double evaluation_ns_per_pair = pairs == 0
        ? 0.0 : row.evaluation_seconds * 1.0e9 /
                    static_cast<double>(pairs);
    stream << options.label << ',' << options.commit << ','
           << geometry_name(options.source) << ','
           << geometry_name(options.target) << ','
           << workload_name(options.workload) << ','
           << options.sources << ',' << options.targets << ',' << pairs << ','
           << (options.identity_map ? 1 : 0) << ','
           << row.precision << ',' << row.backend << ',' << threads << ','
           << std::setprecision(9) << std::scientific
           << row.construction_seconds << ',' << row.validation_seconds << ','
           << row.allocation_seconds << ','
           << row.geometry_preparation_seconds << ','
           << row.classification_seconds << ',' << row.tensor_build_seconds
           << ',' << row.materialisation_seconds << ','
           // The FP32 plan quantises each tensor at the point of store, so no
           // separate conversion pass over a complete FP64 matrix exists.
           << 0.0 << ','
           << row.cuda_host_construction_seconds << ','
           << row.cuda_context_seconds << ',' << row.cuda_allocation_seconds
           << ',' << row.cuda_upload_seconds << ','
           << std::defaultfloat << (row.classified ? 1 : 0) << ','
           << row.built_tensors << ',' << distinct_inputs << ','
           << row.prepared_bodies << ',' << row.matrix_bytes << ','
           << row.prepared_geometry_bytes << ',' << row.class_map_bytes << ','
           << row.unique_tensor_bytes << ',' << row.persistent_device_bytes
           << ',' << row.pinned_host_bytes << ',' << row.peak_rss_bytes << ','
           << std::setprecision(9) << std::scientific
           << row.first_evaluation_seconds << ',' << row.evaluation_seconds
           << ',' << std::defaultfloat << options.evaluations << ','
           << std::setprecision(6) << std::scientific << ns_per_pair << ','
           << evaluation_ns_per_pair << ',' << std::defaultfloat
           << row.checksum << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_options(argc, argv);

#if defined(_OPENMP)
        if (options.threads > 0) {
            omp_set_num_threads(options.threads);
        }
        const int threads = omp_get_max_threads();
#else
        const int threads = 1;
#endif

        std::mt19937 engine(options.seed);
        const BodySet sources = make_body_set(
            options.source, options.sources, options.workload,
            Vec3{0.0, 0.0, 0.0}, engine);
        // Targets are generated independently, so an asymmetric plan is an
        // ordinary configuration rather than a special case.
        const BodySet targets = make_body_set(
            options.target, options.targets, options.workload,
            Vec3{0.0, 0.0, 0.0}, engine);

        // The identity map marks physical self interactions.  It is never
        // inferred from coordinate equality, so it is only meaningful when
        // the two sets describe the same bodies.
        std::vector<int> identities;
        if (options.identity_map && options.sources == options.targets) {
            identities.resize(options.targets);
            for (std::size_t index = 0; index < identities.size(); ++index) {
                identities[index] = static_cast<int>(index);
            }
        }

        const std::size_t pairs = options.sources * options.targets;
        const std::size_t distinct_inputs = options.probe_redundancy
            ? count_distinct_exact_inputs(sources, targets, options.source,
                                          options.target, identities)
            : 0;

        std::ofstream file;
        std::ostream* stream = &std::cout;
        bool write_header = true;
        if (!options.output.empty()) {
            std::ifstream existing(options.output);
            write_header = !existing.good() || existing.peek() == EOF;
            existing.close();
            file.open(options.output, std::ios::app);
            if (!file) {
                throw std::runtime_error("cannot open " + options.output);
            }
            stream = &file;
        }
        if (write_header) {
            *stream << csv_header << '\n';
        }

        // A volatile sink keeps every measured evaluation observable, so the
        // optimiser cannot delete the loop whose time is being reported.
        static volatile double checksum_sink_storage = 0.0;
        double checksum_sink = 0.0;
        for (const StaticPrecision precision : options.precisions) {
            for (const Backend backend : options.backends) {
                if (backend == Backend::OneMkl &&
                    !cdfmm::dense_direct_mkl_available()) {
                    std::cerr << "skipping one-mkl: not enabled in this build\n";
                    continue;
                }
                if (backend == Backend::Cuda &&
                    !cdfmm::cuda_dense_direct_available()) {
                    std::cerr << "skipping cuda: no device available\n";
                    continue;
                }
                const Row row = backend == Backend::Cuda
                    ? measure_cuda_backend(options, sources, targets,
                                           identities, precision,
                                           checksum_sink)
                    : measure_host_backend(options, sources, targets,
                                           identities, precision, backend,
                                           checksum_sink);
                write_row(*stream, options, row, pairs, distinct_inputs,
                          threads);
                stream->flush();
            }
        }
        checksum_sink_storage = checksum_sink;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_dense_direct_construction: " << error.what()
                  << '\n';
        return 1;
    }
}
