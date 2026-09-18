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
// reachable, over workloads chosen to bracket exact redundancy: `lattice`
// (regular positions, one shared body record) has a great deal,
// `lattice-irregular` (regular positions, per-body records) isolates what
// displacement repetition alone is worth, and `random` is the negative
// control where there is essentially none.  Sources and targets are generated
// independently, so `--targets` may differ from `--sources`.
//
// The lattices fill all three axes rather than forming a slab, and the
// `anisotropic` pair of workloads stretches both the spacing and the bodies.
// That is not cosmetic: a cube lattice of cubes is the most symmetric input
// these kernels accept, the far-separation switch keys on a body's
// circumradius, and elongating a body therefore moves the boundary between
// the analytical surface integrals and the 216-node quadrature.  What
// construction costs is measured on stretched geometry rather than inferred
// from the cubic case.
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
    Random,
    /// As `Lattice`, but with anisotropic spacing and elongated bodies.
    Anisotropic,
    /// As `LatticeIrregular`, with anisotropic spacing and elongated bodies.
    AnisotropicIrregular,
    /// A regular grid with one octant refined: two commensurate sublattices
    /// and one shared body record per refinement level.
    Refined,
    /// As `Refined`, with anisotropic spacing and elongated bodies.
    RefinedAnisotropic
};

/// @brief Whether a workload refines part of its grid.
///
/// A uniformly discretised block and a set of independently shaped bodies are
/// the two extremes of exact redundancy.  Real discretisations usually sit
/// between them: a regular grid with a region resolved more finely, which has
/// a handful of distinct body records rather than one or N.  That is the
/// regime in which the classification gate has to decide correctly, so it is
/// measured rather than interpolated between the two extremes.
[[nodiscard]] bool is_refined(const Workload workload)
{
    return workload == Workload::Refined ||
        workload == Workload::RefinedAnisotropic;
}

/// @brief Whether a workload places its bodies on a regular lattice.
[[nodiscard]] bool is_lattice(const Workload workload)
{
    return workload != Workload::Random;
}

/// @brief Whether every body carries its own independently drawn record.
[[nodiscard]] bool has_per_body_records(const Workload workload)
{
    return workload == Workload::LatticeIrregular ||
        workload == Workload::Random ||
        workload == Workload::AnisotropicIrregular;
}

/// @brief Whether a workload stretches the lattice and the bodies.
///
/// An isotropic cube lattice of cubes is the most symmetric input the exact
/// kernels accept, and symmetry is not neutral here: the far-separation
/// switch keys on a body's circumradius, so elongating a body moves the
/// boundary between the analytical surface integrals and the 216-node
/// quadrature, and anisotropic spacing changes how many distinct
/// displacements a lattice has.  Both change what construction costs, so
/// both are measured rather than assumed to behave like the cubic case.
[[nodiscard]] bool is_anisotropic(const Workload workload)
{
    return workload == Workload::Anisotropic ||
        workload == Workload::AnisotropicIrregular ||
        workload == Workload::RefinedAnisotropic;
}

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
    if (value == "anisotropic") {
        return Workload::Anisotropic;
    }
    if (value == "anisotropic-irregular") {
        return Workload::AnisotropicIrregular;
    }
    if (value == "refined") {
        return Workload::Refined;
    }
    if (value == "refined-anisotropic") {
        return Workload::RefinedAnisotropic;
    }
    throw std::invalid_argument(
        "--workload must be lattice, lattice-irregular, random, anisotropic, "
        "anisotropic-irregular, refined or refined-anisotropic");
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
    case Workload::Anisotropic:
        return "anisotropic";
    case Workload::AnisotropicIrregular:
        return "anisotropic-irregular";
    case Workload::Refined:
        return "refined";
    case Workload::RefinedAnisotropic:
        return "refined-anisotropic";
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
        "  --workload lattice|lattice-irregular|random|anisotropic|\n"
        "             anisotropic-irregular|refined|refined-anisotropic\n"
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
    /// Refinement level of each body; empty when the grid is uniform.
    std::vector<int> levels;
};

/// @brief Lattice spacing, fixed so that neighbouring bodies never overlap.
constexpr double lattice_spacing = 1.0;

/// @brief A body's half-extent as a fraction of the lattice spacing.
constexpr double body_fill = 0.35;

/// @brief Per-axis stretch of an anisotropic workload's spacing and bodies.
///
/// Chosen to be a genuinely flattened, elongated cell rather than a nearly
/// cubic one, while leaving every body strictly inside its cell.
const Vec3 anisotropy{1.0, 0.4, 2.2};

/// @brief Returns the per-axis lattice spacing of a workload.
[[nodiscard]] Vec3 spacing_of(const Workload workload)
{
    if (!is_anisotropic(workload)) {
        return {lattice_spacing, lattice_spacing, lattice_spacing};
    }
    return {lattice_spacing * anisotropy.x, lattice_spacing * anisotropy.y,
            lattice_spacing * anisotropy.z};
}

/// @brief Returns `count` positions on a lattice of the given spacing.
///
/// A lattice is the regular case the exact reuse mechanism exists for: the
/// displacement between two bodies takes only a few hundred distinct values
/// however many bodies there are, and every one of them is an exact multiple
/// of the spacing, so equal displacements agree bit for bit rather than
/// approximately.  The lattice fills all three axes; it is never a slab.
[[nodiscard]] std::vector<Vec3> lattice_positions(
    const std::size_t count, const Vec3 origin, const Vec3 spacing)
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
            {origin.x + spacing.x * static_cast<double>(x),
             origin.y + spacing.y * static_cast<double>(y),
             origin.z + spacing.z * static_cast<double>(z)});
    }
    return positions;
}

/// @brief Returns positions on a regular grid with one octant refined.
///
/// Each cell of the refined octant is replaced by its eight children, so the
/// result is the union of two commensurate sublattices.  Every offset is a
/// multiple of a quarter of the spacing and every refined extent is an exact
/// halving, so on an isotropic grid equal index differences still produce
/// bitwise-equal displacements: the workload varies the number of distinct
/// body records without also giving up exact displacement agreement.
[[nodiscard]] std::vector<Vec3> refined_positions(
    const std::size_t count, const Vec3 origin, const Vec3 spacing,
    std::vector<int>& levels)
{
    // Grow the coarse grid until its refined yield covers the request.
    std::size_t side = 1;
    const auto yield_of = [](const std::size_t s) {
        const std::size_t refined = s / 2;
        return s * s * s - refined * refined * refined +
            8 * refined * refined * refined;
    };
    while (yield_of(side) < count) {
        ++side;
    }

    std::vector<Vec3> positions;
    positions.reserve(count);
    levels.clear();
    levels.reserve(count);
    const std::size_t refined_side = side / 2;
    for (std::size_t index = 0; index < side * side * side &&
             positions.size() < count; ++index) {
        const std::size_t x = index % side;
        const std::size_t y = (index / side) % side;
        const std::size_t z = index / (side * side);
        const Vec3 centre{origin.x + spacing.x * static_cast<double>(x),
                          origin.y + spacing.y * static_cast<double>(y),
                          origin.z + spacing.z * static_cast<double>(z)};
        const bool refine =
            x < refined_side && y < refined_side && z < refined_side;
        if (!refine) {
            positions.push_back(centre);
            levels.push_back(0);
            continue;
        }
        for (int child = 0; child < 8 && positions.size() < count; ++child) {
            const double sx = (child & 1) != 0 ? 0.25 : -0.25;
            const double sy = (child & 2) != 0 ? 0.25 : -0.25;
            const double sz = (child & 4) != 0 ? 0.25 : -0.25;
            positions.push_back({centre.x + spacing.x * sx,
                                 centre.y + spacing.y * sy,
                                 centre.z + spacing.z * sz});
            levels.push_back(1);
        }
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
[[nodiscard]] Tetrahedron centred_tetrahedron(const double scale,
                                              const Vec3 stretch)
{
    Tetrahedron tetrahedron{};
    const auto vertex = [&](const double x, const double y, const double z) {
        return Vec3{x * scale * stretch.x, y * scale * stretch.y,
                    z * scale * stretch.z};
    };
    tetrahedron.vertices[0] = vertex(-0.5, -0.5, -0.35);
    tetrahedron.vertices[1] = vertex(0.6, -0.4, -0.3);
    tetrahedron.vertices[2] = vertex(-0.1, 0.7, -0.25);
    tetrahedron.vertices[3] = vertex(0.0, -0.05, 0.6);
    return tetrahedron;
}

[[nodiscard]] BodySet make_body_set(
    const Geometry geometry, const std::size_t count, const Workload workload,
    const Vec3 origin, std::mt19937& engine)
{
    BodySet set;
    const Vec3 spacing = spacing_of(workload);
    if (is_refined(workload)) {
        set.positions = refined_positions(count, origin, spacing, set.levels);
    } else if (is_lattice(workload)) {
        set.positions = lattice_positions(count, origin, spacing);
    } else {
        set.positions = random_positions(count, engine);
    }

    const bool shared_record = !has_per_body_records(workload) &&
        !is_refined(workload);
    // A body is stretched with its cell, so an anisotropic workload keeps the
    // same fill fraction and stays strictly inside its cell on every axis.
    const Vec3 stretch = is_anisotropic(workload)
        ? anisotropy : Vec3{1.0, 1.0, 1.0};
    // A varying record must stay well inside the cell so that an irregular
    // workload differs from its regular counterpart only in the records.
    std::uniform_real_distribution<double> jitter(0.72, 1.0);

    // A refined body is its parent halved, which is exact in binary, so a
    // refined grid has exactly one distinct record per refinement level.
    const auto level_scale = [&](const std::size_t index) {
        return set.levels.empty() || set.levels[index] == 0 ? 1.0 : 0.5;
    };

    if (geometry == Geometry::Prism) {
        const double half = body_fill * lattice_spacing;
        if (shared_record) {
            set.sizes.push_back({half * stretch.x, half * stretch.y,
                                 half * stretch.z});
        } else if (is_refined(workload)) {
            set.sizes.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                const double s = level_scale(index);
                set.sizes.push_back({half * stretch.x * s,
                                     half * stretch.y * s,
                                     half * stretch.z * s});
            }
        } else {
            set.sizes.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                set.sizes.push_back({half * stretch.x * jitter(engine),
                                     half * stretch.y * jitter(engine),
                                     half * stretch.z * jitter(engine)});
            }
        }
    } else if (geometry == Geometry::Tetrahedron) {
        const double scale = body_fill * lattice_spacing;
        if (shared_record) {
            set.tetrahedra.push_back(centred_tetrahedron(scale, stretch));
        } else if (is_refined(workload)) {
            set.tetrahedra.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                set.tetrahedra.push_back(centred_tetrahedron(
                    scale * level_scale(index), stretch));
            }
        } else {
            set.tetrahedra.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                set.tetrahedra.push_back(
                    centred_tetrahedron(scale * jitter(engine), stretch));
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
            // Only a point source has its self interaction omitted, so that,
            // and not the raw identity, is what the tensor depends on.
            const bool omits_identity = !identities.empty() &&
                identities[target_index] == static_cast<int>(source_index) &&
                source == Geometry::Point;
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
            key.push(omits_identity ? 1.0 : 0.0);
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

        // The identity map marks physical self interactions.  It is never
        // inferred from coordinate equality, so it is only meaningful when
        // the two sets describe the same bodies, which is also the only case
        // in which they may share positions: a point source has no self
        // field, and without the map a coincident point pair is an error
        // rather than a measurement.  Independently generated sets therefore
        // sit half a cell apart, which is what distinct sources and distinct
        // field-evaluation targets look like in practice.
        const bool same_bodies =
            options.identity_map && options.sources == options.targets;
        const Vec3 target_origin = same_bodies
            ? Vec3{0.0, 0.0, 0.0}
            : Vec3{0.5 * spacing_of(options.workload).x, 0.0, 0.0};

        std::mt19937 engine(options.seed);
        const BodySet sources = make_body_set(
            options.source, options.sources, options.workload,
            Vec3{0.0, 0.0, 0.0}, engine);
        const BodySet targets = make_body_set(
            options.target, options.targets, options.workload,
            target_origin, engine);

        std::vector<int> identities;
        if (same_bodies) {
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
