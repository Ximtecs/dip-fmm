// SPDX-License-Identifier: Apache-2.0
//
// Systematic near-field correctness matrix.
//
// GEOMETRY BUILDS TENSORS; EXECUTORS APPLY TENSORS.  Every combination of
// source geometry, target geometry, execution backend, precision and P2P
// execution packing must reproduce the same canonical pair tensors.  The
// independent reference is `DenseDirectPlan` in FP64 (all-to-all, exact
// geometry, same identity semantics); the FMM plans below use a single leaf
// so that their result is the pure near field, and a multi-leaf tree so that
// every packing has to assemble dense leaf pairs, dictionary tokens or sparse
// block rows from several leaves.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include <unistd.h>

#include "cdfmm/plan/direct/dense.hpp"
#include "cdfmm/rectangular_prism.hpp"
#include "cdfmm/tetrahedron.hpp"
#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

namespace {

struct GeometryPair {
    SourceGeometry source;
    TargetGeometry target;
    const char* name;
};

constexpr std::array<GeometryPair, 9> geometry_pairs{{
    {SourceGeometry::PointDipole, TargetGeometry::Point, "point->point"},
    {SourceGeometry::PointDipole, TargetGeometry::RectangularPrism,
     "point->prism"},
    {SourceGeometry::PointDipole, TargetGeometry::Tetrahedron,
     "point->tetrahedron"},
    {SourceGeometry::RectangularPrism, TargetGeometry::Point, "prism->point"},
    {SourceGeometry::RectangularPrism, TargetGeometry::RectangularPrism,
     "prism->prism"},
    {SourceGeometry::RectangularPrism, TargetGeometry::Tetrahedron,
     "prism->tetrahedron"},
    {SourceGeometry::Tetrahedron, TargetGeometry::Point, "tetrahedron->point"},
    {SourceGeometry::Tetrahedron, TargetGeometry::RectangularPrism,
     "tetrahedron->prism"},
    {SourceGeometry::Tetrahedron, TargetGeometry::Tetrahedron,
     "tetrahedron->tetrahedron"},
}};

enum class Layout { Irregular, Regular };

const char* name(const Layout layout)
{
    return layout == Layout::Irregular ? "irregular" : "regular";
}

const char* name(const P2PExecutionPacking packing)
{
    switch (packing) {
    case P2PExecutionPacking::Reference: return "Reference";
    case P2PExecutionPacking::CanonicalAos: return "CanonicalAos";
    case P2PExecutionPacking::ParticleRowSoa: return "ParticleRowSoa";
    case P2PExecutionPacking::TensorDictionary: return "TensorDictionary";
    case P2PExecutionPacking::CudaBsr3: return "CudaBsr3";
    case P2PExecutionPacking::LeafBlock: return "LeafBlock";
    case P2PExecutionPacking::PointGeometry: return "PointGeometry";
    case P2PExecutionPacking::Auto: return "Auto";
    }
    return "?";
}

const char* name(const ExecutionBackend backend)
{
    switch (backend) {
    case ExecutionBackend::Auto: return "Auto";
    case ExecutionBackend::CpuReference: return "CpuReference";
    case ExecutionBackend::CpuStatic: return "CpuStatic";
    case ExecutionBackend::CudaM2LP2P: return "CudaPartial";
    case ExecutionBackend::CudaFull: return "CudaFull";
    }
    return "?";
}

// One deterministic scene: every particle is both a source and a target, so
// coincident finite self interactions and point self exclusion are exercised
// by the same identity map.
struct Scene {
    std::vector<Vec3> positions{};
    std::vector<Vec3> moments{};
    std::vector<int> identities{};
    RectangularPrism prism{};
    std::vector<Tetrahedron> tetrahedra{};
};

Tetrahedron base_tetrahedron(const double size, const int orientation)
{
    // Unit right tetrahedron about its centroid, scaled and with the axes
    // cyclically permuted so per-object records differ.
    const std::array<Vec3, 4> vertices{{{-0.25, -0.25, -0.25},
                                        {0.75, -0.25, -0.25},
                                        {-0.25, 0.75, -0.25},
                                        {-0.25, -0.25, 0.75}}};
    Tetrahedron result;
    for (std::size_t vertex = 0; vertex < 4; ++vertex) {
        const Vec3 v = vertices[vertex] * size;
        const std::array<double, 3> c{{v.x, v.y, v.z}};
        result.vertices[vertex] = {
            c[static_cast<std::size_t>(orientation % 3)],
            c[static_cast<std::size_t>((orientation + 1) % 3)],
            c[static_cast<std::size_t>((orientation + 2) % 3)]};
    }
    return result;
}

Scene make_scene(const Layout layout)
{
    Scene scene;
    if (layout == Layout::Irregular) {
        // Linear congruential coordinates: reproducible, no library RNG.
        std::uint64_t state = 0x9E3779B97F4A7C15ULL;
        const auto next = [&state]() {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return -0.85 + 1.7 * static_cast<double>(state >> 11) /
                static_cast<double>(1ULL << 53);
        };
        // UniformFmm canonicalises normalised geometry to a 1e-9 grid of the
        // root side (2.0 here) so cache keys are reproducible; placing the
        // scene on that grid keeps the dense reference bit-comparable.
        const auto canonical = [](const double value) {
            return 2.0 * std::nearbyint(value * 0.5e9) / 1.0e9;
        };
        for (int index = 0; index < 46; ++index) {
            scene.positions.push_back(
                {canonical(next()), canonical(next()), canonical(next())});
        }
        // Two close neighbours well inside one leaf.
        scene.positions.push_back({0.31, -0.22, 0.17});
        scene.positions.push_back({0.33, -0.21, 0.18});
        scene.prism = {0.06, 0.05, 0.04};
        for (std::size_t index = 0; index < scene.positions.size(); ++index) {
            const double size =
                0.05 * (0.6 + 0.4 * static_cast<double>(index % 5) / 4.0);
            scene.tetrahedra.push_back(
                base_tetrahedron(size, static_cast<int>(index % 3)));
        }
    } else {
        for (int iz = 0; iz < 3; ++iz) {
            for (int iy = 0; iy < 4; ++iy) {
                for (int ix = 0; ix < 4; ++ix) {
                    scene.positions.push_back({-0.675 + 0.45 * ix,
                                               -0.675 + 0.45 * iy,
                                               -0.45 + 0.45 * iz});
                }
            }
        }
        scene.prism = {0.2, 0.16, 0.12};
        scene.tetrahedra.push_back(base_tetrahedron(0.16, 0));
    }
    scene.identities.resize(scene.positions.size());
    std::iota(scene.identities.begin(), scene.identities.end(), 0);
    scene.moments.resize(scene.positions.size());
    for (std::size_t index = 0; index < scene.moments.size(); ++index) {
        const double value = static_cast<double>(index);
        scene.moments[index] = {std::sin(1.1 * value + 0.3),
                                std::cos(0.7 * value - 0.2),
                                std::sin(0.4 * value + 1.1)};
    }
    return scene;
}

std::vector<Vec3> dense_reference(const GeometryPair& pair, const Scene& scene)
{
    const std::span<const RectangularPrism> prism{&scene.prism, 1};
    const bool prism_source = pair.source == SourceGeometry::RectangularPrism;
    const bool prism_target = pair.target == TargetGeometry::RectangularPrism;
    const bool tet_source = pair.source == SourceGeometry::Tetrahedron;
    const bool tet_target = pair.target == TargetGeometry::Tetrahedron;
    const DenseDirectPlan plan(
        scene.positions, scene.positions, pair.source, pair.target,
        prism_source ? prism : std::span<const RectangularPrism>{},
        prism_target ? prism : std::span<const RectangularPrism>{},
        scene.identities, StaticPrecision::Float64,
        tet_source ? std::span<const Tetrahedron>(scene.tetrahedra)
                   : std::span<const Tetrahedron>{},
        tet_target ? std::span<const Tetrahedron>(scene.tetrahedra)
                   : std::span<const Tetrahedron>{});
    return plan.evaluate(scene.moments, DenseDirectBackend::Portable);
}

UniformFmmOptions plan_options(const GeometryPair& pair, const Scene& scene,
                               const ExecutionBackend backend,
                               const StaticPrecision precision,
                               const P2PExecutionPacking packing,
                               const int depth)
{
    UniformFmmOptions options;
    options.backend = backend;
    options.precision = precision;
    options.expansion_order = 4;
    options.tree.max_level = depth;
    options.tree.root_centre = Vec3{};
    options.tree.root_half_width = 1.0;
    options.source_geometry = pair.source;
    options.target_geometry = pair.target;
    if (pair.source == SourceGeometry::RectangularPrism) {
        options.source_sizes = {scene.prism};
    } else if (pair.source == SourceGeometry::Tetrahedron) {
        options.source_tetrahedra = scene.tetrahedra;
    }
    if (pair.target == TargetGeometry::RectangularPrism) {
        options.target_sizes = {scene.prism};
    } else if (pair.target == TargetGeometry::Tetrahedron) {
        options.target_tetrahedra = scene.tetrahedra;
    }
    // Exact near field is the subject; the far field uses point models so
    // that every backend shares the identical, cheap hierarchy.
    options.near_field_source_model = SourceModel::ExactGeometry;
    options.near_field_target_model = TargetModel::ExactGeometry;
    options.far_field_source_model = SourceModel::PointDipole;
    options.far_field_target_model = TargetModel::Point;
    options.fixed_target_source_indices = scene.identities;
    options.p2p_packing = packing;
    options.enable_cache = true;
    return options;
}

bool point_pair(const GeometryPair& pair)
{
    return pair.source == SourceGeometry::PointDipole &&
        pair.target == TargetGeometry::Point;
}

std::vector<P2PExecutionPacking> packings_for(const ExecutionBackend backend,
                                              const GeometryPair& pair)
{
    if (backend == ExecutionBackend::CpuStatic) {
        std::vector<P2PExecutionPacking> result{
            P2PExecutionPacking::CanonicalAos,
            P2PExecutionPacking::ParticleRowSoa,
            P2PExecutionPacking::TensorDictionary};
        if (point_pair(pair)) {
            result.push_back(P2PExecutionPacking::PointGeometry);
        }
        return result;
    }
    return {P2PExecutionPacking::CanonicalAos, P2PExecutionPacking::LeafBlock,
            P2PExecutionPacking::CudaBsr3,
            P2PExecutionPacking::TensorDictionary};
}

std::vector<ExecutionBackend> available_backends()
{
    std::vector<ExecutionBackend> result{ExecutionBackend::CpuStatic};
    if (cuda_m2l_p2p_available()) {
        result.push_back(ExecutionBackend::CudaM2LP2P);
    }
    if (cuda_full_available()) {
        result.push_back(ExecutionBackend::CudaFull);
    }
    return result;
}

double field_scale(const std::vector<Vec3>& values)
{
    double scale = 0.0;
    for (const Vec3& value : values) {
        scale = std::max({scale, std::abs(value.x), std::abs(value.y),
                          std::abs(value.z)});
    }
    return scale;
}

double maximum_difference(const std::vector<PotentialField>& actual,
                          const std::vector<Vec3>& expected)
{
    REQUIRE(actual.size() == expected.size());
    double difference = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        difference = std::max(
            {difference, std::abs(actual[index].H.x - expected[index].x),
             std::abs(actual[index].H.y - expected[index].y),
             std::abs(actual[index].H.z - expected[index].z)});
    }
    return difference;
}

double tolerance_for(const StaticPrecision precision)
{
    // FP32 plans quantise tensors and moments to single precision and the
    // CUDA leaf-block kernel accumulates with atomics, so their comparison
    // against the FP64 reference is loose; FP64 packings differ only by
    // summation order.
    return precision == StaticPrecision::Float32 ? 2.0e-4 : 1.0e-10;
}

// Scoped private cache directory so the repeated constructions of one
// geometry pair hit the geometry cache and the cache-hit packing path is
// exercised too.
class ScopedCacheDirectory {
public:
    ScopedCacheDirectory()
        : path_(std::filesystem::temp_directory_path() /
                ("cdfmm-p2p-matrix-" + std::to_string(::getpid())))
    {
        std::filesystem::create_directories(path_);
        ::setenv("CDFMM_CACHE_DIR", path_.c_str(), 1);
    }
    ~ScopedCacheDirectory()
    {
        ::unsetenv("CDFMM_CACHE_DIR");
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

private:
    std::filesystem::path path_;
};

} // namespace

TEST_CASE("every near-field geometry pair reproduces the dense direct reference through every packing",
          "[p2p][geometry-matrix]")
{
    ScopedCacheDirectory cache;
    const std::vector<ExecutionBackend> backends = available_backends();
    for (const Layout layout : {Layout::Irregular, Layout::Regular}) {
        const Scene scene = make_scene(layout);
        for (const GeometryPair& pair : geometry_pairs) {
            const std::vector<Vec3> expected = dense_reference(pair, scene);
            const double scale = field_scale(expected);
            REQUIRE(scale > 0.0);
            for (const ExecutionBackend backend : backends) {
                for (const StaticPrecision precision :
                     {StaticPrecision::Float64, StaticPrecision::Float32}) {
                    for (const P2PExecutionPacking packing :
                         packings_for(backend, pair)) {
                        INFO(name(layout) << " " << pair.name << " "
                             << name(backend) << " "
                             << (precision == StaticPrecision::Float32
                                     ? "fp32" : "fp64")
                             << " " << name(packing));
                        // A single leaf makes the FMM result the pure exact
                        // near field of every pair.
                        UniformFmm fmm(scene.positions, scene.positions,
                                       plan_options(pair, scene, backend,
                                                    precision, packing, 0));
                        REQUIRE(fmm.p2p_execution_packing() == packing);
                        const auto actual = fmm.evaluate(
                            scene.moments, OutputFlags::Field,
                            scene.identities);
                        REQUIRE(maximum_difference(actual, expected) <=
                                tolerance_for(precision) * scale);
                    }
                }
            }
        }
    }
}

TEST_CASE("CUDA dictionary executors agree for every geometry pair",
          "[p2p][geometry-matrix][cuda]")
{
    if (!cuda_m2l_p2p_available() && !cuda_full_available()) {
        SUCCEED("CUDA backends are unavailable");
        return;
    }
    ScopedCacheDirectory cache;
    const std::vector<ExecutionBackend> backends = available_backends();
    const Scene scene = make_scene(Layout::Regular);
    for (const GeometryPair& pair : geometry_pairs) {
        const std::vector<Vec3> expected = dense_reference(pair, scene);
        const double scale = field_scale(expected);
        for (const ExecutionBackend backend : backends) {
            if (backend == ExecutionBackend::CpuStatic) {
                continue;
            }
            for (const StaticPrecision precision :
                 {StaticPrecision::Float64, StaticPrecision::Float32}) {
                for (const int executor : {0, 1, 2}) {
                    INFO(pair.name << " " << name(backend) << " executor "
                         << executor);
                    UniformFmmOptions options = plan_options(
                        pair, scene, backend, precision,
                        P2PExecutionPacking::TensorDictionary, 0);
                    options.cuda_dictionary_target_owned = executor == 1;
                    options.cuda_dictionary_power2_microtiles = executor == 2;
                    UniformFmm fmm(scene.positions, scene.positions, options);
                    REQUIRE(fmm.p2p_execution_packing() ==
                            P2PExecutionPacking::TensorDictionary);
                    const auto actual = fmm.evaluate(
                        scene.moments, OutputFlags::Field, scene.identities);
                    REQUIRE(maximum_difference(actual, expected) <=
                            tolerance_for(precision) * scale);
                }
            }
        }
    }
}

TEST_CASE("multi-leaf packings agree with canonical rows for every geometry pair",
          "[p2p][geometry-matrix]")
{
    // Depth one puts about six particles in each of the eight leaves, so the
    // leaf-pair, dictionary and block-row packings assemble their structures
    // from many (target leaf, source leaf) pairs. The far field is identical
    // for every plan; the FP64 canonical-row CPU plan is the reference.
    ScopedCacheDirectory cache;
    const std::vector<ExecutionBackend> backends = available_backends();
    const Scene scene = make_scene(Layout::Irregular);
    for (const GeometryPair& pair : geometry_pairs) {
        UniformFmm reference(scene.positions, scene.positions,
                             plan_options(pair, scene,
                                          ExecutionBackend::CpuStatic,
                                          StaticPrecision::Float64,
                                          P2PExecutionPacking::CanonicalAos, 1));
        const auto reference_fields = reference.evaluate(
            scene.moments, OutputFlags::Field, scene.identities);
        std::vector<Vec3> expected(reference_fields.size());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            expected[index] = reference_fields[index].H;
        }
        const double scale = field_scale(expected);
        for (const ExecutionBackend backend : backends) {
            for (const StaticPrecision precision :
                 {StaticPrecision::Float64, StaticPrecision::Float32}) {
                for (const P2PExecutionPacking packing :
                     packings_for(backend, pair)) {
                    INFO(pair.name << " " << name(backend) << " "
                         << (precision == StaticPrecision::Float32 ? "fp32"
                                                                    : "fp64")
                         << " " << name(packing));
                    UniformFmm fmm(scene.positions, scene.positions,
                                   plan_options(pair, scene, backend,
                                                precision, packing, 1));
                    REQUIRE(fmm.p2p_execution_packing() == packing);
                    const auto actual = fmm.evaluate(
                        scene.moments, OutputFlags::Field, scene.identities);
                    REQUIRE(maximum_difference(actual, expected) <=
                            tolerance_for(precision) * scale);
                }
            }
        }
    }
}

TEST_CASE("automatic policy executes every geometry pair on every production backend",
          "[p2p][geometry-matrix]")
{
    // No packing is forced: the resolved default of each backend must still
    // reproduce the dense reference, and it must be a stored-tensor packing
    // for every finite pair.
    ScopedCacheDirectory cache;
    const std::vector<ExecutionBackend> backends = available_backends();
    const Scene scene = make_scene(Layout::Irregular);
    for (const GeometryPair& pair : geometry_pairs) {
        const std::vector<Vec3> expected = dense_reference(pair, scene);
        const double scale = field_scale(expected);
        for (const ExecutionBackend backend : backends) {
            for (const StaticPrecision precision :
                 {StaticPrecision::Float64, StaticPrecision::Float32}) {
                INFO(pair.name << " " << name(backend));
                UniformFmm fmm(scene.positions, scene.positions,
                               plan_options(pair, scene, backend, precision,
                                            P2PExecutionPacking::Auto, 0));
                REQUIRE(fmm.requested_p2p_packing() ==
                        P2PExecutionPacking::Auto);
                REQUIRE(fmm.p2p_execution_packing() !=
                        P2PExecutionPacking::Auto);
                if (!point_pair(pair)) {
                    REQUIRE(fmm.p2p_execution_packing() !=
                            P2PExecutionPacking::PointGeometry);
                }
                const auto actual = fmm.evaluate(
                    scene.moments, OutputFlags::Field, scene.identities);
                REQUIRE(maximum_difference(actual, expected) <=
                        tolerance_for(precision) * scale);
            }
        }
    }
    if (one_mkl_available()) {
        // oneMKL changes the M2L executor only; its near field is the CPU
        // static packing and must be identical.
        for (const GeometryPair& pair : geometry_pairs) {
            const std::vector<Vec3> expected = dense_reference(pair, scene);
            UniformFmmOptions options = plan_options(
                pair, scene, ExecutionBackend::CpuStatic,
                StaticPrecision::Float64, P2PExecutionPacking::Auto, 0);
            options.static_matrix_backend = StaticMatrixBackend::OneMkl;
            UniformFmm fmm(scene.positions, scene.positions, options);
            REQUIRE(fmm.execution_plan().m2l == StaticOperatorExecutor::OneMkl);
            const auto actual = fmm.evaluate(scene.moments, OutputFlags::Field,
                                             scene.identities);
            REQUIRE(maximum_difference(actual, expected) <=
                    1.0e-10 * field_scale(expected));
        }
    }
}

TEST_CASE("CpuReference is audited separately from the canonical near field",
          "[p2p][geometry-matrix]")
{
    // The reference backend forms dynamic Cartesian contractions and has no
    // finite P2M/L2P; it rejects exact finite stages explicitly instead of
    // approximating them, and its point near field agrees with the dense
    // reference. This limitation must not leak into the static operator,
    // whose nine pairs are covered above.
    const Scene scene = make_scene(Layout::Irregular);
    for (const GeometryPair& pair : geometry_pairs) {
        UniformFmmOptions options = plan_options(
            pair, scene, ExecutionBackend::CpuReference,
            StaticPrecision::Float64, P2PExecutionPacking::Auto, 0);
        options.expansion_basis = ExpansionBasis::Cartesian;
        options.enable_cache = false;
        if (point_pair(pair)) {
            UniformFmm fmm(scene.positions, scene.positions, options);
            REQUIRE(fmm.p2p_execution_packing() ==
                    P2PExecutionPacking::Reference);
            const std::vector<Vec3> expected = dense_reference(pair, scene);
            const auto actual = fmm.evaluate(scene.moments, OutputFlags::Field,
                                             scene.identities);
            REQUIRE(maximum_difference(actual, expected) <=
                    1.0e-10 * field_scale(expected));
        } else {
            REQUIRE_THROWS_AS(
                UniformFmm(scene.positions, scene.positions, options),
                std::invalid_argument);
        }
    }
}

TEST_CASE("periodic plans agree across the packings that represent image records",
          "[p2p][geometry-matrix][periodic]")
{
    // Periodic image records are representable by the per-pair packings
    // (canonical rows, SoA rows) and by the position-based point executor;
    // the FMM totals of every backend that offers them must agree, for point
    // and finite pairs alike.
    ScopedCacheDirectory cache;
    const std::vector<ExecutionBackend> backends = available_backends();
    Scene scene = make_scene(Layout::Irregular);
    scene.prism = {0.03, 0.025, 0.02};
    for (Tetrahedron& tetrahedron : scene.tetrahedra) {
        for (Vec3& vertex : tetrahedron.vertices) {
            vertex = vertex * 0.5;
        }
    }
    for (const GeometryPair& pair : geometry_pairs) {
        UniformFmmOptions reference_options = plan_options(
            pair, scene, ExecutionBackend::CpuStatic, StaticPrecision::Float64,
            P2PExecutionPacking::ParticleRowSoa, 2);
        reference_options.periodic.enabled = true;
        reference_options.periodic.centre = Vec3{};
        reference_options.periodic.lengths = Vec3{2.0, 2.0, 2.0};
        UniformFmm reference(scene.positions, scene.positions,
                             reference_options);
        REQUIRE(reference.p2p_execution_packing() ==
                P2PExecutionPacking::ParticleRowSoa);
        const auto reference_fields = reference.evaluate(
            scene.moments, OutputFlags::Field, scene.identities);
        std::vector<Vec3> expected(reference_fields.size());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            expected[index] = reference_fields[index].H;
        }
        const double scale = field_scale(expected);
        for (const ExecutionBackend backend : backends) {
            for (const StaticPrecision precision :
                 {StaticPrecision::Float64, StaticPrecision::Float32}) {
                std::vector<P2PExecutionPacking> packings{
                    P2PExecutionPacking::CanonicalAos};
                if (backend == ExecutionBackend::CpuStatic) {
                    packings.push_back(P2PExecutionPacking::ParticleRowSoa);
                    if (point_pair(pair)) {
                        // The position-based executor folds every image
                        // shift and identity marker of the leaf records in.
                        packings.push_back(P2PExecutionPacking::PointGeometry);
                    }
                }
                for (const P2PExecutionPacking packing : packings) {
                    INFO(pair.name << " periodic " << name(backend) << " "
                         << name(packing));
                    UniformFmmOptions options = reference_options;
                    options.backend = backend;
                    options.precision = precision;
                    options.p2p_packing = packing;
                    UniformFmm fmm(scene.positions, scene.positions, options);
                    REQUIRE(fmm.p2p_execution_packing() == packing);
                    const auto actual = fmm.evaluate(
                        scene.moments, OutputFlags::Field, scene.identities);
                    REQUIRE(maximum_difference(actual, expected) <=
                            tolerance_for(precision) * scale);
                }
            }
        }
    }
}
