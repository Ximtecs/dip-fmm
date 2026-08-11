// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"
#include "cuda_fmm_plan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

#ifdef CDFMM_USE_MKL
#include <mkl.h>
#include <mkl_version.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using cdfmm::Vec3;

struct Options {
    int sources{1000};
    int targets{1000};
    int depth{3};
    int order{4};
    int evaluations{10};
    int warmups{1};
    int samples{5};
    int threads{0};
    unsigned int seed{314159U};
    bool direct{true};
    bool workload_comparison{true};
    bool cuda_status{false};
    std::string backend{"cpu-static-matrix"};
    std::string output{};
};

struct WorkloadTiming {
    double construction_seconds{0.0};
    double evaluation_seconds{0.0};
    double total_seconds{0.0};
};

enum class BenchmarkBackend {
    CpuDirect,
    CpuStaticMatrix,
    CpuStaticMatrixMkl,
    CudaDirect
};

Options parse_options(const int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--no-direct") {
            options.direct = false;
            continue;
        }
        if (key == "--no-workload-comparison") {
            options.workload_comparison = false;
            continue;
        }
        if (key == "--cuda-status") {
            options.cuda_status = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + key);
        }
        const std::string value = argv[++index];
        if (key == "--sources") options.sources = std::stoi(value);
        else if (key == "--targets") options.targets = std::stoi(value);
        else if (key == "--depth") options.depth = std::stoi(value);
        else if (key == "--order") options.order = std::stoi(value);
        else if (key == "--evaluations") options.evaluations = std::stoi(value);
        else if (key == "--warmups") options.warmups = std::stoi(value);
        else if (key == "--samples") options.samples = std::stoi(value);
        else if (key == "--threads") options.threads = std::stoi(value);
        else if (key == "--seed") options.seed =
            static_cast<unsigned int>(std::stoul(value));
        else if (key == "--backend") options.backend = value;
        else if (key == "--output") options.output = value;
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (options.sources < 0 || options.targets < 0 || options.depth < 0 ||
        options.order < 0 || options.evaluations < 1 || options.samples < 1) {
        throw std::invalid_argument("Counts, depth, order, and samples are invalid");
    }
    if (options.targets != options.sources) {
        throw std::invalid_argument(
            "The all-to-all benchmark requires equal source and target counts"
        );
    }
    return options;
}

BenchmarkBackend benchmark_backend(const std::string& name)
{
    if (name == "cpu-direct") {
        return BenchmarkBackend::CpuDirect;
    }
    if (name == "cpu-static-matrix") {
        return BenchmarkBackend::CpuStaticMatrix;
    }
    if (name == "cpu-static-matrix-mkl") {
        if (!cdfmm::one_mkl_available()) {
            throw std::runtime_error(
                "The oneMKL static-matrix backend is unavailable"
            );
        }
        return BenchmarkBackend::CpuStaticMatrixMkl;
    }
    if (name == "cuda-direct") {
        if (!cdfmm::cuda_direct_available()) {
            throw std::runtime_error("CUDA direct P2P is unavailable");
        }
        return BenchmarkBackend::CudaDirect;
    }
    throw std::invalid_argument(
        "Unknown backend '" + name +
        "'; expected cpu-direct, cuda-direct, cpu-static-matrix, or "
        "cpu-static-matrix-mkl"
    );
}

cdfmm::UniformFmmOptions cpu_options(
    const cdfmm::UniformFmmOptions& base,
    const BenchmarkBackend backend)
{
    cdfmm::UniformFmmOptions selected = base;
    if (backend == BenchmarkBackend::CpuStaticMatrix) {
        selected.backend = cdfmm::ExecutionBackend::CpuStatic;
        selected.m2l_backend = cdfmm::M2LBackend::Static;
        selected.static_matrix_backend = cdfmm::StaticMatrixBackend::Portable;
        return selected;
    }
    if (backend == BenchmarkBackend::CpuStaticMatrixMkl) {
        selected.backend = cdfmm::ExecutionBackend::CpuStatic;
        selected.m2l_backend = cdfmm::M2LBackend::Static;
        selected.static_matrix_backend = cdfmm::StaticMatrixBackend::OneMkl;
        return selected;
    }
    throw std::invalid_argument("Direct backends do not use UniformFmmOptions");
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

double mean(const std::vector<double>& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
}

WorkloadTiming median_timing(const std::vector<WorkloadTiming>& timings)
{
    std::vector<double> construction;
    std::vector<double> evaluation;
    std::vector<double> total;
    construction.reserve(timings.size());
    evaluation.reserve(timings.size());
    total.reserve(timings.size());
    for (const WorkloadTiming& timing : timings) {
        construction.push_back(timing.construction_seconds);
        evaluation.push_back(timing.evaluation_seconds);
        total.push_back(timing.total_seconds);
    }
    return {median(construction), median(evaluation), median(total)};
}

WorkloadTiming benchmark_fmm_workload(
    const std::vector<Vec3>& source_positions,
    const std::vector<Vec3>& target_positions,
    const std::vector<std::vector<Vec3>>& moment_states,
    const std::vector<int>& source_identities,
    const cdfmm::UniformFmmOptions& options,
    const int evaluation_count,
    const int samples)
{
    std::vector<WorkloadTiming> timings;
    timings.reserve(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample) {
        const auto total_start = Clock::now();
        const auto construction_start = Clock::now();
        cdfmm::UniformFmm fmm(source_positions, target_positions, options);
        std::vector<cdfmm::PotentialField> results(target_positions.size());
        const double construction_seconds = std::chrono::duration<double>(
            Clock::now() - construction_start
        ).count();

        const auto evaluation_start = Clock::now();
        for (int evaluation = 0; evaluation < evaluation_count; ++evaluation) {
            fmm.evaluate_into(
                moment_states[static_cast<std::size_t>(evaluation)],
                results,
                cdfmm::OutputFlags::Field,
                source_identities
            );
        }
        const double evaluation_seconds = std::chrono::duration<double>(
            Clock::now() - evaluation_start
        ).count();
        const double total_seconds = std::chrono::duration<double>(
            Clock::now() - total_start
        ).count();
        timings.push_back(
            {construction_seconds, evaluation_seconds, total_seconds}
        );
    }
    return median_timing(timings);
}

WorkloadTiming benchmark_cuda_direct_workload(
    const std::vector<Vec3>& source_positions,
    const std::vector<Vec3>& target_positions,
    const std::vector<std::vector<Vec3>>& moment_states,
    const std::vector<int>& source_identities,
    const int evaluation_count,
    const int samples)
{
    std::vector<WorkloadTiming> timings;
    timings.reserve(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample) {
        const auto total_start = Clock::now();
        const auto construction_start = Clock::now();
        cdfmm::CudaFmmPlan plan(source_positions, target_positions);
        std::vector<cdfmm::PotentialField> results(target_positions.size());
        const double construction_seconds = std::chrono::duration<double>(
            Clock::now() - construction_start
        ).count();

        const auto evaluation_start = Clock::now();
        for (int evaluation = 0; evaluation < evaluation_count; ++evaluation) {
            plan.evaluate(
                moment_states[static_cast<std::size_t>(evaluation)],
                results,
                cdfmm::OutputFlags::Field,
                source_identities
            );
        }
        const double evaluation_seconds = std::chrono::duration<double>(
            Clock::now() - evaluation_start
        ).count();
        const double total_seconds = std::chrono::duration<double>(
            Clock::now() - total_start
        ).count();
        timings.push_back(
            {construction_seconds, evaluation_seconds, total_seconds}
        );
    }
    return median_timing(timings);
}

WorkloadTiming benchmark_cpu_direct_workload(
    const std::vector<Vec3>& source_positions,
    const std::vector<Vec3>& target_positions,
    const std::vector<std::vector<Vec3>>& moment_states,
    const std::vector<int>& source_identities,
    const int evaluation_count,
    const int samples)
{
    std::vector<WorkloadTiming> timings;
    timings.reserve(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample) {
        const auto evaluation_start = Clock::now();
        for (int evaluation = 0; evaluation < evaluation_count; ++evaluation) {
            const auto results = cdfmm::direct_p2p_reference(
                target_positions,
                source_positions,
                moment_states[static_cast<std::size_t>(evaluation)],
                cdfmm::OutputFlags::Field,
                source_identities
            );
            if (results.size() != target_positions.size()) {
                throw std::runtime_error("CPU direct P2P returned invalid output");
            }
        }
        const double evaluation_seconds = std::chrono::duration<double>(
            Clock::now() - evaluation_start
        ).count();
        timings.push_back({0.0, evaluation_seconds, evaluation_seconds});
    }
    return median_timing(timings);
}

void write_workload(std::ostream& out, const WorkloadTiming& timing)
{
    out << ',' << timing.construction_seconds
        << ',' << timing.evaluation_seconds
        << ',' << timing.total_seconds;
}

std::string progress_bar(const int completed, const int total,
                         const int width = 24)
{
    const int filled = total > 0 ? (width * completed) / total : width;
    return "[" + std::string(static_cast<std::size_t>(filled), '=') +
        std::string(static_cast<std::size_t>(width - filled), ' ') + "]";
}

std::string compiler_name()
{
#if defined(__INTEL_LLVM_COMPILER)
    return "IntelLLVM";
#elif defined(__clang__)
    return "Clang";
#elif defined(__GNUC__)
    return "GNU";
#else
    return "unknown";
#endif
}

std::string compiler_version()
{
#ifdef __VERSION__
    return __VERSION__;
#else
    return "unknown";
#endif
}

std::string static_multiply_backend(const BenchmarkBackend backend)
{
    if (backend == BenchmarkBackend::CpuStaticMatrixMkl) {
        return "oneMKL";
    }
    if (backend == BenchmarkBackend::CpuStaticMatrix) {
        return "portable";
    }
    return "not_applicable";
}

std::string mkl_version()
{
#ifdef CDFMM_USE_MKL
    return std::to_string(__INTEL_MKL__) + "." +
        std::to_string(__INTEL_MKL_MINOR__) + "." +
        std::to_string(__INTEL_MKL_UPDATE__);
#else
    return "not_enabled";
#endif
}

void warm_mkl_runtime()
{
#ifdef CDFMM_USE_MKL
    double left = 1.0;
    double right = 1.0;
    double product = 0.0;
    const int previous_threads = mkl_set_num_threads_local(1);
    cblas_dgemm(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        1,
        1,
        1,
        1.0,
        &left,
        1,
        &right,
        1,
        0.0,
        &product,
        1
    );
    mkl_set_num_threads_local(previous_threads);
#endif
}

} // namespace

int main(int argc, char** argv)
{
    using namespace cdfmm;
    try {
        const Options options = parse_options(argc, argv);
        if (options.cuda_status) {
            std::cout << "cuda_compiled=" << (cuda_compiled() ? 1 : 0) << '\n'
                      << "cuda_available=" << (cuda_available() ? 1 : 0) << '\n'
                      << "cuda_direct_available="
                      << (cuda_direct_available() ? 1 : 0) << '\n'
                      << "cuda_farfield_available="
                      << (cuda_farfield_available() ? 1 : 0) << '\n'
                      << "cuda_full_available="
                      << (cuda_full_available() ? 1 : 0) << '\n'
                      << "one_mkl_available="
                      << (one_mkl_available() ? 1 : 0) << '\n'
                      << "cuda_device=";
            if (cuda_available()) {
                std::cout << cuda_device_description();
            }
            std::cout << '\n';
            return 0;
        }
        const BenchmarkBackend selected_backend = benchmark_backend(
            options.backend
        );
#ifdef CDFMM_USE_OPENMP
        if (options.threads > 0) {
            omp_set_num_threads(options.threads);
        }
        const int thread_count = omp_get_max_threads();
        const int openmp_version = _OPENMP;
        const char* openmp_status = "enabled";
#else
        const int thread_count = 1;
        const int openmp_version = 0;
        const char* openmp_status = "disabled";
#endif

#ifdef CDFMM_USE_OPENMP
        int warmed_threads = 0;
        #pragma omp parallel reduction(+:warmed_threads)
        warmed_threads += 1;
        if (warmed_threads != thread_count) {
            throw std::runtime_error("OpenMP runtime warm-up used wrong team size");
        }
#endif

        std::cerr << "Running benchmark: sources=" << options.sources
                  << " targets=" << options.targets
                  << " order=" << options.order
                  << " depth=" << options.depth
                  << " backend=" << options.backend
                  << " threads=" << thread_count
                  << " evaluations=" << options.evaluations
                  << " samples=" << options.samples << "\n";

        std::mt19937 generator(options.seed);
        std::uniform_real_distribution<double> distribution(-0.95, 0.95);
        std::vector<Vec3> source_positions(
            static_cast<std::size_t>(options.sources)
        );
        for (Vec3& position : source_positions) {
            position = {distribution(generator), distribution(generator),
                        distribution(generator)};
        }

        // Benchmark source-point evaluation: every particle is both a source
        // and a target, with its singular self-pair excluded explicitly.
        const std::vector<Vec3> target_positions = source_positions;
        std::vector<int> source_identities(source_positions.size());
        std::iota(source_identities.begin(), source_identities.end(), 0);

        constexpr int repeated_evaluation_count = 10;
        const int state_count = options.warmups + std::max(
            options.evaluations,
            repeated_evaluation_count
        );
        std::vector<std::vector<Vec3>> moment_states(
            static_cast<std::size_t>(state_count),
            std::vector<Vec3>(source_positions.size())
        );
        for (auto& state : moment_states) {
            for (Vec3& moment : state) {
                moment = {distribution(generator), distribution(generator),
                          distribution(generator)};
            }
        }

        UniformFmmOptions fmm_options;
        fmm_options.expansion_order = options.order;
        fmm_options.tree.max_level = options.depth;
        fmm_options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
        fmm_options.tree.root_half_width = 1.0;

        UniformFmmOptions selected_options = fmm_options;
        if (selected_backend == BenchmarkBackend::CpuStaticMatrix ||
            selected_backend == BenchmarkBackend::CpuStaticMatrixMkl) {
            selected_options = cpu_options(fmm_options, selected_backend);
        }

        const double missing = std::numeric_limits<double>::quiet_NaN();
        WorkloadTiming selected_single{missing, missing, missing};
        WorkloadTiming selected_repeated{missing, missing, missing};
        if (options.workload_comparison) {
            if (selected_backend == BenchmarkBackend::CudaDirect) {
                // Exclude one-time CUDA context and kernel initialisation from
                // both persistent-plan construction workloads.
                CudaFmmPlan warmup_plan(source_positions, target_positions);
                std::vector<PotentialField> warmup_results(
                    target_positions.size()
                );
                warmup_plan.evaluate(
                    moment_states.front(),
                    warmup_results,
                    OutputFlags::Field,
                    source_identities
                );
            }
            if (selected_backend == BenchmarkBackend::CpuStaticMatrixMkl) {
                warm_mkl_runtime();
            }
            std::cerr << "Benchmarking selected 1- and 10-evaluation "
                         "workloads...\n";
            if (selected_backend == BenchmarkBackend::CudaDirect) {
                selected_single = benchmark_cuda_direct_workload(
                    source_positions,
                    target_positions,
                    moment_states,
                    source_identities,
                    1,
                    options.samples
                );
                selected_repeated = benchmark_cuda_direct_workload(
                    source_positions,
                    target_positions,
                    moment_states,
                    source_identities,
                    repeated_evaluation_count,
                    options.samples
                );
            } else if (selected_backend == BenchmarkBackend::CpuDirect) {
                selected_single = benchmark_cpu_direct_workload(
                    source_positions,
                    target_positions,
                    moment_states,
                    source_identities,
                    1,
                    options.samples
                );
                selected_repeated = benchmark_cpu_direct_workload(
                    source_positions,
                    target_positions,
                    moment_states,
                    source_identities,
                    repeated_evaluation_count,
                    options.samples
                );
            } else {
                selected_single = benchmark_fmm_workload(
                    source_positions,
                    target_positions,
                    moment_states,
                    source_identities,
                    selected_options,
                    1,
                    options.samples
                );
                selected_repeated = benchmark_fmm_workload(
                    source_positions,
                    target_positions,
                    moment_states,
                    source_identities,
                    selected_options,
                    repeated_evaluation_count,
                    options.samples
                );
            }
            std::cerr
                << "Workload median totals (1 evaluation / 10 evaluations):\n"
                << "  " << options.backend << ": "
                << selected_single.total_seconds << " s / "
                << selected_repeated.total_seconds << " s\n";
        }

        std::unique_ptr<UniformFmm> fmm;
        std::unique_ptr<CudaFmmPlan> cuda_direct_plan;
        const auto setup_start = Clock::now();
        if (selected_backend == BenchmarkBackend::CudaDirect) {
            cuda_direct_plan = std::make_unique<CudaFmmPlan>(
                source_positions,
                target_positions
            );
        } else if (selected_backend != BenchmarkBackend::CpuDirect) {
            fmm = std::make_unique<UniformFmm>(
                source_positions,
                target_positions,
                selected_options
            );
        }
        const double setup_seconds = std::chrono::duration<double>(
            Clock::now() - setup_start
        ).count();
        std::vector<PotentialField> result(target_positions.size());
        EvaluationTimings direct_timings;
        const auto evaluate_selected = [&](const std::span<const Vec3> moments) {
            if (fmm) {
                fmm->evaluate_into(
                    moments,
                    result,
                    OutputFlags::Field,
                    source_identities
                );
                return;
            }
            if (selected_backend == BenchmarkBackend::CpuDirect) {
                const auto evaluation_start = Clock::now();
                result = direct_p2p_reference(
                    target_positions,
                    source_positions,
                    moments,
                    OutputFlags::Field,
                    source_identities
                );
                const double elapsed = std::chrono::duration<double>(
                    Clock::now() - evaluation_start
                ).count();
                direct_timings.p2p.add(elapsed);
                direct_timings.total.add(elapsed);
                ++direct_timings.evaluations;
                return;
            }
            const auto evaluation_start = Clock::now();
            cuda_direct_plan->evaluate(
                moments,
                result,
                OutputFlags::Field,
                source_identities
            );
            const auto& device = cuda_direct_plan->evaluation_timings();
            direct_timings.cuda_h2d.add(device.h2d_seconds);
            direct_timings.cuda_kernel.add(device.kernel_seconds);
            direct_timings.cuda_d2h.add(device.d2h_seconds);
            direct_timings.total.add(std::chrono::duration<double>(
                Clock::now() - evaluation_start
            ).count());
            ++direct_timings.evaluations;
        };
        if (options.warmups > 0) {
            std::cerr << "Warm-up evaluations: " << options.warmups << "\n";
        }
        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            evaluate_selected(moment_states[static_cast<std::size_t>(warmup)]);
        }

        if (fmm) {
            fmm->reset_timings();
        }
        direct_timings = {};
        std::vector<double> sample_seconds;
        sample_seconds.reserve(static_cast<std::size_t>(options.samples));
        for (int sample = 0; sample < options.samples; ++sample) {
            const auto start = Clock::now();
            for (int evaluation = 0; evaluation < options.evaluations;
                ++evaluation) {
                const int state = options.warmups + evaluation;
                evaluate_selected(
                    moment_states[static_cast<std::size_t>(state)]
                );
            }
            sample_seconds.push_back(
                std::chrono::duration<double>(Clock::now() - start).count() /
                static_cast<double>(options.evaluations)
            );

            const int completed_samples = sample + 1;
            const int completed_evaluations =
                completed_samples * options.evaluations;
            const int total_evaluations =
                options.samples * options.evaluations;
            std::cerr << "\rSamples "
                      << progress_bar(completed_samples, options.samples)
                      << " " << completed_samples << "/" << options.samples
                      << "; timed evaluations " << completed_evaluations
                      << "/" << total_evaluations << std::flush;
        }
        std::cerr << "\n";

        const EvaluationTimings timing = fmm
            ? fmm->aggregate_timings()
            : direct_timings;

        double direct_seconds = std::numeric_limits<double>::quiet_NaN();
        ErrorMetrics metrics;
        if (options.direct) {
            if (selected_backend == BenchmarkBackend::CpuDirect) {
                // The selected algorithm is already the CPU reference. Reusing
                // its median avoids timing the same all-to-all kernel twice.
                direct_seconds = median(sample_seconds);
                std::cerr << "CPU direct backend is the all-to-all reference; "
                             "skipping duplicate reference evaluation.\n";
            } else {
                std::cerr << "Computing direct all-to-all reference..."
                          << std::flush;
                const auto direct_start = Clock::now();
                const auto reference = direct_p2p_reference(
                    target_positions, source_positions,
                    moment_states[static_cast<std::size_t>(options.warmups)],
                    OutputFlags::Field, source_identities
                );
                direct_seconds = std::chrono::duration<double>(
                    Clock::now() - direct_start
                ).count();
                std::cerr << " done (" << direct_seconds << " s)\n";
                evaluate_selected(
                    moment_states[static_cast<std::size_t>(options.warmups)]
                );
                std::vector<Vec3> approximate_fields(result.size());
                std::vector<Vec3> reference_fields(reference.size());
                for (std::size_t index = 0; index < result.size(); ++index) {
                    approximate_fields[index] = result[index].H;
                    reference_fields[index] = reference[index].H;
                }
                metrics = compute_error_metrics(
                    approximate_fields,
                    reference_fields
                );
            }
        }

        std::uint64_t m2l_translations = 0;
        std::uint64_t near_pairs = 0;
        if (fmm) {
            for (const TreeNode& node : fmm->tree().nodes()) {
                if (node.target_count() == 0) {
                    continue;
                }
                for (const int source_index : node.list2) {
                    if (fmm->tree().nodes()[
                            static_cast<std::size_t>(source_index)
                        ].source_count() > 0) {
                        ++m2l_translations;
                    }
                }
                if (!node.is_leaf()) {
                    continue;
                }
                for (const int source_index : node.list1) {
                    near_pairs += node.target_count() *
                        fmm->tree().nodes()[
                            static_cast<std::size_t>(source_index)
                        ].source_count();
                }
            }
        } else {
            near_pairs = static_cast<std::uint64_t>(options.targets) *
                static_cast<std::uint64_t>(std::max(0, options.sources - 1));
        }

        const TreeBuildTimings tree = fmm
            ? fmm->tree().build_timings()
            : TreeBuildTimings{};
        const StaticPlanStatistics static_plan = fmm
            ? fmm->static_plan_statistics()
            : StaticPlanStatistics{};
        const double evaluation_median = median(sample_seconds);
        const double evaluation_mean = mean(sample_seconds);
        const double timed_calls = static_cast<double>(timing.evaluations);
        const auto phase_mean = [timed_calls](const PhaseTiming& phase) {
            return timed_calls > 0.0 ? phase.total_seconds / timed_calls : 0.0;
        };
        const bool static_matrix =
            selected_backend == BenchmarkBackend::CpuStaticMatrix ||
            selected_backend == BenchmarkBackend::CpuStaticMatrixMkl;
        const std::string m2l_strategy = static_matrix
            ? "cached-dense-m2l-matrix-per-transfer-class"
            : "not-applicable-direct-p2p";
        const std::string multiply_backend = static_multiply_backend(
            selected_backend
        );
        const std::string multiply_version =
            selected_backend == BenchmarkBackend::CpuStaticMatrixMkl
            ? mkl_version()
            : "not_applicable";
        const std::size_t total_nodes = fmm ? fmm->tree().nodes().size() : 0;
        const std::size_t occupied_source_leaves = fmm
            ? fmm->tree().occupied_source_leaves().size()
            : 0;
        const std::size_t occupied_target_leaves = fmm
            ? fmm->tree().occupied_target_leaves().size()
            : 0;
        const CudaPlanStatistics cuda_statistics = cuda_direct_plan
            ? cuda_direct_plan->statistics()
            : CudaPlanStatistics{};

        std::ostream* output_stream = &std::cout;
        std::ofstream output_file;
        if (!options.output.empty()) {
            output_file.open(options.output);
            output_stream = &output_file;
        }
        std::ostream& out = *output_stream;
        out << "sources,targets,depth,order,threads,seed,evaluations,samples,"
               "execution_backend,cuda_compiled,cuda_available,cuda_device,"
               "one_mkl_available,"
               "compiler,compiler_version,build_type,openmp_status,openmp_version,"
               "fmm_setup_seconds,tree_total,root_bounds,node_construction,topology,source_morton,"
               "source_sorting,target_morton,target_sorting,ranges,interaction_lists,"
               "evaluation_median,evaluation_mean,evaluations_per_second,"
               "amortised_seconds,moment_permutation,multipole_reset,p2m,m2m,"
               "local_reset,l2l,m2l,m2l_gather,m2l_multiply,m2l_scatter,"
               "l2p,p2p,result_unpermutation,cuda_h2d,"
               "cuda_kernel,cuda_d2h,direct_seconds,"
               "mean_relative_error,rms_relative_error,max_relative_error,total_nodes,"
               "occupied_source_leaves,occupied_target_leaves,m2l_translations,"
               "near_field_pairs,m2l_strategy,static_multiply_backend,mkl_version,"
               "static_plan_seconds,cached_m2l_matrices,cached_m2l_matrix_bytes,"
               "static_interaction_bytes,static_scratch_bytes,static_plan_bytes,"
               "workload_1_creation_median,workload_1_evaluation_median,"
               "workload_1_total_median,workload_10_creation_median,"
               "workload_10_evaluation_median,workload_10_total_median,"
               "cuda_setup_h2d_bytes,"
               "cuda_evaluation_h2d_bytes,cuda_evaluation_d2h_bytes,"
               "cuda_persistent_device_bytes\n";
        const char* build_type =
#ifdef NDEBUG
            "Release";
#else
            "Debug";
#endif
        out << options.sources << ',' << options.targets << ',' << options.depth
            << ',' << options.order << ',' << thread_count << ',' << options.seed
            << ',' << options.evaluations << ',' << options.samples << ','
            << options.backend << ',' << (cuda_compiled() ? 1 : 0) << ','
            << (cuda_available() ? 1 : 0) << ",\""
            << (cuda_available() ? cuda_device_description() : std::string{})
            << "\"," << (one_mkl_available() ? 1 : 0) << ','
            << compiler_name() << ",\"" << compiler_version() << "\","
            << build_type
            << ',' << openmp_status << ',' << openmp_version << ','
            << setup_seconds << ',' << tree.total.total_seconds << ','
            << tree.root_bounds.total_seconds << ','
            << tree.node_construction.total_seconds << ','
            << tree.topology.total_seconds << ','
            << tree.source_morton.total_seconds << ','
            << tree.source_sorting.total_seconds << ','
            << tree.target_morton.total_seconds << ','
            << tree.target_sorting.total_seconds << ','
            << tree.ranges.total_seconds << ','
            << tree.interaction_lists.total_seconds << ','
            << evaluation_median << ',' << evaluation_mean << ','
            << 1.0 / evaluation_median << ','
            << evaluation_median + setup_seconds /
                static_cast<double>(options.evaluations) << ','
            << phase_mean(timing.moment_permutation) << ','
            << phase_mean(timing.multipole_reset) << ','
            << phase_mean(timing.p2m) << ',' << phase_mean(timing.m2m) << ','
            << phase_mean(timing.local_reset) << ',' << phase_mean(timing.l2l)
            << ',' << phase_mean(timing.m2l)
            << ',' << phase_mean(timing.m2l_gather)
            << ',' << phase_mean(timing.m2l_multiply)
            << ',' << phase_mean(timing.m2l_scatter)
            << ',' << phase_mean(timing.l2p)
            << ',' << phase_mean(timing.p2p) << ','
            << phase_mean(timing.result_unpermutation) << ','
            << phase_mean(timing.cuda_h2d) << ','
            << phase_mean(timing.cuda_kernel) << ','
            << phase_mean(timing.cuda_d2h) << ',' << direct_seconds
            << ',' << metrics.mean_relative_error << ','
            << metrics.rms_relative_error << ',' << metrics.max_relative_error
            << ',' << total_nodes << ','
            << occupied_source_leaves << ','
            << occupied_target_leaves << ','
            << m2l_translations << ',' << near_pairs << ','
            << m2l_strategy << ',' << multiply_backend << ',' << multiply_version
            << ',' << static_plan.total.total_seconds
            << ',' << static_plan.transfer_classes
            << ',' << static_plan.operator_bytes
            << ',' << static_plan.interaction_bytes
            << ',' << static_plan.scratch_bytes
            << ',' << static_plan.total_bytes();
        write_workload(out, selected_single);
        write_workload(out, selected_repeated);
        out << ',' << cuda_statistics.setup_h2d_bytes
            << ',' << cuda_statistics.evaluation_h2d_bytes
            << ',' << cuda_statistics.evaluation_d2h_bytes
            << ',' << cuda_statistics.persistent_device_bytes << '\n';

        if (!options.output.empty()) {
            std::cout << "Wrote " << options.output << "\n";
        }
        std::cerr << "compiler=" << compiler_name()
                  << " backend=" << options.backend
                  << " openmp=" << openmp_status
                  << " threads=" << thread_count
                  << " median_ms=" << evaluation_median * 1.0e3 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_uniform_fmm: " << error.what() << '\n';
        return 2;
    }
}
