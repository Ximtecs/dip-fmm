// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

#ifdef CDFMM_USE_MKL
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
    std::string backend{"cpu-static"};
    std::string output{};
};

struct WorkloadTiming {
    double construction_seconds{0.0};
    double evaluation_seconds{0.0};
    double total_seconds{0.0};
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

cdfmm::ExecutionBackend execution_backend(const std::string& name)
{
    using cdfmm::ExecutionBackend;

    if (name == "cpu-reference") {
        return ExecutionBackend::CpuReference;
    }
    if (name == "cpu-static") {
        return ExecutionBackend::CpuStatic;
    }
    if (name == "cuda-farfield-fmm") {
        if (!cdfmm::cuda_farfield_available()) {
            throw std::runtime_error("CUDA far-field FMM is unavailable");
        }
        return ExecutionBackend::CudaFarField;
    }
    throw std::invalid_argument(
        "Unknown backend '" + name +
        "'; expected cpu-reference, cpu-static, or cuda-farfield-fmm"
    );
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

WorkloadTiming benchmark_p2p_workload(
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
                throw std::runtime_error("Direct P2P returned an invalid result");
            }
        }
        const double evaluation_seconds = std::chrono::duration<double>(
            Clock::now() - evaluation_start
        ).count();
        // Direct P2P has no reusable geometry object, so construction is zero.
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

std::string static_multiply_backend()
{
#ifdef CDFMM_USE_MKL
    return "oneMKL";
#else
    return "portable";
#endif
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

} // namespace

int main(int argc, char** argv)
{
    using namespace cdfmm;
    try {
        const Options options = parse_options(argc, argv);
        if (options.cuda_status) {
            std::cout << "cuda_compiled=" << (cuda_compiled() ? 1 : 0) << '\n'
                      << "cuda_available=" << (cuda_available() ? 1 : 0) << '\n'
                      << "cuda_device=";
            if (cuda_available()) {
                std::cout << cuda_device_description();
            }
            std::cout << '\n';
            return 0;
        }
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
        selected_options.backend = execution_backend(options.backend);

        UniformFmmOptions reference_options = fmm_options;
        reference_options.m2l_backend = M2LBackend::Reference;
        reference_options.backend = ExecutionBackend::CpuReference;
        UniformFmmOptions static_options = fmm_options;
        static_options.m2l_backend = M2LBackend::Static;
        static_options.backend = ExecutionBackend::CpuStatic;

        const double missing = std::numeric_limits<double>::quiet_NaN();
        WorkloadTiming p2p_single{0.0, missing, missing};
        WorkloadTiming p2p_repeated{0.0, missing, missing};
        WorkloadTiming reference_single{missing, missing, missing};
        WorkloadTiming reference_repeated{missing, missing, missing};
        WorkloadTiming static_single{missing, missing, missing};
        WorkloadTiming static_repeated{missing, missing, missing};
        if (options.workload_comparison) {
            std::cerr << "Benchmarking creation/evaluation workloads...\n";
            if (options.direct) {
                p2p_single = benchmark_p2p_workload(
                    source_positions,
                    target_positions,
                    moment_states,
                    source_identities,
                    1,
                    options.samples
                );
                p2p_repeated = benchmark_p2p_workload(
                    source_positions,
                    target_positions,
                    moment_states,
                    source_identities,
                    repeated_evaluation_count,
                    options.samples
                );
            }
            reference_single = benchmark_fmm_workload(
                source_positions,
                target_positions,
                moment_states,
                source_identities,
                reference_options,
                1,
                options.samples
            );
            reference_repeated = benchmark_fmm_workload(
                source_positions,
                target_positions,
                moment_states,
                source_identities,
                reference_options,
                repeated_evaluation_count,
                options.samples
            );
            static_single = benchmark_fmm_workload(
                source_positions,
                target_positions,
                moment_states,
                source_identities,
                static_options,
                1,
                options.samples
            );
            static_repeated = benchmark_fmm_workload(
                source_positions,
                target_positions,
                moment_states,
                source_identities,
                static_options,
                repeated_evaluation_count,
                options.samples
            );
            std::cerr
                << "Workload median totals (1 evaluation / 10 evaluations):\n"
                << "  direct all-to-all P2P: "
                << p2p_single.total_seconds << " s / "
                << p2p_repeated.total_seconds << " s\n"
                << "  reference FMM: "
                << reference_single.total_seconds << " s / "
                << reference_repeated.total_seconds << " s\n"
                << "  static FMM (" << static_multiply_backend() << "): "
                << static_single.total_seconds << " s / "
                << static_repeated.total_seconds << " s\n";
        }

        const auto setup_start = Clock::now();
        UniformFmm fmm(source_positions, target_positions, selected_options);
        const double setup_seconds = std::chrono::duration<double>(
            Clock::now() - setup_start
        ).count();
        std::vector<PotentialField> result(target_positions.size());
        if (options.warmups > 0) {
            std::cerr << "Warm-up evaluations: " << options.warmups << "\n";
        }
        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            fmm.evaluate_into(moment_states[static_cast<std::size_t>(warmup)],
                              result, OutputFlags::Field, source_identities);
        }

        fmm.reset_timings();
        std::vector<double> sample_seconds;
        sample_seconds.reserve(static_cast<std::size_t>(options.samples));
        for (int sample = 0; sample < options.samples; ++sample) {
            const auto start = Clock::now();
            for (int evaluation = 0; evaluation < options.evaluations;
                 ++evaluation) {
                const int state = options.warmups + evaluation;
                fmm.evaluate_into(moment_states[static_cast<std::size_t>(state)],
                                  result, OutputFlags::Field,
                                  source_identities);
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

        double direct_seconds = std::numeric_limits<double>::quiet_NaN();
        ErrorMetrics metrics;
        ErrorMetrics reference_metrics;
        if (options.direct) {
            std::cerr << "Computing direct all-to-all reference..." << std::flush;
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
            fmm.evaluate_into(
                moment_states[static_cast<std::size_t>(options.warmups)], result,
                OutputFlags::Field, source_identities
            );
            std::vector<Vec3> approximate_fields(result.size());
            std::vector<Vec3> reference_fields(reference.size());
            for (std::size_t index = 0; index < result.size(); ++index) {
                approximate_fields[index] = result[index].H;
                reference_fields[index] = reference[index].H;
            }
            metrics = compute_error_metrics(approximate_fields, reference_fields);

            if (options.workload_comparison) {
                UniformFmm reference_fmm(
                    source_positions,
                    target_positions,
                    reference_options
                );
                const auto reference_fmm_values = reference_fmm.evaluate(
                    moment_states[static_cast<std::size_t>(options.warmups)],
                    OutputFlags::Field,
                    source_identities
                );
                std::vector<Vec3> reference_fmm_fields(
                    reference_fmm_values.size()
                );
                for (std::size_t index = 0;
                     index < reference_fmm_values.size(); ++index) {
                    reference_fmm_fields[index] = reference_fmm_values[index].H;
                }
                reference_metrics = compute_error_metrics(
                    reference_fmm_fields,
                    reference_fields
                );
            }
        }

        std::uint64_t m2l_translations = 0;
        std::uint64_t near_pairs = 0;
        for (const TreeNode& node : fmm.tree().nodes()) {
            if (node.target_count() == 0) continue;
            for (const int source_index : node.list2) {
                if (fmm.tree().nodes()[static_cast<std::size_t>(source_index)]
                        .source_count() > 0) {
                    ++m2l_translations;
                }
            }
            if (!node.is_leaf()) continue;
            for (const int source_index : node.list1) {
                near_pairs += node.target_count() *
                    fmm.tree().nodes()[static_cast<std::size_t>(source_index)]
                        .source_count();
            }
        }

        const auto& tree = fmm.tree().build_timings();
        const auto& timing = fmm.aggregate_timings();
        const double evaluation_median = median(sample_seconds);
        const double evaluation_mean = mean(sample_seconds);
        const double timed_calls = static_cast<double>(timing.evaluations);
        const auto phase_mean = [timed_calls](const PhaseTiming& phase) {
            return timed_calls > 0.0 ? phase.total_seconds / timed_calls : 0.0;
        };

        std::ostream* output_stream = &std::cout;
        std::ofstream output_file;
        if (!options.output.empty()) {
            output_file.open(options.output);
            output_stream = &output_file;
        }
        std::ostream& out = *output_stream;
        out << "sources,targets,depth,order,threads,seed,evaluations,samples,"
               "execution_backend,cuda_compiled,cuda_available,cuda_device,"
               "compiler,compiler_version,build_type,openmp_status,openmp_version,"
               "fmm_setup_seconds,tree_total,root_bounds,node_construction,topology,source_morton,"
               "source_sorting,target_morton,target_sorting,ranges,interaction_lists,"
               "evaluation_median,evaluation_mean,evaluations_per_second,"
               "amortised_seconds,moment_permutation,multipole_reset,p2m,m2m,"
               "local_reset,l2l,m2l,l2p,p2p,result_unpermutation,cuda_h2d,"
               "cuda_kernel,cuda_d2h,direct_seconds,"
               "mean_relative_error,rms_relative_error,max_relative_error,total_nodes,"
               "occupied_source_leaves,occupied_target_leaves,m2l_translations,"
               "near_field_pairs,static_multiply_backend,mkl_version,"
               "p2p_1_creation_median,p2p_1_evaluation_median,p2p_1_total_median,"
               "p2p_10_creation_median,p2p_10_evaluation_median,p2p_10_total_median,"
               "reference_1_creation_median,reference_1_evaluation_median,"
               "reference_1_total_median,reference_10_creation_median,"
               "reference_10_evaluation_median,reference_10_total_median,"
               "static_1_creation_median,static_1_evaluation_median,"
               "static_1_total_median,static_10_creation_median,"
               "static_10_evaluation_median,static_10_total_median,"
               "reference_mean_relative_error,reference_rms_relative_error,"
               "reference_max_relative_error,cuda_setup_h2d_bytes,"
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
            << "\","
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
            << ',' << phase_mean(timing.m2l) << ',' << phase_mean(timing.l2p)
            << ',' << phase_mean(timing.p2p) << ','
            << phase_mean(timing.result_unpermutation) << ','
            << phase_mean(timing.cuda_h2d) << ','
            << phase_mean(timing.cuda_kernel) << ','
            << phase_mean(timing.cuda_d2h) << ',' << direct_seconds
            << ',' << metrics.mean_relative_error << ','
            << metrics.rms_relative_error << ',' << metrics.max_relative_error
            << ',' << fmm.tree().nodes().size() << ','
            << fmm.tree().occupied_source_leaves().size() << ','
            << fmm.tree().occupied_target_leaves().size() << ','
            << m2l_translations << ',' << near_pairs << ','
            << static_multiply_backend() << ',' << mkl_version();
        write_workload(out, p2p_single);
        write_workload(out, p2p_repeated);
        write_workload(out, reference_single);
        write_workload(out, reference_repeated);
        write_workload(out, static_single);
        write_workload(out, static_repeated);
        out << ',' << reference_metrics.mean_relative_error
            << ',' << reference_metrics.rms_relative_error
            << ',' << reference_metrics.max_relative_error;
        const auto& cuda_statistics = fmm.cuda_plan_statistics();
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
