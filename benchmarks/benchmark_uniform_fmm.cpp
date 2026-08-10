// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
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
    std::string output{};
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
        else if (key == "--output") options.output = value;
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (options.sources < 0 || options.targets < 0 || options.depth < 0 ||
        options.order < 0 || options.evaluations < 1 || options.samples < 1) {
        throw std::invalid_argument("Counts, depth, order, and samples are invalid");
    }
    return options;
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

} // namespace

int main(int argc, char** argv)
{
    using namespace cdfmm;
    try {
        const Options options = parse_options(argc, argv);
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

        std::mt19937 generator(options.seed);
        std::uniform_real_distribution<double> distribution(-0.95, 0.95);
        std::vector<Vec3> source_positions(
            static_cast<std::size_t>(options.sources)
        );
        std::vector<Vec3> target_positions(
            static_cast<std::size_t>(options.targets)
        );
        for (Vec3& position : source_positions) {
            position = {distribution(generator), distribution(generator),
                        distribution(generator)};
        }
        for (Vec3& position : target_positions) {
            position = {distribution(generator), distribution(generator),
                        distribution(generator)};
        }

        const int state_count = options.warmups + options.evaluations;
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

        const auto setup_start = Clock::now();
        UniformFmm fmm(source_positions, target_positions, fmm_options);
        const double setup_seconds = std::chrono::duration<double>(
            Clock::now() - setup_start
        ).count();
        std::vector<PotentialField> result(target_positions.size());
        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            fmm.evaluate_into(moment_states[static_cast<std::size_t>(warmup)],
                              result);
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
                                  result);
            }
            sample_seconds.push_back(
                std::chrono::duration<double>(Clock::now() - start).count() /
                static_cast<double>(options.evaluations)
            );
        }

        double direct_seconds = std::numeric_limits<double>::quiet_NaN();
        ErrorMetrics metrics;
        if (options.direct) {
            const auto direct_start = Clock::now();
            const auto reference = direct_p2p_reference(
                target_positions, source_positions,
                moment_states[static_cast<std::size_t>(options.warmups)]
            );
            direct_seconds = std::chrono::duration<double>(
                Clock::now() - direct_start
            ).count();
            fmm.evaluate_into(
                moment_states[static_cast<std::size_t>(options.warmups)], result
            );
            std::vector<Vec3> approximate_fields(result.size());
            std::vector<Vec3> reference_fields(reference.size());
            for (std::size_t index = 0; index < result.size(); ++index) {
                approximate_fields[index] = result[index].H;
                reference_fields[index] = reference[index].H;
            }
            metrics = compute_error_metrics(approximate_fields, reference_fields);
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
               "compiler,compiler_version,build_type,openmp_status,openmp_version,"
               "tree_total,root_bounds,node_construction,topology,source_morton,"
               "source_sorting,target_morton,target_sorting,ranges,interaction_lists,"
               "evaluation_median,evaluation_mean,evaluations_per_second,"
               "amortised_seconds,moment_permutation,multipole_reset,p2m,m2m,"
               "local_reset,l2l,m2l,l2p,p2p,result_unpermutation,direct_seconds,"
               "mean_relative_error,rms_relative_error,max_relative_error,total_nodes,"
               "occupied_source_leaves,occupied_target_leaves,m2l_translations,"
               "near_field_pairs\n";
        const char* build_type =
#ifdef NDEBUG
            "Release";
#else
            "Debug";
#endif
        out << options.sources << ',' << options.targets << ',' << options.depth
            << ',' << options.order << ',' << thread_count << ',' << options.seed
            << ',' << options.evaluations << ',' << options.samples << ','
            << compiler_name() << ",\"" << compiler_version() << "\","
            << build_type
            << ',' << openmp_status << ',' << openmp_version << ','
            << tree.total.total_seconds << ',' << tree.root_bounds.total_seconds << ','
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
            << phase_mean(timing.result_unpermutation) << ',' << direct_seconds
            << ',' << metrics.mean_relative_error << ','
            << metrics.rms_relative_error << ',' << metrics.max_relative_error
            << ',' << fmm.tree().nodes().size() << ','
            << fmm.tree().occupied_source_leaves().size() << ','
            << fmm.tree().occupied_target_leaves().size() << ','
            << m2l_translations << ',' << near_pairs << '\n';

        if (!options.output.empty()) {
            std::cout << "Wrote " << options.output << "\n";
        }
        std::cerr << "compiler=" << compiler_name()
                  << " openmp=" << openmp_status
                  << " threads=" << thread_count
                  << " median_ms=" << evaluation_median * 1.0e3 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_uniform_fmm: " << error.what() << '\n';
        return 2;
    }
}
