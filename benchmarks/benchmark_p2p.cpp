// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/static_operators.hpp"
#include "cdfmm/uniform_tree.hpp"

#if defined(CDFMM_ENABLE_CUDA)
#include "cdfmm/cuda_p2p.hpp"
#include "cdfmm/uniform_fmm.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(CDFMM_USE_OPENMP)
#include <omp.h>
#endif

#if defined(CDFMM_USE_MKL)
#include <mkl_spblas.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  int depth{2};
  int occupancy{8};
  int particles{0};
  int evaluations{20};
  int signed_target_tile{32};
  bool irregular{false};
  bool sweep{false};
  bool cuda{false};
  int regular_grid_s{0};
  int cuda_target_tile{128};
  int cuda_targets_per_thread{1};
};

#if defined(CDFMM_USE_MKL)
class MklBsrPlan {
public:
  MklBsrPlan(cdfmm::StaticP2PBsrPlan &plan, const int source_count,
             const int evaluations)
      : row_offsets_(plan.row_offsets.begin(), plan.row_offsets.end()),
        source_indices_(plan.source_indices.begin(), plan.source_indices.end()),
        value_count_(plan.values.size()) {
    matrix_descr_.type = SPARSE_MATRIX_TYPE_GENERAL;
    check(mkl_sparse_d_create_bsr(
              &matrix_, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR,
              static_cast<MKL_INT>(plan.target_count),
              static_cast<MKL_INT>(source_count), static_cast<MKL_INT>(3),
              row_offsets_.data(), row_offsets_.data() + 1,
              source_indices_.data(), plan.values.data()),
          "mkl_sparse_d_create_bsr");
    check(mkl_sparse_set_mv_hint(matrix_, SPARSE_OPERATION_NON_TRANSPOSE,
                                 matrix_descr_,
                                 static_cast<MKL_INT>(evaluations)),
          "mkl_sparse_set_mv_hint");
    check(mkl_sparse_optimize(matrix_), "mkl_sparse_optimize");
  }

  ~MklBsrPlan() {
    if (matrix_ != nullptr) {
      mkl_sparse_destroy(matrix_);
    }
  }

  MklBsrPlan(const MklBsrPlan &) = delete;
  MklBsrPlan &operator=(const MklBsrPlan &) = delete;

  [[nodiscard]] cdfmm::StaticP2PMemory memory() const noexcept {
    cdfmm::StaticP2PMemory result;
    result.tensor_bytes = value_count_ * sizeof(double);
    result.index_bytes = source_indices_.size() * sizeof(MKL_INT);
    result.row_metadata_bytes = row_offsets_.size() * sizeof(MKL_INT);
    return result;
  }

  void apply(const std::span<const cdfmm::Vec3> moments,
             const std::span<cdfmm::Vec3> fields) const {
    static_assert(sizeof(cdfmm::Vec3) == 3 * sizeof(double));
    check(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix_,
                          matrix_descr_,
                          reinterpret_cast<const double *>(moments.data()), 0.0,
                          reinterpret_cast<double *>(fields.data())),
          "mkl_sparse_d_mv");
  }

private:
  static void check(const sparse_status_t status, const char *operation) {
    if (status != SPARSE_STATUS_SUCCESS) {
      throw std::runtime_error(std::string(operation) + " failed");
    }
  }

  sparse_matrix_t matrix_{nullptr};
  matrix_descr matrix_descr_{};
  std::vector<MKL_INT> row_offsets_{};
  std::vector<MKL_INT> source_indices_{};
  std::size_t value_count_{0};
};
#endif

struct GeometryPlan {
  cdfmm::UniformTree tree;
  cdfmm::StaticP2POperator canonical;
  cdfmm::StaticP2PCompactPlan compact;
  cdfmm::StaticP2PLeafPlan leaf;
  cdfmm::StaticP2PTensorDictionaryPlan tensor_dictionary;
  cdfmm::StaticP2PSignedTensorDictionaryPlan signed_tensor_dictionary;
  cdfmm::FloatStaticP2PCompactPlan compact_float;
  cdfmm::FloatStaticP2PSignedTensorDictionaryPlan signed_tensor_dictionary_float;
  cdfmm::StaticP2PBsrPlan bsr;
#if defined(CDFMM_USE_MKL)
  std::unique_ptr<MklBsrPlan> mkl_bsr;
#endif
  std::vector<cdfmm::Vec3> moments;
  std::vector<cdfmm::FloatVec3> moments_float;
  std::vector<int> identities;
  double canonical_setup_seconds{0.0};
  double compact_setup_seconds{0.0};
  double leaf_setup_seconds{0.0};
  double tensor_dictionary_setup_seconds{0.0};
  double signed_tensor_dictionary_setup_seconds{0.0};
  double bsr_setup_seconds{0.0};
  double mkl_setup_seconds{0.0};
};

struct Result {
  std::string name;
  double seconds{0.0};
  cdfmm::StaticP2PMemory memory{};
  double checksum{0.0};
};

[[nodiscard]] std::array<double, 2> signed_work_balance(
    const cdfmm::StaticP2PSignedTensorDictionaryPlan &plan,
    const int thread_count) {
  std::vector<double> costs(plan.tile_leaf_indices.size());
  for (std::size_t work = 0; work < costs.size(); ++work) {
    const int leaf = plan.tile_leaf_indices[work];
    const int local_begin = plan.tile_target_offsets[work];
    const int lanes = std::min(
        plan.target_tile_size,
        plan.target_counts[static_cast<std::size_t>(leaf)] - local_begin);
    std::size_t sources = 0;
    for (int block = plan.leaf_row_offsets[static_cast<std::size_t>(leaf)];
         block < plan.leaf_row_offsets[static_cast<std::size_t>(leaf) + 1];
         ++block) {
      sources += static_cast<std::size_t>(
          plan.blocks[static_cast<std::size_t>(block)].source_count);
    }
    costs[work] = static_cast<double>(lanes) * sources;
  }
  const double total = std::accumulate(costs.begin(), costs.end(), 0.0);
  const double mean = costs.empty() ? 0.0 : total / costs.size();
  const double item_ratio = mean == 0.0
      ? 0.0
      : *std::max_element(costs.begin(), costs.end()) / mean;
  const int workers = std::max(1, std::min(thread_count,
                                           static_cast<int>(costs.size())));
  std::vector<double> thread_costs(static_cast<std::size_t>(workers));
  const int quotient = static_cast<int>(costs.size()) / workers;
  const int remainder = static_cast<int>(costs.size()) % workers;
  int begin = 0;
  for (int worker = 0; worker < workers; ++worker) {
    const int count = quotient + (worker < remainder ? 1 : 0);
    thread_costs[static_cast<std::size_t>(worker)] = std::accumulate(
        costs.begin() + begin, costs.begin() + begin + count, 0.0);
    begin += count;
  }
  const double thread_mean = total / workers;
  const double thread_ratio = thread_mean == 0.0
      ? 0.0
      : *std::max_element(thread_costs.begin(), thread_costs.end()) /
            thread_mean;
  return {item_ratio, thread_ratio};
}

[[nodiscard]] double elapsed(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

[[nodiscard]] int parse_integer(const char *value, const char *option) {
  const int parsed = std::atoi(value);
  if (parsed <= 0) {
    throw std::invalid_argument(std::string(option) + " must be positive");
  }
  return parsed;
}

[[nodiscard]] Options parse_options(const int argc, char **argv) {
  Options options;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string_view option(argv[argument]);
    if (option == "--irregular") {
      options.irregular = true;
    } else if (option == "--sweep") {
      options.sweep = true;
    } else if (option == "--cuda") {
      options.cuda = true;
    } else if (argument + 1 < argc && option == "--depth") {
      options.depth = parse_integer(argv[++argument], "--depth");
    } else if (argument + 1 < argc && option == "--occupancy") {
      options.occupancy = parse_integer(argv[++argument], "--occupancy");
    } else if (argument + 1 < argc && option == "--particles") {
      options.particles = parse_integer(argv[++argument], "--particles");
    } else if (argument + 1 < argc && option == "--evaluations") {
      options.evaluations = parse_integer(argv[++argument], "--evaluations");
    } else if (argument + 1 < argc && option == "--signed-target-tile") {
      options.signed_target_tile =
          parse_integer(argv[++argument], "--signed-target-tile");
    } else if (argument + 1 < argc && option == "--regular-grid-s") {
      options.regular_grid_s =
          parse_integer(argv[++argument], "--regular-grid-s");
    } else if (argument + 1 < argc && option == "--cuda-target-tile") {
      options.cuda_target_tile =
          parse_integer(argv[++argument], "--cuda-target-tile");
    } else if (argument + 1 < argc && option == "--cuda-targets-per-thread") {
      options.cuda_targets_per_thread =
          parse_integer(argv[++argument], "--cuda-targets-per-thread");
    } else {
      throw std::invalid_argument("unknown or incomplete benchmark option");
    }
  }
  if (options.sweep && options.particles != 0) {
    throw std::invalid_argument("--particles cannot be combined with --sweep");
  }
  return options;
}

[[nodiscard]] std::vector<cdfmm::Vec3>
make_positions(const int depth, const int occupancy,
               const int requested_particles, const bool irregular,
               const int regular_grid_s) {
  const int boxes_per_axis = 1 << depth;
  const int leaf_count = boxes_per_axis * boxes_per_axis * boxes_per_axis;
  const int particle_count =
      requested_particles == 0 ? leaf_count * occupancy : requested_particles;
  std::vector<cdfmm::Vec3> positions;
  positions.reserve(static_cast<std::size_t>(particle_count));
  std::mt19937 generator(4919u + static_cast<unsigned>(particle_count) +
                         17u * static_cast<unsigned>(depth));

  if (regular_grid_s > 0) {
    if (depth != 2 || requested_particles != 0 ||
        occupancy != regular_grid_s * regular_grid_s * regular_grid_s) {
      throw std::invalid_argument(
          "--regular-grid-s requires depth 2 and occupancy s^3");
    }
    const int cells_per_axis = boxes_per_axis * regular_grid_s;
    positions.clear();
    positions.reserve(static_cast<std::size_t>(cells_per_axis) *
                      cells_per_axis * cells_per_axis);
    const double spacing = 2.0 / cells_per_axis;
    for (int iz = 0; iz < cells_per_axis; ++iz) {
      for (int iy = 0; iy < cells_per_axis; ++iy) {
        for (int ix = 0; ix < cells_per_axis; ++ix) {
          positions.push_back({-1.0 + (ix + 0.5) * spacing,
                               -1.0 + (iy + 0.5) * spacing,
                               -1.0 + (iz + 0.5) * spacing});
        }
      }
    }
    return positions;
  }

  if (irregular) {
    std::uniform_real_distribution<double> coordinate(-0.999, 0.999);
    for (int particle = 0; particle < particle_count; ++particle) {
      positions.push_back({coordinate(generator), coordinate(generator),
                           coordinate(generator)});
    }
    return positions;
  }

  std::uniform_real_distribution<double> local(-0.42, 0.42);
  const double leaf_width = 2.0 / boxes_per_axis;
  for (int iz = 0; iz < boxes_per_axis; ++iz) {
    for (int iy = 0; iy < boxes_per_axis; ++iy) {
      for (int ix = 0; ix < boxes_per_axis; ++ix) {
        const int leaf = (iz * boxes_per_axis + iy) * boxes_per_axis + ix;
        // Distribute an exact requested count as evenly as possible.
        // When particles are fewer than leaves this selects spatially
        // separated leaves rather than filling one corner first.
        const int leaf_begin = static_cast<int>(static_cast<long long>(leaf) *
                                                particle_count / leaf_count);
        const int leaf_end = static_cast<int>(static_cast<long long>(leaf + 1) *
                                              particle_count / leaf_count);
        const int leaf_occupancy = leaf_end - leaf_begin;
        const cdfmm::Vec3 centre{-1.0 + (ix + 0.5) * leaf_width,
                                 -1.0 + (iy + 0.5) * leaf_width,
                                 -1.0 + (iz + 0.5) * leaf_width};
        for (int particle = 0; particle < leaf_occupancy; ++particle) {
          positions.push_back({centre.x + local(generator) * leaf_width,
                               centre.y + local(generator) * leaf_width,
                               centre.z + local(generator) * leaf_width});
        }
      }
    }
  }
  return positions;
}

[[nodiscard]] GeometryPlan build_geometry(const int depth, const int occupancy,
                                          const int particle_count,
                                          const bool irregular,
                                          const int regular_grid_s,
                                          const int signed_target_tile) {
  const std::vector<cdfmm::Vec3> positions = make_positions(
      depth, occupancy, particle_count, irregular, regular_grid_s);
  cdfmm::UniformTreeOptions tree_options;
  tree_options.max_level = depth;
  tree_options.root_centre = cdfmm::Vec3{};
  tree_options.root_half_width = 1.0;
  GeometryPlan result{cdfmm::UniformTree(positions, positions, tree_options)};

  const auto nodes = result.tree.nodes();
  std::vector<std::array<int, 2>> interactions;
  std::vector<cdfmm::StaticP2PLeafPair> leaf_pairs;
  for (const int leaf_index : result.tree.occupied_target_leaves()) {
    const cdfmm::TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    for (const int neighbour_index : leaf.list1) {
      const cdfmm::TreeNode &neighbour =
          nodes[static_cast<std::size_t>(neighbour_index)];
      if (neighbour.source_count() == 0) {
        continue;
      }
      leaf_pairs.push_back({static_cast<int>(leaf.target_begin),
                            static_cast<int>(leaf.target_count()),
                            static_cast<int>(neighbour.source_begin),
                            static_cast<int>(neighbour.source_count())});
      for (std::size_t target = leaf.target_begin; target < leaf.target_end;
           ++target) {
        for (std::size_t source = neighbour.source_begin;
             source < neighbour.source_end; ++source) {
          interactions.push_back(
              {static_cast<int>(target), static_cast<int>(source)});
        }
      }
    }
  }

  auto start = Clock::now();
  result.canonical = cdfmm::build_static_p2p_operator(
      result.tree.sorted_target_positions(),
      result.tree.sorted_source_positions(), interactions);
  result.canonical_setup_seconds = elapsed(start);
  start = Clock::now();
  result.compact = cdfmm::build_static_p2p_compact_plan(result.canonical);
  result.compact_setup_seconds = elapsed(start);
  start = Clock::now();
  result.leaf = cdfmm::build_static_p2p_leaf_plan(result.canonical, leaf_pairs);
  result.leaf_setup_seconds = elapsed(start);
  result.identities.resize(positions.size());
  for (std::size_t index = 0; index < positions.size(); ++index) {
    result.identities[index] = static_cast<int>(index);
  }
  start = Clock::now();
  result.tensor_dictionary = cdfmm::build_static_p2p_tensor_dictionary_plan(
      result.canonical, leaf_pairs);
  result.tensor_dictionary_setup_seconds = elapsed(start);
  start = Clock::now();
  result.signed_tensor_dictionary =
      cdfmm::build_static_p2p_signed_tensor_dictionary_plan(
          result.canonical, leaf_pairs, result.identities, signed_target_tile);
  result.signed_tensor_dictionary_setup_seconds = elapsed(start);
  result.compact_float = cdfmm::quantise_static_p2p_compact_plan(result.compact);
  result.signed_tensor_dictionary_float =
      cdfmm::quantise_static_p2p_signed_tensor_dictionary_plan(
          result.signed_tensor_dictionary);
  start = Clock::now();
  result.bsr =
      cdfmm::build_static_p2p_bsr_plan(result.canonical, result.identities);
  result.bsr_setup_seconds = elapsed(start);
#if defined(CDFMM_USE_MKL)
  start = Clock::now();
  result.mkl_bsr = std::make_unique<MklBsrPlan>(
      result.bsr, result.canonical.source_count, 100);
  result.mkl_setup_seconds = elapsed(start);
#endif

  result.moments.resize(positions.size());
  std::mt19937 generator(813u);
  std::uniform_real_distribution<double> component(-1.0, 1.0);
  for (std::size_t index = 0; index < positions.size(); ++index) {
    result.moments[index] = {component(generator), component(generator),
                             component(generator)};
  }
  result.moments_float.reserve(result.moments.size());
  for (const cdfmm::Vec3 moment : result.moments) {
    result.moments_float.push_back(
        {static_cast<float>(moment.x), static_cast<float>(moment.y),
         static_cast<float>(moment.z)});
  }
  return result;
}

template <typename Apply>
[[nodiscard]] Result
measure(std::string name, const cdfmm::StaticP2PMemory memory,
        GeometryPlan &geometry, const int evaluations, Apply apply) {
  std::vector<cdfmm::Vec3> fields(geometry.moments.size());
  apply(fields);
  double best = std::numeric_limits<double>::max();
  for (int evaluation = 0; evaluation < evaluations; ++evaluation) {
    std::fill(fields.begin(), fields.end(), cdfmm::Vec3{});
    const auto start = Clock::now();
    apply(fields);
    best = std::min(best, elapsed(start));
  }
  double checksum = 0.0;
  for (const cdfmm::Vec3 field : fields) {
    checksum += field.x + field.y + field.z;
  }
  return {std::move(name), best, memory, checksum};
}

template <typename Apply>
[[nodiscard]] Result
measure_float(std::string name, const cdfmm::StaticP2PMemory memory,
              GeometryPlan &geometry, const int evaluations, Apply apply) {
  std::vector<cdfmm::FloatVec3> fields(geometry.moments_float.size());
  apply(fields);
  double best = std::numeric_limits<double>::max();
  for (int evaluation = 0; evaluation < evaluations; ++evaluation) {
    std::fill(fields.begin(), fields.end(), cdfmm::FloatVec3{});
    const auto start = Clock::now();
    apply(fields);
    best = std::min(best, elapsed(start));
  }
  double checksum = 0.0;
  for (const cdfmm::FloatVec3 field : fields) {
    checksum += static_cast<double>(field.x) + field.y + field.z;
  }
  return {std::move(name), best, memory, checksum};
}

void print_result(const Result &result, const std::size_t interactions,
                  const double baseline_seconds) {
  std::cout << result.name << ',' << std::setprecision(9) << result.seconds
            << ',' << static_cast<double>(interactions) / result.seconds << ','
            << baseline_seconds / result.seconds << ','
            << result.memory.tensor_bytes << ',' << result.memory.index_bytes
            << ',' << result.memory.row_metadata_bytes << ','
            << result.memory.leaf_metadata_bytes << ','
            << result.memory.scratch_bytes << ',' << result.memory.total_bytes()
            << ',' << result.checksum << '\n';
}

#if defined(CDFMM_ENABLE_CUDA)
struct CudaResult {
  std::string name;
  double setup_seconds{0.0};
  double h2d_seconds{0.0};
  double kernel_seconds{0.0};
  double d2h_seconds{0.0};
  double device_total_seconds{0.0};
  double host_total_seconds{0.0};
  cdfmm::CudaPlanStatistics statistics{};
  double checksum{0.0};
};

template <typename Factory>
[[nodiscard]] CudaResult measure_cuda(std::string name, GeometryPlan &geometry,
                                      const int evaluations, Factory factory) {
  CudaResult result;
  result.name = std::move(name);
  auto start = Clock::now();
  std::unique_ptr<cdfmm::CudaP2PPlan> plan = factory();
  result.setup_seconds = elapsed(start);

  std::vector<cdfmm::Vec3> fields(geometry.moments.size());
  plan->evaluate(geometry.moments, geometry.identities, fields);
  result.device_total_seconds = std::numeric_limits<double>::max();
  result.host_total_seconds = std::numeric_limits<double>::max();
  for (int evaluation = 0; evaluation < evaluations; ++evaluation) {
    geometry.moments[0].x += std::numeric_limits<double>::epsilon();
    start = Clock::now();
    plan->evaluate(geometry.moments, geometry.identities, fields);
    const double host_seconds = elapsed(start);
    const cdfmm::CudaEvaluationTimings &timing = plan->timings();
    const double device_seconds =
        timing.h2d_seconds + timing.kernel_seconds + timing.d2h_seconds;
    if (device_seconds < result.device_total_seconds) {
      result.h2d_seconds = timing.h2d_seconds;
      result.kernel_seconds = timing.kernel_seconds;
      result.d2h_seconds = timing.d2h_seconds;
      result.device_total_seconds = device_seconds;
    }
    result.host_total_seconds =
        std::min(result.host_total_seconds, host_seconds);
  }
  result.statistics = plan->statistics();
  for (const cdfmm::Vec3 field : fields) {
    result.checksum += field.x + field.y + field.z;
  }
  return result;
}

void print_cuda_result(const CudaResult &result, const std::size_t interactions,
                       const CudaResult &baseline) {
  const cdfmm::CudaPlanStatistics &memory = result.statistics;
  std::cout << result.name << ',' << std::setprecision(9)
            << result.setup_seconds << ',' << result.h2d_seconds << ','
            << result.kernel_seconds << ',' << result.d2h_seconds << ','
            << result.device_total_seconds << ',' << result.host_total_seconds
            << ',' << baseline.kernel_seconds / result.kernel_seconds << ','
            << baseline.device_total_seconds / result.device_total_seconds
            << ',' << static_cast<double>(interactions) / result.kernel_seconds
            << ',' << memory.p2p_tensor_bytes << ',' << memory.p2p_index_bytes
            << ',' << memory.p2p_row_metadata_bytes << ','
            << memory.p2p_leaf_metadata_bytes << ','
            << memory.p2p_identity_bytes << ',' << memory.p2p_scratch_bytes
            << ',' << memory.p2p_threads_per_block << ','
            << memory.persistent_device_bytes << ',' << result.checksum << '\n';
}

void run_cuda_case(GeometryPlan &geometry, const Options &options,
                   const double expected_checksum) {
  if (!cdfmm::cuda_m2l_p2p_available()) {
    throw std::runtime_error(
        "--cuda requested but no CUDA P2P runtime is available");
  }
  const CudaResult canonical =
      measure_cuda("cuda-canonical-aos", geometry, options.evaluations, [&] {
        return std::make_unique<cdfmm::CudaP2PPlan>(geometry.canonical,
                                                    geometry.identities);
      });
  const CudaResult compact =
      measure_cuda("cuda-particle-row-soa", geometry, options.evaluations, [&] {
        return std::make_unique<cdfmm::CudaP2PPlan>(geometry.compact,
                                                    geometry.identities);
      });
  const CudaResult leaf = measure_cuda(
      "cuda-leaf-block-compact", geometry, options.evaluations, [&] {
        return std::make_unique<cdfmm::CudaP2PPlan>(geometry.leaf,
                                                    geometry.identities);
      });
  const CudaResult bsr =
      measure_cuda("cuda-cusparse-bsr3", geometry, options.evaluations, [&] {
        return std::make_unique<cdfmm::CudaP2PPlan>(geometry.bsr);
      });

  const auto require_matching_checksum = [&](const CudaResult &candidate) {
    const double scale = std::max(1.0, std::abs(expected_checksum));
    if (std::abs(candidate.checksum - expected_checksum) > 2.0e-11 * scale) {
      throw std::runtime_error(candidate.name +
                               " does not match the canonical CPU checksum");
    }
  };
  require_matching_checksum(canonical);
  require_matching_checksum(compact);
  require_matching_checksum(leaf);
  require_matching_checksum(bsr);

  std::cout << "cuda_implementation,setup_s,h2d_s,kernel_s,d2h_s,"
               "device_total_s,host_total_s,kernel_speedup,total_speedup,"
               "interactions_per_kernel_s,tensor_bytes,index_bytes,"
               "row_metadata_bytes,leaf_metadata_bytes,identity_bytes,"
               "scratch_bytes,"
               "threads_per_block,persistent_device_bytes,checksum\n";
  const std::size_t interactions = geometry.canonical.blocks.size();
  print_cuda_result(canonical, interactions, canonical);
  print_cuda_result(compact, interactions, canonical);
  print_cuda_result(leaf, interactions, canonical);
  print_cuda_result(bsr, interactions, canonical);
}
#endif

void run_case(const Options &options, const int occupancy) {
  GeometryPlan geometry =
      build_geometry(options.depth, occupancy, options.particles,
                     options.irregular, options.regular_grid_s,
                     options.signed_target_tile);
  cdfmm::StaticP2PMemory canonical_memory;
#if defined(CDFMM_USE_OPENMP)
  const int signed_threads = omp_get_max_threads();
#else
  const int signed_threads = 1;
#endif
  const auto balance = signed_work_balance(
      geometry.signed_tensor_dictionary, signed_threads);
  canonical_memory.tensor_bytes =
      geometry.canonical.blocks.size() * 6 * sizeof(double);
  canonical_memory.index_bytes =
      geometry.canonical.blocks.size() * 2 * sizeof(int);
  canonical_memory.row_metadata_bytes =
      geometry.canonical.row_offsets.size() * sizeof(int);

  const Result canonical = measure(
      "canonical-aos", canonical_memory, geometry, options.evaluations,
      [&](const std::span<cdfmm::Vec3> fields) {
        cdfmm::apply_static_p2p_operator(geometry.canonical, geometry.moments,
                                         fields, geometry.identities);
      });
  const Result compact = measure(
      "particle-row-soa", geometry.compact.memory(), geometry,
      options.evaluations, [&](const std::span<cdfmm::Vec3> fields) {
        cdfmm::apply_static_p2p_compact_plan(geometry.compact, geometry.moments,
                                             fields, geometry.identities);
      });
  const Result leaf = measure(
      "leaf-block-compact", geometry.leaf.memory(), geometry,
      options.evaluations, [&](const std::span<cdfmm::Vec3> fields) {
        cdfmm::apply_static_p2p_leaf_plan(geometry.leaf, geometry.moments,
                                          fields, geometry.identities);
      });
  const Result bsr =
      measure("expanded-bsr3-portable", geometry.bsr.memory(), geometry,
              options.evaluations, [&](const std::span<cdfmm::Vec3> fields) {
                cdfmm::apply_static_p2p_bsr_plan(geometry.bsr, geometry.moments,
                                                 fields, geometry.identities);
              });
  const Result tensor_dictionary = measure(
      "tensor-dictionary-legacy", geometry.tensor_dictionary.memory(), geometry,
      options.evaluations, [&](const std::span<cdfmm::Vec3> fields) {
        cdfmm::apply_static_p2p_tensor_dictionary_plan(
            geometry.tensor_dictionary, geometry.moments, fields,
            geometry.identities);
      });
  const Result signed_tensor_dictionary_current = measure(
      "tensor-dictionary-signed-whole-tile-fp64",
      geometry.signed_tensor_dictionary.memory(),
      geometry, options.evaluations, [&](const std::span<cdfmm::Vec3> fields) {
        cdfmm::apply_static_p2p_signed_tensor_dictionary_plan_whole_tile(
            geometry.signed_tensor_dictionary, geometry.moments, fields);
      });
  const Result signed_tensor_dictionary = measure(
      "tensor-dictionary-signed-microtile-fp64",
      geometry.signed_tensor_dictionary.memory(),
      geometry, options.evaluations, [&](const std::span<cdfmm::Vec3> fields) {
        cdfmm::apply_static_p2p_signed_tensor_dictionary_plan(
            geometry.signed_tensor_dictionary, geometry.moments, fields);
      });
  const Result compact_float = measure_float(
      "particle-row-soa-fp32", geometry.compact_float.memory(), geometry,
      options.evaluations,
      [&](const std::span<cdfmm::FloatVec3> fields) {
        cdfmm::apply_static_p2p_compact_plan(
            geometry.compact_float, geometry.moments_float, fields,
            geometry.identities);
      });
  const Result signed_tensor_dictionary_float_current = measure_float(
      "tensor-dictionary-signed-whole-tile-fp32",
      geometry.signed_tensor_dictionary_float.memory(), geometry,
      options.evaluations,
      [&](const std::span<cdfmm::FloatVec3> fields) {
        cdfmm::apply_static_p2p_signed_tensor_dictionary_plan_whole_tile(
            geometry.signed_tensor_dictionary_float, geometry.moments_float,
            fields);
      });
  const Result signed_tensor_dictionary_float = measure_float(
      "tensor-dictionary-signed-microtile-fp32",
      geometry.signed_tensor_dictionary_float.memory(), geometry,
      options.evaluations,
      [&](const std::span<cdfmm::FloatVec3> fields) {
        cdfmm::apply_static_p2p_signed_tensor_dictionary_plan(
            geometry.signed_tensor_dictionary_float, geometry.moments_float,
            fields);
      });
#if defined(CDFMM_USE_MKL)
  const Result mkl_bsr =
      measure("onemkl-bsr3", geometry.mkl_bsr->memory(), geometry,
              options.evaluations, [&](const std::span<cdfmm::Vec3> fields) {
                geometry.mkl_bsr->apply(geometry.moments, fields);
              });
#endif
  const auto require_matching_checksum = [&](const Result &candidate) {
    const double scale = std::max(1.0, std::abs(canonical.checksum));
    if (std::abs(candidate.checksum - canonical.checksum) > 5.0e-12 * scale) {
      throw std::runtime_error(candidate.name +
                               " does not match the canonical checksum");
    }
  };
  require_matching_checksum(compact);
  require_matching_checksum(leaf);
  require_matching_checksum(bsr);
  require_matching_checksum(tensor_dictionary);
  require_matching_checksum(signed_tensor_dictionary_current);
  require_matching_checksum(signed_tensor_dictionary);
  const auto require_matching_float_checksum = [&](const Result &candidate) {
    const double scale = std::max(1.0, std::abs(compact_float.checksum));
    if (std::abs(candidate.checksum - compact_float.checksum) > 2.0e-5 * scale) {
      throw std::runtime_error(candidate.name +
                               " does not match the FP32 particle-row checksum");
    }
  };
  require_matching_float_checksum(signed_tensor_dictionary_float_current);
  require_matching_float_checksum(signed_tensor_dictionary_float);
#if defined(CDFMM_USE_MKL)
  require_matching_checksum(mkl_bsr);
#endif

  std::cout << "case,particles=" << geometry.moments.size()
            << ",depth=" << options.depth
            << ",occupied_leaves=" << geometry.leaf.target_begins.size()
            << ",mean_occupancy=" << geometry.leaf.mean_occupancy
            << ",max_occupancy=" << geometry.leaf.maximum_occupancy
            << ",unique_occupancies=" << geometry.leaf.unique_occupancies
            << ",uniform=" << (geometry.leaf.uniform_occupancy ? "yes" : "no")
            << ",particle_interactions=" << geometry.canonical.blocks.size()
            << ",leaf_interactions=" << geometry.leaf.blocks.size()
            << ",canonical_setup_s=" << geometry.canonical_setup_seconds
            << ",compact_setup_s=" << geometry.compact_setup_seconds
            << ",leaf_setup_s=" << geometry.leaf_setup_seconds;
  std::cout << ",tensor_dictionary_setup_s="
            << geometry.tensor_dictionary_setup_seconds;
  std::cout << ",signed_tensor_dictionary_setup_s="
            << geometry.signed_tensor_dictionary_setup_seconds
            << ",signed_variant_count="
            << geometry.signed_tensor_dictionary.variant_count()
            << ",signed_token_width_bytes="
            << static_cast<int>(geometry.signed_tensor_dictionary.token_width_bytes)
            << ",signed_target_tile="
            << geometry.signed_tensor_dictionary.target_tile_size
            << ",signed_simd_path=" << cdfmm::static_p2p_signed_simd_path()
            << ",signed_simd_width_fp64="
            << cdfmm::static_p2p_signed_simd_width(false)
            << ",signed_simd_width_fp32="
            << cdfmm::static_p2p_signed_simd_width(true)
            << ",signed_work_max_over_mean=" << balance[0]
            << ",signed_static_thread_max_over_mean=" << balance[1]
            << ",legacy_variant_count="
            << geometry.tensor_dictionary.tensors[0].size()
            << ",legacy_token_width_bytes=4";
  std::cout << ",bsr_setup_s=" << geometry.bsr_setup_seconds;
#if defined(CDFMM_USE_MKL)
  std::cout << ",mkl_setup_s=" << geometry.mkl_setup_seconds;
#endif
#if defined(CDFMM_USE_OPENMP)
  std::cout << ",threads=" << omp_get_max_threads();
#else
  std::cout << ",threads=1";
#endif
  std::cout << '\n';
  std::cout << "implementation,evaluation_s,interactions_per_s,speedup,"
               "tensor_bytes,index_bytes,row_metadata_bytes,"
               "leaf_metadata_bytes,scratch_bytes,total_bytes,checksum\n";
  print_result(canonical, geometry.canonical.blocks.size(), canonical.seconds);
  print_result(compact, geometry.canonical.blocks.size(), canonical.seconds);
  print_result(leaf, geometry.canonical.blocks.size(), canonical.seconds);
  print_result(tensor_dictionary, geometry.canonical.blocks.size(),
               canonical.seconds);
  print_result(signed_tensor_dictionary_current,
               geometry.canonical.blocks.size(), canonical.seconds);
  print_result(signed_tensor_dictionary, geometry.canonical.blocks.size(),
               canonical.seconds);
  print_result(compact_float, geometry.canonical.blocks.size(),
               compact_float.seconds);
  print_result(signed_tensor_dictionary_float_current,
               geometry.canonical.blocks.size(), compact_float.seconds);
  print_result(signed_tensor_dictionary_float,
               geometry.canonical.blocks.size(), compact_float.seconds);
  print_result(bsr, geometry.canonical.blocks.size(), canonical.seconds);
#if defined(CDFMM_USE_MKL)
  print_result(mkl_bsr, geometry.canonical.blocks.size(), canonical.seconds);
#endif
  if (options.cuda) {
#if defined(CDFMM_ENABLE_CUDA)
    run_cuda_case(geometry, options, canonical.checksum);
#else
    throw std::runtime_error(
        "--cuda requires a build configured with CDFMM_ENABLE_CUDA=ON");
#endif
  }
}

} // namespace

int main(const int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.sweep) {
      for (const int occupancy : {1, 2, 4, 8, 16, 32, 64}) {
        run_case(options, occupancy);
      }
    } else {
      run_case(options, options.occupancy);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "benchmark_p2p: " << error.what() << '\n';
    return 1;
  }
}
