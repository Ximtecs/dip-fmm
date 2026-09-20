// SPDX-License-Identifier: Apache-2.0
//
// Device-resident far-field operators of the complete CUDA FMM: P2M, M2M, L2L
// and L2P.  Construction converts the canonical operators into the device
// layouts below and uploads them once; `enqueue_*` then only launch kernels
// on the caller's stream, so a repeated evaluation performs no host work
// beyond the launches.  M2L is a separate plan (backend/cuda/m2l).
//
// Layouts: P2M and L2P are sparse coefficient maps stored CSR-by-output, or
// (point models) the procedural representation that holds positions and a
// factor table and recomputes the rows in the kernel.  M2M and L2L are the
// eight child-class matrices of the universal bank, pre-scaled per level, with
// interactions grouped by (level, target) so that one launch per level owns
// disjoint output rows and needs no atomics.  The kernels themselves are in
// entries.cuh, procedural.cuh and translation.cuh.

#include "backend/cuda/common/error.hpp"
#include "backend/cuda/execution_policy.hpp"
#include "backend/cuda/far_field/entries.cuh"
#include "backend/cuda/far_field/internal.hpp"
#include "backend/cuda/far_field/procedural.cuh"
#include "backend/cuda/far_field/translation.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cdfmm::cuda_far_field_detail {

namespace {

using cuda_detail::check_cuda;

// Lane group per CSR row; see the kernel comment.
inline constexpr int entry_row_lanes = 8;
// The two instantiated translation lane widths; the CUDA execution policy
// (backend/cuda/execution_policy.hpp) chooses per level launch.
inline constexpr int translation_lanes = 4;
inline constexpr int wide_translation_lanes = 32;

} // namespace

// Sparse coefficient map stored as CSR by output on the device.
template <typename Scalar> struct DeviceCsrOperator {
  int *row_offsets{nullptr};
  int *inputs{nullptr};
  Scalar *values{nullptr};
  int row_count{0};
  std::size_t entry_count{0};

  void release() noexcept {
    cudaFree(row_offsets);
    cudaFree(inputs);
    cudaFree(values);
  }
};

// Procedural point P2M or L2P: the occupied leaves, every point's
// displacement from its leaf centre, and the per-mode factor table. No
// coefficient rows are resident; the kernels recompute the operator.
template <typename Scalar> struct DeviceProcedural {
  ProceduralLeaf *leaves{nullptr};
  ProceduralPoint<Scalar> *displacements{nullptr};
  Scalar *factors{nullptr};
  int leaf_count{0};
  int lanes_per_leaf{32};

  void release() noexcept {
    cudaFree(leaves);
    cudaFree(displacements);
    cudaFree(factors);
  }
};

// One translation direction (M2M or L2L): level-scaled child-class matrices
// and the per-level target grouping consumed by translate_targets_kernel.
template <typename Scalar> struct DeviceTranslation {
  // Per level block of 8 CSR matrices: `8 * (n + 1)` row offsets, relative to
  // the level's entry base of `8 * entries_per_matrix`.
  int *matrix_row_offsets{nullptr};
  int *matrix_inputs{nullptr};
  Scalar *matrix_values{nullptr};
  int *targets{nullptr};
  int *target_interaction_offsets{nullptr};
  int *sources{nullptr};
  int *classes{nullptr};
  // Host bookkeeping: target range of every level within `targets`.
  std::vector<int> level_target_begin{};
  std::vector<int> level_target_count{};
  int entries_per_matrix{0};

  void release() noexcept {
    cudaFree(matrix_row_offsets);
    cudaFree(matrix_inputs);
    cudaFree(matrix_values);
    cudaFree(targets);
    cudaFree(target_interaction_offsets);
    cudaFree(sources);
    cudaFree(classes);
  }
};

template <typename Scalar, typename Entry>
struct CudaFarFieldExecutionPlan<Scalar, Entry>::Implementation {
  DeviceCsrOperator<Scalar> p2m{};
  DeviceCsrOperator<Scalar> l2p{};
  DeviceProcedural<Scalar> procedural_p2m{};
  DeviceProcedural<Scalar> procedural_l2p{};
  bool use_procedural_p2m{false};
  bool use_procedural_l2p{false};
  int expansion_order{0};
  DeviceTranslation<Scalar> m2m{};
  DeviceTranslation<Scalar> l2l{};
  int coefficient_count{0};
  int maximum_level{0};
  CudaPlanStatistics statistics{};

  ~Implementation() {
    p2m.release();
    l2p.release();
    procedural_p2m.release();
    procedural_l2p.release();
    m2m.release();
    l2l.release();
  }
};

namespace {

template <typename T>
void allocate(T **pointer, const std::size_t bytes, const char *description) {
  check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer),
                        std::max(bytes, std::size_t{1})),
             description);
}

template <typename T>
void upload(T *destination, std::span<const T> values, cudaStream_t stream,
            const char *description) {
  if (!values.empty()) {
    check_cuda(cudaMemcpyAsync(destination, values.data(), values.size_bytes(),
                               cudaMemcpyHostToDevice, stream),
               description);
  }
}

// Builds and uploads the procedural data of one stage: the leaf ranges, the
// displacement of every point from its leaf centre (computed in FP64 and
// narrowed once) and the factor table. Returns the uploaded bytes.
template <typename Scalar>
std::size_t build_procedural(DeviceProcedural<Scalar> &device,
                             std::span<const StaticLeafRange> leaves,
                             std::span<const StaticFmmTopology::Node> nodes,
                             std::span<const Vec3> positions,
                             const std::vector<double> &factors,
                             const int lanes_per_leaf,
                             const char *description) {
  std::vector<ProceduralLeaf> leaf_records;
  leaf_records.reserve(leaves.size());
  std::vector<ProceduralPoint<Scalar>> displacements(positions.size());
  for (const StaticLeafRange &leaf : leaves) {
    leaf_records.push_back({leaf.node, static_cast<int>(leaf.begin),
                            static_cast<int>(leaf.count), 0});
    const Vec3 centre = nodes[static_cast<std::size_t>(leaf.node)].centre;
    for (std::size_t point = leaf.begin; point < leaf.begin + leaf.count;
         ++point) {
      const Vec3 d = positions[point] - centre;
      displacements[point] = {static_cast<Scalar>(d.x),
                              static_cast<Scalar>(d.y),
                              static_cast<Scalar>(d.z), Scalar{0}};
    }
  }
  std::vector<Scalar> narrowed(factors.size());
  for (std::size_t index = 0; index < factors.size(); ++index) {
    narrowed[index] = static_cast<Scalar>(factors[index]);
  }
  device.leaf_count = static_cast<int>(leaf_records.size());
  device.lanes_per_leaf = lanes_per_leaf;
  const std::size_t leaf_bytes = leaf_records.size() * sizeof(ProceduralLeaf);
  const std::size_t point_bytes =
      displacements.size() * sizeof(ProceduralPoint<Scalar>);
  const std::size_t factor_bytes = narrowed.size() * sizeof(Scalar);
  allocate(&device.leaves, leaf_bytes, description);
  allocate(&device.displacements, point_bytes, description);
  allocate(&device.factors, factor_bytes, description);
  const auto copy = [&](void *destination, const void *source,
                        const std::size_t bytes) {
    if (bytes != 0) {
      check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                 description);
    }
  };
  copy(device.leaves, leaf_records.data(), leaf_bytes);
  copy(device.displacements, displacements.data(), point_bytes);
  copy(device.factors, narrowed.data(), factor_bytes);
  return leaf_bytes + point_bytes + factor_bytes;
}

// Host-side CSR-by-output image of a flat entry list.
template <typename Scalar> struct HostCsrOperator {
  std::vector<int> row_offsets{};
  std::vector<int> inputs{};
  std::vector<Scalar> values{};

  [[nodiscard]] std::size_t bytes() const noexcept {
    return row_offsets.size() * sizeof(int) + inputs.size() * sizeof(int) +
           values.size() * sizeof(Scalar);
  }
};

// Counting sort of entries by output row. `scale(entry)` supplies the stored
// value so translation matrices can fold their level scaling in here.
template <typename Scalar, typename Entry, typename Scale>
HostCsrOperator<Scalar> build_csr_by_output(std::span<const Entry> entries,
                                            const int row_count,
                                            const Scale &scale) {
  HostCsrOperator<Scalar> csr;
  csr.row_offsets.assign(static_cast<std::size_t>(row_count) + 1, 0);
  for (const Entry &entry : entries) {
    ++csr.row_offsets[static_cast<std::size_t>(entry.output) + 1];
  }
  for (int row = 0; row < row_count; ++row) {
    csr.row_offsets[static_cast<std::size_t>(row) + 1] +=
        csr.row_offsets[static_cast<std::size_t>(row)];
  }
  csr.inputs.resize(entries.size());
  csr.values.resize(entries.size());
  std::vector<int> cursor(csr.row_offsets.begin(), csr.row_offsets.end() - 1);
  for (const Entry &entry : entries) {
    const std::size_t slot = static_cast<std::size_t>(
        cursor[static_cast<std::size_t>(entry.output)]++);
    csr.inputs[slot] = entry.input;
    csr.values[slot] = scale(entry);
  }
  return csr;
}

template <typename Scalar, typename Entry>
int output_row_count(std::span<const Entry> entries) {
  int rows = 0;
  for (const Entry &entry : entries) {
    if (entry.output < 0 || entry.input < 0) {
      throw std::invalid_argument("CUDA far-field entry index is negative");
    }
    rows = std::max(rows, entry.output + 1);
  }
  return rows;
}

template <typename Scalar>
std::size_t upload_csr(DeviceCsrOperator<Scalar> &device,
                       const HostCsrOperator<Scalar> &host,
                       cudaStream_t stream, const char *description) {
  device.row_count = static_cast<int>(host.row_offsets.size()) - 1;
  device.entry_count = host.inputs.size();
  allocate(&device.row_offsets, host.row_offsets.size() * sizeof(int),
           description);
  allocate(&device.inputs, host.inputs.size() * sizeof(int), description);
  allocate(&device.values, host.values.size() * sizeof(Scalar), description);
  upload(device.row_offsets, std::span<const int>(host.row_offsets), stream,
         description);
  upload(device.inputs, std::span<const int>(host.inputs), stream,
         description);
  upload(device.values, std::span<const Scalar>(host.values), stream,
         description);
  return host.bytes();
}

// Builds one translation direction: eight level-scaled CSR matrices per level
// and interactions grouped by (level, target node) so one launch per level
// owns disjoint target rows.
template <typename Scalar, typename Entry>
std::size_t build_translation(
    DeviceTranslation<Scalar> &device, std::span<const Entry> matrices,
    std::span<const CudaTranslationInteraction> interactions,
    const int entries_per_matrix, const int coefficient_count,
    const int maximum_level, std::span<const int> coefficient_degrees,
    cudaStream_t stream, const char *description) {
  device.entries_per_matrix = entries_per_matrix;
  device.level_target_begin.assign(static_cast<std::size_t>(maximum_level) + 1,
                                   0);
  device.level_target_count.assign(static_cast<std::size_t>(maximum_level) + 1,
                                   0);
  if (entries_per_matrix == 0 || interactions.empty() || maximum_level < 1) {
    return 0;
  }
  const int matrix_count =
      static_cast<int>(matrices.size()) / entries_per_matrix;
  for (const CudaTranslationInteraction &interaction : interactions) {
    if (interaction.level < 1 || interaction.level > maximum_level ||
        interaction.matrix_id < 0 || interaction.matrix_id >= matrix_count ||
        interaction.source_node < 0 || interaction.target_node < 0) {
      throw std::invalid_argument("CUDA far-field translation is invalid");
    }
  }

  // Level-scaled matrices: ldexp by a power of two is exact, so folding the
  // scaling in at construction reproduces the CPU reference arithmetic.
  std::vector<int> matrix_row_offsets;
  std::vector<int> matrix_inputs;
  std::vector<Scalar> matrix_values;
  matrix_row_offsets.reserve(static_cast<std::size_t>(maximum_level) *
                             matrix_count * (coefficient_count + 1));
  matrix_inputs.reserve(static_cast<std::size_t>(maximum_level) *
                        matrices.size());
  matrix_values.reserve(static_cast<std::size_t>(maximum_level) *
                        matrices.size());
  for (int level = 1; level <= maximum_level; ++level) {
    int level_base = 0;
    for (int matrix = 0; matrix < matrix_count; ++matrix) {
      const std::span<const Entry> entries = matrices.subspan(
          static_cast<std::size_t>(matrix) * entries_per_matrix,
          static_cast<std::size_t>(entries_per_matrix));
      for (const Entry &entry : entries) {
        if (entry.output < 0 || entry.output >= coefficient_count ||
            entry.input < 0 || entry.input >= coefficient_count) {
          throw std::invalid_argument(
              "CUDA far-field translation matrix entry is invalid");
        }
      }
      const HostCsrOperator<Scalar> csr = build_csr_by_output<Scalar, Entry>(
          entries, coefficient_count, [&](const Entry &entry) {
            const int degree_difference =
                coefficient_degrees[static_cast<std::size_t>(entry.output)] -
                coefficient_degrees[static_cast<std::size_t>(entry.input)];
            const int power = std::abs(degree_difference);
            return std::ldexp(static_cast<Scalar>(entry.value),
                              -(level - 1) * power);
          });
      for (const int offset : csr.row_offsets) {
        matrix_row_offsets.push_back(level_base + offset);
      }
      matrix_inputs.insert(matrix_inputs.end(), csr.inputs.begin(),
                           csr.inputs.end());
      matrix_values.insert(matrix_values.end(), csr.values.begin(),
                           csr.values.end());
      level_base += entries_per_matrix;
    }
  }

  // Interactions grouped by (level, target): stable counting sort by level,
  // then by target within a level, preserving the plan's interaction order
  // among a target's children.
  std::vector<std::size_t> order(interactions.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](const std::size_t left, const std::size_t right) {
                     const auto &a = interactions[left];
                     const auto &b = interactions[right];
                     return a.level != b.level ? a.level < b.level
                                               : a.target_node < b.target_node;
                   });
  std::vector<int> targets;
  std::vector<int> target_interaction_offsets;
  std::vector<int> sources(interactions.size());
  std::vector<int> classes(interactions.size());
  targets.reserve(interactions.size());
  target_interaction_offsets.reserve(interactions.size() + 1);
  int previous_level = 0;
  int previous_target = -1;
  for (std::size_t position = 0; position < order.size(); ++position) {
    const CudaTranslationInteraction &interaction = interactions[order[position]];
    if (interaction.level != previous_level ||
        interaction.target_node != previous_target) {
      if (interaction.level != previous_level) {
        for (int level = previous_level + 1; level <= interaction.level;
             ++level) {
          device.level_target_begin[static_cast<std::size_t>(level)] =
              static_cast<int>(targets.size());
        }
        previous_level = interaction.level;
      }
      targets.push_back(interaction.target_node);
      target_interaction_offsets.push_back(static_cast<int>(position));
      ++device.level_target_count[static_cast<std::size_t>(interaction.level)];
      previous_target = interaction.target_node;
    }
    sources[position] = interaction.source_node;
    classes[position] = interaction.matrix_id;
  }
  target_interaction_offsets.push_back(static_cast<int>(order.size()));
  for (int level = previous_level + 1; level <= maximum_level; ++level) {
    device.level_target_begin[static_cast<std::size_t>(level)] =
        static_cast<int>(targets.size());
  }

  allocate(&device.matrix_row_offsets, matrix_row_offsets.size() * sizeof(int),
           description);
  allocate(&device.matrix_inputs, matrix_inputs.size() * sizeof(int),
           description);
  allocate(&device.matrix_values, matrix_values.size() * sizeof(Scalar),
           description);
  allocate(&device.targets, targets.size() * sizeof(int), description);
  allocate(&device.target_interaction_offsets,
           target_interaction_offsets.size() * sizeof(int), description);
  allocate(&device.sources, sources.size() * sizeof(int), description);
  allocate(&device.classes, classes.size() * sizeof(int), description);
  upload(device.matrix_row_offsets, std::span<const int>(matrix_row_offsets),
         stream, description);
  upload(device.matrix_inputs, std::span<const int>(matrix_inputs), stream,
         description);
  upload(device.matrix_values, std::span<const Scalar>(matrix_values), stream,
         description);
  upload(device.targets, std::span<const int>(targets), stream, description);
  upload(device.target_interaction_offsets,
         std::span<const int>(target_interaction_offsets), stream, description);
  upload(device.sources, std::span<const int>(sources), stream, description);
  upload(device.classes, std::span<const int>(classes), stream, description);
  return (matrix_row_offsets.size() + matrix_inputs.size() + targets.size() +
          target_interaction_offsets.size() + sources.size() +
          classes.size()) *
             sizeof(int) +
         matrix_values.size() * sizeof(Scalar);
}

template <typename Scalar, int lanes>
void launch_translation_level(const DeviceTranslation<Scalar> &translation,
                              const int level, const int coefficient_count,
                              const Scalar *input, Scalar *output,
                              cudaStream_t stream, const char *description) {
  const int target_count =
      translation.level_target_count[static_cast<std::size_t>(level)];
  const int target_begin =
      translation.level_target_begin[static_cast<std::size_t>(level)];
  const std::size_t level_matrix_rows =
      static_cast<std::size_t>(8) * (coefficient_count + 1);
  const std::size_t level_matrix_entries =
      static_cast<std::size_t>(8) * translation.entries_per_matrix;
  const std::size_t items =
      static_cast<std::size_t>(target_count) * coefficient_count * lanes;
  translate_targets_kernel<Scalar, lanes>
      <<<(items + far_field_threads - 1) / far_field_threads,
         far_field_threads, 0, stream>>>(
      translation.targets + target_begin,
      translation.target_interaction_offsets + target_begin,
      translation.sources, translation.classes,
      translation.matrix_row_offsets +
          static_cast<std::size_t>(level - 1) * level_matrix_rows,
      translation.matrix_inputs +
          static_cast<std::size_t>(level - 1) * level_matrix_entries,
      translation.matrix_values +
          static_cast<std::size_t>(level - 1) * level_matrix_entries,
      target_count, coefficient_count, input, output);
  check_cuda(cudaGetLastError(), description);
}

// Pick the lane width for one level from the number of outputs it produces:
// wide lanes keep small levels (few nodes, near the root) busy, narrow lanes
// give large levels more independent work items.  The rule lives in the
// execution policy so the CPU-side summary can report it.
template <typename Scalar>
void enqueue_translation_level(const DeviceTranslation<Scalar> &translation,
                               const int level, const int coefficient_count,
                               const Scalar *input, Scalar *output,
                               cudaStream_t stream, const char *description) {
  const int target_count =
      translation.level_target_count[static_cast<std::size_t>(level)];
  if (target_count == 0) {
    return;
  }
  const std::size_t outputs =
      static_cast<std::size_t>(target_count) * coefficient_count;
  if (cuda_policy::translation_lanes_for_outputs(outputs) ==
      wide_translation_lanes) {
    launch_translation_level<Scalar, wide_translation_lanes>(
        translation, level, coefficient_count, input, output, stream,
        description);
  } else {
    launch_translation_level<Scalar, translation_lanes>(
        translation, level, coefficient_count, input, output, stream,
        description);
  }
}

} // namespace

// Validate the static data, then build and upload each stage.  The uploads
// are asynchronous on `stream`; the owner synchronises once after every plan
// of the evaluation has been constructed.
template <typename Scalar, typename Entry>
CudaFarFieldExecutionPlan<Scalar, Entry>::CudaFarFieldExecutionPlan(
    const CudaFarFieldStaticData<Entry> &data, cudaStream_t stream)
    : implementation_(new Implementation{}) {
  try {
    auto &plan = *implementation_;
    if (data.coefficient_count < 0 || data.maximum_level < -1 ||
        data.m2m_entries_per_matrix < 0 || data.l2l_entries_per_matrix < 0 ||
        data.m2m_matrix_count < 0 || data.l2l_matrix_count < 0) {
      throw std::invalid_argument("CUDA far-field dimensions are invalid");
    }
    if (data.coefficient_degrees.size() !=
        static_cast<std::size_t>(data.coefficient_count)) {
      throw std::invalid_argument("CUDA far-field degree metadata is invalid");
    }
    if (data.m2m_entries_per_matrix != 0 &&
        data.m2m_matrices.size() %
                static_cast<std::size_t>(data.m2m_entries_per_matrix) !=
            0) {
      throw std::invalid_argument("CUDA far-field M2M matrix data is invalid");
    }
    if (data.l2l_entries_per_matrix != 0 &&
        data.l2l_matrices.size() %
                static_cast<std::size_t>(data.l2l_entries_per_matrix) !=
            0) {
      throw std::invalid_argument("CUDA far-field L2L matrix data is invalid");
    }
    if (data.m2m_entries_per_matrix == 0 && !data.m2m_interactions.empty()) {
      throw std::invalid_argument(
          "CUDA far-field M2M interactions are invalid");
    }
    if (data.l2l_entries_per_matrix == 0 && !data.l2l_interactions.empty()) {
      throw std::invalid_argument(
          "CUDA far-field L2L interactions are invalid");
    }
    // The child-class matrices are addressed as eight per level.
    if ((data.m2m_entries_per_matrix != 0 &&
         data.m2m_matrices.size() !=
             static_cast<std::size_t>(8) * data.m2m_entries_per_matrix) ||
        (data.l2l_entries_per_matrix != 0 &&
         data.l2l_matrices.size() !=
             static_cast<std::size_t>(8) * data.l2l_entries_per_matrix)) {
      throw std::invalid_argument(
          "CUDA far-field translation requires eight child-class matrices");
    }

    plan.coefficient_count = data.coefficient_count;
    plan.maximum_level = data.maximum_level;

    const auto identity = [](const Entry &entry) {
      return static_cast<Scalar>(entry.value);
    };
    std::size_t uploaded_bytes = 0;
    plan.use_procedural_p2m = data.procedural_p2m;
    plan.use_procedural_l2p = data.procedural_l2p;
    plan.expansion_order = data.expansion_order;
    if ((data.procedural_p2m || data.procedural_l2p) &&
        data.topology == nullptr) {
      throw std::invalid_argument(
          "CUDA procedural point expansion needs the plan topology");
    }
    if (data.procedural_p2m) {
      uploaded_bytes += build_procedural<Scalar>(
          plan.procedural_p2m, data.topology->source_leaves,
          data.topology->nodes, data.topology->sorted_source_positions,
          operators::point_expansion::p2m_mode_factors(data.expansion_order),
          data.p2m_lanes_per_leaf, "upload CUDA procedural P2M data");
    } else {
      const HostCsrOperator<Scalar> p2m = build_csr_by_output<Scalar, Entry>(
          data.p2m, output_row_count<Scalar, Entry>(data.p2m), identity);
      uploaded_bytes += upload_csr(plan.p2m, p2m, stream,
                                   "upload CUDA far-field P2M entries");
    }
    if (data.procedural_l2p) {
      uploaded_bytes += build_procedural<Scalar>(
          plan.procedural_l2p, data.topology->target_leaves,
          data.topology->nodes, data.topology->sorted_target_positions,
          operators::point_expansion::l2p_field_mode_factors(
              data.expansion_order),
          data.l2p_lanes_per_leaf, "upload CUDA procedural L2P data");
    } else {
      const HostCsrOperator<Scalar> l2p = build_csr_by_output<Scalar, Entry>(
          data.l2p, output_row_count<Scalar, Entry>(data.l2p), identity);
      uploaded_bytes += upload_csr(plan.l2p, l2p, stream,
                                   "upload CUDA far-field L2P entries");
    }
    const std::size_t m2m_bytes = build_translation<Scalar, Entry>(
        plan.m2m, data.m2m_matrices, data.m2m_interactions,
        data.m2m_entries_per_matrix, data.coefficient_count,
        data.maximum_level, data.coefficient_degrees, stream,
        "upload CUDA far-field M2M translation");
    const std::size_t l2l_bytes = build_translation<Scalar, Entry>(
        plan.l2l, data.l2l_matrices, data.l2l_interactions,
        data.l2l_entries_per_matrix, data.coefficient_count,
        data.maximum_level, data.coefficient_degrees, stream,
        "upload CUDA far-field L2L translation");

    plan.statistics.scalar_bytes = sizeof(Scalar);
    plan.statistics.m2m_unique_matrix_count = data.m2m_matrix_count;
    plan.statistics.m2m_matrix_bytes = m2m_bytes;
    plan.statistics.l2l_unique_matrix_count = data.l2l_matrix_count;
    plan.statistics.l2l_matrix_bytes = l2l_bytes;
    plan.statistics.setup_h2d_bytes = uploaded_bytes + m2m_bytes + l2l_bytes;
    plan.statistics.persistent_device_bytes = plan.statistics.setup_h2d_bytes;
  } catch (...) {
    delete implementation_;
    implementation_ = nullptr;
    throw;
  }
}

template <typename Scalar, typename Entry>
CudaFarFieldExecutionPlan<Scalar, Entry>::~CudaFarFieldExecutionPlan() {
  if (implementation_ == nullptr) {
    return;
  }
  delete implementation_;
}

// P2M: `input` is the sorted moment array viewed as scalars, `output` the
// multipole coefficients of every node (zeroed by the caller).  Procedural
// and stored representations produce the same coefficients up to rounding.
template <typename Scalar, typename Entry>
void CudaFarFieldExecutionPlan<Scalar, Entry>::enqueue_p2m(
    const Scalar *input, Scalar *output, cudaStream_t stream) const {
  const auto &plan = *implementation_;
  if (plan.use_procedural_p2m) {
    using Vector =
        std::conditional_t<std::is_same_v<Scalar, double>, Vec3, FloatVec3>;
    launch_procedural_p2m<Scalar, Vector>(
        plan.expansion_order, plan.procedural_p2m.leaves,
        plan.procedural_p2m.leaf_count, plan.procedural_p2m.lanes_per_leaf,
        plan.procedural_p2m.displacements,
        reinterpret_cast<const Vector *>(input), plan.procedural_p2m.factors,
        output, stream);
    return;
  }
  if (plan.p2m.entry_count != 0) {
    const std::size_t items =
        static_cast<std::size_t>(plan.p2m.row_count) * entry_row_lanes;
    apply_csr_rows_kernel<Scalar, entry_row_lanes>
        <<<(items + far_field_threads - 1) / far_field_threads,
           far_field_threads, 0, stream>>>(
        plan.p2m.row_offsets, plan.p2m.inputs, plan.p2m.values,
        plan.p2m.row_count, input, output);
    check_cuda(cudaGetLastError(), "launch CUDA far-field P2M kernel");
  }
}

template <typename Scalar, typename Entry>
void CudaFarFieldExecutionPlan<Scalar, Entry>::enqueue_m2m(
    const Scalar *input, Scalar *output, cudaStream_t stream) const {
  const auto &plan = *implementation_;
  // Deep to shallow: every parent level consumes completed child multipoles.
  for (int level = plan.maximum_level; level >= 1; --level) {
    enqueue_translation_level(plan.m2m, level, plan.coefficient_count, input,
                              output, stream,
                              "launch CUDA far-field M2M kernel");
  }
}

template <typename Scalar, typename Entry>
void CudaFarFieldExecutionPlan<Scalar, Entry>::enqueue_l2l(
    const Scalar *input, Scalar *output, cudaStream_t stream) const {
  const auto &plan = *implementation_;
  // Shallow to deep: every child level consumes completed parent locals.
  for (int level = 1; level <= plan.maximum_level; ++level) {
    enqueue_translation_level(plan.l2l, level, plan.coefficient_count, input,
                              output, stream,
                              "launch CUDA far-field L2L kernel");
  }
}

// L2P: `input` is the local coefficients of every node, `output` the sorted
// far field viewed as scalars (zeroed by the caller).
template <typename Scalar, typename Entry>
void CudaFarFieldExecutionPlan<Scalar, Entry>::enqueue_l2p(
    const Scalar *input, Scalar *output, cudaStream_t stream) const {
  const auto &plan = *implementation_;
  if (plan.use_procedural_l2p) {
    using Vector =
        std::conditional_t<std::is_same_v<Scalar, double>, Vec3, FloatVec3>;
    launch_procedural_l2p<Scalar, Vector>(
        plan.expansion_order, plan.procedural_l2p.leaves,
        plan.procedural_l2p.leaf_count, plan.procedural_l2p.lanes_per_leaf,
        plan.procedural_l2p.displacements, input, plan.procedural_l2p.factors,
        reinterpret_cast<Vector *>(output), stream);
    return;
  }
  if (plan.l2p.entry_count != 0) {
    const std::size_t items =
        static_cast<std::size_t>(plan.l2p.row_count) * entry_row_lanes;
    apply_csr_rows_kernel<Scalar, entry_row_lanes>
        <<<(items + far_field_threads - 1) / far_field_threads,
           far_field_threads, 0, stream>>>(
        plan.l2p.row_offsets, plan.l2p.inputs, plan.l2p.values,
        plan.l2p.row_count, input, output);
    check_cuda(cudaGetLastError(), "launch CUDA far-field L2P kernel");
  }
}

template <typename Scalar, typename Entry>
const CudaPlanStatistics &
CudaFarFieldExecutionPlan<Scalar, Entry>::statistics() const noexcept {
  return implementation_->statistics;
}

template class CudaFarFieldExecutionPlan<double, StaticOperatorEntry>;
template class CudaFarFieldExecutionPlan<float, FloatStaticOperatorEntry>;

} // namespace cdfmm::cuda_far_field_detail
