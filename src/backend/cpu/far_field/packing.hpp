// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/plan/static_plan.hpp"

namespace cdfmm::detail::cpu {

/**
 * @brief Level-scaled M2M or L2L translation matrices packed by input column.
 *
 * The eight canonical child-class operators are sparse coefficient maps whose
 * level dependence is the exact power-of-two factor
 * `2^(-(level - 1) * |degree(output) - degree(input)|)`.  For every child
 * level the factor is folded into the values once at construction, so the
 * evaluation loop needs no degree lookup and no `ldexp`.  Values are stored
 * per input column over the contiguous output range that holds that column's
 * entries (`[column_begin, column_end)`), which turns every translation into
 * unit-stride axpy updates of the output expansion.  The column structure is
 * shared by all levels of one class.
 */
template <typename Scalar>
struct PackedTranslationBank {
  int coefficient_count{0};
  /// Child levels 1..level_count are packed; block `(level - 1) * 8 + class`.
  int level_count{0};
  std::array<std::vector<int>, 8> column_begin{};
  std::array<std::vector<int>, 8> column_end{};
  std::array<std::vector<std::size_t>, 8> column_offset{};
  std::array<std::size_t, 8> class_values{};
  std::vector<std::size_t> block_offset{};
  std::vector<Scalar> values{};

  [[nodiscard]] std::size_t memory_bytes() const noexcept {
    std::size_t bytes = values.size() * sizeof(Scalar) +
                        block_offset.size() * sizeof(std::size_t);
    for (int child_class = 0; child_class < 8; ++child_class) {
      const auto index = static_cast<std::size_t>(child_class);
      bytes += (column_begin[index].size() + column_end[index].size()) *
                   sizeof(int) +
               column_offset[index].size() * sizeof(std::size_t);
    }
    return bytes;
  }
};

/**
 * @brief Dense P2M coefficient rows for every sorted source.
 *
 * `values[((source * 3) + component) * C + mode]` is the coefficient of the
 * source's moment component in multipole mode `mode`; zeros that the sparse
 * canonical operator omitted are stored explicitly.  A leaf's sources are
 * contiguous in sorted order, so a leaf's block is one contiguous range.
 */
template <typename Scalar>
struct PackedP2M {
  int coefficient_count{0};
  std::vector<Scalar> values{};

  [[nodiscard]] std::size_t memory_bytes() const noexcept {
    return values.size() * sizeof(Scalar);
  }
};

/**
 * @brief Flat L2P rows for every sorted target.
 *
 * `field[((target * 3) + component) * C + index]` holds the three field rows
 * and `potential[target * C + index]` the potential row, so a field-only
 * evaluation streams the field rows only.
 */
template <typename Scalar>
struct PackedL2P {
  int coefficient_count{0};
  std::vector<Scalar> field{};
  std::vector<Scalar> potential{};

  [[nodiscard]] std::size_t memory_bytes() const noexcept {
    return (field.size() + potential.size()) * sizeof(Scalar);
  }
};

/** @brief CPU execution packing of the far-field hierarchy for one precision. */
template <typename Scalar>
struct FarFieldPacking {
  PackedP2M<Scalar> p2m{};
  PackedTranslationBank<Scalar> m2m{};
  PackedTranslationBank<Scalar> l2l{};
  PackedL2P<Scalar> l2p{};
};

/**
 * @brief Builds the level-scaled column bank from eight canonical operators.
 *
 * @param degrees   Expansion degree of every coefficient.
 * @param level_count  Deepest child level to pack (levels 1..level_count).
 */
PackedTranslationBank<double> pack_translation_bank(
    std::span<const StaticCoefficientOperator> operators,
    std::span<const int> degrees, int level_count);
PackedTranslationBank<float> pack_translation_bank(
    std::span<const FloatStaticCoefficientOperator> operators,
    std::span<const int> degrees, int level_count);

PackedP2M<double> pack_p2m(std::span<const P2MPlan> plans,
                           std::size_t source_count, int coefficient_count);
PackedP2M<float> pack_p2m(std::span<const FloatP2MPlan> plans,
                          std::size_t source_count, int coefficient_count);

PackedL2P<double> pack_l2p(std::span<const StaticL2PEvaluator> evaluators,
                           int coefficient_count);
PackedL2P<float> pack_l2p(std::span<const FloatStaticL2PEvaluator> evaluators,
                          int coefficient_count);

/**
 * @brief Accumulates one leaf's P2M: `M += sum_s G_s m_s` over the leaf range.
 *
 * `moments` is the leaf's slice of the sorted moments; `first_source` is the
 * sorted index of its first source.  Rows are visited in source order and the
 * three components of one source are fused into one pass over the modes.
 */
template <typename Scalar, typename Moment>
inline void apply_packed_p2m(const PackedP2M<Scalar>& packing,
                             const std::size_t first_source,
                             const std::span<const Moment> moments,
                             Scalar* M) {
  const int n = packing.coefficient_count;
  const Scalar* rows =
      packing.values.data() + first_source * 3 * static_cast<std::size_t>(n);
  for (std::size_t source = 0; source < moments.size(); ++source) {
    const Moment moment = moments[source];
    const Scalar mx = static_cast<Scalar>(moment.x);
    const Scalar my = static_cast<Scalar>(moment.y);
    const Scalar mz = static_cast<Scalar>(moment.z);
    const Scalar* gx = rows + (source * 3) * static_cast<std::size_t>(n);
    const Scalar* gy = gx + n;
    const Scalar* gz = gy + n;
#pragma omp simd
    for (int mode = 0; mode < n; ++mode) {
      M[mode] += gx[mode] * mx + gy[mode] * my + gz[mode] * mz;
    }
  }
}

/**
 * @brief Accumulates one level-scaled translation: `output += A(level, class) input`.
 */
template <typename Scalar>
inline void apply_packed_translation(
    const PackedTranslationBank<Scalar>& bank, const int level,
    const int child_class, const Scalar* input, Scalar* output) {
  const int n = bank.coefficient_count;
  const auto class_index = static_cast<std::size_t>(child_class);
  const Scalar* block =
      bank.values.data() +
      bank.block_offset[static_cast<std::size_t>(level - 1) * 8 + class_index];
  const int* begin = bank.column_begin[class_index].data();
  const int* end = bank.column_end[class_index].data();
  const std::size_t* offset = bank.column_offset[class_index].data();
  for (int in = 0; in < n; ++in) {
    const Scalar x = input[in];
    const Scalar* column = block + offset[in];
    const int first = begin[in];
    const int last = end[in];
    Scalar* out = output + first;
#pragma omp simd
    for (int row = 0; row < last - first; ++row) {
      out[row] += column[row] * x;
    }
  }
}

/** @brief Field of one target from its flat L2P rows. */
template <typename Scalar>
inline void apply_packed_l2p_field(const PackedL2P<Scalar>& packing,
                                   const std::size_t target,
                                   const Scalar* L, Scalar& Hx, Scalar& Hy,
                                   Scalar& Hz) {
  const int n = packing.coefficient_count;
  const Scalar* fx =
      packing.field.data() + target * 3 * static_cast<std::size_t>(n);
  const Scalar* fy = fx + n;
  const Scalar* fz = fy + n;
  Scalar sx = Scalar{0};
  Scalar sy = Scalar{0};
  Scalar sz = Scalar{0};
#pragma omp simd reduction(+ : sx, sy, sz)
  for (int index = 0; index < n; ++index) {
    sx += fx[index] * L[index];
    sy += fy[index] * L[index];
    sz += fz[index] * L[index];
  }
  Hx = sx;
  Hy = sy;
  Hz = sz;
}

/** @brief Potential of one target from its flat L2P row. */
template <typename Scalar>
inline Scalar apply_packed_l2p_potential(const PackedL2P<Scalar>& packing,
                                         const std::size_t target,
                                         const Scalar* L) {
  const int n = packing.coefficient_count;
  const Scalar* row =
      packing.potential.data() + target * static_cast<std::size_t>(n);
  Scalar phi = Scalar{0};
#pragma omp simd reduction(+ : phi)
  for (int index = 0; index < n; ++index) {
    phi += row[index] * L[index];
  }
  return phi;
}

} // namespace cdfmm::detail::cpu
