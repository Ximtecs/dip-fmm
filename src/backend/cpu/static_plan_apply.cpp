// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cpu/static_plan_apply.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "cdfmm/tensor_dictionary.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace cdfmm {

namespace {

void validate_operator_dimensions(
    const StaticCoefficientOperator& operator_map,
    const std::span<const double> input,
    const std::span<double> output)
{
    if (input.size() != static_cast<std::size_t>(operator_map.input_size) ||
        output.size() != static_cast<std::size_t>(operator_map.output_size)) {
        throw std::invalid_argument(
            "static operator dimensions are inconsistent");
    }
}

} // namespace

void apply_static_m2l_plan(const StaticM2LPlan &plan, const int level,
    const std::span<const double> multipoles,
                           const std::span<double> locals) {
    const int n = plan.coefficient_count;
    const bool explicit_targets = !plan.target_level_offsets.empty();
    const int target_begin = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level)]
        : plan.level_target_begin[static_cast<std::size_t>(level)];
    const int target_end = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level + 1)]
        : plan.level_target_end[static_cast<std::size_t>(level)];

    // The target schedule owns one output row at a time.  Iterating only this
    // target level avoids revisiting other rows; one iteration owns one target
    // coefficient and therefore needs no atomic accumulation.
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(target_end - target_begin) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = explicit_targets
            ? plan.target_nodes_by_level[static_cast<std::size_t>(
                target_begin + static_cast<int>(output / n))]
            : target_begin + static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        double value = 0.0;
        const int row_begin = plan.target_row_offsets[target];
        const int row_end = plan.target_row_offsets[target + 1];
        for (int interaction = row_begin; interaction < row_end; ++interaction) {
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
            const int target_level = plan.target_levels.empty()
                ? level : plan.target_levels[interaction];
            if (target_level != level) {
                continue;
            }
            const int source_level = plan.source_levels.empty()
                ? level : plan.source_levels[interaction];
            const double *source_scale = plan.multipole_scaling.data() +
                static_cast<std::size_t>(source_level) * n;
            const double *matrix_column =
                plan.matrices.data() +
                static_cast<std::size_t>(matrix_id) * n * n + beta;
            const double *source_M =
                multipoles.data() + static_cast<std::size_t>(source) * n;
            for (int alpha = 0; alpha < n; ++alpha) {
                value += matrix_column[static_cast<std::size_t>(alpha) * n] *
                    source_scale[alpha] * source_M[alpha];
            }
        }
        locals[static_cast<std::size_t>(target) * n + beta] +=
            plan.local_scaling[static_cast<std::size_t>(level) * n + beta] * value;
    }
}

void apply_static_m2l_plan(const FloatStaticM2LPlan &plan, const int level,
    const std::span<const float> multipoles,
                           const std::span<float> locals) {
    const int n = plan.coefficient_count;
    const bool explicit_targets = !plan.target_level_offsets.empty();
    const int target_begin = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level)]
        : plan.level_target_begin[static_cast<std::size_t>(level)];
    const int target_end = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level + 1)]
        : plan.level_target_end[static_cast<std::size_t>(level)];
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(target_end - target_begin) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = explicit_targets
            ? plan.target_nodes_by_level[static_cast<std::size_t>(
                target_begin + static_cast<int>(output / n))]
            : target_begin + static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        float value = 0.0F;
        const int row_begin = plan.target_row_offsets[target];
        const int row_end = plan.target_row_offsets[target + 1];
        for (int interaction = row_begin; interaction < row_end;
             ++interaction) {
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
            const int target_level = plan.target_levels.empty()
                ? level : plan.target_levels[interaction];
            if (target_level != level) {
                continue;
            }
            const int source_level = plan.source_levels.empty()
                ? level : plan.source_levels[interaction];
            const float* matrix_column = plan.matrices.data() +
                static_cast<std::size_t>(matrix_id) * n * n + beta;
            const float* source_M = multipoles.data() +
                static_cast<std::size_t>(source) * n;
            for (int alpha = 0; alpha < n; ++alpha) {
                value += matrix_column[static_cast<std::size_t>(alpha) * n] *
                    plan.multipole_scaling[static_cast<std::size_t>(source_level) * n + alpha] *
                    source_M[alpha];
            }
        }
        locals[static_cast<std::size_t>(target) * n + beta] +=
            plan.local_scaling[static_cast<std::size_t>(level) * n + beta] * value;
    }
}

void apply_static_m2l_plan(
    const StaticM2LPlan &plan, const int level,
    const std::span<const std::vector<double>> multipoles,
    const std::span<std::vector<double>> locals) {
    const int n = plan.coefficient_count;
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(locals.size()) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        double value = 0.0;
        for (int interaction = plan.target_row_offsets[target];
         interaction < plan.target_row_offsets[target + 1]; ++interaction) {
            const int target_level = plan.target_levels.empty()
                ? (plan.interaction_levels.empty() ? level
                   : plan.interaction_levels[interaction])
                : plan.target_levels[interaction];
            if (target_level != level) {
                continue;
            }
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
            const int source_level = plan.source_levels.empty()
                ? level : plan.source_levels[interaction];
            const double local_scale = plan.local_scaling[
                static_cast<std::size_t>(target_level) * n + beta];
            for (int alpha = 0; alpha < n; ++alpha) {
                const std::size_t matrix_index =
                    (static_cast<std::size_t>(matrix_id) * n + alpha) * n + beta;
                value += local_scale * plan.matrices[matrix_index] *
                 plan.multipole_scaling[static_cast<std::size_t>(source_level) * n +
                                        alpha] *
                    multipoles[source][alpha];
            }
        }
        locals[target][beta] += value;
    }
}


void apply_static_p2p_operator(
    const StaticP2POperator& operator_map,
    const std::span<const Vec3> dipole_moments,
    const std::span<Vec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(operator_map.source_count) ||
        H.size() != static_cast<std::size_t>(operator_map.target_count) ||
        (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
        throw std::invalid_argument("static P2P dimensions are inconsistent");
    }
    #pragma omp parallel for schedule(static) if(operator_map.target_count >= 64)
    for (int target = 0; target < operator_map.target_count; ++target) {
        Vec3 field{};
        const int self = target_source_indices.empty()
            ? -1 : target_source_indices[static_cast<std::size_t>(target)];
        for (int entry = operator_map.row_offsets[static_cast<std::size_t>(target)];
             entry < operator_map.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const StaticDipoleBlock& block = operator_map.blocks[
                static_cast<std::size_t>(entry)];
            if (block.skip_for_identity != 0 && block.source == self) {
                continue;
            }
            const Vec3 m = dipole_moments[static_cast<std::size_t>(block.source)];
            accumulate_static_dipole_block(block, m, field);
        }
        H[static_cast<std::size_t>(target)] += field;
    }
}

void apply_static_p2p_compact_plan(
    const StaticP2PCompactPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H,
    const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      plan.skip_for_identity.size() != plan.source_indices.size() ||
      (!target_source_indices.empty() &&
       target_source_indices.size() != H.size())) {
    throw std::invalid_argument("compact P2P dimensions are inconsistent");
  }

#pragma omp parallel for schedule(static) if (plan.target_count >= 64)
  for (int target = 0; target < plan.target_count; ++target) {
    double Hx = 0.0;
    double Hy = 0.0;
    double Hz = 0.0;
    const int self =
        target_source_indices.empty()
            ? -1
            : target_source_indices[static_cast<std::size_t>(target)];
    const int begin = plan.row_offsets[static_cast<std::size_t>(target)];
    const int end = plan.row_offsets[static_cast<std::size_t>(target) + 1];
#pragma omp simd reduction(+ : Hx, Hy, Hz)
    for (int entry = begin; entry < end; ++entry) {
      const int source = plan.source_indices[static_cast<std::size_t>(entry)];
      if (plan.skip_for_identity[static_cast<std::size_t>(entry)] != 0 &&
          source == self) {
        continue;
      }
      const Vec3 m = dipole_moments[static_cast<std::size_t>(source)];
      const std::size_t index = static_cast<std::size_t>(entry);
      Hx += plan.tensors[0][index] * m.x + plan.tensors[1][index] * m.y +
            plan.tensors[2][index] * m.z;
      Hy += plan.tensors[1][index] * m.x + plan.tensors[3][index] * m.y +
            plan.tensors[4][index] * m.z;
      Hz += plan.tensors[2][index] * m.x + plan.tensors[4][index] * m.y +
            plan.tensors[5][index] * m.z;
    }
    H[static_cast<std::size_t>(target)] += Vec3{Hx, Hy, Hz};
  }
}

void apply_static_p2p_operator(
    const FloatStaticP2POperator& operator_map,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() !=
            static_cast<std::size_t>(operator_map.source_count) ||
        H.size() != static_cast<std::size_t>(operator_map.target_count) ||
        (!target_source_indices.empty() &&
         target_source_indices.size() != H.size())) {
        throw std::invalid_argument("static FP32 P2P dimensions are inconsistent");
    }
#pragma omp parallel for schedule(static) if(operator_map.target_count >= 64)
    for (int target = 0; target < operator_map.target_count; ++target) {
        FloatVec3 field{};
        const int self = target_source_indices.empty()
            ? -1 : target_source_indices[static_cast<std::size_t>(target)];
        for (int entry = operator_map.row_offsets[static_cast<std::size_t>(target)];
             entry < operator_map.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const FloatStaticDipoleBlock& block =
                operator_map.blocks[static_cast<std::size_t>(entry)];
            if (block.skip_for_identity != 0 && block.source == self) {
                continue;
            }
            accumulate_static_dipole_block(
          block, dipole_moments[static_cast<std::size_t>(block.source)], field);
        }
        H[static_cast<std::size_t>(target)] += field;
    }
}

void apply_static_p2p_leaf_plan(
    const FloatStaticP2PLeafPlan& plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
        H.size() != static_cast<std::size_t>(plan.target_count) ||
        (!target_source_indices.empty() &&
         target_source_indices.size() != H.size())) {
        throw std::invalid_argument("leaf FP32 P2P dimensions are inconsistent");
    }
    const int target_leaf_count = static_cast<int>(plan.target_begins.size());
#pragma omp parallel for schedule(static) if(target_leaf_count >= 8)
    for (int target_leaf = 0; target_leaf < target_leaf_count; ++target_leaf) {
        const int target_begin =
            plan.target_begins[static_cast<std::size_t>(target_leaf)];
        const int target_count =
            plan.target_counts[static_cast<std::size_t>(target_leaf)];
        for (int local_target = 0; local_target < target_count; ++local_target) {
            const int target = target_begin + local_target;
            const int self = target_source_indices.empty()
                ? -1 : target_source_indices[static_cast<std::size_t>(target)];
            FloatVec3 field{};
            for (int block_index =
                     plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
                 block_index < plan.leaf_row_offsets[
                     static_cast<std::size_t>(target_leaf) + 1];
                 ++block_index) {
                const StaticP2PLeafBlock& block =
                    plan.blocks[static_cast<std::size_t>(block_index)];
                const std::size_t tensor_begin = block.tensor_offset +
                    static_cast<std::size_t>(local_target) * block.source_count;
                for (int local_source = 0; local_source < block.source_count;
                     ++local_source) {
                    const int source = block.source_begin + local_source;
                    if (source == self) {
                        continue;
                    }
                    const std::size_t index = tensor_begin + local_source;
                    const FloatVec3 moment =
                        dipole_moments[static_cast<std::size_t>(source)];
                    field.x += plan.tensors[0][index] * moment.x +
                        plan.tensors[1][index] * moment.y +
                        plan.tensors[2][index] * moment.z;
                    field.y += plan.tensors[1][index] * moment.x +
                        plan.tensors[3][index] * moment.y +
                        plan.tensors[4][index] * moment.z;
                    field.z += plan.tensors[2][index] * moment.x +
                        plan.tensors[4][index] * moment.y +
                        plan.tensors[5][index] * moment.z;
                }
            }
            H[static_cast<std::size_t>(target)] += field;
        }
    }
}

void apply_static_p2p_tensor_dictionary_plan(
    const FloatStaticP2PTensorDictionaryPlan &plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
    throw std::invalid_argument("FP32 Tensor6 dictionary P2P dimensions are inconsistent");
  }
#pragma omp parallel for schedule(static) if (plan.target_begins.size() >= 8)
  for (int target_leaf = 0;
       target_leaf < static_cast<int>(plan.target_begins.size()); ++target_leaf) {
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_target = 0; local_target < target_count; ++local_target) {
      const int target = target_begin + local_target;
      const int self = target_source_indices.empty() ? -1 : target_source_indices[static_cast<std::size_t>(target)];
      FloatVec3 field{};
      for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
           block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1]; ++block_index) {
        const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
        const std::size_t begin = block.tensor_offset + static_cast<std::size_t>(local_target) * block.source_count;
        for (int local_source = 0; local_source < block.source_count; ++local_source) {
          const int source = block.source_begin + local_source;
          if (plan.skip_for_identity && source == self) continue;
          const std::uint32_t token = plan.tokens[begin + local_source];
          const auto id = tensor6_token_id(token);
          const auto signs = tensor6_token_sign_mask(token);
          const auto value = [&](int component) {
            const float coefficient = plan.tensors[static_cast<std::size_t>(component)][id];
            return (signs & (1U << component)) != 0U ? -coefficient : coefficient;
          };
          const auto moment = dipole_moments[static_cast<std::size_t>(source)];
          const float xx = value(0), xy = value(1), xz = value(2),
                      yy = value(3), yz = value(4), zz = value(5);
          field.x += xx * moment.x + xy * moment.y + xz * moment.z;
          field.y += xy * moment.x + yy * moment.y + yz * moment.z;
          field.z += xz * moment.x + yz * moment.y + zz * moment.z;
        }
      }
      H[static_cast<std::size_t>(target)] += field;
    }
  }
}

void apply_static_p2p_bsr_plan(
    const FloatStaticP2PBsrPlan& plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
        H.size() != static_cast<std::size_t>(plan.target_count) ||
        (!target_source_indices.empty() &&
         !std::equal(target_source_indices.begin(), target_source_indices.end(),
                     plan.target_source_indices.begin(),
                     plan.target_source_indices.end()))) {
        throw std::invalid_argument(
            "FP32 BSR P2P dimensions or identities are inconsistent");
    }
#pragma omp parallel for schedule(static) if(plan.target_count >= 64)
    for (int target = 0; target < plan.target_count; ++target) {
        FloatVec3 field{};
        for (int entry = plan.row_offsets[static_cast<std::size_t>(target)];
             entry < plan.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const FloatVec3 moment = dipole_moments[static_cast<std::size_t>(
                plan.source_indices[static_cast<std::size_t>(entry)])];
            const float* block =
                plan.values.data() + static_cast<std::size_t>(entry) * 9;
            field.x += block[0] * moment.x + block[1] * moment.y +
                block[2] * moment.z;
            field.y += block[3] * moment.x + block[4] * moment.y +
                block[5] * moment.z;
            field.z += block[6] * moment.x + block[7] * moment.y +
                block[8] * moment.z;
        }
        H[static_cast<std::size_t>(target)] += field;
    }
}

void apply_static_p2p_leaf_plan(
    const StaticP2PLeafPlan &plan, const std::span<const Vec3> dipole_moments,
    const std::span<Vec3> H, const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() &&
       target_source_indices.size() != H.size())) {
    throw std::invalid_argument("leaf P2P dimensions are inconsistent");
  }

  const int target_leaf_count = static_cast<int>(plan.target_begins.size());
#pragma omp parallel for schedule(static) if (target_leaf_count >= 8)
  for (int target_leaf = 0; target_leaf < target_leaf_count; ++target_leaf) {
    const int target_begin =
        plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count =
        plan.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_target = 0; local_target < target_count; ++local_target) {
      const int target = target_begin + local_target;
      const int self =
          target_source_indices.empty()
              ? -1
              : target_source_indices[static_cast<std::size_t>(target)];
      double Hx = 0.0;
      double Hy = 0.0;
      double Hz = 0.0;
      for (int block_index =
               plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
           block_index <
           plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
           ++block_index) {
        const StaticP2PLeafBlock &block =
            plan.blocks[static_cast<std::size_t>(block_index)];
        const std::size_t tensor_begin =
            block.tensor_offset +
            static_cast<std::size_t>(local_target) * block.source_count;
#pragma omp simd reduction(+ : Hx, Hy, Hz)
        for (int local_source = 0; local_source < block.source_count;
             ++local_source) {
          const int source = block.source_begin + local_source;
          if (source == self) {
            continue;
          }
          const std::size_t index = tensor_begin + local_source;
          const Vec3 m = dipole_moments[static_cast<std::size_t>(source)];
          Hx += plan.tensors[0][index] * m.x + plan.tensors[1][index] * m.y +
                plan.tensors[2][index] * m.z;
          Hy += plan.tensors[1][index] * m.x + plan.tensors[3][index] * m.y +
                plan.tensors[4][index] * m.z;
          Hz += plan.tensors[2][index] * m.x + plan.tensors[4][index] * m.y +
                plan.tensors[5][index] * m.z;
        }
      }
      H[static_cast<std::size_t>(target)] += Vec3{Hx, Hy, Hz};
    }
  }
}

void apply_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H,
    const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
    throw std::invalid_argument("Tensor6 dictionary P2P dimensions are inconsistent");
  }
#pragma omp parallel for schedule(static) if (plan.target_begins.size() >= 8)
  for (int target_leaf = 0;
       target_leaf < static_cast<int>(plan.target_begins.size()); ++target_leaf) {
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_target = 0; local_target < target_count; ++local_target) {
      const int target = target_begin + local_target;
      const int self = target_source_indices.empty() ? -1 :
          target_source_indices[static_cast<std::size_t>(target)];
      Vec3 field{};
      for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
           block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
           ++block_index) {
        const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
        const std::size_t begin = block.tensor_offset +
            static_cast<std::size_t>(local_target) * block.source_count;
        for (int local_source = 0; local_source < block.source_count; ++local_source) {
          const int source = block.source_begin + local_source;
          if (plan.skip_for_identity && source == self) {
            continue;
          }
          const std::uint32_t token = plan.tokens[begin + local_source];
          const std::uint32_t id = tensor6_token_id(token);
          const std::uint8_t signs = tensor6_token_sign_mask(token);
          const auto signed_value = [&](const int component) {
            const double value = plan.tensors[static_cast<std::size_t>(component)][id];
            return (signs & (1U << component)) != 0U ? -value : value;
          };
          const Vec3 moment = dipole_moments[static_cast<std::size_t>(source)];
          const double xx = signed_value(0), xy = signed_value(1),
                       xz = signed_value(2), yy = signed_value(3),
                       yz = signed_value(4), zz = signed_value(5);
          field.x += xx * moment.x + xy * moment.y + xz * moment.z;
          field.y += xy * moment.x + yy * moment.y + yz * moment.z;
          field.z += xz * moment.x + yz * moment.y + zz * moment.z;
        }
      }
      H[static_cast<std::size_t>(target)] += field;
    }
  }
}

namespace {

template <typename Scalar, typename Vector, typename Plan, typename Token>
void apply_signed_tensor_dictionary_whole_tile_impl(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const std::span<const Token> tokens) {
  constexpr int max_tile_size = 128;
  const int work_count = static_cast<int>(plan.tile_leaf_indices.size());
  const Scalar *const xx = plan.tensors[0].data();
  const Scalar *const xy = plan.tensors[1].data();
  const Scalar *const xz = plan.tensors[2].data();
  const Scalar *const yy = plan.tensors[3].data();
  const Scalar *const yz = plan.tensors[4].data();
  const Scalar *const zz = plan.tensors[5].data();
  const Token *const token_data = tokens.data();
#pragma omp parallel for schedule(static) if (work_count >= 8)
  for (int work = 0; work < work_count; ++work) {
    const int target_leaf = plan.tile_leaf_indices[static_cast<std::size_t>(work)];
    const int local_begin = plan.tile_target_offsets[static_cast<std::size_t>(work)];
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    const int lanes = std::min(plan.target_tile_size, target_count - local_begin);
    std::array<Scalar, max_tile_size> Hx{};
    std::array<Scalar, max_tile_size> Hy{};
    std::array<Scalar, max_tile_size> Hz{};
    for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
         block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
         ++block_index) {
      const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
      for (int local_source = 0; local_source < block.source_count; ++local_source) {
        const Vector moment = moments[static_cast<std::size_t>(block.source_begin + local_source)];
        const Token *const block_tokens = token_data + block.tensor_offset +
            static_cast<std::size_t>(local_source) * target_count + local_begin;
#pragma omp simd
        for (int lane = 0; lane < lanes; ++lane) {
          const std::size_t variant = block_tokens[lane];
          Hx[static_cast<std::size_t>(lane)] += xx[variant] * moment.x +
              xy[variant] * moment.y + xz[variant] * moment.z;
          Hy[static_cast<std::size_t>(lane)] += xy[variant] * moment.x +
              yy[variant] * moment.y + yz[variant] * moment.z;
          Hz[static_cast<std::size_t>(lane)] += xz[variant] * moment.x +
              yz[variant] * moment.y + zz[variant] * moment.z;
        }
      }
    }
    for (int lane = 0; lane < lanes; ++lane) {
      fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
          Vector{Hx[static_cast<std::size_t>(lane)], Hy[static_cast<std::size_t>(lane)],
                 Hz[static_cast<std::size_t>(lane)]};
    }
  }
}

#if defined(__AVX2__) && defined(__FMA__)

template <typename Token>
[[nodiscard]] inline __m128i load_four_variant_ids(
    const Token *const tokens) noexcept {
  if constexpr (sizeof(Token) == 1) {
    std::uint32_t packed{};
    std::memcpy(&packed, tokens, sizeof(packed));
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(packed)));
  } else if constexpr (sizeof(Token) == 2) {
    std::uint64_t packed{};
    std::memcpy(&packed, tokens, sizeof(packed));
    return _mm_cvtepu16_epi32(_mm_cvtsi64_si128(static_cast<long long>(packed)));
  } else {
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(tokens));
  }
}

template <typename Token>
[[nodiscard]] inline __m256i load_eight_variant_ids(
    const Token *const tokens) noexcept {
  if constexpr (sizeof(Token) == 1) {
    std::uint64_t packed{};
    std::memcpy(&packed, tokens, sizeof(packed));
    return _mm256_cvtepu8_epi32(
        _mm_cvtsi64_si128(static_cast<long long>(packed)));
  } else if constexpr (sizeof(Token) == 2) {
    return _mm256_cvtepu16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(tokens)));
  } else {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tokens));
  }
}

#if defined(__GNUC__) || defined(__clang__)
#define CDFMM_SIGNED_P2P_NOINLINE __attribute__((noinline))
#else
#define CDFMM_SIGNED_P2P_NOINLINE
#endif

template <typename Vector, typename Plan, typename Token>
CDFMM_SIGNED_P2P_NOINLINE void apply_signed_microtile_avx2_f64(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const Token *const token_data,
    const int target_leaf, const int target_begin, const int target_count,
    const int local_begin) {
  const double *const xx = plan.tensors[0].data();
  const double *const xy = plan.tensors[1].data();
  const double *const xz = plan.tensors[2].data();
  const double *const yy = plan.tensors[3].data();
  const double *const yz = plan.tensors[4].data();
  const double *const zz = plan.tensors[5].data();
  __m256d hx = _mm256_setzero_pd();
  __m256d hy = _mm256_setzero_pd();
  __m256d hz = _mm256_setzero_pd();
  for (int block_index =
           plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
       block_index <
       plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
       ++block_index) {
    const StaticP2PLeafBlock &block =
        plan.blocks[static_cast<std::size_t>(block_index)];
    for (int local_source = 0; local_source < block.source_count;
         ++local_source) {
      const Vector moment =
          moments[static_cast<std::size_t>(block.source_begin + local_source)];
      const Token *const block_tokens = token_data + block.tensor_offset +
          static_cast<std::size_t>(local_source) * target_count + local_begin;
      const __m128i ids = load_four_variant_ids(block_tokens);
      const __m256d vxx = _mm256_i32gather_pd(xx, ids, 8);
      const __m256d vxy = _mm256_i32gather_pd(xy, ids, 8);
      const __m256d vxz = _mm256_i32gather_pd(xz, ids, 8);
      const __m256d vyy = _mm256_i32gather_pd(yy, ids, 8);
      const __m256d vyz = _mm256_i32gather_pd(yz, ids, 8);
      const __m256d vzz = _mm256_i32gather_pd(zz, ids, 8);
      const __m256d mx = _mm256_set1_pd(moment.x);
      const __m256d my = _mm256_set1_pd(moment.y);
      const __m256d mz = _mm256_set1_pd(moment.z);
      hx = _mm256_fmadd_pd(vxz, mz,
                           _mm256_fmadd_pd(vxy, my,
                                          _mm256_fmadd_pd(vxx, mx, hx)));
      hy = _mm256_fmadd_pd(vyz, mz,
                           _mm256_fmadd_pd(vyy, my,
                                          _mm256_fmadd_pd(vxy, mx, hy)));
      hz = _mm256_fmadd_pd(vzz, mz,
                           _mm256_fmadd_pd(vyz, my,
                                          _mm256_fmadd_pd(vxz, mx, hz)));
    }
  }
  alignas(32) std::array<double, 4> hx_values{}, hy_values{}, hz_values{};
  _mm256_store_pd(hx_values.data(), hx);
  _mm256_store_pd(hy_values.data(), hy);
  _mm256_store_pd(hz_values.data(), hz);
  for (int lane = 0; lane < 4; ++lane) {
    fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
        Vector{hx_values[static_cast<std::size_t>(lane)],
               hy_values[static_cast<std::size_t>(lane)],
               hz_values[static_cast<std::size_t>(lane)]};
  }
}

template <typename Vector, typename Plan, typename Token>
CDFMM_SIGNED_P2P_NOINLINE void apply_signed_microtile_avx2_f32(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const Token *const token_data,
    const int target_leaf, const int target_begin, const int target_count,
    const int local_begin) {
  const float *const xx = plan.tensors[0].data();
  const float *const xy = plan.tensors[1].data();
  const float *const xz = plan.tensors[2].data();
  const float *const yy = plan.tensors[3].data();
  const float *const yz = plan.tensors[4].data();
  const float *const zz = plan.tensors[5].data();
  __m256 hx = _mm256_setzero_ps();
  __m256 hy = _mm256_setzero_ps();
  __m256 hz = _mm256_setzero_ps();
  for (int block_index =
           plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
       block_index <
       plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
       ++block_index) {
    const StaticP2PLeafBlock &block =
        plan.blocks[static_cast<std::size_t>(block_index)];
    for (int local_source = 0; local_source < block.source_count;
         ++local_source) {
      const Vector moment =
          moments[static_cast<std::size_t>(block.source_begin + local_source)];
      const Token *const block_tokens = token_data + block.tensor_offset +
          static_cast<std::size_t>(local_source) * target_count + local_begin;
      const __m256i ids = load_eight_variant_ids(block_tokens);
      const __m256 vxx = _mm256_i32gather_ps(xx, ids, 4);
      const __m256 vxy = _mm256_i32gather_ps(xy, ids, 4);
      const __m256 vxz = _mm256_i32gather_ps(xz, ids, 4);
      const __m256 vyy = _mm256_i32gather_ps(yy, ids, 4);
      const __m256 vyz = _mm256_i32gather_ps(yz, ids, 4);
      const __m256 vzz = _mm256_i32gather_ps(zz, ids, 4);
      const __m256 mx = _mm256_set1_ps(moment.x);
      const __m256 my = _mm256_set1_ps(moment.y);
      const __m256 mz = _mm256_set1_ps(moment.z);
      hx = _mm256_fmadd_ps(vxz, mz,
                           _mm256_fmadd_ps(vxy, my,
                                          _mm256_fmadd_ps(vxx, mx, hx)));
      hy = _mm256_fmadd_ps(vyz, mz,
                           _mm256_fmadd_ps(vyy, my,
                                          _mm256_fmadd_ps(vxy, mx, hy)));
      hz = _mm256_fmadd_ps(vzz, mz,
                           _mm256_fmadd_ps(vyz, my,
                                          _mm256_fmadd_ps(vxz, mx, hz)));
    }
  }
  alignas(32) std::array<float, 8> hx_values{}, hy_values{}, hz_values{};
  _mm256_store_ps(hx_values.data(), hx);
  _mm256_store_ps(hy_values.data(), hy);
  _mm256_store_ps(hz_values.data(), hz);
  for (int lane = 0; lane < 8; ++lane) {
    fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
        Vector{hx_values[static_cast<std::size_t>(lane)],
               hy_values[static_cast<std::size_t>(lane)],
               hz_values[static_cast<std::size_t>(lane)]};
  }
}

#undef CDFMM_SIGNED_P2P_NOINLINE
#endif

template <typename Scalar, typename Vector, typename Plan, typename Token>
inline void apply_signed_microtile_portable(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const Token *const token_data,
    const int target_leaf, const int target_begin, const int target_count,
    const int local_begin, const int lanes) {
  constexpr int width = std::is_same_v<Scalar, float> ? 8 : 4;
  std::array<Scalar, width> hx{}, hy{}, hz{};
  for (int block_index =
           plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
       block_index <
       plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
       ++block_index) {
    const StaticP2PLeafBlock &block =
        plan.blocks[static_cast<std::size_t>(block_index)];
    for (int local_source = 0; local_source < block.source_count;
         ++local_source) {
      const Vector moment =
          moments[static_cast<std::size_t>(block.source_begin + local_source)];
      const Token *const block_tokens = token_data + block.tensor_offset +
          static_cast<std::size_t>(local_source) * target_count + local_begin;
#pragma omp simd
      for (int lane = 0; lane < lanes; ++lane) {
        const std::size_t variant = block_tokens[lane];
        hx[static_cast<std::size_t>(lane)] +=
            plan.tensors[0][variant] * moment.x +
            plan.tensors[1][variant] * moment.y +
            plan.tensors[2][variant] * moment.z;
        hy[static_cast<std::size_t>(lane)] +=
            plan.tensors[1][variant] * moment.x +
            plan.tensors[3][variant] * moment.y +
            plan.tensors[4][variant] * moment.z;
        hz[static_cast<std::size_t>(lane)] +=
            plan.tensors[2][variant] * moment.x +
            plan.tensors[4][variant] * moment.y +
            plan.tensors[5][variant] * moment.z;
      }
    }
  }
  for (int lane = 0; lane < lanes; ++lane) {
    fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
        Vector{hx[static_cast<std::size_t>(lane)],
               hy[static_cast<std::size_t>(lane)],
               hz[static_cast<std::size_t>(lane)]};
  }
}

template <typename Scalar, typename Vector, typename Plan, typename Token>
void apply_signed_tensor_dictionary_microtile_impl(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const std::span<const Token> tokens) {
  constexpr int width = std::is_same_v<Scalar, float> ? 8 : 4;
  const int work_count = static_cast<int>(plan.tile_leaf_indices.size());
  const Token *const token_data = tokens.data();
#pragma omp parallel for schedule(static) if (work_count >= 8)
  for (int work = 0; work < work_count; ++work) {
    const int target_leaf = plan.tile_leaf_indices[static_cast<std::size_t>(work)];
    const int tile_begin = plan.tile_target_offsets[static_cast<std::size_t>(work)];
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    const int tile_end =
        std::min(target_count, tile_begin + plan.target_tile_size);
    int local_begin = tile_begin;
    for (; local_begin + width <= tile_end; local_begin += width) {
#if defined(__AVX2__) && defined(__FMA__)
      if constexpr (std::is_same_v<Scalar, float>) {
        apply_signed_microtile_avx2_f32(
            plan, moments, fields, token_data, target_leaf, target_begin,
            target_count, local_begin);
      } else {
        apply_signed_microtile_avx2_f64(
            plan, moments, fields, token_data, target_leaf, target_begin,
            target_count, local_begin);
      }
#else
      apply_signed_microtile_portable<Scalar>(
          plan, moments, fields, token_data, target_leaf, target_begin,
          target_count, local_begin, width);
#endif
    }
    if (local_begin < tile_end) {
      apply_signed_microtile_portable<Scalar>(
          plan, moments, fields, token_data, target_leaf, target_begin,
          target_count, local_begin, tile_end - local_begin);
    }
  }
}

template <typename Scalar, typename Vector, typename Plan>
void validate_signed_tensor_dictionary(const Plan &plan,
                                       const std::span<const Vector> moments,
                                       const std::span<Vector> fields) {
  if (moments.size() != static_cast<std::size_t>(plan.source_count) ||
      fields.size() != static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("signed Tensor6 dictionary P2P dimensions are inconsistent");
  }
  if (plan.target_tile_size <= 0 || plan.target_tile_size > 128 ||
      plan.tile_leaf_indices.size() != plan.tile_target_offsets.size()) {
    throw std::invalid_argument("signed Tensor6 dictionary plan is malformed");
  }
  if (plan.token_width_bytes != 1 && plan.token_width_bytes != 2 &&
      plan.token_width_bytes != 4) {
    throw std::invalid_argument("signed Tensor6 dictionary token width is invalid");
  }
}

template <typename Scalar, typename Vector, typename Plan>
void apply_signed_tensor_dictionary(const Plan &plan,
                                    const std::span<const Vector> moments,
                                    const std::span<Vector> fields) {
  validate_signed_tensor_dictionary<Scalar>(plan, moments, fields);
  if (plan.token_width_bytes == 1) {
    apply_signed_tensor_dictionary_microtile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens8});
  } else if (plan.token_width_bytes == 2) {
    apply_signed_tensor_dictionary_microtile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens16});
  } else {
    apply_signed_tensor_dictionary_microtile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens32});
  }
}

template <typename Scalar, typename Vector, typename Plan>
void apply_signed_tensor_dictionary_whole_tile(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields) {
  validate_signed_tensor_dictionary<Scalar>(plan, moments, fields);
  if (plan.token_width_bytes == 1) {
    apply_signed_tensor_dictionary_whole_tile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens8});
  } else if (plan.token_width_bytes == 2) {
    apply_signed_tensor_dictionary_whole_tile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens16});
  } else {
    apply_signed_tensor_dictionary_whole_tile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens32});
  }
}

} // namespace

void apply_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H) {
  apply_signed_tensor_dictionary<double>(plan, dipole_moments, H);
}

void apply_static_p2p_signed_tensor_dictionary_plan_whole_tile(
    const StaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H) {
  apply_signed_tensor_dictionary_whole_tile<double>(plan, dipole_moments, H);
}

void apply_static_p2p_signed_tensor_dictionary_plan(
    const FloatStaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H) {
  apply_signed_tensor_dictionary<float>(plan, dipole_moments, H);
}

void apply_static_p2p_signed_tensor_dictionary_plan_whole_tile(
    const FloatStaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H) {
  apply_signed_tensor_dictionary_whole_tile<float>(plan, dipole_moments, H);
}

const char *static_p2p_signed_simd_path() noexcept {
#if defined(__AVX2__) && defined(__FMA__)
  return "avx2-gather-fma";
#else
  return "compiler-simd-portable";
#endif
}

int static_p2p_signed_simd_width(const bool single_precision) noexcept {
  return single_precision ? 8 : 4;
}

void apply_static_p2p_bsr_plan(
    const StaticP2PBsrPlan &plan, const std::span<const Vec3> dipole_moments,
    const std::span<Vec3> H, const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() &&
       !std::equal(target_source_indices.begin(), target_source_indices.end(),
                   plan.target_source_indices.begin(),
                   plan.target_source_indices.end()))) {
    throw std::invalid_argument(
        "BSR P2P dimensions or fixed identities are inconsistent");
  }

#pragma omp parallel for schedule(static) if (plan.target_count >= 64)
  for (int target = 0; target < plan.target_count; ++target) {
    Vec3 field{};
    for (int entry = plan.row_offsets[static_cast<std::size_t>(target)];
         entry < plan.row_offsets[static_cast<std::size_t>(target) + 1];
         ++entry) {
      const Vec3 moment = dipole_moments[static_cast<std::size_t>(
          plan.source_indices[static_cast<std::size_t>(entry)])];
      const double *block =
          plan.values.data() + static_cast<std::size_t>(entry) * 9;
      field.x +=
          block[0] * moment.x + block[1] * moment.y + block[2] * moment.z;
      field.y +=
          block[3] * moment.x + block[4] * moment.y + block[5] * moment.z;
      field.z +=
          block[6] * moment.x + block[7] * moment.y + block[8] * moment.z;
    }
    H[static_cast<std::size_t>(target)] += field;
  }
}

void apply_static_operator(
    const StaticCoefficientOperator& operator_map,
    const std::span<const double> input,
    const std::span<double> output)
{
    validate_operator_dimensions(operator_map, input, output);
    for (const StaticOperatorEntry& entry : operator_map.entries) {
        output[static_cast<std::size_t>(entry.output)] += entry.value *
            input[static_cast<std::size_t>(entry.input)];
    }
}

PotentialField apply_static_l2p_evaluator(
    const StaticL2PEvaluator& evaluator,
    const std::span<const double> L,
    const OutputFlags output)
{
    if (evaluator.potential.size() != L.size()) {
        throw std::invalid_argument("static L2P dimensions are inconsistent");
    }
    PotentialField result;
    for (std::size_t index = 0; index < L.size(); ++index) {
        if (has_flag(output, OutputFlags::Potential)) {
            result.phi += evaluator.potential[index] * L[index];
        }
        if (has_flag(output, OutputFlags::Field)) {
            result.H.x += evaluator.field[0][index] * L[index];
            result.H.y += evaluator.field[1][index] * L[index];
            result.H.z += evaluator.field[2][index] * L[index];
        }
    }
    return result;
}

void apply_static_coefficient_matrix(const std::span<const double> matrix,
    const std::span<const double> input,
                                     const std::span<double> output) {
    const std::size_t coefficient_count = input.size();
    if (output.size() != coefficient_count ||
        matrix.size() != coefficient_count * coefficient_count) {
        throw std::invalid_argument(
        "static coefficient matrix dimensions are inconsistent");
    }
    for (std::size_t alpha = 0; alpha < coefficient_count; ++alpha) {
        for (std::size_t beta = 0; beta < coefficient_count; ++beta) {
      output[beta] += matrix[beta + coefficient_count * alpha] * input[alpha];
        }
    }
}


void apply_static_operator(
    const FloatStaticCoefficientOperator& operator_map,
    const std::span<const float> input,
    const std::span<float> output)
{
    if (input.size() != static_cast<std::size_t>(operator_map.input_size) ||
        output.size() != static_cast<std::size_t>(operator_map.output_size)) {
        throw std::invalid_argument("static FP32 operator dimensions are inconsistent");
    }
    for (const FloatStaticOperatorEntry& entry : operator_map.entries) {
        output[static_cast<std::size_t>(entry.output)] +=
            entry.value * input[static_cast<std::size_t>(entry.input)];
    }
}

FloatPotentialField
apply_static_l2p_evaluator(const FloatStaticL2PEvaluator &evaluator,
    const std::span<const float> L,
                           const OutputFlags output) {
    if (evaluator.potential.size() != L.size()) {
        throw std::invalid_argument("static FP32 L2P dimensions are inconsistent");
    }
    FloatPotentialField result;
    for (std::size_t index = 0; index < L.size(); ++index) {
        if (has_flag(output, OutputFlags::Potential)) {
            result.phi += evaluator.potential[index] * L[index];
        }
        if (has_flag(output, OutputFlags::Field)) {
            result.H.x += evaluator.field[0][index] * L[index];
            result.H.y += evaluator.field[1][index] * L[index];
            result.H.z += evaluator.field[2][index] * L[index];
        }
    }
    return result;
}

void apply_static_p2p_compact_plan(
    const FloatStaticP2PCompactPlan& plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
        H.size() != static_cast<std::size_t>(plan.target_count) ||
        plan.skip_for_identity.size() != plan.source_indices.size() ||
        (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
        throw std::invalid_argument("compact FP32 P2P dimensions are inconsistent");
    }
#pragma omp parallel for schedule(static) if(plan.target_count >= 64)
    for (int target = 0; target < plan.target_count; ++target) {
        FloatVec3 field{};
        const int self = target_source_indices.empty()
            ? -1 : target_source_indices[static_cast<std::size_t>(target)];
        for (int entry = plan.row_offsets[static_cast<std::size_t>(target)];
             entry < plan.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const std::size_t index = static_cast<std::size_t>(entry);
            const int source = plan.source_indices[index];
            if (plan.skip_for_identity[index] != 0 && source == self) {
                continue;
            }
            const FloatVec3 moment = dipole_moments[static_cast<std::size_t>(source)];
            field.x += plan.tensors[0][index] * moment.x +
                       plan.tensors[1][index] * moment.y +
                       plan.tensors[2][index] * moment.z;
            field.y += plan.tensors[1][index] * moment.x +
                       plan.tensors[3][index] * moment.y +
                       plan.tensors[4][index] * moment.z;
            field.z += plan.tensors[2][index] * moment.x +
                       plan.tensors[4][index] * moment.y +
                       plan.tensors[5][index] * moment.z;
        }
        H[static_cast<std::size_t>(target)] += field;
    }
}

} // namespace cdfmm
