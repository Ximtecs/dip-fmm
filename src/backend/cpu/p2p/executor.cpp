// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cpu/p2p.hpp"

#include <algorithm>
#include <stdexcept>

namespace cdfmm {
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
                    if (block.skip_for_identity != 0 && source == self) {
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
          if (block.skip_for_identity != 0 && source == self) {
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
