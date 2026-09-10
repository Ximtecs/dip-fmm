// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/p2p/bsr.hpp"
#include "cdfmm/plan/p2p/compact.hpp"
#include "cdfmm/plan/p2p/dictionary.hpp"
#include "cdfmm/plan/p2p/leaf.hpp"
#include "cdfmm/plan/p2p/signed_dictionary.hpp"

namespace cdfmm {

std::size_t StaticP2PMemory::total_bytes() const noexcept
{
    return tensor_bytes + index_bytes + row_metadata_bytes +
        leaf_metadata_bytes + scratch_bytes;
}

std::size_t StaticP2POperator::memory_bytes() const noexcept
{
    return row_offsets.size() * sizeof(int) +
        blocks.size() * sizeof(StaticDipoleBlock);
}

std::size_t FloatStaticP2POperator::memory_bytes() const noexcept
{
    return row_offsets.size() * sizeof(int) +
        blocks.size() * sizeof(FloatStaticDipoleBlock);
}

StaticP2PMemory StaticP2PCompactPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 9 * sizeof(double);
    result.index_bytes = source_indices.size() * sizeof(int) +
        skip_for_identity.size() * sizeof(unsigned char);
    result.row_metadata_bytes = row_offsets.size() * sizeof(int);
    return result;
}

StaticP2PMemory FloatStaticP2PCompactPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 9 * sizeof(float);
    result.index_bytes = source_indices.size() * sizeof(int) +
        skip_for_identity.size() * sizeof(unsigned char);
    result.row_metadata_bytes = row_offsets.size() * sizeof(int);
    return result;
}

StaticP2PMemory StaticP2PLeafPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 6 * sizeof(double);
    result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
    result.leaf_metadata_bytes =
        (target_begins.size() + target_counts.size()) * sizeof(int) +
        blocks.size() * sizeof(StaticP2PLeafBlock);
    return result;
}

StaticP2PMemory FloatStaticP2PLeafPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 6 * sizeof(float);
    result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
    result.leaf_metadata_bytes =
        (target_begins.size() + target_counts.size()) * sizeof(int) +
        blocks.size() * sizeof(StaticP2PLeafBlock);
    return result;
}

StaticP2PMemory StaticP2PTensorDictionaryPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 6 * sizeof(double);
    result.index_bytes = tokens.size() * sizeof(std::uint32_t);
    result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
    result.leaf_metadata_bytes =
        (target_begins.size() + target_counts.size()) * sizeof(int) +
        blocks.size() * sizeof(StaticP2PLeafBlock);
    return result;
}

StaticP2PMemory FloatStaticP2PTensorDictionaryPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 6 * sizeof(float);
    result.index_bytes = tokens.size() * sizeof(std::uint32_t);
    result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
    result.leaf_metadata_bytes =
        (target_begins.size() + target_counts.size()) * sizeof(int) +
        blocks.size() * sizeof(StaticP2PLeafBlock);
    return result;
}

StaticP2PMemory StaticP2PSignedTensorDictionaryPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 6 * sizeof(double);
    result.index_bytes = tokens8.size() * sizeof(std::uint8_t) +
        tokens16.size() * sizeof(std::uint16_t) +
        tokens32.size() * sizeof(std::uint32_t);
    result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
    result.leaf_metadata_bytes =
        (target_begins.size() + target_counts.size() +
         tile_leaf_indices.size() + tile_target_offsets.size()) * sizeof(int) +
        blocks.size() * sizeof(StaticP2PLeafBlock);
    return result;
}

StaticP2PMemory FloatStaticP2PSignedTensorDictionaryPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 6 * sizeof(float);
    result.index_bytes = tokens8.size() * sizeof(std::uint8_t) +
        tokens16.size() * sizeof(std::uint16_t) +
        tokens32.size() * sizeof(std::uint32_t);
    result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
    result.leaf_metadata_bytes =
        (target_begins.size() + target_counts.size() +
         tile_leaf_indices.size() + tile_target_offsets.size()) * sizeof(int) +
        blocks.size() * sizeof(StaticP2PLeafBlock);
    return result;
}

StaticP2PMemory StaticP2PBsrPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = values.size() * sizeof(double);
    result.index_bytes = source_indices.size() * sizeof(int);
    result.row_metadata_bytes =
        (row_offsets.size() + target_source_indices.size()) * sizeof(int);
    return result;
}

StaticP2PMemory FloatStaticP2PBsrPlan::memory() const noexcept
{
    StaticP2PMemory result;
    result.tensor_bytes = values.size() * sizeof(float);
    result.index_bytes = source_indices.size() * sizeof(int);
    result.row_metadata_bytes =
        (row_offsets.size() + target_source_indices.size()) * sizeof(int);
    return result;
}

} // namespace cdfmm
