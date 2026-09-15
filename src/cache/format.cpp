// SPDX-License-Identifier: Apache-2.0

#include "cache/internal.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

// Field-wise records for the solver types a cache payload holds: coefficient
// operators, value arrays, and canonical P2P blocks. Both the universal and
// the geometry payloads share these facilities.
//
// WARNING(cdfmm): field order, record strides, and the FP32/FP64 branches are
// part of the persistent format. The FP32 paths decode directly into FP32
// storage and must not be routed through FP64.
namespace cdfmm::detail::cache {

void write_operator(Writer& writer, const StaticCoefficientOperator& value,
                    const StaticPrecision precision) {
  writer.scalar(value.input_size);
  writer.scalar(value.output_size);
  writer.scalar<std::uint64_t>(value.entries.size());
  const std::size_t record_bytes =
      2 * sizeof(int) +
      (precision == StaticPrecision::Float32 ? sizeof(float) : sizeof(double));
  unsigned char* raw = writer.append_uninitialized(
      checked_bytes(value.entries.size(), record_bytes));
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(value.entries.size());
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const StaticOperatorEntry& entry = value.entries[static_cast<std::size_t>(index)];
    unsigned char* record = raw + static_cast<std::size_t>(index) * record_bytes;
    store_unaligned<int>(record, entry.output);
    store_unaligned<int>(record + sizeof(int), entry.input);
    if (precision == StaticPrecision::Float32) {
      store_unaligned<float>(record + 2 * sizeof(int),
                             static_cast<float>(entry.value));
    } else {
      store_unaligned<double>(record + 2 * sizeof(int), entry.value);
    }
  }
}

StaticCoefficientOperator read_operator(Reader& reader,
                                        const StaticPrecision precision) {
  StaticCoefficientOperator value;
  value.input_size = reader.scalar<int>();
  value.output_size = reader.scalar<int>();
  const std::uint64_t count64 = reader.scalar<std::uint64_t>();
  if (count64 > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("operator cache entry count too large");
  }
  const std::size_t record_bytes =
      2 * sizeof(int) +
      (precision == StaticPrecision::Float32 ? sizeof(float) : sizeof(double));
  const unsigned char* raw = reader.take_bytes(checked_bytes(count64, record_bytes));
  value.entries.resize(static_cast<std::size_t>(count64));
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(count64);
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const unsigned char* record =
        raw + static_cast<std::size_t>(index) * record_bytes;
    StaticOperatorEntry& entry = value.entries[static_cast<std::size_t>(index)];
    entry.output = load_unaligned<int>(record);
    entry.input = load_unaligned<int>(record + sizeof(int));
    entry.value = precision == StaticPrecision::Float32
        ? static_cast<double>(load_unaligned<float>(record + 2 * sizeof(int)))
        : load_unaligned<double>(record + 2 * sizeof(int));
  }
  return value;
}

FloatStaticCoefficientOperator read_operator_float(Reader& reader) {
  FloatStaticCoefficientOperator value;
  value.input_size = reader.scalar<int>();
  value.output_size = reader.scalar<int>();
  const std::uint64_t count64 = reader.scalar<std::uint64_t>();
  if (count64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("operator cache entry count too large");
  }
  constexpr std::size_t record_bytes = 2 * sizeof(int) + sizeof(float);
  const unsigned char* raw =
      reader.take_bytes(checked_bytes(count64, record_bytes));
  value.entries.resize(static_cast<std::size_t>(count64));
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(count64);
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const unsigned char* record =
        raw + static_cast<std::size_t>(index) * record_bytes;
    FloatStaticOperatorEntry& entry =
        value.entries[static_cast<std::size_t>(index)];
    entry.output = load_unaligned<int>(record);
    entry.input = load_unaligned<int>(record + sizeof(int));
    entry.value = load_unaligned<float>(record + 2 * sizeof(int));
  }
  return value;
}

void write_values(Writer& writer, const std::span<const double> values,
                  const StaticPrecision precision) {
  writer.scalar<std::uint64_t>(values.size());
  if (precision == StaticPrecision::Float64) {
    writer.append_raw(values.data(), values.size_bytes());
    return;
  }
  unsigned char* raw =
      writer.append_uninitialized(checked_bytes(values.size(), sizeof(float)));
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(values.size());
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    store_unaligned<float>(raw + static_cast<std::size_t>(index) * sizeof(float),
                           static_cast<float>(values[static_cast<std::size_t>(index)]));
  }
}

void write_values(Writer& writer, const std::vector<double>& values,
                  const StaticPrecision precision) {
  write_values(writer, std::span<const double>(values), precision);
}

std::vector<double> read_values(Reader& reader,
                                const StaticPrecision precision) {
  const std::uint64_t count64 = reader.scalar<std::uint64_t>();
  if (count64 > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("cache value count too large");
  }
  const std::size_t count = static_cast<std::size_t>(count64);
  std::vector<double> values(count);
  if (precision == StaticPrecision::Float64) {
    const std::size_t bytes = checked_bytes(count64, sizeof(double));
    if (bytes != 0) {
      std::memcpy(values.data(), reader.take_bytes(bytes), bytes);
    }
    return values;
  }

  const unsigned char* raw =
      reader.take_bytes(checked_bytes(count64, sizeof(float)));
  const std::ptrdiff_t parallel_count = static_cast<std::ptrdiff_t>(count64);
#pragma omp parallel for schedule(static) if (parallel_count >= 100000)
  for (std::ptrdiff_t index = 0; index < parallel_count; ++index) {
    values[static_cast<std::size_t>(index)] = static_cast<double>(
        load_unaligned<float>(raw + static_cast<std::size_t>(index) * sizeof(float)));
  }
  return values;
}

std::vector<float> read_values_float(Reader& reader) {
  const std::uint64_t count64 = reader.scalar<std::uint64_t>();
  const std::size_t bytes = checked_bytes(count64, sizeof(float));
  std::vector<float> values(static_cast<std::size_t>(count64));
  if (bytes != 0) {
    std::memcpy(values.data(), reader.take_bytes(bytes), bytes);
  }
  return values;
}

void write_p2p_blocks(Writer& writer,
                      const std::span<const StaticDipoleBlock> blocks,
                      const StaticPrecision precision) {
  const std::size_t record_bytes = p2p_record_bytes(precision);
  unsigned char* raw =
      writer.append_uninitialized(checked_bytes(blocks.size(), record_bytes));
  if (blocks.size() > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("P2P cache block count too large");
  }
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(blocks.size());
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const StaticDipoleBlock& block = blocks[static_cast<std::size_t>(index)];
    unsigned char* record = raw + static_cast<std::size_t>(index) * record_bytes;
    store_unaligned<int>(record, block.target);
    store_unaligned<int>(record + sizeof(int), block.source);
    const std::array<double, 9> values{
        block.px, block.py, block.pz, block.xx, block.xy,
        block.xz, block.yy, block.yz, block.zz};
    std::size_t offset = 2 * sizeof(int);
    if (precision == StaticPrecision::Float32) {
      for (const double value : values) {
        store_unaligned<float>(record + offset, static_cast<float>(value));
        offset += sizeof(float);
      }
    } else {
      for (const double value : values) {
        store_unaligned<double>(record + offset, value);
        offset += sizeof(double);
      }
    }
    store_unaligned<int>(record + offset, block.skip_for_identity);
  }
}

void read_p2p_blocks(Reader& reader,
                     const std::uint64_t block_count64,
                     const StaticPrecision precision,
                     StaticP2POperator& operator_map,
                     StaticP2PCompactPlan& compact_plan) {
  if (block_count64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("P2P cache block count too large");
  }
  const std::size_t block_count = static_cast<std::size_t>(block_count64);
  const std::size_t record_bytes = p2p_record_bytes(precision);
  const unsigned char* raw =
      reader.take_bytes(checked_bytes(block_count64, record_bytes));

  operator_map.blocks.resize(block_count);

  compact_plan.source_count = operator_map.source_count;
  compact_plan.target_count = operator_map.target_count;
  compact_plan.row_offsets = operator_map.row_offsets;
  compact_plan.source_indices.resize(block_count);
  compact_plan.skip_for_identity.resize(block_count);
  for (auto& row : compact_plan.potential) {
    row.resize(block_count);
  }
  for (auto& row : compact_plan.tensors) {
    row.resize(block_count);
  }

  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(block_count64);
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const std::size_t slot = static_cast<std::size_t>(index);
    const unsigned char* record = raw + slot * record_bytes;
    StaticDipoleBlock& block = operator_map.blocks[slot];
    block.target = load_unaligned<int>(record);
    block.source = load_unaligned<int>(record + sizeof(int));
    std::size_t offset = 2 * sizeof(int);
    std::array<double, 9> values{};
    if (precision == StaticPrecision::Float32) {
      for (double& value : values) {
        value = static_cast<double>(load_unaligned<float>(record + offset));
        offset += sizeof(float);
      }
    } else {
      for (double& value : values) {
        value = load_unaligned<double>(record + offset);
        offset += sizeof(double);
      }
    }
    block.px = values[0];
    block.py = values[1];
    block.pz = values[2];
    block.xx = values[3];
    block.xy = values[4];
    block.xz = values[5];
    block.yy = values[6];
    block.yz = values[7];
    block.zz = values[8];
    block.skip_for_identity = load_unaligned<int>(record + offset);

    // Deliberately fused with the canonical decode above: plan/p2p/compact.hpp
    // owns the canonical-to-compact row rule, this loop owns only reading the
    // persisted bytes and choosing to build both representations in one pass.
    assign_static_p2p_compact_row(compact_plan, slot, block);
  }
}

void read_p2p_blocks_float(Reader& reader,
                           const std::uint64_t block_count64,
                           FloatStaticP2POperator& operator_map) {
  if (block_count64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("P2P cache block count too large");
  }
  const std::size_t block_count = static_cast<std::size_t>(block_count64);
  const std::size_t record_bytes = p2p_record_bytes(StaticPrecision::Float32);
  const unsigned char* raw =
      reader.take_bytes(checked_bytes(block_count64, record_bytes));
  operator_map.blocks.resize(block_count);

  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(block_count64);
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const std::size_t slot = static_cast<std::size_t>(index);
    const unsigned char* record = raw + slot * record_bytes;
    FloatStaticDipoleBlock& block = operator_map.blocks[slot];
    block.target = load_unaligned<int>(record);
    block.source = load_unaligned<int>(record + sizeof(int));
    std::size_t offset = 2 * sizeof(int);
    block.px = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.py = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.pz = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.xx = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.xy = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.xz = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.yy = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.yz = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.zz = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.skip_for_identity = load_unaligned<int>(record + offset);
  }
}

} // namespace cdfmm::detail::cache
