// SPDX-License-Identifier: Apache-2.0

#include "backend/cpu/direct/dense.hpp"

#include <algorithm>
#include <type_traits>

namespace cdfmm::detail::dense_direct::cpu {
namespace {

template <typename Scalar>
void gemv(
    const std::vector<Scalar>& matrix,
    const std::size_t rows,
    const std::size_t columns,
    const std::vector<Scalar>& input,
    std::vector<Scalar>& output,
    const bool add)
{
    if (!add) {
        std::fill(output.begin(), output.end(), Scalar{0});
    }
    for (std::size_t row = 0; row < rows; ++row) {
        Scalar value = Scalar{0};
        for (std::size_t column = 0; column < columns; ++column) {
            value += matrix[row * columns + column] * input[column];
        }
        output[row] += value;
    }
}

template <typename Scalar>
void apply_impl(
    const std::array<std::vector<Scalar>, 6>& matrices,
    const std::size_t target_count,
    const std::size_t source_count,
    const std::span<const cdfmm::Vec3> total_moments,
    DenseDirectWorkspace& workspace,
    std::vector<cdfmm::Vec3>& result)
{
    auto& moments = [&]() -> auto& {
        if constexpr (std::is_same_v<Scalar, float>) {
            return workspace.float_moments;
        } else {
            return workspace.double_moments;
        }
    }();
    auto& fields = [&]() -> auto& {
        if constexpr (std::is_same_v<Scalar, float>) {
            return workspace.float_fields;
        } else {
            return workspace.double_fields;
        }
    }();

    for (auto& component : moments) {
        component.resize(source_count);
    }
    for (auto& component : fields) {
        component.resize(target_count);
    }
    for (std::size_t source = 0; source < source_count; ++source) {
        moments[0][source] = static_cast<Scalar>(total_moments[source].x);
        moments[1][source] = static_cast<Scalar>(total_moments[source].y);
        moments[2][source] = static_cast<Scalar>(total_moments[source].z);
    }

    // Preserve target-major storage, component order, and the original
    // nine-GEMV accumulation sequence exactly.
    gemv(matrices[0], target_count, source_count, moments[0], fields[0], false);
    gemv(matrices[1], target_count, source_count, moments[1], fields[0], true);
    gemv(matrices[2], target_count, source_count, moments[2], fields[0], true);
    gemv(matrices[1], target_count, source_count, moments[0], fields[1], false);
    gemv(matrices[3], target_count, source_count, moments[1], fields[1], true);
    gemv(matrices[4], target_count, source_count, moments[2], fields[1], true);
    gemv(matrices[2], target_count, source_count, moments[0], fields[2], false);
    gemv(matrices[4], target_count, source_count, moments[1], fields[2], true);
    gemv(matrices[5], target_count, source_count, moments[2], fields[2], true);

    result.resize(target_count);
    for (std::size_t target = 0; target < target_count; ++target) {
        result[target] = {static_cast<double>(fields[0][target]),
                          static_cast<double>(fields[1][target]),
                          static_cast<double>(fields[2][target])};
    }
}

} // namespace

void apply(
    const std::array<std::vector<float>, 6>& matrices,
    const std::size_t target_count,
    const std::size_t source_count,
    const std::span<const cdfmm::Vec3> total_moments,
    DenseDirectWorkspace& workspace,
    std::vector<cdfmm::Vec3>& result)
{
    apply_impl(matrices, target_count, source_count, total_moments, workspace,
               result);
}

void apply(
    const std::array<std::vector<double>, 6>& matrices,
    const std::size_t target_count,
    const std::size_t source_count,
    const std::span<const cdfmm::Vec3> total_moments,
    DenseDirectWorkspace& workspace,
    std::vector<cdfmm::Vec3>& result)
{
    apply_impl(matrices, target_count, source_count, total_moments, workspace,
               result);
}

} // namespace cdfmm::detail::dense_direct::cpu
