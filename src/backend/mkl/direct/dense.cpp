// SPDX-License-Identifier: Apache-2.0

#include "backend/mkl/direct/dense.hpp"

#include <stdexcept>
#include <type_traits>

#ifdef CDFMM_USE_MKL
#include <mkl.h>
#endif

namespace cdfmm {

bool dense_direct_mkl_available() noexcept
{
#ifdef CDFMM_USE_MKL
    return true;
#else
    return false;
#endif
}

namespace detail::dense_direct::mkl {
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
#ifdef CDFMM_USE_MKL
    if constexpr (std::is_same_v<Scalar, float>) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<MKL_INT>(rows),
                    static_cast<MKL_INT>(columns), 1.0F, matrix.data(),
                    static_cast<MKL_INT>(columns), input.data(), 1,
                    add ? 1.0F : 0.0F, output.data(), 1);
    } else {
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<MKL_INT>(rows),
                    static_cast<MKL_INT>(columns), 1.0, matrix.data(),
                    static_cast<MKL_INT>(columns), input.data(), 1,
                    add ? 1.0 : 0.0, output.data(), 1);
    }
#else
    static_cast<void>(matrix);
    static_cast<void>(rows);
    static_cast<void>(columns);
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(add);
    throw std::invalid_argument("unsupported dense direct backend");
#endif
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

} // namespace detail::dense_direct::mkl
} // namespace cdfmm
