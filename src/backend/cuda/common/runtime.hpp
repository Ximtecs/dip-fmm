// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace cdfmm {

[[nodiscard]] bool cuda_runtime_available() noexcept;
[[nodiscard]] bool cuda_compiled() noexcept;
[[nodiscard]] std::string cuda_runtime_description();

} // namespace cdfmm
