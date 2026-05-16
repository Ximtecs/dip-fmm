// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "cdfmm/multi_index.hpp"
#include <vector>
namespace cdfmm {
std::vector<double> laplace_derivatives_raw(const MultiIndexSet &basis,
                                            const Vec3 &r);
}
