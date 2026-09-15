// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/plan/static_plan.hpp"

namespace cdfmm {

void apply_static_m2l_plan(
    const StaticM2LPlan& plan, int level, std::span<const double> multipoles,
    std::span<double> locals);
void apply_static_m2l_plan(
    const FloatStaticM2LPlan& plan, int level, std::span<const float> multipoles,
    std::span<float> locals);
void apply_static_m2l_plan(
    const StaticM2LPlan& plan, int level,
    std::span<const std::vector<double>> multipoles,
    std::span<std::vector<double>> locals);

} // namespace cdfmm
