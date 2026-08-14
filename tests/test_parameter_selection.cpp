#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "cdfmm/parameter_selection.hpp"

using namespace cdfmm;

TEST_CASE("Parameter selection helpers are deterministic") {
  const auto first = deterministic_target_sample(100, 7);
  const auto second = deterministic_target_sample(100, 7);

  REQUIRE(first == second);
  REQUIRE(first.size() == 7);
  REQUIRE(std::is_sorted(first.begin(), first.end()));
  REQUIRE(first.back() < 100);
}

TEST_CASE("Near and far balance ratio is symmetric") {
  REQUIRE(branch_balance_ratio(2.0, 8.0) == Catch::Approx(4.0));
  REQUIRE(branch_balance_ratio(8.0, 2.0) == Catch::Approx(4.0));
  REQUIRE(std::isinf(branch_balance_ratio(0.0, 2.0)));
}
