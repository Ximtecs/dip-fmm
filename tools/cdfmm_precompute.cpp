// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cdfmm/uniform_fmm.hpp"

namespace {

std::vector<int> parse_orders(const std::string& text) {
  std::vector<int> orders;
  std::istringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const int order = std::stoi(item);
    if (order < 1) {
      throw std::invalid_argument("orders must be positive");
    }
    orders.push_back(order);
  }
  if (orders.empty()) {
    throw std::invalid_argument("at least one order is required");
  }
  return orders;
}

void usage() {
  std::cout
      << "Usage: cdfmm-precompute --basis spherical|cartesian "
         "--orders 4,6,8 --precision f32|f64 [--periodic] "
         "[--periodic-tolerance 1e-12]\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    cdfmm::ExpansionBasis basis = cdfmm::ExpansionBasis::Spherical;
    cdfmm::StaticPrecision precision = cdfmm::StaticPrecision::Float32;
    std::vector<int> orders{4, 6, 8};
    bool periodic = false;
    double periodic_tolerance = 1.0e-12;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      const auto next = [&]() -> std::string {
        if (++index >= argc) {
          throw std::invalid_argument("missing value after " + argument);
        }
        return argv[index];
      };
      if (argument == "--basis") {
        const std::string value = next();
        if (value == "spherical") {
          basis = cdfmm::ExpansionBasis::Spherical;
        } else if (value == "cartesian") {
          basis = cdfmm::ExpansionBasis::Cartesian;
        } else {
          throw std::invalid_argument("unsupported basis: " + value);
        }
      } else if (argument == "--orders") {
        orders = parse_orders(next());
      } else if (argument == "--precision") {
        const std::string value = next();
        if (value == "f32") {
          precision = cdfmm::StaticPrecision::Float32;
        } else if (value == "f64") {
          precision = cdfmm::StaticPrecision::Float64;
        } else {
          throw std::invalid_argument("unsupported precision: " + value);
        }
      } else if (argument == "--periodic") {
        periodic = true;
      } else if (argument == "--periodic-tolerance") {
        periodic_tolerance = std::stod(next());
      } else if (argument == "--help" || argument == "-h") {
        usage();
        return EXIT_SUCCESS;
      } else {
        throw std::invalid_argument("unknown argument: " + argument);
      }
    }

    const std::vector<cdfmm::Vec3> position{{0.0, 0.0, 0.0}};
    for (const int order : orders) {
      cdfmm::UniformFmmOptions options;
      options.expansion_basis = basis;
      options.expansion_order = order;
      options.precision = precision;
      options.tree.max_level = 0;
      options.fixed_target_source_indices = std::vector<int>{0};
      options.periodic.enabled = periodic;
      options.periodic.setup_tolerance = periodic_tolerance;
      cdfmm::UniformFmm plan(position, position, options);
      const auto& statistics = plan.static_plan_statistics();
      std::cout << "universal=" << plan.universal_cache_key()
                << " hit=" << std::boolalpha
                << statistics.universal_cache_hit;
      if (periodic) {
        std::cout << " periodic=" << plan.periodic_cache_key()
                  << " hit=" << statistics.periodic_cache_hit;
      }
      std::cout << " bytes_read=" << statistics.cache_bytes_read
                << " bytes_written=" << statistics.cache_bytes_written
                << '\n';
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "cdfmm-precompute: " << error.what() << '\n';
    usage();
    return EXIT_FAILURE;
  }
}
