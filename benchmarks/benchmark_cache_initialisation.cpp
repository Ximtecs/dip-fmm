// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cdfmm/uniform_fmm.hpp"

namespace {

struct Arguments {
  int depth{3};
  int grid{8};
  std::string backend{"portable"};
  std::filesystem::path cache_directory{};
};

Arguments parse(const int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto next = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--depth") {
      result.depth = std::stoi(next());
    } else if (argument == "--grid") {
      result.grid = std::stoi(next());
    } else if (argument == "--backend") {
      result.backend = next();
    } else if (argument == "--cache-dir") {
      result.cache_directory = next();
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (result.depth < 0 || result.grid <= 0 ||
      result.cache_directory.empty()) {
    throw std::invalid_argument(
        "use --depth D --grid N --cache-dir EMPTY_DIRECTORY");
  }
  return result;
}

void report(const char* state, const std::string& backend, const int depth,
            const cdfmm::UniformFmm& plan) {
  const auto& value = plan.static_plan_statistics();
  std::cout << "CACHE_BENCH," << state << ',' << backend << ',' << depth
            << ',' << std::boolalpha << value.universal_cache_hit << ','
            << value.geometry_cache_hit << ',' << value.total_setup.total_seconds
            << ',' << value.normalisation.total_seconds << ','
            << value.tree_construction.total_seconds << ','
            << value.universal_cache_lookup.total_seconds << ','
            << value.universal_cache_load.total_seconds << ','
            << value.universal_operator_build.total_seconds << ','
            << value.universal_cache_write.total_seconds << ','
            << value.geometry_hash.total_seconds << ','
            << value.geometry_cache_lookup.total_seconds << ','
            << value.geometry_cache_load.total_seconds << ','
            << value.geometry_cache_write.total_seconds << ','
            << value.p2m_plan.total_seconds << ','
            << value.m2m_plan.total_seconds << ','
            << value.m2l_plan.total_seconds << ','
            << value.l2l_plan.total_seconds << ','
            << value.l2p_plan.total_seconds << ','
            << value.p2p_tensor_plan.total_seconds << ','
            << value.backend_packing.total_seconds << ','
            << value.cuda_upload.total_seconds << ',' << value.cache_bytes_read
            << ',' << value.cache_bytes_written << '\n';
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parse(argc, argv);
    if (std::filesystem::exists(arguments.cache_directory) &&
        !std::filesystem::is_empty(arguments.cache_directory)) {
      throw std::invalid_argument("--cache-dir must be empty for a cold run");
    }
    std::filesystem::create_directories(arguments.cache_directory);
    if (::setenv("CDFMM_CACHE_DIR", arguments.cache_directory.c_str(), 1) != 0) {
      throw std::runtime_error("failed to set CDFMM_CACHE_DIR");
    }

    std::vector<cdfmm::Vec3> positions;
    positions.reserve(static_cast<std::size_t>(arguments.grid) * arguments.grid *
                      arguments.grid);
    const double spacing = 1.0 / static_cast<double>(arguments.grid);
    for (int z = 0; z < arguments.grid; ++z) {
      for (int y = 0; y < arguments.grid; ++y) {
        for (int x = 0; x < arguments.grid; ++x) {
          positions.push_back({(x + 0.5) * spacing, (y + 0.5) * spacing,
                               (z + 0.5) * spacing});
        }
      }
    }
    std::vector<int> identities(positions.size());
    for (std::size_t index = 0; index < identities.size(); ++index) {
      identities[index] = static_cast<int>(index);
    }

    cdfmm::UniformFmmOptions options;
    options.expansion_basis = cdfmm::ExpansionBasis::Spherical;
    options.expansion_order = 6;
    options.precision = cdfmm::StaticPrecision::Float32;
    options.tree.max_level = arguments.depth;
    options.source_geometry = cdfmm::SourceGeometry::UniformCuboid;
    options.source_sizes = {{spacing, spacing, spacing}};
    options.fixed_target_source_indices = identities;
    if (arguments.backend == "onemkl") {
      options.static_matrix_backend = cdfmm::StaticMatrixBackend::OneMkl;
    } else if (arguments.backend == "cuda-full") {
      options.backend = cdfmm::ExecutionBackend::CudaFull;
    } else if (arguments.backend != "portable") {
      throw std::invalid_argument("backend must be portable, onemkl, or cuda-full");
    }

    std::cout << "CACHE_BENCH_HEADER,state,backend,depth,universal_hit,"
                 "geometry_hit,total_setup_s,normalisation_s,tree_s,"
                 "universal_lookup_s,universal_load_s,universal_build_s,"
                 "universal_write_s,geometry_hash_s,geometry_lookup_s,"
                 "geometry_load_s,geometry_write_s,"
                 "p2m_s,m2m_s,m2l_s,l2l_s,l2p_s,p2p_s,backend_packing_s,"
                 "cuda_upload_s,bytes_read,bytes_written\n";
    cdfmm::UniformFmm cold(positions, positions, options);
    report("cold", arguments.backend, arguments.depth, cold);
    cdfmm::UniformFmm warm(positions, positions, options);
    report("warm", arguments.backend, arguments.depth, warm);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_cache_initialisation: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
