// SPDX-License-Identifier: Apache-2.0
//
// benchmark_operator_representation: precomputed versus procedural execution
// of the exact near-field and far-field operators (Phase 3B.5b).
//
// For a chosen operator (P2P, P2M or L2P), geometry pair and layout the
// benchmark measures, separately,
//
//   * construction of the exact operator with the production builders,
//   * application of the precomputed representation through the production
//     execution packings (SoA rows, dense leaf blocks, tensor dictionaries,
//     packed coefficient rows), and
//   * procedural execution: the identical operator reconstructed from the
//     retained geometry during every update and applied immediately, with no
//     operator storage.
//
// Two modes bracket the memory behaviour: `hot` repeats one small set of
// operators that stays cache resident (intrinsic arithmetic cost) and
// `streaming` traverses a complete list-1 neighbourhood whose stored operator
// exceeds the private caches (realistic FMM behaviour).  Every representation
// is checked against the FP64 canonical operator on the final moments, and
// every row of the CSV output carries the metadata needed to reproduce it.
//
// The benchmark never changes production execution; the procedural finite
// executors here are the experimental representations under study.

#include "cdfmm/backend/cpu/p2p.hpp"
#include "cdfmm/math/spherical_harmonics.hpp"
#include "cdfmm/operators/l2p.hpp"
#include "cdfmm/operators/p2m.hpp"
#include "cdfmm/operators/p2p.hpp"
#include "cdfmm/plan/precision.hpp"
#include "cdfmm/plan/static_plan.hpp"
#include "cdfmm/tree/uniform_tree.hpp"

#include "backend/cpu/far_field/packing.hpp"
#include "geometry/primitives/rectangular_prism_point_kernel.hpp"
#include "geometry/primitives/tetrahedron_detail.hpp"

#if defined(CDFMM_ENABLE_CUDA)
#include "benchmark_operator_representation_cuda.hpp"
#include "cdfmm/backend/cuda/availability.hpp"
#include "cdfmm/backend/cuda/p2p.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(CDFMM_USE_OPENMP)
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using cdfmm::PairTensor;
using cdfmm::RectangularPrism;
using cdfmm::Tetrahedron;
using cdfmm::Vec3;

//------------------------------------------------------------------------------
// Options
//------------------------------------------------------------------------------

enum class Geometry { Point, Prism, Tetrahedron };
enum class Stage { P2P, P2M, L2P };
enum class Mode { Hot, Streaming };
enum class Separation { Lattice, Self, Adjacent, List1, SmallFar, FarSafeguard };

struct Options {
  Stage stage{Stage::P2P};
  Geometry source{Geometry::Prism};
  Geometry target{Geometry::Prism};
  bool irregular{false};
  Mode mode{Mode::Streaming};
  Separation separation{Separation::Lattice};
  int depth{3};
  int bodies_per_leaf_axis{2};
  double fill{1.0};
  double jitter{0.2};
  int pairs{512};
  bool fp32{true};
  bool fp64{true};
  std::vector<int> orders{4, 6, 8};
  int evaluations{20};
  int warmups{2};
  int samples{5};
  double procedural_budget_seconds{20.0};
  std::string output{};
  std::string label{};
  std::string commit{};
  bool cuda{false};
  bool cpu_precomputed{true};
  bool cpu_procedural{true};
};

[[nodiscard]] Geometry parse_geometry(const std::string_view value) {
  if (value == "point") {
    return Geometry::Point;
  }
  if (value == "prism") {
    return Geometry::Prism;
  }
  if (value == "tetrahedron" || value == "tetra") {
    return Geometry::Tetrahedron;
  }
  throw std::invalid_argument("geometry must be point, prism or tetrahedron");
}

[[nodiscard]] const char* geometry_name(const Geometry geometry) {
  switch (geometry) {
  case Geometry::Point:
    return "point";
  case Geometry::Prism:
    return "prism";
  case Geometry::Tetrahedron:
    return "tetrahedron";
  }
  return "?";
}

[[nodiscard]] Separation parse_separation(const std::string_view value) {
  if (value == "lattice") {
    return Separation::Lattice;
  }
  if (value == "self") {
    return Separation::Self;
  }
  if (value == "adjacent") {
    return Separation::Adjacent;
  }
  if (value == "list1") {
    return Separation::List1;
  }
  if (value == "small-far") {
    return Separation::SmallFar;
  }
  if (value == "far-safeguard") {
    return Separation::FarSafeguard;
  }
  throw std::invalid_argument(
      "separation must be lattice, self, adjacent, list1, small-far or "
      "far-safeguard");
}

[[nodiscard]] const char* separation_name(const Separation separation) {
  switch (separation) {
  case Separation::Lattice:
    return "lattice";
  case Separation::Self:
    return "self";
  case Separation::Adjacent:
    return "adjacent";
  case Separation::List1:
    return "list1";
  case Separation::SmallFar:
    return "small-far";
  case Separation::FarSafeguard:
    return "far-safeguard";
  }
  return "?";
}

void print_usage() {
  std::cout
      << "benchmark_operator_representation [options]\n"
         "  --stage p2p|p2m|l2p            operator family (default p2p)\n"
         "  --source point|prism|tetrahedron\n"
         "  --target point|prism|tetrahedron\n"
         "  --layout regular|irregular     identical lattice bodies or "
         "jittered, per-body records\n"
         "  --mode hot|streaming           cache-resident set or complete "
         "list-1 neighbourhood\n"
         "  --separation lattice|self|adjacent|list1|small-far|far-safeguard\n"
         "                                 pair class of the hot P2P set\n"
         "  --depth D                      uniform tree depth (default 3)\n"
         "  --bodies-per-leaf-axis S       S^3 bodies per leaf (default 2)\n"
         "  --fill F                       body extent / lattice spacing "
         "(default 1.0; irregular scales it by 0.6-1.0)\n"
         "  --pairs N                      pairs (P2P) or bodies (P2M/L2P) of "
         "the hot set (default 512)\n"
         "  --precision fp32|fp64|both     (default both)\n"
         "  --orders 4,6,8                 spherical orders for P2M/L2P\n"
         "  --evaluations K --warmups W --samples S\n"
         "  --procedural-budget-seconds T  cap for one procedural streaming "
         "sample; the set is truncated and scaled beyond it\n"
         "  --cuda                         also measure the CUDA stored "
         "representations (P2P)\n"
         "  --no-cpu-precomputed           skip the CPU precomputed rows\n"
         "  --no-cpu-procedural            skip the CPU procedural rows\n"
         "  --output FILE                  append CSV rows (header written to "
         "an empty file)\n"
         "  --label TEXT --commit SHA      metadata columns\n";
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
  Options options;
  auto next = [&](int& index, const char* name) -> std::string {
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string(name) + " needs a value");
    }
    return argv[++index];
  };
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      print_usage();
      std::exit(0);
    } else if (argument == "--stage") {
      const std::string value = next(index, "--stage");
      if (value == "p2p") {
        options.stage = Stage::P2P;
      } else if (value == "p2m") {
        options.stage = Stage::P2M;
      } else if (value == "l2p") {
        options.stage = Stage::L2P;
      } else {
        throw std::invalid_argument("--stage must be p2p, p2m or l2p");
      }
    } else if (argument == "--source") {
      options.source = parse_geometry(next(index, "--source"));
    } else if (argument == "--target") {
      options.target = parse_geometry(next(index, "--target"));
    } else if (argument == "--layout") {
      const std::string value = next(index, "--layout");
      if (value == "regular") {
        options.irregular = false;
      } else if (value == "irregular") {
        options.irregular = true;
      } else {
        throw std::invalid_argument("--layout must be regular or irregular");
      }
    } else if (argument == "--mode") {
      const std::string value = next(index, "--mode");
      if (value == "hot") {
        options.mode = Mode::Hot;
      } else if (value == "streaming") {
        options.mode = Mode::Streaming;
      } else {
        throw std::invalid_argument("--mode must be hot or streaming");
      }
    } else if (argument == "--separation") {
      options.separation = parse_separation(next(index, "--separation"));
    } else if (argument == "--depth") {
      options.depth = std::stoi(next(index, "--depth"));
    } else if (argument == "--bodies-per-leaf-axis") {
      options.bodies_per_leaf_axis =
          std::stoi(next(index, "--bodies-per-leaf-axis"));
    } else if (argument == "--fill") {
      options.fill = std::stod(next(index, "--fill"));
    } else if (argument == "--jitter") {
      options.jitter = std::stod(next(index, "--jitter"));
    } else if (argument == "--pairs") {
      options.pairs = std::stoi(next(index, "--pairs"));
    } else if (argument == "--precision") {
      const std::string value = next(index, "--precision");
      options.fp32 = value == "fp32" || value == "both";
      options.fp64 = value == "fp64" || value == "both";
      if (!options.fp32 && !options.fp64) {
        throw std::invalid_argument("--precision must be fp32, fp64 or both");
      }
    } else if (argument == "--orders") {
      options.orders.clear();
      std::stringstream stream(next(index, "--orders"));
      std::string item;
      while (std::getline(stream, item, ',')) {
        options.orders.push_back(std::stoi(item));
      }
    } else if (argument == "--evaluations") {
      options.evaluations = std::stoi(next(index, "--evaluations"));
    } else if (argument == "--warmups") {
      options.warmups = std::stoi(next(index, "--warmups"));
    } else if (argument == "--samples") {
      options.samples = std::stoi(next(index, "--samples"));
    } else if (argument == "--procedural-budget-seconds") {
      options.procedural_budget_seconds =
          std::stod(next(index, "--procedural-budget-seconds"));
    } else if (argument == "--output") {
      options.output = next(index, "--output");
    } else if (argument == "--label") {
      options.label = next(index, "--label");
    } else if (argument == "--commit") {
      options.commit = next(index, "--commit");
    } else if (argument == "--cuda") {
      options.cuda = true;
    } else if (argument == "--no-cpu-precomputed") {
      options.cpu_precomputed = false;
    } else if (argument == "--no-cpu-procedural") {
      options.cpu_procedural = false;
    } else {
      throw std::invalid_argument("unknown option " + std::string(argument));
    }
  }
  if (options.mode == Mode::Streaming) {
    options.separation = Separation::Lattice;
  } else if (options.separation == Separation::Lattice) {
    options.separation = Separation::List1;
  }
  return options;
}

//------------------------------------------------------------------------------
// Metadata and timing helpers
//------------------------------------------------------------------------------

[[nodiscard]] double elapsed(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

[[nodiscard]] double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1] + values[middle]);
}

[[nodiscard]] int thread_count() {
#if defined(CDFMM_USE_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

[[nodiscard]] std::string compiler_name() {
#if defined(__INTEL_LLVM_COMPILER)
  return "icpx";
#elif defined(__clang__)
  return "clang++";
#elif defined(__GNUC__)
  return "g++";
#else
  return "unknown";
#endif
}

[[nodiscard]] std::string compiler_version() {
#if defined(__VERSION__)
  return __VERSION__;
#else
  return "";
#endif
}

[[nodiscard]] std::string cpu_model() {
  std::ifstream stream("/proc/cpuinfo");
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind("model name", 0) == 0) {
      const std::size_t colon = line.find(':');
      if (colon != std::string::npos) {
        std::string value = line.substr(colon + 1);
        const std::size_t first = value.find_first_not_of(' ');
        return first == std::string::npos ? value : value.substr(first);
      }
    }
  }
  return "unknown";
}

[[nodiscard]] std::string gpu_model() {
#if defined(CDFMM_ENABLE_CUDA)
  if (cdfmm::cuda_available()) {
    return cdfmm::cuda_device_description();
  }
#endif
  return "";
}

[[nodiscard]] std::string csv_quote(const std::string& value) {
  std::string result = "\"";
  for (const char character : value) {
    if (character == '"') {
      result += '"';
    }
    result += character;
  }
  result += '"';
  return result;
}

/** @brief One CSV row; every measurement of the benchmark is one of these. */
struct Row {
  std::string stage;
  std::string source_geometry;
  std::string target_geometry;
  std::string layout;
  std::string separation;
  std::string mode;
  std::string precision;
  int order{0};
  std::string backend;
  std::string representation;
  std::string representation_kind;  // precomputed | procedural | reference
  std::size_t bodies{0};
  std::size_t leaves{0};
  std::size_t items{0};              // pairs (P2P) or bodies (P2M/L2P)
  int threads{1};
  double build_seconds{0.0};
  double build_seconds_per_item{0.0};
  double update_seconds{0.0};        // median wall time of one complete update
  double ns_per_item{0.0};
  double measured_fraction{1.0};     // procedural streaming truncation
  std::size_t geometry_bytes{0};
  std::size_t topology_bytes{0};
  std::size_t operator_bytes{0};
  std::size_t index_bytes{0};
  std::size_t metadata_bytes{0};
  std::size_t invariant_bytes{0};
  std::size_t scratch_bytes{0};
  std::size_t unique_operators{0};
  double checksum{0.0};
  double max_relative_error{0.0};
  std::string note;
};

const char* const csv_header =
    "label,commit,stage,source_geometry,target_geometry,layout,separation,"
    "mode,precision,order,backend,representation,representation_kind,bodies,"
    "leaves,items,threads,build_seconds,build_seconds_per_item,"
    "update_seconds,ns_per_item,measured_fraction,geometry_bytes,"
    "topology_bytes,operator_bytes,index_bytes,metadata_bytes,"
    "invariant_bytes,scratch_bytes,total_persistent_bytes,unique_operators,"
    "checksum,max_relative_error,compiler,compiler_version,build_type,"
    "cpu_model,gpu_model,note";

class RowSink {
public:
  explicit RowSink(const Options& options) : options_(options) {
    if (!options.output.empty()) {
      std::ifstream existing(options.output, std::ios::ate);
      const bool empty = !existing.good() || existing.tellg() <= 0;
      existing.close();
      file_.open(options.output, std::ios::app);
      if (!file_) {
        throw std::runtime_error("cannot open " + options.output);
      }
      if (empty) {
        file_ << csv_header << '\n';
      }
    }
  }

  void emit(const Row& row) {
    print_console(row);
    if (file_) {
      write_csv(row);
    }
  }

private:
  void print_console(const Row& row) const {
    std::cout << std::left << std::setw(11) << row.stage << std::setw(28)
              << (row.source_geometry + "->" + row.target_geometry + " " +
                  row.layout)
              << std::setw(15) << row.separation << std::setw(10) << row.mode
              << std::setw(5) << row.precision;
    if (row.order > 0) {
      std::cout << " p" << std::setw(3) << row.order;
    } else {
      std::cout << std::setw(5) << "";
    }
    std::cout << std::setw(6) << row.backend << std::setw(30)
              << row.representation << std::right << std::setw(12)
              << std::fixed << std::setprecision(2) << row.ns_per_item
              << " ns/item" << std::setw(12) << std::setprecision(4)
              << row.build_seconds << " s build" << std::setw(12)
              << (row.operator_bytes + row.index_bytes + row.metadata_bytes +
                  row.invariant_bytes) /
                     1.0e6
              << " MB" << std::setw(11) << std::scientific
              << std::setprecision(2) << row.max_relative_error << " err";
    if (row.measured_fraction < 1.0) {
      std::cout << "  (measured " << std::fixed << std::setprecision(3)
                << row.measured_fraction << " of the set)";
    }
    if (!row.note.empty()) {
      std::cout << "  " << row.note;
    }
    std::cout << std::defaultfloat << '\n';
  }

  void write_csv(const Row& row) {
    file_ << csv_quote(options_.label) << ',' << csv_quote(options_.commit)
          << ',' << row.stage << ',' << row.source_geometry << ','
          << row.target_geometry << ',' << row.layout << ',' << row.separation
          << ',' << row.mode << ',' << row.precision << ',' << row.order << ','
          << row.backend << ',' << row.representation << ','
          << row.representation_kind << ',' << row.bodies << ',' << row.leaves
          << ',' << row.items << ',' << row.threads << ','
          << std::setprecision(9) << row.build_seconds << ','
          << row.build_seconds_per_item << ',' << row.update_seconds << ','
          << row.ns_per_item << ',' << row.measured_fraction << ','
          << row.geometry_bytes << ',' << row.topology_bytes << ','
          << row.operator_bytes << ',' << row.index_bytes << ','
          << row.metadata_bytes << ',' << row.invariant_bytes << ','
          << row.scratch_bytes << ','
          << (row.geometry_bytes + row.topology_bytes + row.operator_bytes +
              row.index_bytes + row.metadata_bytes + row.invariant_bytes)
          << ',' << row.unique_operators << ',' << std::setprecision(17)
          << row.checksum << ',' << std::setprecision(6)
          << row.max_relative_error << ',' << compiler_name() << ','
          << csv_quote(compiler_version()) << ','
#if defined(NDEBUG)
          << "Release"
#else
          << "Debug"
#endif
          << ',' << csv_quote(cpu_model()) << ',' << csv_quote(gpu_model())
          << ',' << csv_quote(row.note) << '\n';
    file_.flush();
  }

  const Options& options_;
  std::ofstream file_;
};

//------------------------------------------------------------------------------
// Bodies
//------------------------------------------------------------------------------

/** @brief Deterministic per-body size factor and orientation of a layout. */
[[nodiscard]] double body_factor(const std::size_t index, const bool irregular) {
  return irregular
             ? 0.6 + 0.4 * static_cast<double>((index * 7919U) % 101U) / 100.0
             : 1.0;
}

// A prism spans `fill * spacing` per axis; a tetrahedron is the centred unit
// right simplex scaled so that its farthest vertex stays inside the same
// extent.  These match the finite-body workloads of benchmark_uniform_fmm.
[[nodiscard]] RectangularPrism make_prism(const double spacing,
                                          const double fill,
                                          const std::size_t index,
                                          const bool irregular) {
  const double extent = fill * body_factor(index, irregular) * spacing;
  return {extent, extent, extent};
}

[[nodiscard]] Tetrahedron make_tetrahedron(const double spacing,
                                           const double fill,
                                           const std::size_t index,
                                           const bool irregular) {
  const double size = fill * body_factor(index, irregular) * spacing / 1.5;
  const std::array<Vec3, 4> base{{{-0.25, -0.25, -0.25},
                                  {0.75, -0.25, -0.25},
                                  {-0.25, 0.75, -0.25},
                                  {-0.25, -0.25, 0.75}}};
  const int orientation = irregular ? static_cast<int>(index % 3) : 0;
  Tetrahedron result;
  for (std::size_t vertex = 0; vertex < 4; ++vertex) {
    const Vec3 v = base[vertex] * size;
    const std::array<double, 3> c{{v.x, v.y, v.z}};
    result.vertices[vertex] = {
        c[static_cast<std::size_t>(orientation % 3)],
        c[static_cast<std::size_t>((orientation + 1) % 3)],
        c[static_cast<std::size_t>((orientation + 2) % 3)]};
  }
  return result;
}

[[nodiscard]] double circumradius(const RectangularPrism& prism) {
  return 0.5 * std::sqrt(prism.hx * prism.hx + prism.hy * prism.hy +
                         prism.hz * prism.hz);
}

[[nodiscard]] double circumradius(const Tetrahedron& tetrahedron) {
  double result = 0.0;
  for (const Vec3& vertex : tetrahedron.vertices) {
    result = std::max(result, std::sqrt(dot(vertex, vertex)));
  }
  return result;
}

/**
 * @brief The bodies of one role (source or target) in sorted order.
 *
 * `common` bodies share one record, as the production plan does for a regular
 * lattice; otherwise every body has its own.  Prepared forms are the
 * per-body invariants a procedural executor is allowed to retain.
 */
struct BodySet {
  Geometry geometry{Geometry::Point};
  std::vector<Vec3> positions{};
  std::vector<RectangularPrism> prisms{};
  std::vector<Tetrahedron> tetrahedra{};
  std::vector<cdfmm::detail::PreparedTetrahedron> prepared_tetrahedra{};
  std::vector<cdfmm::detail::PolyhedronBody> polyhedra{};
  bool common{true};

  [[nodiscard]] const RectangularPrism& prism(const std::size_t index) const {
    return prisms[common ? 0 : index];
  }
  [[nodiscard]] const Tetrahedron& tetrahedron(const std::size_t index) const {
    return tetrahedra[common ? 0 : index];
  }
  [[nodiscard]] const cdfmm::detail::PreparedTetrahedron&
  prepared(const std::size_t index) const {
    return prepared_tetrahedra[common ? 0 : index];
  }
  [[nodiscard]] const cdfmm::detail::PolyhedronBody&
  polyhedron(const std::size_t index) const {
    return polyhedra[common ? 0 : index];
  }
  [[nodiscard]] double body_circumradius(const std::size_t index) const {
    switch (geometry) {
    case Geometry::Point:
      return 0.0;
    case Geometry::Prism:
      return circumradius(prism(index));
    case Geometry::Tetrahedron:
      return circumradius(tetrahedron(index));
    }
    return 0.0;
  }

  /// Retained geometry bytes: positions plus the body records.
  [[nodiscard]] std::size_t geometry_bytes() const {
    return positions.size() * sizeof(Vec3) +
           prisms.size() * sizeof(RectangularPrism) +
           tetrahedra.size() * sizeof(Tetrahedron);
  }

  /// Retained bytes of the prepared per-body invariants.
  [[nodiscard]] std::size_t invariant_bytes() const {
    std::size_t bytes = prepared_tetrahedra.size() *
                        sizeof(cdfmm::detail::PreparedTetrahedron);
    for (const auto& body : polyhedra) {
      bytes += sizeof(cdfmm::detail::PolyhedronBody) +
               body.surface.faces.size() * sizeof(std::array<Vec3, 3>) +
               body.surface.outward_normals.size() * sizeof(Vec3);
    }
    return bytes;
  }
};

void prepare_bodies(BodySet& set, const bool needs_polyhedra) {
  if (set.geometry == Geometry::Tetrahedron) {
    set.prepared_tetrahedra.reserve(set.tetrahedra.size());
    for (const Tetrahedron& tetrahedron : set.tetrahedra) {
      set.prepared_tetrahedra.push_back(
          cdfmm::detail::prepare_tetrahedron(tetrahedron));
    }
  }
  if (needs_polyhedra) {
    if (set.geometry == Geometry::Prism) {
      for (const RectangularPrism& prism : set.prisms) {
        set.polyhedra.push_back(cdfmm::detail::prepare_polyhedron_body(prism));
      }
    } else if (set.geometry == Geometry::Tetrahedron) {
      for (const Tetrahedron& tetrahedron : set.tetrahedra) {
        set.polyhedra.push_back(
            cdfmm::detail::prepare_polyhedron_body(tetrahedron));
      }
    }
  }
}

void fill_records(BodySet& set, const Geometry geometry, const double spacing,
                  const double fill, const bool irregular,
                  const std::size_t count, const bool common) {
  set.geometry = geometry;
  set.common = common;
  const std::size_t records = common ? 1 : count;
  if (geometry == Geometry::Prism) {
    for (std::size_t index = 0; index < records; ++index) {
      set.prisms.push_back(make_prism(spacing, fill, index, irregular));
    }
  } else if (geometry == Geometry::Tetrahedron) {
    for (std::size_t index = 0; index < records; ++index) {
      set.tetrahedra.push_back(
          make_tetrahedron(spacing, fill, index, irregular));
    }
  }
}

//------------------------------------------------------------------------------
// Pair-tensor reconstruction (the procedural P2P representation)
//------------------------------------------------------------------------------

/**
 * @brief Reconstructs one exact pair tensor from the retained bodies.
 *
 * Exactly the per-pair functions the canonical builder in
 * `src/operators/p2p.cpp` calls, so the procedural and the precomputed
 * representation share one mathematical definition.  Prepared bodies
 * (tetrahedron faces and normals, polyhedron surfaces) are the hoisted
 * per-body invariants; nothing per pair is retained.
 */
struct PairReconstruction {
  /// Reconstruction precision of the prism <-> point kernel: the production
  /// `long double`, or the precision-generic kernel in double / float.
  enum class Math { Production, Double, Float };

  const BodySet* sources{nullptr};
  const BodySet* targets{nullptr};
  Math math{Math::Production};

  template <typename Real>
  [[nodiscard]] static PairTensor prism_point(const RectangularPrism& prism,
                                              const Vec3& displacement) {
    Real tensor[6];
    cdfmm::detail::prism_kernel::rectangular_prism_point_tensor_components<Real>(
        static_cast<Real>(prism.hx), static_cast<Real>(prism.hy),
        static_cast<Real>(prism.hz), static_cast<Real>(displacement.x),
        static_cast<Real>(displacement.y), static_cast<Real>(displacement.z),
        tensor);
    return {static_cast<double>(tensor[0]), static_cast<double>(tensor[1]),
            static_cast<double>(tensor[2]), static_cast<double>(tensor[3]),
            static_cast<double>(tensor[4]), static_cast<double>(tensor[5])};
  }

  [[nodiscard]] PairTensor operator()(const std::size_t target,
                                      const std::size_t source,
                                      const Vec3& displacement) const {
    const Geometry sg = sources->geometry;
    const Geometry tg = targets->geometry;
    if (math != Math::Production &&
        ((sg == Geometry::Prism && tg == Geometry::Point) ||
         (sg == Geometry::Point && tg == Geometry::Prism))) {
      // Reciprocity: the point -> prism tensor is the target prism's point
      // tensor at the same displacement (as in the canonical builder).
      const RectangularPrism& prism =
          sg == Geometry::Prism ? sources->prism(source) : targets->prism(target);
      return math == Math::Double ? prism_point<double>(prism, displacement)
                                  : prism_point<float>(prism, displacement);
    }
    if (sg == Geometry::Tetrahedron && tg == Geometry::Tetrahedron) {
      // The same sorted index names the same body (targets are sources).
      const bool coincident =
          target == source && dot(displacement, displacement) == 0.0;
      return cdfmm::detail::tetrahedron_tetrahedron_tensor_prepared(
          displacement, sources->prepared(source), targets->prepared(target),
          coincident);
    }
    if (sg == Geometry::Tetrahedron && tg == Geometry::Prism) {
      return cdfmm::detail::polyhedron_pair_tensor(
          displacement, sources->polyhedron(source),
          targets->polyhedron(target));
    }
    if (sg == Geometry::Prism && tg == Geometry::Tetrahedron) {
      return cdfmm::detail::polyhedron_pair_tensor(
          displacement, sources->polyhedron(source),
          targets->polyhedron(target));
    }
    if (sg == Geometry::Tetrahedron) {
      return cdfmm::tetrahedron_point_tensor(displacement,
                                             sources->tetrahedron(source));
    }
    if (tg == Geometry::Tetrahedron) {
      return cdfmm::point_tetrahedron_tensor(displacement,
                                             targets->tetrahedron(target));
    }
    const cdfmm::SourceGeometry source_geometry =
        sg == Geometry::Prism ? cdfmm::SourceGeometry::RectangularPrism
                              : cdfmm::SourceGeometry::PointDipole;
    const cdfmm::TargetGeometry target_geometry =
        tg == Geometry::Prism ? cdfmm::TargetGeometry::RectangularPrism
                              : cdfmm::TargetGeometry::Point;
    const RectangularPrism source_prism =
        sg == Geometry::Prism ? sources->prism(source) : RectangularPrism{};
    const RectangularPrism target_prism =
        tg == Geometry::Prism ? targets->prism(target) : RectangularPrism{};
    return cdfmm::operators::p2p::build_pair(
        targets->positions[target], sources->positions[source],
        source_geometry, target_geometry, source_prism, target_prism);
  }
};

template <typename Scalar, typename Moment, typename Field>
inline void accumulate_tensor(const PairTensor& tensor, const Moment& m,
                              Field& H) {
  const Scalar xx = static_cast<Scalar>(tensor.xx);
  const Scalar xy = static_cast<Scalar>(tensor.xy);
  const Scalar xz = static_cast<Scalar>(tensor.xz);
  const Scalar yy = static_cast<Scalar>(tensor.yy);
  const Scalar yz = static_cast<Scalar>(tensor.yz);
  const Scalar zz = static_cast<Scalar>(tensor.zz);
  H.x += xx * m.x + xy * m.y + xz * m.z;
  H.y += xy * m.x + yy * m.y + yz * m.z;
  H.z += xz * m.x + yz * m.y + zz * m.z;
}

//------------------------------------------------------------------------------
// Moments, fields and checksums
//------------------------------------------------------------------------------

template <typename Vector>
void fill_random(std::vector<Vector>& values, const unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> component(-1.0, 1.0);
  for (Vector& value : values) {
    value.x = static_cast<decltype(value.x)>(component(generator));
    value.y = static_cast<decltype(value.y)>(component(generator));
    value.z = static_cast<decltype(value.z)>(component(generator));
  }
}

// Every update sees different moments: the base moments are scaled by a
// per-update factor and rotated among the components so that no executor can
// reuse a previous result.
template <typename Vector>
void update_moments(const std::vector<Vector>& base, std::vector<Vector>& out,
                    const int update) {
  using Scalar = decltype(base[0].x);
  const Scalar scale =
      static_cast<Scalar>(1.0 + 0.001 * static_cast<double>(update % 97));
  const int rotation = update % 3;
  for (std::size_t index = 0; index < base.size(); ++index) {
    const Scalar components[3] = {base[index].x, base[index].y, base[index].z};
    out[index].x = scale * components[(0 + rotation) % 3];
    out[index].y = scale * components[(1 + rotation) % 3];
    out[index].z = scale * components[(2 + rotation) % 3];
  }
}

template <typename Vector>
[[nodiscard]] double checksum(const std::vector<Vector>& fields) {
  long double sum = 0.0L;
  for (const Vector& field : fields) {
    sum += static_cast<long double>(field.x) +
           2.0L * static_cast<long double>(field.y) +
           3.0L * static_cast<long double>(field.z);
  }
  return static_cast<double>(sum);
}

template <typename Vector>
[[nodiscard]] double max_relative_error(const std::vector<Vector>& fields,
                                        const std::vector<Vec3>& reference) {
  double scale = 0.0;
  for (const Vec3& value : reference) {
    scale = std::max({scale, std::abs(value.x), std::abs(value.y),
                      std::abs(value.z)});
  }
  double error = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    error = std::max(
        {error,
         std::abs(static_cast<double>(fields[index].x) - reference[index].x),
         std::abs(static_cast<double>(fields[index].y) - reference[index].y),
         std::abs(static_cast<double>(fields[index].z) - reference[index].z)});
  }
  return scale > 0.0 ? error / scale : error;
}

/**
 * @brief Times repeated updates of one representation.
 *
 * `apply(update)` performs one complete update with the moments of that
 * update already in place; the moment rewrite is excluded from the timing.
 * Returns the median over the samples of the mean update time.
 */
template <typename Apply, typename Prepare>
[[nodiscard]] double time_updates(const Options& options, Prepare prepare,
                                  Apply apply, const int evaluations) {
  int update = 0;
  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    prepare(update);
    apply();
    ++update;
  }
  std::vector<double> samples;
  for (int sample = 0; sample < options.samples; ++sample) {
    double total = 0.0;
    for (int evaluation = 0; evaluation < evaluations; ++evaluation) {
      prepare(update);
      const auto start = Clock::now();
      apply();
      total += elapsed(start);
      ++update;
    }
    samples.push_back(total / evaluations);
  }
  return median(samples);
}

//------------------------------------------------------------------------------
// P2P scenes
//------------------------------------------------------------------------------

/** @brief One P2P working set: bodies, list-1 topology and canonical rows. */
struct P2PScene {
  BodySet sources;
  BodySet targets;
  std::vector<std::array<int, 2>> interactions;
  std::vector<cdfmm::StaticP2PLeafPair> leaf_pairs;
  // Target-leaf loop of the procedural executor: leaf -> [(target range),
  // (source ranges)].  For a hot set every pair is its own "leaf".
  struct LeafWork {
    int target_begin{0};
    int target_count{0};
    std::vector<std::array<int, 2>> source_ranges;  // begin, count
  };
  std::vector<LeafWork> leaf_work;
  std::vector<int> identities;
  std::size_t leaves{0};
  double spacing{0.0};
};

[[nodiscard]] std::size_t topology_bytes(const P2PScene& scene) {
  std::size_t bytes = 0;
  for (const auto& work : scene.leaf_work) {
    bytes += 2 * sizeof(int) + work.source_ranges.size() * 2 * sizeof(int);
  }
  return bytes;
}

// A complete list-1 neighbourhood on a lattice of `S^3` bodies per leaf,
// jittered for the irregular layout, exactly as the uniform tree would see it.
[[nodiscard]] P2PScene make_lattice_scene(const Options& options) {
  P2PScene scene;
  const int boxes_per_axis = 1 << options.depth;
  const int cells_per_axis = boxes_per_axis * options.bodies_per_leaf_axis;
  const double spacing = 2.0 / cells_per_axis;
  scene.spacing = spacing;
  std::vector<Vec3> positions;
  positions.reserve(static_cast<std::size_t>(cells_per_axis) * cells_per_axis *
                    cells_per_axis);
  std::mt19937 generator(314159u);
  std::uniform_real_distribution<double> jitter(-options.jitter,
                                                options.jitter);
  for (int iz = 0; iz < cells_per_axis; ++iz) {
    for (int iy = 0; iy < cells_per_axis; ++iy) {
      for (int ix = 0; ix < cells_per_axis; ++ix) {
        Vec3 position{-1.0 + (ix + 0.5) * spacing, -1.0 + (iy + 0.5) * spacing,
                      -1.0 + (iz + 0.5) * spacing};
        if (options.irregular) {
          position.x += jitter(generator) * spacing;
          position.y += jitter(generator) * spacing;
          position.z += jitter(generator) * spacing;
        }
        positions.push_back(position);
      }
    }
  }
  cdfmm::UniformTreeOptions tree_options;
  tree_options.max_level = options.depth;
  tree_options.root_centre = Vec3{};
  tree_options.root_half_width = 1.0;
  const cdfmm::UniformTree tree(positions, positions, tree_options);
  const auto sorted = tree.sorted_source_positions();
  scene.sources.positions.assign(sorted.begin(), sorted.end());
  scene.targets.positions = scene.sources.positions;
  const std::size_t count = scene.sources.positions.size();
  const bool common = !options.irregular;
  fill_records(scene.sources, options.source, spacing, options.fill,
               options.irregular, count, common);
  fill_records(scene.targets, options.target, spacing, options.fill,
               options.irregular, count, common);

  const auto nodes = tree.nodes();
  for (const int leaf_index : tree.occupied_target_leaves()) {
    const cdfmm::TreeNode& leaf = nodes[static_cast<std::size_t>(leaf_index)];
    P2PScene::LeafWork work;
    work.target_begin = static_cast<int>(leaf.target_begin);
    work.target_count = static_cast<int>(leaf.target_count());
    for (const int neighbour_index : leaf.list1) {
      const cdfmm::TreeNode& neighbour =
          nodes[static_cast<std::size_t>(neighbour_index)];
      if (neighbour.source_count() == 0) {
        continue;
      }
      scene.leaf_pairs.push_back({work.target_begin, work.target_count,
                                  static_cast<int>(neighbour.source_begin),
                                  static_cast<int>(neighbour.source_count())});
      work.source_ranges.push_back(
          {static_cast<int>(neighbour.source_begin),
           static_cast<int>(neighbour.source_count())});
      for (std::size_t target = leaf.target_begin; target < leaf.target_end;
           ++target) {
        for (std::size_t source = neighbour.source_begin;
             source < neighbour.source_end; ++source) {
          scene.interactions.push_back(
              {static_cast<int>(target), static_cast<int>(source)});
        }
      }
    }
    scene.leaf_work.push_back(std::move(work));
  }
  scene.leaves = scene.leaf_work.size();
  scene.identities.resize(count);
  std::iota(scene.identities.begin(), scene.identities.end(), 0);
  return scene;
}

// `pairs` independent (target i, source i) pairs of one separation class.
// The spacing is that of the default lattice so that the classes are
// comparable with the streaming set.
[[nodiscard]] P2PScene make_hot_scene(const Options& options) {
  P2PScene scene;
  const int boxes_per_axis = 1 << options.depth;
  const int cells_per_axis = boxes_per_axis * options.bodies_per_leaf_axis;
  const double spacing = 2.0 / cells_per_axis;
  scene.spacing = spacing;
  const std::size_t count = static_cast<std::size_t>(options.pairs);
  const bool common = !options.irregular;
  double fill = options.fill;
  if (options.separation == Separation::SmallFar) {
    fill *= 0.25;
  }
  fill_records(scene.sources, options.source, spacing, fill, options.irregular,
               count, common);
  fill_records(scene.targets, options.target, spacing, fill, options.irregular,
               count, common);

  std::mt19937 generator(271828u);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::uniform_int_distribution<int> lattice_offset(-3, 3);
  std::uniform_int_distribution<int> axis(0, 2);
  std::uniform_int_distribution<int> sign(0, 1);
  // Sources sit on their own random lattice-scale sites.
  for (std::size_t index = 0; index < count; ++index) {
    scene.sources.positions.push_back({spacing * (unit(generator) * 4.0),
                                       spacing * (unit(generator) * 4.0),
                                       spacing * (unit(generator) * 4.0)});
  }
  for (std::size_t index = 0; index < count; ++index) {
    // The self class pairs a body with itself (finite sources only; the
    // point self pair is excluded by identity).  Every other class pairs
    // target `index` with source `index + 1` so that the identity map, which
    // names body `index` as its own self, never excludes a measured pair.
    const std::size_t source_index = options.separation == Separation::Self
                                         ? index
                                         : (index + 1) % count;
    const Vec3 source = scene.sources.positions[source_index];
    Vec3 offset{};
    switch (options.separation) {
    case Separation::Self:
      break;
    case Separation::Adjacent: {
      const int direction = axis(generator);
      offset[direction] = (sign(generator) == 0 ? -1.0 : 1.0) * spacing;
      break;
    }
    case Separation::Lattice:
    case Separation::List1:
    case Separation::SmallFar: {
      do {
        offset = {spacing * lattice_offset(generator),
                  spacing * lattice_offset(generator),
                  spacing * lattice_offset(generator)};
      } while (dot(offset, offset) == 0.0);
      break;
    }
    case Separation::FarSafeguard: {
      const double radii = scene.sources.body_circumradius(source_index) +
                           scene.targets.body_circumradius(index);
      const double distance =
          radii > 0.0
              ? 10.0 * radii
              : 10.0 * spacing;  // point-point: the same relative scale
      Vec3 direction{unit(generator), unit(generator), unit(generator)};
      const double norm = std::sqrt(dot(direction, direction));
      offset = direction * (distance / (norm > 0.0 ? norm : 1.0));
      break;
    }
    }
    if (options.irregular && options.separation != Separation::Self) {
      offset.x += options.jitter * spacing * unit(generator);
      offset.y += options.jitter * spacing * unit(generator);
      offset.z += options.jitter * spacing * unit(generator);
    }
    scene.targets.positions.push_back(source + offset);
    scene.interactions.push_back(
        {static_cast<int>(index), static_cast<int>(source_index)});
    scene.leaf_pairs.push_back(
        {static_cast<int>(index), 1, static_cast<int>(source_index), 1});
    P2PScene::LeafWork work;
    work.target_begin = static_cast<int>(index);
    work.target_count = 1;
    work.source_ranges.push_back({static_cast<int>(source_index), 1});
    scene.leaf_work.push_back(std::move(work));
  }
  scene.leaves = count;
  scene.identities.resize(count);
  std::iota(scene.identities.begin(), scene.identities.end(), 0);
  return scene;
}

[[nodiscard]] cdfmm::SourceGeometry source_enum(const Geometry geometry) {
  switch (geometry) {
  case Geometry::Point:
    return cdfmm::SourceGeometry::PointDipole;
  case Geometry::Prism:
    return cdfmm::SourceGeometry::RectangularPrism;
  case Geometry::Tetrahedron:
    return cdfmm::SourceGeometry::Tetrahedron;
  }
  return cdfmm::SourceGeometry::PointDipole;
}

[[nodiscard]] cdfmm::TargetGeometry target_enum(const Geometry geometry) {
  switch (geometry) {
  case Geometry::Point:
    return cdfmm::TargetGeometry::Point;
  case Geometry::Prism:
    return cdfmm::TargetGeometry::RectangularPrism;
  case Geometry::Tetrahedron:
    return cdfmm::TargetGeometry::Tetrahedron;
  }
  return cdfmm::TargetGeometry::Point;
}

/** @brief The production canonical construction, timed. */
[[nodiscard]] cdfmm::StaticP2POperator build_canonical(const P2PScene& scene,
                                                       double& seconds) {
  const auto start = Clock::now();
  cdfmm::StaticP2POperator result = cdfmm::build_static_p2p_operator(
      scene.targets.positions, scene.sources.positions, scene.interactions,
      source_enum(scene.sources.geometry), scene.sources.prisms,
      scene.sources.tetrahedra, target_enum(scene.targets.geometry),
      scene.targets.prisms, scene.targets.tetrahedra,
      cdfmm::SourceModel::ExactGeometry, cdfmm::TargetModel::ExactGeometry);
  seconds = elapsed(start);
  return result;
}

/**
 * @brief Procedural P2P over the scene's leaf work list.
 *
 * Parallel over target leaves like the production executors; each pair
 * tensor is reconstructed and applied at once.  `leaf_limit` truncates the
 * traversal when a complete procedural update would exceed the time budget.
 */
template <typename Scalar, typename Moment, typename Field>
void apply_procedural_p2p(const P2PScene& scene,
                          const PairReconstruction& reconstruct,
                          const std::vector<Moment>& moments,
                          std::vector<Field>& fields,
                          const std::size_t leaf_limit,
                          std::atomic<std::size_t>& failures) {
  // A point source's coincident self pair is singular and excluded by
  // identity in the canonical operator whatever the target geometry; a
  // finite source's self tensor is physical and applied.
  const bool point_source = scene.sources.geometry == Geometry::Point;
  const int leaf_count = static_cast<int>(std::min(leaf_limit, scene.leaves));
#pragma omp parallel for schedule(dynamic, 1) if (leaf_count >= 8)
  for (int leaf = 0; leaf < leaf_count; ++leaf) {
    const P2PScene::LeafWork& work = scene.leaf_work[static_cast<std::size_t>(leaf)];
    for (int t = 0; t < work.target_count; ++t) {
      const std::size_t target = static_cast<std::size_t>(work.target_begin + t);
      const Vec3& target_position = scene.targets.positions[target];
      Field H{};
      for (const auto& range : work.source_ranges) {
        for (int s = 0; s < range[1]; ++s) {
          const std::size_t source = static_cast<std::size_t>(range[0] + s);
          if (point_source && source == target) {
            continue;  // the singular point self pair is excluded by identity
          }
          const Vec3 displacement =
              target_position - scene.sources.positions[source];
          try {
            const PairTensor tensor = reconstruct(target, source, displacement);
            accumulate_tensor<Scalar>(tensor, moments[source], H);
          } catch (const std::exception&) {
            failures.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
      fields[target] = H;
    }
  }
}

[[nodiscard]] std::size_t unique_tensor_count(
    const cdfmm::StaticP2POperator& canonical) {
  std::vector<std::array<double, 6>> tensors;
  tensors.reserve(canonical.blocks.size());
  for (const auto& block : canonical.blocks) {
    tensors.push_back({std::abs(block.xx), std::abs(block.xy), std::abs(block.xz),
                       std::abs(block.yy), std::abs(block.yz), std::abs(block.zz)});
  }
  std::sort(tensors.begin(), tensors.end());
  return static_cast<std::size_t>(
      std::unique(tensors.begin(), tensors.end()) - tensors.begin());
}

template <typename Plan>
void fill_memory(Row& row, const Plan& plan) {
  const cdfmm::StaticP2PMemory memory = plan.memory();
  row.operator_bytes = memory.tensor_bytes;
  row.index_bytes = memory.index_bytes;
  row.metadata_bytes = memory.row_metadata_bytes + memory.leaf_metadata_bytes;
  row.scratch_bytes = memory.scratch_bytes;
}

void run_p2p(const Options& options, RowSink& sink) {
  P2PScene scene = options.mode == Mode::Streaming ? make_lattice_scene(options)
                                                   : make_hot_scene(options);
  const bool needs_polyhedra =
      (scene.sources.geometry == Geometry::Prism &&
       scene.targets.geometry == Geometry::Tetrahedron) ||
      (scene.sources.geometry == Geometry::Tetrahedron &&
       scene.targets.geometry == Geometry::Prism);
  prepare_bodies(scene.sources, needs_polyhedra);
  prepare_bodies(scene.targets, needs_polyhedra);
  const std::size_t bodies = scene.sources.positions.size();
  const std::size_t pairs = scene.interactions.size();
  const bool point_source = scene.sources.geometry == Geometry::Point;
  // A streaming lattice pairs every body with itself once; point sources
  // skip that singular pair, so it is not an applied pair.
  const std::size_t applied_pairs =
      point_source && options.mode == Mode::Streaming ? pairs - bodies : pairs;
  if (point_source && options.separation == Separation::Self) {
    throw std::invalid_argument(
        "a point source at the target representative is the excluded "
        "singular self pair; finite sources define the self tensor");
  }

  std::cout << "# P2P " << geometry_name(scene.sources.geometry) << " -> "
            << geometry_name(scene.targets.geometry) << ", "
            << (options.irregular ? "irregular" : "regular") << ", "
            << (options.mode == Mode::Hot ? "hot" : "streaming") << " ("
            << separation_name(options.separation) << "), bodies " << bodies
            << ", leaves " << scene.leaves << ", pairs " << pairs
            << ", threads " << thread_count() << ", spacing " << scene.spacing
            << '\n';

  Row base;
  base.stage = "p2p";
  base.source_geometry = geometry_name(scene.sources.geometry);
  base.target_geometry = geometry_name(scene.targets.geometry);
  base.layout = options.irregular ? "irregular" : "regular";
  base.separation = separation_name(options.separation);
  base.mode = options.mode == Mode::Hot ? "hot" : "streaming";
  base.bodies = bodies;
  base.leaves = scene.leaves;
  base.items = applied_pairs;
  base.threads = options.mode == Mode::Hot ? 1 : thread_count();
  base.geometry_bytes =
      scene.sources.geometry_bytes() +
      (scene.targets.geometry == scene.sources.geometry ? 0
                                                        : scene.targets.geometry_bytes());
  base.topology_bytes = topology_bytes(scene);

  // Construction: the production canonical operator.  A hot set is small
  // enough for the first build to be dominated by frequency ramp-up and
  // first-touch allocation, so it is built twice and the second time is
  // reported; a streaming set is built once.
  double build_seconds = 0.0;
  if (options.mode == Mode::Hot) {
    (void)build_canonical(scene, build_seconds);
  }
  const cdfmm::StaticP2POperator canonical = build_canonical(scene, build_seconds);
  const std::size_t unique_tensors = unique_tensor_count(canonical);
  std::cout << "# canonical build " << std::fixed << std::setprecision(3)
            << build_seconds << " s (" << std::setprecision(1)
            << 1.0e6 * build_seconds / std::max<std::size_t>(pairs, 1)
            << " us/pair), unique |tensors| " << unique_tensors << " of "
            << pairs << std::defaultfloat << '\n';

  // Moments and the FP64 reference fields (canonical rows, final moments).
  std::vector<Vec3> base_moments(bodies);
  fill_random(base_moments, 813u);
  std::vector<Vec3> moments(bodies);
  std::vector<cdfmm::FloatVec3> base_moments_float(bodies);
  for (std::size_t index = 0; index < bodies; ++index) {
    base_moments_float[index] = {static_cast<float>(base_moments[index].x),
                                 static_cast<float>(base_moments[index].y),
                                 static_cast<float>(base_moments[index].z)};
  }
  std::vector<cdfmm::FloatVec3> moments_float(bodies);
  // The update counter of the last timed update is deterministic:
  // warmups + samples * evaluations - 1.
  const int final_update =
      options.warmups + options.samples * options.evaluations - 1;
  update_moments(base_moments, moments, final_update);
  update_moments(base_moments_float, moments_float, final_update);
  std::vector<Vec3> reference(bodies);
  cdfmm::apply_static_p2p_operator(canonical, moments, reference,
                                   scene.identities);

  auto emit_fp64 = [&](const std::string& name, const std::string& kind,
                       const double representation_build_seconds,
                       const std::vector<Vec3>& fields, const double seconds,
                       const double fraction, Row row) {
    row.precision = "fp64";
    if (row.backend.empty()) {
      row.backend = "cpu";
    }
    row.representation = name;
    row.representation_kind = kind;
    row.build_seconds = representation_build_seconds;
    row.build_seconds_per_item =
        representation_build_seconds / std::max<std::size_t>(pairs, 1);
    row.update_seconds = seconds / fraction;
    row.ns_per_item = 1.0e9 * seconds /
                      (fraction * static_cast<double>(std::max<std::size_t>(applied_pairs, 1)));
    row.measured_fraction = fraction;
    row.checksum = checksum(fields);
    row.max_relative_error = max_relative_error(fields, reference);
    sink.emit(row);
  };
  auto emit_fp32 = [&](const std::string& name, const std::string& kind,
                       const double representation_build_seconds,
                       const std::vector<cdfmm::FloatVec3>& fields,
                       const double seconds, const double fraction, Row row) {
    row.precision = "fp32";
    if (row.backend.empty()) {
      row.backend = "cpu";
    }
    row.representation = name;
    row.representation_kind = kind;
    row.build_seconds = representation_build_seconds;
    row.build_seconds_per_item =
        representation_build_seconds / std::max<std::size_t>(pairs, 1);
    row.update_seconds = seconds / fraction;
    row.ns_per_item = 1.0e9 * seconds /
                      (fraction * static_cast<double>(std::max<std::size_t>(applied_pairs, 1)));
    row.measured_fraction = fraction;
    row.checksum = checksum(fields);
    row.max_relative_error = max_relative_error(fields, reference);
    sink.emit(row);
  };

  // Hot mode runs single threaded: the intrinsic cost of one operator.
#if defined(CDFMM_USE_OPENMP)
  const int saved_threads = omp_get_max_threads();
  if (options.mode == Mode::Hot) {
    omp_set_num_threads(1);
  }
#endif

  std::vector<Vec3> fields(bodies);
  std::vector<cdfmm::FloatVec3> fields_float(bodies);
  // The production apply functions accumulate into the field buffer, so the
  // (untimed) preparation of every update also clears it.
  auto prepare_fp64 = [&](const int update) {
    update_moments(base_moments, moments, update);
    std::fill(fields.begin(), fields.end(), Vec3{});
  };
  auto prepare_fp32 = [&](const int update) {
    update_moments(base_moments_float, moments_float, update);
    std::fill(fields_float.begin(), fields_float.end(), cdfmm::FloatVec3{});
  };

  if (options.cpu_precomputed) {
    // Canonical AoS rows.
    if (options.fp64) {
      const double seconds = time_updates(
          options, prepare_fp64,
          [&] {
            cdfmm::apply_static_p2p_operator(canonical, moments, fields,
                                             scene.identities);
          },
          options.evaluations);
      Row row = base;
      row.operator_bytes = canonical.blocks.size() * 9 * sizeof(double);
      row.index_bytes = canonical.blocks.size() * 3 * sizeof(int);
      row.metadata_bytes = canonical.row_offsets.size() * sizeof(int);
      row.unique_operators = unique_tensors;
      emit_fp64("canonical-aos", "precomputed", build_seconds, fields, seconds,
                1.0, row);
    }

    // Particle-row SoA: the CPU production packing for finite bodies.
    auto start = Clock::now();
    const cdfmm::StaticP2PCompactPlan compact =
        cdfmm::build_static_p2p_compact_plan(canonical);
    const double compact_seconds = elapsed(start);
    if (options.fp64) {
      const double seconds = time_updates(
          options, prepare_fp64,
          [&] {
            cdfmm::apply_static_p2p_compact_plan(compact, moments, fields,
                                                 scene.identities);
          },
          options.evaluations);
      Row row = base;
      fill_memory(row, compact);
      row.unique_operators = unique_tensors;
      emit_fp64("particle-row-soa", "precomputed",
                build_seconds + compact_seconds, fields, seconds, 1.0, row);
    }
    if (options.fp32) {
      start = Clock::now();
      const cdfmm::FloatStaticP2PCompactPlan compact_float =
          cdfmm::quantise_static_p2p_compact_plan(compact);
      const double quantise_seconds = elapsed(start);
      const double seconds = time_updates(
          options, prepare_fp32,
          [&] {
            cdfmm::apply_static_p2p_compact_plan(compact_float, moments_float,
                                                 fields_float, scene.identities);
          },
          options.evaluations);
      Row row = base;
      fill_memory(row, compact_float);
      row.unique_operators = unique_tensors;
      emit_fp32("particle-row-soa", "precomputed",
                build_seconds + compact_seconds + quantise_seconds, fields_float,
                seconds, 1.0, row);
    }

    // Dense leaf blocks (the CUDA default packing, also executable on the CPU).
    start = Clock::now();
    const cdfmm::StaticP2PLeafPlan leaf =
        cdfmm::build_static_p2p_leaf_plan(canonical, scene.leaf_pairs);
    const double leaf_seconds = elapsed(start);
    if (options.fp64) {
      const double seconds = time_updates(
          options, prepare_fp64,
          [&] {
            cdfmm::apply_static_p2p_leaf_plan(leaf, moments, fields,
                                              scene.identities);
          },
          options.evaluations);
      Row row = base;
      fill_memory(row, leaf);
      row.unique_operators = unique_tensors;
      emit_fp64("leaf-block", "precomputed", build_seconds + leaf_seconds,
                fields, seconds, 1.0, row);
    }
    if (options.fp32) {
      const cdfmm::FloatStaticP2PLeafPlan leaf_float =
          cdfmm::quantise_static_p2p_leaf_plan(leaf);
      const double seconds = time_updates(
          options, prepare_fp32,
          [&] {
            cdfmm::apply_static_p2p_leaf_plan(leaf_float, moments_float,
                                              fields_float, scene.identities);
          },
          options.evaluations);
      Row row = base;
      fill_memory(row, leaf_float);
      row.unique_operators = unique_tensors;
      emit_fp32("leaf-block", "precomputed", build_seconds + leaf_seconds,
                fields_float, seconds, 1.0, row);
    }

    // Signed tensor dictionary: the compressed packing a regular lattice
    // selects automatically; on an irregular layout it degenerates to one
    // variant per pair and is shown for completeness.
    start = Clock::now();
    const cdfmm::StaticP2PSignedTensorDictionaryPlan dictionary =
        cdfmm::build_static_p2p_signed_tensor_dictionary_plan(
            canonical, scene.leaf_pairs, scene.identities, 32);
    const double dictionary_seconds = elapsed(start);
    const std::string dictionary_name =
        "tensor-dictionary-" + std::to_string(dictionary.token_width_bytes) +
        "B";
    if (options.fp64) {
      const double seconds = time_updates(
          options, prepare_fp64,
          [&] {
            cdfmm::apply_static_p2p_signed_tensor_dictionary_plan(
                dictionary, moments, fields);
          },
          options.evaluations);
      Row row = base;
      fill_memory(row, dictionary);
      row.unique_operators = dictionary.variant_count();
      emit_fp64(dictionary_name, "precomputed",
                build_seconds + dictionary_seconds, fields, seconds, 1.0, row);
    }
    if (options.fp32) {
      const cdfmm::FloatStaticP2PSignedTensorDictionaryPlan dictionary_float =
          cdfmm::quantise_static_p2p_signed_tensor_dictionary_plan(dictionary);
      const double seconds = time_updates(
          options, prepare_fp32,
          [&] {
            cdfmm::apply_static_p2p_signed_tensor_dictionary_plan(
                dictionary_float, moments_float, fields_float);
          },
          options.evaluations);
      Row row = base;
      fill_memory(row, dictionary_float);
      row.unique_operators = dictionary_float.variant_count();
      emit_fp32(dictionary_name, "precomputed",
                build_seconds + dictionary_seconds, fields_float, seconds, 1.0,
                row);
    }
  }

#if defined(CDFMM_ENABLE_CUDA)
  if (options.cuda && cdfmm::cuda_available()) {
    // Stored CUDA representations: the leaf-block default, the canonical
    // baseline and the source-warp / target-owned dictionaries.  The
    // synchronous evaluate includes the moment upload and field download,
    // as the CudaPartial near field does; the kernel time is the device
    // event measurement of the last update.
    const cdfmm::StaticP2PLeafPlan leaf =
        cdfmm::build_static_p2p_leaf_plan(canonical, scene.leaf_pairs);
    const cdfmm::StaticP2PSignedTensorDictionaryPlan dictionary =
        cdfmm::build_static_p2p_signed_tensor_dictionary_plan(
            canonical, scene.leaf_pairs, scene.identities, 32);
    const std::string dictionary_name =
        "tensor-dictionary-" + std::to_string(dictionary.token_width_bytes) +
        "B";
    auto measure_cuda = [&](cdfmm::CudaP2PPlan& plan, const std::string& name,
                            const bool single, const std::size_t unique,
                            const std::size_t operator_bytes) {
      Row row = base;
      row.backend = "cuda";
      row.threads = 1;
      row.unique_operators = unique;
      const cdfmm::CudaPlanStatistics& statistics = plan.statistics();
      row.operator_bytes = statistics.p2p_tensor_bytes > 0
                               ? statistics.p2p_tensor_bytes
                               : operator_bytes;
      row.index_bytes = statistics.p2p_index_bytes;
      row.metadata_bytes = statistics.p2p_row_metadata_bytes +
                           statistics.p2p_leaf_metadata_bytes +
                           statistics.p2p_identity_bytes;
      row.scratch_bytes = statistics.p2p_scratch_bytes;
      double seconds = 0.0;
      if (single) {
        seconds = time_updates(
            options, prepare_fp32,
            [&] {
              plan.evaluate(moments_float, scene.identities, fields_float);
            },
            options.evaluations);
        row.note = "kernel_seconds=" +
                   std::to_string(plan.timings().kernel_seconds);
        emit_fp32(name, "precomputed", build_seconds, fields_float, seconds,
                  1.0, row);
      } else {
        seconds = time_updates(
            options, prepare_fp64,
            [&] { plan.evaluate(moments, scene.identities, fields); },
            options.evaluations);
        row.note = "kernel_seconds=" +
                   std::to_string(plan.timings().kernel_seconds);
        emit_fp64(name, "precomputed", build_seconds, fields, seconds, 1.0,
                  row);
      }
    };
    if (options.fp32) {
      const cdfmm::FloatStaticP2PLeafPlan leaf_float =
          cdfmm::quantise_static_p2p_leaf_plan(leaf);
      cdfmm::CudaP2PPlan plan(leaf_float, scene.identities);
      measure_cuda(plan, "leaf-block", true, unique_tensors,
                   leaf_float.memory().tensor_bytes);
      const cdfmm::FloatStaticP2PSignedTensorDictionaryPlan dictionary_float =
          cdfmm::quantise_static_p2p_signed_tensor_dictionary_plan(dictionary);
      cdfmm::CudaP2PPlan warp(dictionary_float, false, false);
      measure_cuda(warp, dictionary_name + "-source-warp", true,
                   dictionary_float.variant_count(),
                   dictionary_float.memory().tensor_bytes);
      cdfmm::CudaP2PPlan owned(dictionary_float, true, false);
      measure_cuda(owned, dictionary_name + "-target-owned", true,
                   dictionary_float.variant_count(),
                   dictionary_float.memory().tensor_bytes);
      const cdfmm::FloatStaticP2POperator canonical_float =
          cdfmm::quantise_static_p2p_operator(canonical);
      cdfmm::CudaP2PPlan aos(canonical_float, scene.identities);
      measure_cuda(aos, "canonical-aos", true, unique_tensors,
                   canonical_float.blocks.size() * 9 * sizeof(float));
    }
    if (options.fp64) {
      cdfmm::CudaP2PPlan plan(leaf, scene.identities);
      measure_cuda(plan, "leaf-block", false, unique_tensors,
                   leaf.memory().tensor_bytes);
      cdfmm::CudaP2PPlan warp(dictionary, false, false);
      measure_cuda(warp, dictionary_name + "-source-warp", false,
                   dictionary.variant_count(), dictionary.memory().tensor_bytes);
      cdfmm::CudaP2PPlan owned(dictionary, true, false);
      measure_cuda(owned, dictionary_name + "-target-owned", false,
                   dictionary.variant_count(), dictionary.memory().tensor_bytes);
    }
  } else if (options.cuda) {
    std::cout << "# CUDA requested but no device is available\n";
  }
#endif

  // Procedural: reconstruct every pair tensor from the retained bodies.
  const PairReconstruction reconstruct{&scene.sources, &scene.targets};
  std::atomic<std::size_t> failures{0};
  const bool prism_point_pair =
      (scene.sources.geometry == Geometry::Prism &&
       scene.targets.geometry == Geometry::Point) ||
      (scene.sources.geometry == Geometry::Point &&
       scene.targets.geometry == Geometry::Prism);
  Row procedural_base = base;
  procedural_base.operator_bytes = 0;
  procedural_base.index_bytes = 0;
  procedural_base.metadata_bytes = 0;
  procedural_base.invariant_bytes =
      scene.sources.invariant_bytes() +
      (scene.targets.geometry == scene.sources.geometry ? 0
                                                        : scene.targets.invariant_bytes());
  procedural_base.unique_operators = 0;

  // A calibration pass on the first leaves sizes the traversal against the
  // time budget; the measured fraction scales the reported update time.
  std::size_t leaf_limit = scene.leaves;
  double fraction = 1.0;
  int procedural_evaluations = options.evaluations;
  {
    const std::size_t probe = std::min<std::size_t>(scene.leaves, 8);
    const auto start = Clock::now();
    apply_procedural_p2p<double>(scene, reconstruct, moments, fields, probe,
                                 failures);
    const double probe_seconds = elapsed(start);
    std::size_t probe_pairs = 0;
    for (std::size_t leaf = 0; leaf < probe; ++leaf) {
      const auto& work = scene.leaf_work[leaf];
      for (const auto& range : work.source_ranges) {
        probe_pairs += static_cast<std::size_t>(work.target_count) *
                       static_cast<std::size_t>(range[1]);
      }
    }
    const double estimate = probe_seconds * static_cast<double>(pairs) /
                            static_cast<double>(std::max<std::size_t>(probe_pairs, 1)) /
                            (probe < 8 ? 1.0 : static_cast<double>(thread_count()));
    // The budget bounds one precision's complete procedural measurement
    // (warmups plus samples times evaluations); fewer evaluations come
    // first, a truncated traversal only when one update alone is too slow.
    const int minimal_updates = options.warmups + options.samples;
    const double per_update_budget =
        options.procedural_budget_seconds / static_cast<double>(minimal_updates);
    if (options.mode == Mode::Streaming && estimate > 0.0) {
      procedural_evaluations = std::max(
          1, std::min(options.evaluations,
                      static_cast<int>(options.procedural_budget_seconds /
                                       (estimate * options.samples))));
    }
    if (options.mode == Mode::Streaming && estimate > per_update_budget &&
        pairs > 0) {
      const double wanted_pairs =
          static_cast<double>(pairs) * per_update_budget / estimate;
      std::size_t accumulated = 0;
      leaf_limit = 0;
      while (leaf_limit < scene.leaves &&
             static_cast<double>(accumulated) < wanted_pairs) {
        const auto& work = scene.leaf_work[leaf_limit];
        for (const auto& range : work.source_ranges) {
          accumulated += static_cast<std::size_t>(work.target_count) *
                         static_cast<std::size_t>(range[1]);
        }
        ++leaf_limit;
      }
      leaf_limit = std::max<std::size_t>(leaf_limit, std::min<std::size_t>(scene.leaves, 8));
      accumulated = 0;
      for (std::size_t leaf = 0; leaf < leaf_limit; ++leaf) {
        const auto& work = scene.leaf_work[leaf];
        for (const auto& range : work.source_ranges) {
          accumulated += static_cast<std::size_t>(work.target_count) *
                         static_cast<std::size_t>(range[1]);
        }
      }
      fraction = static_cast<double>(accumulated) / static_cast<double>(pairs);
      std::cout << "# procedural estimate " << std::fixed << std::setprecision(1)
                << estimate << " s per update exceeds the budget; measuring "
                << leaf_limit << " of " << scene.leaves << " leaves ("
                << std::setprecision(4) << fraction << " of the pairs)"
                << std::defaultfloat << '\n';
    }
  }
  if (procedural_evaluations != options.evaluations) {
    std::cout << "# procedural measurement uses " << procedural_evaluations
              << " evaluations per sample\n";
  }
  const int procedural_final_update =
      options.warmups + options.samples * procedural_evaluations - 1;

  if (procedural_evaluations != options.evaluations) {
    // The final moments differ from the precomputed rows' final update.
    update_moments(base_moments, moments, procedural_final_update);
    std::fill(reference.begin(), reference.end(), Vec3{});
    cdfmm::apply_static_p2p_operator(canonical, moments, reference,
                                     scene.identities);
  }
  if (options.fp64 && options.cpu_procedural) {
    failures = 0;
    const double seconds = time_updates(
        options, prepare_fp64,
        [&] {
          apply_procedural_p2p<double>(scene, reconstruct, moments, fields,
                                       leaf_limit, failures);
        },
        procedural_evaluations);
    Row row = procedural_base;
    if (fraction < 1.0) {
      // Compare the measured leaves only.
      std::vector<Vec3> partial_reference = reference;
      const std::size_t covered_targets =
          static_cast<std::size_t>(scene.leaf_work[leaf_limit - 1].target_begin +
                                   scene.leaf_work[leaf_limit - 1].target_count);
      partial_reference.resize(covered_targets);
      std::vector<Vec3> partial_fields(fields.begin(),
                                       fields.begin() + static_cast<std::ptrdiff_t>(covered_targets));
      row.note = "truncated set; error over measured targets";
      row.max_relative_error =
          max_relative_error(partial_fields, partial_reference);
      row.precision = "fp64";
      row.backend = "cpu";
      row.representation = "procedural";
      row.representation_kind = "procedural";
      row.build_seconds = 0.0;
      row.update_seconds = seconds / fraction;
      row.ns_per_item = 1.0e9 * seconds / (fraction * static_cast<double>(applied_pairs));
      row.measured_fraction = fraction;
      row.checksum = checksum(partial_fields);
      if (failures > 0) {
        row.note += "; failures=" + std::to_string(failures.load());
      }
      sink.emit(row);
    } else {
      if (failures > 0) {
        row.note = "failures=" + std::to_string(failures.load());
      }
      emit_fp64("procedural", "procedural", 0.0, fields, seconds, 1.0, row);
    }
  }
  if (options.fp32 && options.cpu_procedural) {
    failures = 0;
    const double seconds = time_updates(
        options, prepare_fp32,
        [&] {
          apply_procedural_p2p<float>(scene, reconstruct, moments_float,
                                      fields_float, leaf_limit, failures);
        },
        procedural_evaluations);
    Row row = procedural_base;
    if (fraction < 1.0) {
      std::vector<Vec3> partial_reference = reference;
      const std::size_t covered_targets =
          static_cast<std::size_t>(scene.leaf_work[leaf_limit - 1].target_begin +
                                   scene.leaf_work[leaf_limit - 1].target_count);
      partial_reference.resize(covered_targets);
      std::vector<cdfmm::FloatVec3> partial_fields(
          fields_float.begin(),
          fields_float.begin() + static_cast<std::ptrdiff_t>(covered_targets));
      row.note = "truncated set; error over measured targets";
      row.max_relative_error =
          max_relative_error(partial_fields, partial_reference);
      row.precision = "fp32";
      row.backend = "cpu";
      row.representation = "procedural";
      row.representation_kind = "procedural";
      row.build_seconds = 0.0;
      row.update_seconds = seconds / fraction;
      row.ns_per_item = 1.0e9 * seconds / (fraction * static_cast<double>(applied_pairs));
      row.measured_fraction = fraction;
      row.checksum = checksum(partial_fields);
      if (failures > 0) {
        row.note += "; failures=" + std::to_string(failures.load());
      }
      sink.emit(row);
    } else {
      if (failures > 0) {
        row.note = "failures=" + std::to_string(failures.load());
      }
      emit_fp32("procedural", "procedural", 0.0, fields_float, seconds, 1.0,
                row);
    }
  }

  // Prism <-> point only: the same reconstruction through the
  // precision-generic kernel in double and float, which quantifies the cost
  // of the production `long double` evaluation and the accuracy a device
  // kernel would have in either precision.
  if (prism_point_pair && fraction >= 1.0 && options.cpu_procedural) {
    for (const auto math : {PairReconstruction::Math::Double,
                            PairReconstruction::Math::Float}) {
      const PairReconstruction variant{&scene.sources, &scene.targets, math};
      const std::string name = math == PairReconstruction::Math::Double
                                   ? "procedural-fp64-math"
                                   : "procedural-fp32-math";
      if (options.fp64) {
        failures = 0;
        const double seconds = time_updates(
            options, prepare_fp64,
            [&] {
              apply_procedural_p2p<double>(scene, variant, moments, fields,
                                           leaf_limit, failures);
            },
            procedural_evaluations);
        emit_fp64(name, "procedural", 0.0, fields, seconds, 1.0,
                  procedural_base);
      }
      if (options.fp32) {
        failures = 0;
        const double seconds = time_updates(
            options, prepare_fp32,
            [&] {
              apply_procedural_p2p<float>(scene, variant, moments_float,
                                          fields_float, leaf_limit, failures);
            },
            procedural_evaluations);
        emit_fp32(name, "procedural", 0.0, fields_float, seconds, 1.0,
                  procedural_base);
      }
    }
  }

#if defined(CDFMM_ENABLE_CUDA)
  if (options.cuda && cdfmm::cuda_available() && prism_point_pair) {
    cdfmm_bench::ProceduralPrismScene device_scene;
    device_scene.source_positions = scene.sources.positions;
    device_scene.target_positions = scene.targets.positions;
    device_scene.prism_is_source = scene.sources.geometry == Geometry::Prism;
    device_scene.skip_point_self_pair = point_source;
    device_scene.prisms = device_scene.prism_is_source ? scene.sources.prisms
                                                       : scene.targets.prisms;
    for (const auto& work : scene.leaf_work) {
      for (int t = 0; t < work.target_count; ++t) {
        for (const auto& range : work.source_ranges) {
          device_scene.items.push_back({work.target_begin + t, range[0], range[1]});
        }
      }
    }
    cdfmm_bench::ProceduralPrismCudaPlan plan(device_scene);
    Row cuda_base = procedural_base;
    cuda_base.backend = "cuda";
    cuda_base.threads = 1;
    cuda_base.topology_bytes = device_scene.items.size() *
                               sizeof(cdfmm_bench::ProceduralWorkItem);
    auto measure = [&](const std::string& name, const bool single,
                       const bool double_math) {
      double kernel_seconds = 0.0;
      if (single) {
        const double seconds = time_updates(
            options, prepare_fp32,
            [&] {
              kernel_seconds =
                  plan.evaluate(moments_float, fields_float, double_math);
            },
            options.evaluations);
        Row row = cuda_base;
        row.note = "kernel_seconds=" + std::to_string(kernel_seconds);
        emit_fp32(name, "procedural", 0.0, fields_float, seconds, 1.0, row);
      } else {
        const double seconds = time_updates(
            options, prepare_fp64,
            [&] { kernel_seconds = plan.evaluate(moments, fields); },
            options.evaluations);
        Row row = cuda_base;
        row.note = "kernel_seconds=" + std::to_string(kernel_seconds);
        emit_fp64(name, "procedural", 0.0, fields, seconds, 1.0, row);
      }
    };
    if (options.fp32) {
      measure("procedural-cuda-fp64-math", true, true);
      measure("procedural-cuda-fp32-math", true, false);
    }
    if (options.fp64) {
      measure("procedural-cuda", false, true);
    }
  }
#endif

#if defined(CDFMM_USE_OPENMP)
  omp_set_num_threads(saved_threads);
#endif
}

//------------------------------------------------------------------------------
// P2M / L2P scenes
//------------------------------------------------------------------------------

/** @brief Sorted bodies grouped into leaves with their expansion centres. */
struct ExpansionScene {
  BodySet bodies;
  struct Leaf {
    Vec3 centre{};
    std::size_t begin{0};
    std::size_t count{0};
  };
  std::vector<Leaf> leaves;
  double spacing{0.0};
};

[[nodiscard]] ExpansionScene make_expansion_scene(const Options& options,
                                                  const Geometry geometry) {
  ExpansionScene scene;
  const int boxes_per_axis = 1 << options.depth;
  const int cells_per_axis = boxes_per_axis * options.bodies_per_leaf_axis;
  const double spacing = 2.0 / cells_per_axis;
  scene.spacing = spacing;
  std::vector<Vec3> positions;
  std::mt19937 generator(161803u);
  std::uniform_real_distribution<double> jitter(-options.jitter,
                                                options.jitter);
  for (int iz = 0; iz < cells_per_axis; ++iz) {
    for (int iy = 0; iy < cells_per_axis; ++iy) {
      for (int ix = 0; ix < cells_per_axis; ++ix) {
        Vec3 position{-1.0 + (ix + 0.5) * spacing, -1.0 + (iy + 0.5) * spacing,
                      -1.0 + (iz + 0.5) * spacing};
        if (options.irregular) {
          position.x += jitter(generator) * spacing;
          position.y += jitter(generator) * spacing;
          position.z += jitter(generator) * spacing;
        }
        positions.push_back(position);
      }
    }
  }
  cdfmm::UniformTreeOptions tree_options;
  tree_options.max_level = options.depth;
  tree_options.root_centre = Vec3{};
  tree_options.root_half_width = 1.0;
  const cdfmm::UniformTree tree(positions, positions, tree_options);
  const auto sorted = tree.sorted_source_positions();
  scene.bodies.positions.assign(sorted.begin(), sorted.end());
  fill_records(scene.bodies, geometry, spacing, options.fill, options.irregular,
               scene.bodies.positions.size(), !options.irregular);
  const auto nodes = tree.nodes();
  for (const int leaf_index : tree.occupied_source_leaves()) {
    const cdfmm::TreeNode& leaf = nodes[static_cast<std::size_t>(leaf_index)];
    scene.leaves.push_back({leaf.centre, leaf.source_begin, leaf.source_count()});
  }
  if (options.mode == Mode::Hot) {
    // One leaf's worth of bodies (at least `pairs`), repeated.
    std::vector<ExpansionScene::Leaf> kept;
    std::size_t bodies = 0;
    for (const auto& leaf : scene.leaves) {
      kept.push_back(leaf);
      bodies += leaf.count;
      if (bodies >= static_cast<std::size_t>(options.pairs)) {
        break;
      }
    }
    scene.leaves = kept;
  }
  return scene;
}

// Total Cartesian monomial terms over all modes: the minimal work of any
// table-driven finite expansion evaluator is proportional to this count.
[[nodiscard]] std::size_t polynomial_term_count(
    const cdfmm::SphericalHarmonicBasis& basis) {
  std::size_t terms = 0;
  for (int mode = 0; mode < basis.size(); ++mode) {
    terms += basis.polynomial(mode).size();
  }
  return terms;
}

[[nodiscard]] std::size_t expansion_body_count(const ExpansionScene& scene) {
  std::size_t count = 0;
  for (const auto& leaf : scene.leaves) {
    count += leaf.count;
  }
  return count;
}

// Builder of one leaf's exact P2M operator in the production form.
[[nodiscard]] cdfmm::StaticCoefficientOperator build_leaf_p2m(
    const cdfmm::SphericalHarmonicBasis& basis, const ExpansionScene& scene,
    const ExpansionScene::Leaf& leaf) {
  const std::span<const Vec3> positions(scene.bodies.positions.data() + leaf.begin,
                                        leaf.count);
  if (scene.bodies.geometry == Geometry::Prism) {
    const std::span<const RectangularPrism> sizes =
        scene.bodies.common
            ? std::span<const RectangularPrism>(scene.bodies.prisms)
            : std::span<const RectangularPrism>(scene.bodies.prisms.data() + leaf.begin,
                                                leaf.count);
    return cdfmm::operators::p2m::build_cuboid(basis, leaf.centre, positions,
                                               sizes);
  }
  if (scene.bodies.geometry == Geometry::Tetrahedron) {
    const std::span<const Tetrahedron> tetrahedra =
        scene.bodies.common
            ? std::span<const Tetrahedron>(scene.bodies.tetrahedra)
            : std::span<const Tetrahedron>(scene.bodies.tetrahedra.data() + leaf.begin,
                                           leaf.count);
    return cdfmm::operators::p2m::build_tetrahedron(basis, leaf.centre,
                                                    positions, tetrahedra);
  }
  return cdfmm::operators::p2m::build(basis, leaf.centre, positions);
}

[[nodiscard]] cdfmm::StaticL2PEvaluator build_target_l2p(
    const cdfmm::SphericalHarmonicBasis& basis, const ExpansionScene& scene,
    const Vec3& centre, const std::size_t target) {
  const Vec3& position = scene.bodies.positions[target];
  if (scene.bodies.geometry == Geometry::Prism) {
    return cdfmm::operators::l2p::build_cuboid(basis, centre, position,
                                               scene.bodies.prism(target));
  }
  if (scene.bodies.geometry == Geometry::Tetrahedron) {
    return cdfmm::operators::l2p::build_tetrahedron(
        basis, centre, position, scene.bodies.tetrahedron(target));
  }
  return cdfmm::operators::l2p::build(basis, centre, position);
}

template <typename Scalar>
struct ExpansionBuffers {
  using Moment = std::conditional_t<std::is_same_v<Scalar, float>,
                                    cdfmm::FloatVec3, Vec3>;
  std::vector<Moment> base_moments;
  std::vector<Moment> moments;
  std::vector<Scalar> base_locals;   // leaves * C
  std::vector<Scalar> locals;
  std::vector<Scalar> multipoles;    // leaves * C
  std::vector<Moment> fields;
};

template <typename Scalar>
void update_locals(const std::vector<Scalar>& base, std::vector<Scalar>& out,
                   const int update) {
  const Scalar scale =
      static_cast<Scalar>(1.0 + 0.001 * static_cast<double>(update % 97));
  const std::size_t shift = static_cast<std::size_t>(update % 5);
  for (std::size_t index = 0; index < base.size(); ++index) {
    out[index] = scale * base[(index + shift) % base.size()];
  }
}

template <typename Scalar>
[[nodiscard]] double checksum_scalars(const std::vector<Scalar>& values) {
  long double sum = 0.0L;
  for (std::size_t index = 0; index < values.size(); ++index) {
    sum += static_cast<long double>(values[index]) *
           static_cast<long double>(1 + index % 7);
  }
  return static_cast<double>(sum);
}

template <typename Scalar>
[[nodiscard]] double max_relative_error_scalars(
    const std::vector<Scalar>& values, const std::vector<double>& reference) {
  double scale = 0.0;
  for (const double value : reference) {
    scale = std::max(scale, std::abs(value));
  }
  double error = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    error = std::max(error, std::abs(static_cast<double>(values[index]) -
                                     reference[index]));
  }
  return scale > 0.0 ? error / scale : error;
}

/**
 * @brief Evaluation count and leaf truncation of a procedural expansion
 * measurement within the time budget.
 *
 * The exact builder is the procedural reconstruction, so the serial build
 * time (divided by the threads of a streaming run) predicts one procedural
 * update.  Fewer evaluations come first; the leaf traversal is truncated
 * only when one update alone exceeds its share of the budget.
 */
struct ProceduralPlanning {
  int evaluations{1};
  std::size_t leaf_limit{0};
  double fraction{1.0};
};

[[nodiscard]] ProceduralPlanning plan_procedural(const Options& options,
                                                 const ExpansionScene& scene,
                                                 const double build_seconds) {
  ProceduralPlanning planning;
  planning.evaluations = options.evaluations;
  planning.leaf_limit = scene.leaves.size();
  const double threads =
      options.mode == Mode::Streaming ? static_cast<double>(thread_count()) : 1.0;
  const double estimate = build_seconds / threads;
  if (estimate <= 0.0) {
    return planning;
  }
  planning.evaluations = std::max(
      1, std::min(options.evaluations,
                  static_cast<int>(options.procedural_budget_seconds /
                                   (estimate * options.samples))));
  const double per_update_budget =
      options.procedural_budget_seconds /
      static_cast<double>(options.warmups + options.samples);
  if (estimate > per_update_budget) {
    const std::size_t total = expansion_body_count(scene);
    const double wanted = static_cast<double>(total) * per_update_budget / estimate;
    std::size_t accumulated = 0;
    std::size_t limit = 0;
    while (limit < scene.leaves.size() && static_cast<double>(accumulated) < wanted) {
      accumulated += scene.leaves[limit].count;
      ++limit;
    }
    limit = std::max<std::size_t>(limit, std::min<std::size_t>(scene.leaves.size(), 8));
    accumulated = 0;
    for (std::size_t leaf = 0; leaf < limit; ++leaf) {
      accumulated += scene.leaves[leaf].count;
    }
    planning.leaf_limit = limit;
    planning.fraction = static_cast<double>(accumulated) / static_cast<double>(total);
    std::cout << "# procedural estimate " << std::fixed << std::setprecision(1)
              << estimate << " s per update exceeds the budget; measuring "
              << limit << " of " << scene.leaves.size() << " leaves ("
              << std::setprecision(4) << planning.fraction << " of the bodies)"
              << std::defaultfloat << '\n';
  }
  if (planning.evaluations != options.evaluations) {
    std::cout << "# procedural measurement uses " << planning.evaluations
              << " evaluations per sample\n";
  }
  return planning;
}

template <typename Scalar>
void run_p2m_precision(const Options& options, RowSink& sink,
                       const ExpansionScene& scene,
                       const cdfmm::SphericalHarmonicBasis& basis,
                       const std::vector<cdfmm::P2MPlan>& plans,
                       const double build_seconds, const Row& base,
                       const std::vector<double>& reference_multipoles) {
  using Buffers = ExpansionBuffers<Scalar>;
  using Moment = typename Buffers::Moment;
  const int C = basis.size();
  const std::size_t bodies = scene.bodies.positions.size();
  const std::size_t leaves = scene.leaves.size();
  Buffers buffers;
  buffers.base_moments.resize(bodies);
  fill_random(buffers.base_moments, 813u);
  buffers.moments.resize(bodies);
  buffers.multipoles.assign(leaves * static_cast<std::size_t>(C), Scalar{0});
  const std::size_t items = expansion_body_count(scene);

  const auto prepare = [&](const int update) {
    update_moments(buffers.base_moments, buffers.moments, update);
  };
  const auto finish_row = [&](Row row, const std::string& name,
                              const std::string& kind, const double seconds,
                              const double representation_build_seconds) {
    row.precision = std::is_same_v<Scalar, float> ? "fp32" : "fp64";
    if (row.backend.empty()) {
      row.backend = "cpu";
    }
    row.representation = name;
    row.representation_kind = kind;
    row.build_seconds = representation_build_seconds;
    row.build_seconds_per_item =
        representation_build_seconds / static_cast<double>(std::max<std::size_t>(items, 1));
    row.update_seconds = seconds;
    row.ns_per_item = 1.0e9 * seconds / static_cast<double>(std::max<std::size_t>(items, 1));
    row.checksum = checksum_scalars(buffers.multipoles);
    row.max_relative_error =
        max_relative_error_scalars(buffers.multipoles, reference_multipoles);
    sink.emit(row);
  };

  // Precomputed: the production dense packing and kernel.
  {
    auto start = Clock::now();
    cdfmm::detail::cpu::PackedP2M<Scalar> packing;
    double quantise_seconds = 0.0;
    if constexpr (std::is_same_v<Scalar, float>) {
      std::vector<cdfmm::FloatP2MPlan> float_plans;
      float_plans.reserve(plans.size());
      for (const auto& plan : plans) {
        float_plans.push_back({plan.leaf, plan.begin, plan.count,
                               cdfmm::quantise_static_operator(plan.operator_map)});
      }
      quantise_seconds = elapsed(start);
      start = Clock::now();
      packing = cdfmm::detail::cpu::pack_p2m(float_plans, bodies, C);
    } else {
      packing = cdfmm::detail::cpu::pack_p2m(plans, bodies, C);
    }
    const double pack_seconds = elapsed(start);
    const int leaf_count = static_cast<int>(leaves);
    const bool parallel = options.mode == Mode::Streaming;
    const double seconds = time_updates(
        options, prepare,
        [&] {
          std::fill(buffers.multipoles.begin(), buffers.multipoles.end(),
                    Scalar{0});
#pragma omp parallel for schedule(static) if (parallel && leaf_count >= 8)
          for (int leaf = 0; leaf < leaf_count; ++leaf) {
            const auto& range = scene.leaves[static_cast<std::size_t>(leaf)];
            cdfmm::detail::cpu::apply_packed_p2m<Scalar, Moment>(
                packing, range.begin,
                std::span<const Moment>(buffers.moments.data() + range.begin,
                                        range.count),
                buffers.multipoles.data() + static_cast<std::size_t>(leaf) * C);
          }
        },
        options.evaluations);
    Row row = base;
    // The packing stores every body of the scene; the hot set uses a slice.
    row.operator_bytes = items * 3 * static_cast<std::size_t>(C) * sizeof(Scalar);
    finish_row(row, "packed-rows", "precomputed", seconds,
               build_seconds + quantise_seconds + pack_seconds);
  }

  // Procedural: the exact builder every update, applied entry by entry.
  {
    const ProceduralPlanning planning =
        plan_procedural(options, scene, build_seconds);
    const int leaf_count = static_cast<int>(planning.leaf_limit);
    const bool parallel = options.mode == Mode::Streaming;
    std::atomic<std::size_t> scratch_entries{0};
    // The reference multipoles belong to the final update of the planned
    // evaluation count.
    std::vector<double> planned_reference = reference_multipoles;
    if (planning.evaluations != options.evaluations) {
      std::vector<Vec3> moments_fp64(bodies);
      std::vector<Vec3> base_fp64(bodies);
      fill_random(base_fp64, 813u);
      update_moments(base_fp64, moments_fp64,
                     options.warmups + options.samples * planning.evaluations - 1);
      std::fill(planned_reference.begin(), planned_reference.end(), 0.0);
      for (std::size_t leaf = 0; leaf < plans.size(); ++leaf) {
        const auto& plan = plans[leaf];
        for (const auto& entry : plan.operator_map.entries) {
          const std::size_t source =
              plan.begin + static_cast<std::size_t>(entry.input / 3);
          planned_reference[leaf * static_cast<std::size_t>(C) +
                            static_cast<std::size_t>(entry.output)] +=
              entry.value * moments_fp64[source][entry.input % 3];
        }
      }
    }
    const double seconds = time_updates(
        options, prepare,
        [&] {
          std::fill(buffers.multipoles.begin(), buffers.multipoles.end(),
                    Scalar{0});
          std::size_t entries_seen = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : entries_seen) if (parallel && leaf_count >= 8)
          for (int leaf = 0; leaf < leaf_count; ++leaf) {
            const auto& range = scene.leaves[static_cast<std::size_t>(leaf)];
            const cdfmm::StaticCoefficientOperator op =
                build_leaf_p2m(basis, scene, range);
            Scalar* M = buffers.multipoles.data() + static_cast<std::size_t>(leaf) * C;
            const Moment* m = buffers.moments.data() + range.begin;
            for (const auto& entry : op.entries) {
              const int source = entry.input / 3;
              const int component = entry.input % 3;
              const Scalar value = static_cast<Scalar>(entry.value);
              const Scalar moment = component == 0 ? m[source].x
                                    : component == 1 ? m[source].y
                                                     : m[source].z;
              M[entry.output] += value * moment;
            }
            entries_seen += op.entries.size();
          }
          scratch_entries = entries_seen;
        },
        planning.evaluations);
    Row row = base;
    row.operator_bytes = 0;
    row.scratch_bytes =
        scratch_entries.load() * sizeof(cdfmm::StaticOperatorEntry) /
        std::max<std::size_t>(planning.leaf_limit, 1);  // one leaf's entries
    row.measured_fraction = planning.fraction;
    if (planning.fraction < 1.0) {
      row.note = "truncated set; error over measured leaves";
    }
    // Compare the measured leaves only.
    const std::size_t covered = planning.leaf_limit * static_cast<std::size_t>(C);
    std::vector<Scalar> measured(buffers.multipoles.begin(),
                                 buffers.multipoles.begin() + static_cast<std::ptrdiff_t>(covered));
    planned_reference.resize(covered);
    row.precision = std::is_same_v<Scalar, float> ? "fp32" : "fp64";
    row.backend = "cpu";
    row.representation = "procedural-builder";
    row.representation_kind = "procedural";
    row.build_seconds = 0.0;
    row.build_seconds_per_item = 0.0;
    row.update_seconds = seconds / planning.fraction;
    row.ns_per_item = 1.0e9 * seconds /
                      (planning.fraction * static_cast<double>(std::max<std::size_t>(items, 1)));
    row.checksum = checksum_scalars(measured);
    row.max_relative_error = max_relative_error_scalars(measured, planned_reference);
    sink.emit(row);
  }
}

void run_p2m(const Options& options, RowSink& sink) {
  const ExpansionScene scene = make_expansion_scene(options, options.source);
  const std::size_t bodies = scene.bodies.positions.size();
  const std::size_t items = expansion_body_count(scene);
  std::cout << "# P2M " << geometry_name(scene.bodies.geometry) << " source, "
            << (options.irregular ? "irregular" : "regular") << ", "
            << (options.mode == Mode::Hot ? "hot" : "streaming") << ", bodies "
            << items << " in " << scene.leaves.size() << " leaves, threads "
            << (options.mode == Mode::Hot ? 1 : thread_count()) << '\n';
#if defined(CDFMM_USE_OPENMP)
  const int saved_threads = omp_get_max_threads();
  if (options.mode == Mode::Hot) {
    omp_set_num_threads(1);
  }
#endif
  for (const int order : options.orders) {
    const cdfmm::SphericalHarmonicBasis basis(order);
    const int C = basis.size();
    // Construction: every leaf's exact operator, serially like the plan
    // preparation, in P2MPlan form.
    const auto start = Clock::now();
    std::vector<cdfmm::P2MPlan> plans;
    plans.reserve(scene.leaves.size());
    for (std::size_t leaf = 0; leaf < scene.leaves.size(); ++leaf) {
      const auto& range = scene.leaves[leaf];
      plans.push_back({static_cast<int>(leaf), range.begin, range.count,
                       build_leaf_p2m(basis, scene, range)});
    }
    const double build_seconds = elapsed(start);
    std::cout << "# p=" << order << " C=" << C << " polynomial terms "
              << polynomial_term_count(basis) << " build " << std::fixed
              << std::setprecision(3) << build_seconds << " s ("
              << std::setprecision(2) << 1.0e6 * build_seconds / items
              << " us/source)" << std::defaultfloat << '\n';

    Row base;
    base.stage = "p2m";
    base.source_geometry = geometry_name(scene.bodies.geometry);
    base.target_geometry = "-";
    base.layout = options.irregular ? "irregular" : "regular";
    base.separation = "lattice";
    base.mode = options.mode == Mode::Hot ? "hot" : "streaming";
    base.order = order;
    base.bodies = items;
    base.leaves = scene.leaves.size();
    base.items = items;
    base.threads = options.mode == Mode::Hot ? 1 : thread_count();
    base.geometry_bytes = items * sizeof(Vec3) +
                          (scene.bodies.common ? 1 : items) *
                              (scene.bodies.geometry == Geometry::Prism
                                   ? sizeof(RectangularPrism)
                                   : scene.bodies.geometry == Geometry::Tetrahedron
                                         ? sizeof(Tetrahedron)
                                         : 0);
    base.topology_bytes = scene.leaves.size() * sizeof(ExpansionScene::Leaf);

    // FP64 reference multipoles from the canonical sparse operators on the
    // final moments.
    const int final_update =
        options.warmups + options.samples * options.evaluations - 1;
    std::vector<Vec3> base_moments(bodies);
    fill_random(base_moments, 813u);
    std::vector<Vec3> moments(bodies);
    update_moments(base_moments, moments, final_update);
    std::vector<double> reference(scene.leaves.size() * static_cast<std::size_t>(C), 0.0);
    for (std::size_t leaf = 0; leaf < plans.size(); ++leaf) {
      const auto& plan = plans[leaf];
      for (const auto& entry : plan.operator_map.entries) {
        const std::size_t source = plan.begin + static_cast<std::size_t>(entry.input / 3);
        const int component = entry.input % 3;
        reference[leaf * static_cast<std::size_t>(C) + static_cast<std::size_t>(entry.output)] +=
            entry.value * moments[source][component];
      }
    }
    if (options.fp64) {
      run_p2m_precision<double>(options, sink, scene, basis, plans,
                                build_seconds, base, reference);
    }
    if (options.fp32) {
      run_p2m_precision<float>(options, sink, scene, basis, plans,
                               build_seconds, base, reference);
    }
  }
#if defined(CDFMM_USE_OPENMP)
  omp_set_num_threads(saved_threads);
#endif
}

template <typename Scalar>
void run_l2p_precision(const Options& options, RowSink& sink,
                       const ExpansionScene& scene,
                       const cdfmm::SphericalHarmonicBasis& basis,
                       const std::vector<cdfmm::StaticL2PEvaluator>& evaluators,
                       const double build_seconds, const Row& base,
                       const std::vector<Vec3>& reference_fields) {
  using Buffers = ExpansionBuffers<Scalar>;
  using Moment = typename Buffers::Moment;
  const int C = basis.size();
  const std::size_t bodies = scene.bodies.positions.size();
  const std::size_t leaves = scene.leaves.size();
  const std::size_t items = expansion_body_count(scene);
  Buffers buffers;
  buffers.base_locals.resize(leaves * static_cast<std::size_t>(C));
  {
    std::mt19937 generator(577215u);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    for (Scalar& local : buffers.base_locals) {
      local = static_cast<Scalar>(value(generator));
    }
  }
  buffers.locals.resize(buffers.base_locals.size());
  buffers.fields.assign(bodies, Moment{});
  const auto prepare = [&](const int update) {
    update_locals(buffers.base_locals, buffers.locals, update);
  };
  const auto finish_row = [&](Row row, const std::string& name,
                              const std::string& kind, const double seconds,
                              const double representation_build_seconds) {
    row.precision = std::is_same_v<Scalar, float> ? "fp32" : "fp64";
    if (row.backend.empty()) {
      row.backend = "cpu";
    }
    row.representation = name;
    row.representation_kind = kind;
    row.build_seconds = representation_build_seconds;
    row.build_seconds_per_item =
        representation_build_seconds / static_cast<double>(std::max<std::size_t>(items, 1));
    row.update_seconds = seconds;
    row.ns_per_item = 1.0e9 * seconds / static_cast<double>(std::max<std::size_t>(items, 1));
    row.checksum = checksum(buffers.fields);
    row.max_relative_error = max_relative_error(buffers.fields, reference_fields);
    sink.emit(row);
  };

  {
    auto start = Clock::now();
    cdfmm::detail::cpu::PackedL2P<Scalar> packing;
    double quantise_seconds = 0.0;
    if constexpr (std::is_same_v<Scalar, float>) {
      std::vector<cdfmm::FloatStaticL2PEvaluator> float_evaluators;
      float_evaluators.reserve(evaluators.size());
      for (const auto& evaluator : evaluators) {
        float_evaluators.push_back(cdfmm::quantise_static_l2p_evaluator(evaluator));
      }
      quantise_seconds = elapsed(start);
      start = Clock::now();
      packing = cdfmm::detail::cpu::pack_l2p(float_evaluators, C);
    } else {
      packing = cdfmm::detail::cpu::pack_l2p(evaluators, C);
    }
    const double pack_seconds = elapsed(start);
    const int leaf_count = static_cast<int>(leaves);
    const bool parallel = options.mode == Mode::Streaming;
    const double seconds = time_updates(
        options, prepare,
        [&] {
#pragma omp parallel for schedule(static) if (parallel && leaf_count >= 8)
          for (int leaf = 0; leaf < leaf_count; ++leaf) {
            const auto& range = scene.leaves[static_cast<std::size_t>(leaf)];
            const Scalar* L = buffers.locals.data() + static_cast<std::size_t>(leaf) * C;
            for (std::size_t target = range.begin; target < range.begin + range.count;
                 ++target) {
              Scalar Hx{};
              Scalar Hy{};
              Scalar Hz{};
              cdfmm::detail::cpu::apply_packed_l2p_field(packing, target, L, Hx,
                                                         Hy, Hz);
              buffers.fields[target] = {Hx, Hy, Hz};
            }
          }
        },
        options.evaluations);
    Row row = base;
    row.operator_bytes = items * 4 * static_cast<std::size_t>(C) * sizeof(Scalar);
    finish_row(row, "packed-rows", "precomputed", seconds,
               build_seconds + quantise_seconds + pack_seconds);
  }

  {
    const ProceduralPlanning planning =
        plan_procedural(options, scene, build_seconds);
    const int leaf_count = static_cast<int>(planning.leaf_limit);
    const bool parallel = options.mode == Mode::Streaming;
    // The reference fields belong to the final update of the planned
    // evaluation count.
    std::vector<Vec3> planned_reference = reference_fields;
    if (planning.evaluations != options.evaluations) {
      std::vector<double> base_fp64(buffers.base_locals.size());
      {
        std::mt19937 generator(577215u);
        std::uniform_real_distribution<double> value(-1.0, 1.0);
        for (double& local : base_fp64) {
          local = value(generator);
        }
      }
      std::vector<double> locals_fp64(base_fp64.size());
      update_locals(base_fp64, locals_fp64,
                    options.warmups + options.samples * planning.evaluations - 1);
      for (std::size_t leaf = 0; leaf < scene.leaves.size(); ++leaf) {
        const auto& range = scene.leaves[leaf];
        const double* L = locals_fp64.data() + leaf * static_cast<std::size_t>(C);
        for (std::size_t target = range.begin; target < range.begin + range.count;
             ++target) {
          Vec3 H{};
          for (int index = 0; index < C; ++index) {
            H.x += evaluators[target].field[0][static_cast<std::size_t>(index)] * L[index];
            H.y += evaluators[target].field[1][static_cast<std::size_t>(index)] * L[index];
            H.z += evaluators[target].field[2][static_cast<std::size_t>(index)] * L[index];
          }
          planned_reference[target] = H;
        }
      }
    }
    const double seconds = time_updates(
        options, prepare,
        [&] {
#pragma omp parallel for schedule(dynamic, 1) if (parallel && leaf_count >= 8)
          for (int leaf = 0; leaf < leaf_count; ++leaf) {
            const auto& range = scene.leaves[static_cast<std::size_t>(leaf)];
            const Scalar* L = buffers.locals.data() + static_cast<std::size_t>(leaf) * C;
            for (std::size_t target = range.begin; target < range.begin + range.count;
                 ++target) {
              const cdfmm::StaticL2PEvaluator evaluator =
                  build_target_l2p(basis, scene, range.centre, target);
              Scalar Hx{};
              Scalar Hy{};
              Scalar Hz{};
              for (int index = 0; index < C; ++index) {
                const Scalar l = L[index];
                Hx += static_cast<Scalar>(evaluator.field[0][static_cast<std::size_t>(index)]) * l;
                Hy += static_cast<Scalar>(evaluator.field[1][static_cast<std::size_t>(index)]) * l;
                Hz += static_cast<Scalar>(evaluator.field[2][static_cast<std::size_t>(index)]) * l;
              }
              buffers.fields[target] = {Hx, Hy, Hz};
            }
          }
        },
        planning.evaluations);
    Row row = base;
    row.operator_bytes = 0;
    row.scratch_bytes = 4 * static_cast<std::size_t>(C) * sizeof(double);
    row.measured_fraction = planning.fraction;
    if (planning.fraction < 1.0) {
      row.note = "truncated set; error over measured targets";
    }
    const auto& last = scene.leaves[planning.leaf_limit - 1];
    const std::size_t covered = last.begin + last.count;
    std::vector<Moment> measured(buffers.fields.begin(),
                                 buffers.fields.begin() + static_cast<std::ptrdiff_t>(covered));
    planned_reference.resize(covered);
    row.precision = std::is_same_v<Scalar, float> ? "fp32" : "fp64";
    row.backend = "cpu";
    row.representation = "procedural-builder";
    row.representation_kind = "procedural";
    row.build_seconds = 0.0;
    row.build_seconds_per_item = 0.0;
    row.update_seconds = seconds / planning.fraction;
    row.ns_per_item = 1.0e9 * seconds /
                      (planning.fraction * static_cast<double>(std::max<std::size_t>(items, 1)));
    row.checksum = checksum(measured);
    row.max_relative_error = max_relative_error(measured, planned_reference);
    sink.emit(row);
  }
}

void run_l2p(const Options& options, RowSink& sink) {
  const ExpansionScene scene = make_expansion_scene(options, options.target);
  const std::size_t bodies = scene.bodies.positions.size();
  const std::size_t items = expansion_body_count(scene);
  std::cout << "# L2P " << geometry_name(scene.bodies.geometry) << " target, "
            << (options.irregular ? "irregular" : "regular") << ", "
            << (options.mode == Mode::Hot ? "hot" : "streaming") << ", bodies "
            << items << " in " << scene.leaves.size() << " leaves, threads "
            << (options.mode == Mode::Hot ? 1 : thread_count()) << '\n';
#if defined(CDFMM_USE_OPENMP)
  const int saved_threads = omp_get_max_threads();
  if (options.mode == Mode::Hot) {
    omp_set_num_threads(1);
  }
#endif
  for (const int order : options.orders) {
    const cdfmm::SphericalHarmonicBasis basis(order);
    const int C = basis.size();
    const auto start = Clock::now();
    std::vector<cdfmm::StaticL2PEvaluator> evaluators(bodies);
    for (const auto& range : scene.leaves) {
      for (std::size_t target = range.begin; target < range.begin + range.count;
           ++target) {
        evaluators[target] = build_target_l2p(basis, scene, range.centre, target);
      }
    }
    // Targets outside the hot slice keep empty rows of the right size so the
    // production packing accepts the vector.
    for (auto& evaluator : evaluators) {
      if (evaluator.potential.empty()) {
        evaluator.potential.assign(static_cast<std::size_t>(C), 0.0);
        for (auto& row : evaluator.field) {
          row.assign(static_cast<std::size_t>(C), 0.0);
        }
      }
    }
    const double build_seconds = elapsed(start);
    std::cout << "# p=" << order << " C=" << C << " polynomial terms "
              << polynomial_term_count(basis) << " build " << std::fixed
              << std::setprecision(3) << build_seconds << " s ("
              << std::setprecision(2) << 1.0e6 * build_seconds / items
              << " us/target)" << std::defaultfloat << '\n';

    Row base;
    base.stage = "l2p";
    base.source_geometry = "-";
    base.target_geometry = geometry_name(scene.bodies.geometry);
    base.layout = options.irregular ? "irregular" : "regular";
    base.separation = "lattice";
    base.mode = options.mode == Mode::Hot ? "hot" : "streaming";
    base.order = order;
    base.bodies = items;
    base.leaves = scene.leaves.size();
    base.items = items;
    base.threads = options.mode == Mode::Hot ? 1 : thread_count();
    base.geometry_bytes = items * sizeof(Vec3) +
                          (scene.bodies.common ? 1 : items) *
                              (scene.bodies.geometry == Geometry::Prism
                                   ? sizeof(RectangularPrism)
                                   : scene.bodies.geometry == Geometry::Tetrahedron
                                         ? sizeof(Tetrahedron)
                                         : 0);
    base.topology_bytes = scene.leaves.size() * sizeof(ExpansionScene::Leaf);

    // FP64 reference fields from the canonical evaluators on the final locals.
    const int final_update =
        options.warmups + options.samples * options.evaluations - 1;
    std::vector<double> base_locals(scene.leaves.size() * static_cast<std::size_t>(C));
    {
      std::mt19937 generator(577215u);
      std::uniform_real_distribution<double> value(-1.0, 1.0);
      for (double& local : base_locals) {
        local = value(generator);
      }
    }
    std::vector<double> locals(base_locals.size());
    update_locals(base_locals, locals, final_update);
    std::vector<Vec3> reference(bodies);
    for (std::size_t leaf = 0; leaf < scene.leaves.size(); ++leaf) {
      const auto& range = scene.leaves[leaf];
      const double* L = locals.data() + leaf * static_cast<std::size_t>(C);
      for (std::size_t target = range.begin; target < range.begin + range.count;
           ++target) {
        Vec3 H{};
        for (int index = 0; index < C; ++index) {
          H.x += evaluators[target].field[0][static_cast<std::size_t>(index)] * L[index];
          H.y += evaluators[target].field[1][static_cast<std::size_t>(index)] * L[index];
          H.z += evaluators[target].field[2][static_cast<std::size_t>(index)] * L[index];
        }
        reference[target] = H;
      }
    }
    if (options.fp64) {
      run_l2p_precision<double>(options, sink, scene, basis, evaluators,
                                build_seconds, base, reference);
    }
    if (options.fp32) {
      run_l2p_precision<float>(options, sink, scene, basis, evaluators,
                               build_seconds, base, reference);
    }
  }
#if defined(CDFMM_USE_OPENMP)
  omp_set_num_threads(saved_threads);
#endif
}

} // namespace

int main(const int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    RowSink sink(options);
    std::cout << "# benchmark_operator_representation: " << compiler_name()
              << ' ' << compiler_version() << ", threads " << thread_count()
              << ", cpu " << cpu_model();
    const std::string gpu = gpu_model();
    if (!gpu.empty()) {
      std::cout << ", gpu " << gpu;
    }
    std::cout << '\n';
    switch (options.stage) {
    case Stage::P2P:
      run_p2p(options, sink);
      break;
    case Stage::P2M:
      run_p2m(options, sink);
      break;
    case Stage::L2P:
      run_l2p(options, sink);
      break;
    }
  } catch (const std::exception& error) {
    std::cerr << "benchmark_operator_representation: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
