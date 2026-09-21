// SPDX-License-Identifier: Apache-2.0
//
// UniformFmm construction: input validation, choice of the physical root box,
// normalisation of every coordinate and finite-body record into the unit root
// [-1/2, 1/2]^3 on the canonical 1e-9 grid, and the constructor chain that
// builds the trees and topology before handing over to `initialise_execution`
// (src/fmm/execution_setup.cpp).
//
// Normalisation is what makes plans reusable: two geometries that differ only
// by a physical translation or a uniform scale produce bitwise-identical
// normalised coordinates, hence identical operators and identical cache keys.
// Everything downstream of this unit works in normalised units; the physical
// root centre and side length are retained only to scale results, keys and
// the inspection API back.

#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/tree/uniform_topology.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmm/internal.hpp"
#include "phase_stopwatch.hpp"

namespace cdfmm {

namespace {

double canonicalise_normalised_value(const double value) {
  // Keep supplied-topology cuboid sizes on the same canonical grid as the
  // ordinary physical-geometry constructor.
  constexpr double resolution = 1.0e9;
  return std::nearbyint(value * resolution) / resolution;
}

// A prebuilt topology (for example from AdaptiveTree) already holds normalised
// sorted positions, but the caller's finite-body records are still physical.
// Scale them by the topology's coordinate scale, snap them to the canonical
// grid, and verify that every body lies inside the normalised root.  Records
// are indexed in user order, so `permutation[sorted]` maps a sorted position
// back to its record.
UniformFmmOptions normalise_supplied_topology_options(
    const StaticFmmTopology& topology, const UniformFmmOptions& options) {
  UniformFmmOptions normalised = options;
  const auto& root = topology.nodes[static_cast<std::size_t>(topology.root)];
  const double inverse_scale = 1.0 / topology.coordinate_scale;
  const auto normalise_tetrahedra =
      [&](std::vector<Tetrahedron>& tetrahedra,
          const std::vector<Vec3>& positions,
          const std::vector<int>& permutation, const char* description) {
    if (tetrahedra.size() != 1 && tetrahedra.size() != positions.size()) {
      throw std::invalid_argument(
          std::string(description) +
          " geometries must contain one or one per object");
    }
    for (Tetrahedron& tetrahedron : tetrahedra) {
      static_cast<void>(tetrahedron_volume(tetrahedron));
      for (Vec3& vertex : tetrahedron.vertices) {
        vertex = vertex * inverse_scale;
        vertex = {canonicalise_normalised_value(vertex.x),
                  canonicalise_normalised_value(vertex.y),
                  canonicalise_normalised_value(vertex.z)};
      }
    }
    constexpr double tolerance = 2.0e-9;
    for (std::size_t sorted = 0; sorted < positions.size(); ++sorted) {
      const std::size_t geometry_index = tetrahedra.size() == 1
          ? 0
          : static_cast<std::size_t>(permutation[sorted]);
      const Tetrahedron& tetrahedron = tetrahedra[geometry_index];
      const Vec3 distance = positions[sorted] - root.centre;
      for (const Vec3& vertex : tetrahedron.vertices) {
        if (std::abs(distance.x + vertex.x) > root.half_width + tolerance ||
            std::abs(distance.y + vertex.y) > root.half_width + tolerance ||
            std::abs(distance.z + vertex.z) > root.half_width + tolerance) {
          throw std::invalid_argument(
              std::string(description) +
              " geometry lies outside the supplied topology root");
        }
      }
    }
  };
  const auto normalise_sizes = [&](std::vector<CuboidSize>& sizes,
                                   const std::vector<Vec3>& positions,
                                   const std::vector<int>& permutation,
                                   const char* description) {
    if (sizes.size() != 1 && sizes.size() != positions.size()) {
      throw std::invalid_argument(std::string(description) +
                                  " sizes must contain one or one per object");
    }
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      const CuboidSize physical = sizes[index];
      if (!std::isfinite(physical.hx) || !std::isfinite(physical.hy) ||
          !std::isfinite(physical.hz) || physical.hx <= 0.0 ||
          physical.hy <= 0.0 || physical.hz <= 0.0) {
        throw std::invalid_argument(std::string(description) +
                                    " dimensions must be finite and positive");
      }
      sizes[index] = {
          canonicalise_normalised_value(physical.hx * inverse_scale),
          canonicalise_normalised_value(physical.hy * inverse_scale),
          canonicalise_normalised_value(physical.hz * inverse_scale)};
    }

    // The stored coordinates and normalised sizes use a 1e-9 canonical grid.
    // Permit its worst-case independent rounding at a root face.
    constexpr double tolerance = 2.0e-9;
    for (std::size_t sorted = 0; sorted < positions.size(); ++sorted) {
      const std::size_t size_index = sizes.size() == 1
          ? 0
          : static_cast<std::size_t>(permutation[sorted]);
      const CuboidSize size = sizes[size_index];
      const Vec3 extent{0.5 * size.hx, 0.5 * size.hy, 0.5 * size.hz};
      const Vec3 distance = positions[sorted] - root.centre;
      if (std::abs(distance.x) + extent.x > root.half_width + tolerance ||
          std::abs(distance.y) + extent.y > root.half_width + tolerance ||
          std::abs(distance.z) + extent.z > root.half_width + tolerance) {
        throw std::invalid_argument(std::string(description) +
                                    " geometry lies outside the supplied "
                                    "topology root");
      }
    }
  };

  if (normalised.source_geometry == SourceGeometry::RectangularPrism) {
    normalise_sizes(normalised.source_sizes,
                    topology.sorted_source_positions,
                    topology.source_permutation, "source cuboid");
  }
  if (normalised.target_geometry == TargetGeometry::RectangularPrism) {
    normalise_sizes(normalised.target_sizes,
                    topology.sorted_target_positions,
                    topology.target_permutation, "target cuboid");
  }
  if (normalised.source_geometry == SourceGeometry::Tetrahedron) {
    normalise_tetrahedra(normalised.source_tetrahedra,
                         topology.sorted_source_positions,
                         topology.source_permutation, "source tetrahedron");
  }
  if (normalised.target_geometry == TargetGeometry::Tetrahedron) {
    normalise_tetrahedra(normalised.target_tetrahedra,
                         topology.sorted_target_positions,
                         topology.target_permutation, "target tetrahedron");
  }
  return normalised;
}

} // namespace

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------

// Everything the delegating constructor needs, computed once by
// `normalise_geometry`: the wrapped physical positions (for the physical
// inspection tree), the normalised positions (for the plan tree), the options
// with every record rewritten into normalised units, and the physical root so
// results can be mapped back.
struct UniformFmm::NormalisedGeometry {
  std::vector<Vec3> physical_source_positions{};
  std::vector<Vec3> physical_target_positions{};
  UniformTreeOptions physical_tree_options{};
  std::vector<Vec3> source_positions{};
  std::vector<Vec3> target_positions{};
  UniformFmmOptions options{};
  Vec3 physical_root_centre{};
  double physical_root_side_length{1.0};
  double normalisation_seconds{0.0};
  // The whole-constructor clock (TimingLevel::Coarse) starts before
  // normalisation, which is why the geometry carries it.
  detail::PhaseStopwatch construction_clock{false};
};

// Steps, in order: validate positions and records; bound the geometry
// including finite extents; choose the physical root (the periodic cell, the
// caller's explicit root, or the tight bounding cube); wrap periodic positions
// into the cell and check finite bodies stay inside it; scale by the inverse
// side length and snap to the canonical grid; rewrite the options so the rest
// of the solver sees a unit root at the origin.
UniformFmm::NormalisedGeometry UniformFmm::normalise_geometry(
    const std::vector<Vec3>& source_positions,
    const std::vector<Vec3>& target_positions,
    const UniformFmmOptions& options) {
  // Two clocks start here: the `Coarse` whole-constructor clock, which the
  // returned geometry carries into the constructor body, and the `Detailed`
  // normalisation clock read at the end of this function.
  detail::PhaseStopwatch construction_clock(
      options.timing_level != TimingLevel::Off);
  construction_clock.start();
  detail::PhaseStopwatch normalisation_clock(
      options.timing_level == TimingLevel::Detailed);
  normalisation_clock.start();
  validate_periodic_cell(options.periodic);

  const auto finite_position = [](const Vec3& position) {
    return std::isfinite(position.x) && std::isfinite(position.y) &&
        std::isfinite(position.z);
  };
  const auto validate_sizes = [](const std::vector<CuboidSize>& sizes,
                                 const std::size_t count,
                                 const char* description) {
    if (sizes.size() != 1 && sizes.size() != count) {
      throw std::invalid_argument(std::string(description) +
                                  " sizes must contain one or one per object");
    }
    for (const CuboidSize& size : sizes) {
      if (!std::isfinite(size.hx) || !std::isfinite(size.hy) ||
          !std::isfinite(size.hz) || size.hx <= 0.0 || size.hy <= 0.0 ||
          size.hz <= 0.0) {
        throw std::invalid_argument(std::string(description) +
                                    " dimensions must be finite and positive");
      }
    }
  };
  const auto validate_tetrahedra =
      [](const std::vector<Tetrahedron>& tetrahedra,
         const std::size_t count, const char* description) {
    if (tetrahedra.size() != 1 && tetrahedra.size() != count) {
      throw std::invalid_argument(
          std::string(description) +
          " geometries must contain one or one per object");
    }
    for (const Tetrahedron& tetrahedron : tetrahedra) {
      static_cast<void>(tetrahedron_volume(tetrahedron));
    }
  };

  for (const Vec3& position : source_positions) {
    if (!finite_position(position)) {
      throw std::invalid_argument("source positions must be finite");
    }
  }
  for (const Vec3& position : target_positions) {
    if (!finite_position(position)) {
      throw std::invalid_argument("target positions must be finite");
    }
  }
  if (options.source_geometry == SourceGeometry::RectangularPrism) {
    validate_sizes(options.source_sizes, source_positions.size(),
                   "source cuboid");
  }
  if (options.target_geometry == TargetGeometry::RectangularPrism) {
    validate_sizes(options.target_sizes, target_positions.size(),
                   "target cuboid");
  }
  if (options.source_geometry == SourceGeometry::Tetrahedron) {
    validate_tetrahedra(options.source_tetrahedra, source_positions.size(),
                        "source tetrahedron");
  }
  if (options.target_geometry == TargetGeometry::Tetrahedron) {
    validate_tetrahedra(options.target_tetrahedra, target_positions.size(),
                        "target tetrahedron");
  }

  // Bounding box of the complete geometry.  Finite bodies contribute their
  // full extent, not just their representative point, so the root box always
  // contains every body the exact near field will integrate over.
  Vec3 minimum{std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity()};
  Vec3 maximum{-std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity()};
  bool has_geometry = false;
  const auto include_population = [&](const std::vector<Vec3>& positions,
                                      const std::vector<CuboidSize>& sizes,
                                      const bool cuboids) {
    for (std::size_t index = 0; index < positions.size(); ++index) {
      const CuboidSize size = cuboids
          ? sizes[sizes.size() == 1 ? 0 : index]
          : CuboidSize{};
      const Vec3 half_extent{0.5 * size.hx, 0.5 * size.hy,
                             0.5 * size.hz};
      const Vec3 lower = positions[index] - half_extent;
      const Vec3 upper = positions[index] + half_extent;
      minimum.x = std::min(minimum.x, lower.x);
      minimum.y = std::min(minimum.y, lower.y);
      minimum.z = std::min(minimum.z, lower.z);
      maximum.x = std::max(maximum.x, upper.x);
      maximum.y = std::max(maximum.y, upper.y);
      maximum.z = std::max(maximum.z, upper.z);
      has_geometry = true;
    }
  };
  const auto include_tetrahedra = [&](const std::vector<Vec3>& positions,
                                      const std::vector<Tetrahedron>& tetrahedra) {
    for (std::size_t index = 0; index < positions.size(); ++index) {
      const Tetrahedron& tetrahedron =
          tetrahedra[tetrahedra.size() == 1 ? 0 : index];
      for (const Vec3& vertex : tetrahedron.vertices) {
        const Vec3 point = positions[index] + vertex;
        minimum.x = std::min(minimum.x, point.x);
        minimum.y = std::min(minimum.y, point.y);
        minimum.z = std::min(minimum.z, point.z);
        maximum.x = std::max(maximum.x, point.x);
        maximum.y = std::max(maximum.y, point.y);
        maximum.z = std::max(maximum.z, point.z);
      }
      has_geometry = true;
    }
  };
  include_population(source_positions, options.source_sizes,
                     options.source_geometry == SourceGeometry::RectangularPrism);
  include_population(
      target_positions, options.target_sizes,
      options.target_geometry == TargetGeometry::RectangularPrism);
  if (options.source_geometry == SourceGeometry::Tetrahedron) {
    include_tetrahedra(source_positions, options.source_tetrahedra);
  }
  if (options.target_geometry == TargetGeometry::Tetrahedron) {
    include_tetrahedra(target_positions, options.target_tetrahedra);
  }

  NormalisedGeometry geometry;
  geometry.construction_clock = construction_clock;
  geometry.options = options;
  // The periodic cell *is* the root box: the image topology assumes the unit
  // cell and the root coincide.  Otherwise the root is the caller's explicit
  // box, or the tightest cube about the geometry (or its chosen centre).
  if (options.periodic.enabled) {
    geometry.physical_root_centre = options.periodic.centre;
    geometry.physical_root_side_length = options.periodic.lengths.x;
  } else {
    if (!has_geometry) {
      minimum = {};
      maximum = {};
    }
    geometry.physical_root_centre = options.tree.root_centre.value_or(
        (minimum + maximum) * 0.5);
    const Vec3 upper_distance = maximum - geometry.physical_root_centre;
    const Vec3 lower_distance = geometry.physical_root_centre - minimum;
    double required_half_width = has_geometry
        ? std::max({upper_distance.x, upper_distance.y, upper_distance.z,
                    lower_distance.x, lower_distance.y, lower_distance.z})
        : 0.0;
    if (required_half_width < 0.0 || !std::isfinite(required_half_width)) {
      throw std::invalid_argument("invalid physical root geometry");
    }
    if (options.tree.root_half_width.has_value()) {
      const double requested = *options.tree.root_half_width;
      if (!std::isfinite(requested) || requested <= 0.0) {
        throw std::invalid_argument(
            "UniformTreeOptions.root_half_width must be finite and positive");
      }
      const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
          std::max({1.0, requested, required_half_width});
      if (requested + tolerance < required_half_width) {
        throw std::invalid_argument(
            "complete geometry lies outside the requested root box");
      }
      required_half_width = requested;
    } else if (required_half_width == 0.0) {
      // A finite convention is required for empty or coincident point sets.
      required_half_width = 0.5;
    }
    geometry.physical_root_side_length = 2.0 * required_half_width;
  }

  const double inverse_length = 1.0 / geometry.physical_root_side_length;
  const auto canonicalise = [](const double value) {
    // Inputs translated far from a small geometry lose a few low mantissa
    // bits during subtraction.  A nanounit grid removes that representation
    // noise so physically translated/rescaled copies have identical canonical
    // coordinates and therefore identical reusable plans.
    return canonicalise_normalised_value(value);
  };
  const auto physical_positions = [&](const std::vector<Vec3>& positions) {
    std::vector<Vec3> wrapped;
    wrapped.reserve(positions.size());
    for (Vec3 position : positions) {
      if (options.periodic.enabled) {
        position = wrap_periodic_position(position, options.periodic);
      }
      wrapped.push_back(position);
    }
    return wrapped;
  };
  // Representative points are wrapped into the periodic cell, but finite
  // bodies are not split across its faces: a body whose extent crosses the
  // cell boundary after wrapping is rejected rather than silently imaged.
  if (options.periodic.enabled) {
    const std::vector<Vec3> wrapped_sources =
        physical_positions(source_positions);
    const std::vector<Vec3> wrapped_targets =
        physical_positions(target_positions);
    const double half_width = 0.5 * options.periodic.lengths.x;
    const double tolerance =
        32.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, half_width, options.periodic.lengths.x});
    const auto check_point = [&](const Vec3& point, const char* description) {
      const Vec3 distance = point - options.periodic.centre;
      if (std::abs(distance.x) > half_width + tolerance ||
          std::abs(distance.y) > half_width + tolerance ||
          std::abs(distance.z) > half_width + tolerance) {
        throw std::invalid_argument(std::string(description) +
                                    " geometry lies outside the periodic root");
      }
    };
    const auto check_prisms = [&](const std::vector<Vec3>& positions,
                                  const std::vector<CuboidSize>& sizes,
                                  const char* description) {
      for (std::size_t index = 0; index < positions.size(); ++index) {
        const CuboidSize& size = sizes[sizes.size() == 1 ? 0 : index];
        const Vec3 extent{0.5 * size.hx, 0.5 * size.hy, 0.5 * size.hz};
        check_point(positions[index] - extent, description);
        check_point(positions[index] + extent, description);
      }
    };
    const auto check_tetrahedra = [&](const std::vector<Vec3>& positions,
                                      const std::vector<Tetrahedron>& tetrahedra,
                                      const char* description) {
      for (std::size_t index = 0; index < positions.size(); ++index) {
        const Tetrahedron& tetrahedron =
            tetrahedra[tetrahedra.size() == 1 ? 0 : index];
        for (const Vec3& vertex : tetrahedron.vertices) {
          check_point(positions[index] + vertex, description);
        }
      }
    };
    if (options.source_geometry == SourceGeometry::RectangularPrism) {
      check_prisms(wrapped_sources, options.source_sizes, "source prism");
    } else if (options.source_geometry == SourceGeometry::Tetrahedron) {
      check_tetrahedra(wrapped_sources, options.source_tetrahedra,
                       "source tetrahedron");
    }
    if (options.target_geometry == TargetGeometry::RectangularPrism) {
      check_prisms(wrapped_targets, options.target_sizes, "target prism");
    } else if (options.target_geometry == TargetGeometry::Tetrahedron) {
      check_tetrahedra(wrapped_targets, options.target_tetrahedra,
                       "target tetrahedron");
    }
  }
  const auto normalise_sizes = [inverse_length](
                                   std::vector<CuboidSize>& sizes) {
    for (CuboidSize& size : sizes) {
      size.hx *= inverse_length;
      size.hy *= inverse_length;
      size.hz *= inverse_length;
    }
  };
  const auto normalise_tetrahedra = [inverse_length, canonicalise](
                                        std::vector<Tetrahedron>& tetrahedra) {
    for (Tetrahedron& tetrahedron : tetrahedra) {
      for (Vec3& vertex : tetrahedron.vertices) {
        vertex = vertex * inverse_length;
        vertex = {canonicalise(vertex.x), canonicalise(vertex.y),
                   canonicalise(vertex.z)};
      }
    }
  };

  // Normalised coordinate = (physical - root centre) / side length, snapped to
  // the grid.  Sizes are scaled but not offset; tetrahedron vertices are
  // representative-relative and are scaled and snapped like coordinates.
  geometry.physical_source_positions = physical_positions(source_positions);
  geometry.physical_target_positions = physical_positions(target_positions);
  geometry.source_positions.reserve(geometry.physical_source_positions.size());
  for (const Vec3& position : geometry.physical_source_positions) {
    const Vec3 value =
        (position - geometry.physical_root_centre) * inverse_length;
    geometry.source_positions.push_back(
        {canonicalise(value.x), canonicalise(value.y), canonicalise(value.z)});
  }
  geometry.target_positions.reserve(geometry.physical_target_positions.size());
  for (const Vec3& position : geometry.physical_target_positions) {
    const Vec3 value =
        (position - geometry.physical_root_centre) * inverse_length;
    geometry.target_positions.push_back(
        {canonicalise(value.x), canonicalise(value.y), canonicalise(value.z)});
  }
  geometry.physical_tree_options = options.tree;
  geometry.physical_tree_options.root_centre = geometry.physical_root_centre;
  geometry.physical_tree_options.root_half_width =
      0.5 * geometry.physical_root_side_length;
  normalise_sizes(geometry.options.source_sizes);
  normalise_sizes(geometry.options.target_sizes);
  normalise_tetrahedra(geometry.options.source_tetrahedra);
  normalise_tetrahedra(geometry.options.target_tetrahedra);
  const auto canonicalise_sizes = [canonicalise](
                                      std::vector<CuboidSize>& sizes) {
    for (CuboidSize& size : sizes) {
      size.hx = canonicalise(size.hx);
      size.hy = canonicalise(size.hy);
      size.hz = canonicalise(size.hz);
    }
  };
  canonicalise_sizes(geometry.options.source_sizes);
  canonicalise_sizes(geometry.options.target_sizes);
  // From here on the solver only ever sees the unit root at the origin.
  geometry.options.tree.root_centre = Vec3{};
  geometry.options.tree.root_half_width = 0.5;
  if (geometry.options.periodic.enabled) {
    geometry.options.periodic.centre = {};
    geometry.options.periodic.lengths = {1.0, 1.0, 1.0};
  }
  geometry.normalisation_seconds = normalisation_clock.elapsed();
  return geometry;
}

// Public constructors delegate through `normalise_geometry` to the private
// constructor below.  An empty target list means "targets are the sources".
UniformFmm::UniformFmm(const std::vector<Vec3> &source_positions,
                       const UniformFmmOptions &options)
    : UniformFmm(source_positions, std::vector<Vec3>{}, options) {}

UniformFmm::UniformFmm(const std::vector<Vec3> &source_positions,
                       const std::vector<Vec3> &target_positions,
                       const UniformFmmOptions &options)
    : UniformFmm(normalise_geometry(source_positions, target_positions, options),
                 options) {}

// Two trees are built: `physical_tree_` in physical units backs the public
// inspection API, `tree_` in normalised units is the one the plan is built
// from.  `coordinate_scale_` (the physical side length) is what FP32 plans
// and the cache keys use to relate the two.  `physical_options` is kept only
// for the initialisation summary, which reports what the caller asked for.
UniformFmm::UniformFmm(NormalisedGeometry geometry,
                       const UniformFmmOptions& physical_options)
    : physical_tree_(std::in_place, geometry.physical_source_positions,
                     geometry.physical_target_positions,
                     geometry.physical_tree_options),
      tree_(std::in_place, geometry.source_positions, geometry.target_positions,
            geometry.options.tree),
      physical_root_centre_(geometry.physical_root_centre),
      physical_root_side_length_(geometry.physical_root_side_length),
      physical_periodic_(physical_options.periodic),
      periodic_(geometry.options.periodic),
      basis_(std::max(geometry.options.expansion_order, 0)),
      spherical_basis_(std::max(geometry.options.expansion_order, 0)),
      expansion_basis_(geometry.options.expansion_basis),
      spherical_m2l_backend_(geometry.options.spherical_m2l_backend),
      m2l_backend_(geometry.options.m2l_backend),
      static_matrix_backend_(geometry.options.static_matrix_backend),
      precision_(geometry.options.precision),
      coordinate_scale_(geometry.physical_root_side_length),
      timing_level_(geometry.options.timing_level) {
  const UniformFmmOptions& options = geometry.options;
  static_plan_statistics_.timing_level = timing_level_;
  detail::PhaseStopwatch topology_clock(detailed_timing());
  topology_clock.start();
  topology_ = std::make_shared<const StaticFmmTopology>(build_uniform_fmm_topology(*tree_, periodic_));
  topology_clock.record(static_plan_statistics_.topology_construction);
  if (detailed_timing()) {
    static_plan_statistics_.normalisation.add(geometry.normalisation_seconds);
    // The trees keep their own build timings regardless of the level; they
    // are surfaced here only when the plan collects detailed construction
    // timings.
    static_plan_statistics_.tree_construction = tree_->build_timings().total;
    static_plan_statistics_.tree_construction.add(
        physical_tree_->build_timings().total.total_seconds);
  }
  initialise_execution(options);
  geometry.construction_clock.record(static_plan_statistics_.total_setup);
  print_initialisation_summary(physical_options);
}

// Construction from a prebuilt topology (the AdaptiveTree route).  There is no
// UniformTree in this case, so `tree_`/`physical_tree_` stay empty and the
// uniform-tree inspection API is unavailable; the topology must already be
// normalised to the unit root and non-periodic.
UniformFmm::UniformFmm(std::shared_ptr<const StaticFmmTopology> topology,
                       const UniformFmmOptions& options)
    : topology_(std::move(topology)), supplied_topology_(true),
      basis_(std::max(options.expansion_order, 0)),
      spherical_basis_(std::max(options.expansion_order, 0)),
      expansion_basis_(options.expansion_basis),
      spherical_m2l_backend_(options.spherical_m2l_backend),
      m2l_backend_(options.m2l_backend),
      static_matrix_backend_(options.static_matrix_backend),
      precision_(options.precision), timing_level_(options.timing_level) {
  static_plan_statistics_.timing_level = timing_level_;
  detail::PhaseStopwatch setup_clock(coarse_timing());
  setup_clock.start();
  if (!topology_) throw std::invalid_argument("topology must not be null");
  topology_->validate();
  const auto& root = topology_->nodes[topology_->root];
  if (root.level != 0 || root.half_width != 0.5 ||
      !std::isfinite(topology_->coordinate_scale) || topology_->coordinate_scale <= 0.0) {
    throw std::invalid_argument("prebuilt topology requires a unit-width normalised root and positive physical scale");
  }
  if (options.periodic.enabled) {
    throw std::invalid_argument(
        "prebuilt topology currently supports only non-periodic execution");
  }
  detail::PhaseStopwatch normalisation_clock(detailed_timing());
  normalisation_clock.start();
  const UniformFmmOptions normalised_options =
      normalise_supplied_topology_options(*topology_, options);
  normalisation_clock.record(static_plan_statistics_.normalisation);
  physical_root_centre_ = topology_->coordinate_origin +
      root.centre * topology_->coordinate_scale;
  coordinate_scale_ = topology_->coordinate_scale;
  physical_root_side_length_ = coordinate_scale_;
  initialise_execution(normalised_options);
  setup_clock.record(static_plan_statistics_.total_setup);
}


} // namespace cdfmm
