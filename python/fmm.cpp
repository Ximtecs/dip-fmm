// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

namespace cdfmm::python_detail {

void bind_fmm_options(py::module_& module)
{
  py::class_<UniformFmmOptions>(module, "UniformFmmOptions")
      .def(py::init<>())
      .def_readwrite("periodic", &UniformFmmOptions::periodic)
      .def_readwrite("precision", &UniformFmmOptions::precision)
      .def_readwrite("expansion_order", &UniformFmmOptions::expansion_order)
      .def_property(
          "expansion_basis",
          [](const UniformFmmOptions& options) { return options.expansion_basis; },
          [](UniformFmmOptions& options, py::object value) {
            if (py::isinstance<py::str>(value)) {
              const std::string name = py::cast<std::string>(value);
              if (name == "cartesian") {
                options.expansion_basis = ExpansionBasis::Cartesian;
                return;
              }
              if (name == "spherical") {
                options.expansion_basis = ExpansionBasis::Spherical;
                return;
              }
              throw std::invalid_argument(
                  "expansion_basis must be 'cartesian' or 'spherical'");
            }
            options.expansion_basis = py::cast<ExpansionBasis>(value);
          })
      .def_readwrite("spherical_m2l_backend",
                     &UniformFmmOptions::spherical_m2l_backend)
      .def_readwrite("tree", &UniformFmmOptions::tree)
      .def_readwrite("m2l_backend", &UniformFmmOptions::m2l_backend)
      .def_readwrite("static_matrix_backend",
                     &UniformFmmOptions::static_matrix_backend)
      .def_readwrite("backend", &UniformFmmOptions::backend)
      .def_readwrite("source_geometry", &UniformFmmOptions::source_geometry)
      .def_readwrite("source_sizes", &UniformFmmOptions::source_sizes)
      .def_readwrite("source_tetrahedra", &UniformFmmOptions::source_tetrahedra)
      .def_readwrite("target_geometry", &UniformFmmOptions::target_geometry)
      .def_readwrite("target_sizes", &UniformFmmOptions::target_sizes)
      .def_readwrite("target_tetrahedra", &UniformFmmOptions::target_tetrahedra)
      .def_readwrite("near_field_source_model",
                     &UniformFmmOptions::near_field_source_model)
      .def_readwrite("near_field_target_model",
                     &UniformFmmOptions::near_field_target_model)
      .def_readwrite("far_field_source_model",
                     &UniformFmmOptions::far_field_source_model)
      .def_readwrite("far_field_target_model",
                     &UniformFmmOptions::far_field_target_model)
      .def_readwrite("fixed_target_source_indices",
                     &UniformFmmOptions::fixed_target_source_indices)
      .def_readwrite("cuda_p2p_bsr_max_bytes",
                     &UniformFmmOptions::cuda_p2p_bsr_max_bytes)
      .def_readwrite("spatial_layout", &UniformFmmOptions::spatial_layout)
      .def_readwrite("use_reduced_symmetry_p2p",
                     &UniformFmmOptions::use_reduced_symmetry_p2p)
      .def_readwrite("p2p_packing", &UniformFmmOptions::p2p_packing)
      .def_readwrite("point_expansion_execution",
                     &UniformFmmOptions::point_expansion_execution)
      .def_readwrite("cuda_dictionary_target_owned",
                     &UniformFmmOptions::cuda_dictionary_target_owned)
      .def_readwrite("cuda_dictionary_power2_microtiles",
                     &UniformFmmOptions::cuda_dictionary_power2_microtiles)
      .def_readwrite("signed_p2p_target_tile_size",
                     &UniformFmmOptions::signed_p2p_target_tile_size)
      .def_readwrite("enable_cache", &UniformFmmOptions::enable_cache);
}

void bind_fmm(py::module_& module)
{
  module.def(
      "suggest_depth_for_performance",
      [](py::object source_positions, py::object target_positions,
         py::object dipole_moments, const int order,
         const ExecutionBackend backend, py::object candidate_depths,
         const int repetitions, py::object target_source_indices) {
        const auto sources =
            parse_vec3_array(source_positions, "source_positions");
        const auto targets =
            parse_vec3_array(target_positions, "target_positions");
        const auto moments =
            parse_vec3_array(dipole_moments, "dipole_moments");
        const std::vector<int> depths = candidate_depths.is_none()
                                            ? std::vector<int>{}
                                            : py::cast<std::vector<int>>(
                                                  candidate_depths);
        const std::vector<int> identities = target_source_indices.is_none()
                                                ? std::vector<int>{}
                                                : py::cast<std::vector<int>>(
                                                      target_source_indices);
        const PerformanceSuggestion suggestion = suggest_depth_for_performance(
            sources, targets, moments, order, backend, depths, repetitions,
            identities);
        py::dict result;
        result["suggested_depth"] = suggestion.suggested_depth;
        result["order"] = suggestion.order;
        result["branches_concurrent"] = suggestion.branches_concurrent;
        py::list candidates;
        for (const auto& candidate : suggestion.candidates) {
          candidates.append(performance_candidate_to_dict(candidate));
        }
        result["candidates"] = candidates;
        if (suggestion.suggested_depth >= 0) {
          const auto selected = std::find_if(
              suggestion.candidates.begin(), suggestion.candidates.end(),
              [&suggestion](const auto& candidate) {
                return candidate.depth == suggestion.suggested_depth;
              });
          const py::dict diagnostics = performance_candidate_to_dict(*selected);
          for (const auto& item : diagnostics) {
            if (py::cast<std::string>(item.first) != "depth") {
              result[item.first] = item.second;
            }
          }
        }
        return result;
      },
      py::arg("source_positions"), py::arg("target_positions"),
      py::arg("dipole_moments"), py::arg("order") = 6,
      py::arg("backend") = ExecutionBackend::Auto,
      py::arg("candidate_depths") = py::none(), py::arg("repetitions") = 3,
      py::arg("target_source_indices") = py::none(),
      "Empirically suggest a depth without changing UniformFmm defaults.");

  module.def(
      "suggest_parameters_for_accuracy",
      [](py::object source_positions, py::object target_positions,
         py::object dipole_moments, const double desired_accuracy,
         const ExecutionBackend backend, py::object candidate_orders,
         py::object candidate_depths, const std::size_t sample_size,
         const int repetitions, py::object target_source_indices) {
        const auto sources =
            parse_vec3_array(source_positions, "source_positions");
        const auto targets =
            parse_vec3_array(target_positions, "target_positions");
        const auto moments =
            parse_vec3_array(dipole_moments, "dipole_moments");
        const std::vector<int> orders = candidate_orders.is_none()
                                            ? std::vector<int>{}
                                            : py::cast<std::vector<int>>(
                                                  candidate_orders);
        const std::vector<int> depths = candidate_depths.is_none()
                                            ? std::vector<int>{}
                                            : py::cast<std::vector<int>>(
                                                  candidate_depths);
        const std::vector<int> identities = target_source_indices.is_none()
                                                ? std::vector<int>{}
                                                : py::cast<std::vector<int>>(
                                                      target_source_indices);
        const AccuracySuggestion suggestion = suggest_parameters_for_accuracy(
            sources, targets, moments, desired_accuracy, backend, orders,
            depths, sample_size, repetitions, identities);
        py::dict result;
        result["suggested_order"] = suggestion.suggested_order;
        result["suggested_depth"] = suggestion.suggested_depth;
        result["requested_accuracy"] = suggestion.requested_accuracy;
        result["reference_target_count"] = suggestion.reference_target_count;
        result["reference_target_indices"] = suggestion.reference_target_indices;
        py::list candidates;
        for (const auto& candidate : suggestion.candidates) {
          candidates.append(accuracy_candidate_to_dict(candidate));
        }
        result["candidates"] = candidates;
        if (suggestion.suggested_order >= 0) {
          const auto selected = std::find_if(
              suggestion.candidates.begin(), suggestion.candidates.end(),
              [&suggestion](const auto& candidate) {
                return candidate.order == suggestion.suggested_order &&
                       candidate.depth == suggestion.suggested_depth;
              });
          result["estimated_rms_relative_error"] = selected->rms_relative_error;
          result["evaluation_seconds"] = selected->evaluation_seconds;
        }
        return result;
      },
      py::arg("source_positions"), py::arg("target_positions"),
      py::arg("dipole_moments"), py::arg("desired_accuracy") = 1.0e-4,
      py::arg("backend") = ExecutionBackend::Auto,
      py::arg("candidate_orders") = py::none(),
      py::arg("candidate_depths") = py::none(), py::arg("sample_size") = 128,
      py::arg("repetitions") = 3,
      py::arg("target_source_indices") = py::none(),
      "Suggest the fastest tested pair meeting sampled RMS field accuracy.");

  py::class_<UniformFmm>(module, "UniformFmm")
      .def(py::init([](py::object source_positions,
                       const UniformFmmOptions& options) {
        return UniformFmm(parse_vec3_array(source_positions, "source_positions"),
                          options);
      }),
           py::arg("source_positions"),
           py::arg("options") = UniformFmmOptions{})
      .def(py::init([](py::object source_positions, py::object target_positions,
                       const UniformFmmOptions& options) {
        return UniformFmm(parse_vec3_array(source_positions, "source_positions"),
                          parse_vec3_array(target_positions, "target_positions"),
                          options);
      }),
           py::arg("source_positions"), py::arg("target_positions"),
           py::arg("options") = UniformFmmOptions{})
      .def("upward_pass", [](UniformFmm& fmm, py::object dipole_moments) {
        fmm.upward_pass(parse_vec3_array(dipole_moments, "dipole_moments"));
      }, py::arg("dipole_moments"),
           "Replace all node multipoles using moments in original source order.")
      .def("downward_pass", &UniformFmm::downward_pass)
      .def("evaluate_components",
           [](UniformFmm& fmm, py::object moments,
              std::vector<int> identities) {
             const auto result =
                 fmm.evaluate_components(parse_tree_points(moments), identities);
             py::dict output;
             output["H_far"] = points_to_array(result.far);
             output["H_p2p"] = points_to_array(result.p2p);
             output["H_total"] = points_to_array(result.total);
             return output;
           },
           py::arg("moments"),
           py::arg("target_source_indices") = std::vector<int>{})
      .def("evaluate", [](UniformFmm& fmm, py::object dipole_moments,
                           const std::string& output,
                           py::object target_source_indices) {
        const std::vector<Vec3> moments =
            parse_vec3_array(dipole_moments, "dipole_moments");
        std::vector<int> identities;
        if (!target_source_indices.is_none()) {
          identities = py::cast<std::vector<int>>(target_source_indices);
        }
        const OutputFlags output_flags = parse_output(output);
        if (fmm.precision() == StaticPrecision::Float32) {
          return potential_fields_to_dict(
              fmm.evaluate_float32(moments, output_flags, identities));
        }
        return potential_fields_to_dict(fmm.evaluate(moments, output_flags,
                                                     identities));
      },
           py::arg("dipole_moments"), py::arg("output") = "field",
           py::arg("target_source_indices") = py::none(),
           "Run the complete FMM and return values in target order.")
      .def_property_readonly("topology", [](const UniformFmm& plan) {
        return std::const_pointer_cast<StaticFmmTopology>(plan.shared_topology());
      })
      .def_property_readonly("tree", &UniformFmm::tree,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("periodic_cell",
                             [](const UniformFmm& fmm) {
                               return fmm.periodic_cell();
                             })
      .def_property_readonly("expansion_order", &UniformFmm::expansion_order)
      .def_property_readonly("expansion_basis", &UniformFmm::expansion_basis)
      .def_property_readonly("coefficient_count", &UniformFmm::coefficient_count)
      .def_property_readonly("spherical_m2l_backend",
                             &UniformFmm::spherical_m2l_backend)
      .def_property_readonly("m2l_backend", &UniformFmm::m2l_backend)
      .def_property_readonly("backend", &UniformFmm::backend)
      .def_property_readonly("precision", &UniformFmm::precision)
      .def_property_readonly("last_timings", [](const UniformFmm& fmm) {
        return evaluation_timings_to_dict(fmm.last_timings());
      })
      .def_property_readonly("aggregate_timings", [](const UniformFmm& fmm) {
        return evaluation_timings_to_dict(fmm.aggregate_timings());
      })
      .def_property_readonly("p2p_execution_packing",
                             &UniformFmm::p2p_execution_packing)
      .def_property_readonly("requested_p2p_packing",
                             &UniformFmm::requested_p2p_packing)
      .def_property_readonly("requested_point_expansion_execution",
                             &UniformFmm::requested_point_expansion_execution)
      .def_property_readonly("p2m_execution", &UniformFmm::p2m_execution)
      .def_property_readonly("l2p_execution", &UniformFmm::l2p_execution)
      .def_property_readonly("spatial_layout", &UniformFmm::spatial_layout)
      .def_property_readonly("execution_plan", &UniformFmm::execution_plan)
      .def_property_readonly("cuda_plan_statistics", [](const UniformFmm& fmm) {
        const CudaPlanStatistics& statistics = fmm.cuda_plan_statistics();
        py::dict result;
        result["scalar_bytes"] = statistics.scalar_bytes;
        result["m2m_unique_matrix_count"] = statistics.m2m_unique_matrix_count;
        result["m2m_matrix_bytes"] = statistics.m2m_matrix_bytes;
        result["m2l_unique_matrix_count"] = statistics.m2l_unique_matrix_count;
        result["m2l_matrix_bytes"] = statistics.m2l_matrix_bytes;
        result["m2l_interaction_metadata_bytes"] = statistics.m2l_interaction_metadata_bytes;
        result["l2l_unique_matrix_count"] = statistics.l2l_unique_matrix_count;
        result["l2l_matrix_bytes"] = statistics.l2l_matrix_bytes;
        result["setup_h2d_bytes"] = statistics.setup_h2d_bytes;
        result["evaluation_h2d_bytes"] = statistics.evaluation_h2d_bytes;
        result["evaluation_d2h_bytes"] = statistics.evaluation_d2h_bytes;
        result["evaluation_h2d_calls"] = statistics.evaluation_h2d_calls;
        result["evaluation_d2h_calls"] = statistics.evaluation_d2h_calls;
        result["persistent_device_bytes"] = statistics.persistent_device_bytes;
        result["p2p_interaction_count"] = statistics.p2p_interaction_count;
        result["p2p_tensor_bytes"] = statistics.p2p_tensor_bytes;
        result["p2p_index_bytes"] = statistics.p2p_index_bytes;
        result["p2p_row_metadata_bytes"] = statistics.p2p_row_metadata_bytes;
        result["p2p_leaf_metadata_bytes"] = statistics.p2p_leaf_metadata_bytes;
        result["p2p_identity_bytes"] = statistics.p2p_identity_bytes;
        result["p2p_scratch_bytes"] = statistics.p2p_scratch_bytes;
        result["p2p_threads_per_block"] = statistics.p2p_threads_per_block;
        result["plan_generation_count"] = statistics.plan_generation_count;
        result["static_upload_count"] = statistics.static_upload_count;
        result["static_m2l_upload_count"] = statistics.static_m2l_upload_count;
        result["static_p2p_upload_count"] = statistics.static_p2p_upload_count;
        result["geometry_upload_count"] = statistics.geometry_upload_count;
        return result;
      })
      .def_property_readonly("static_plan_statistics", [](const UniformFmm& fmm) {
        const StaticPlanStatistics& statistics = fmm.static_plan_statistics();
        py::dict result;
        result["expansion_order"] = statistics.expansion_order;
        result["coefficient_count"] = statistics.coefficient_count;
        result["spherical"] = statistics.spherical;
        result["scalar_bytes"] = statistics.scalar_bytes;
        result["transfer_classes"] = statistics.transfer_classes;
        result["interactions"] = statistics.interactions;
        result["operator_bytes"] = statistics.operator_bytes;
        result["p2m_operator_bytes"] = statistics.p2m_operator_bytes;
        result["interaction_bytes"] = statistics.interaction_bytes;
        result["scratch_bytes"] = statistics.scratch_bytes;
        result["state_bytes"] = statistics.state_bytes;
        result["multipole_state_bytes"] = statistics.multipole_state_bytes;
        result["local_state_bytes"] = statistics.local_state_bytes;
        result["other_state_bytes"] = statistics.other_state_bytes;
        result["m2m_operators"] = statistics.m2m_operators;
        result["m2m_theoretical_interactions"] = statistics.m2m_theoretical_interactions;
        result["m2m_operator_bytes"] = statistics.m2m_operator_bytes;
        result["m2l_operators"] = statistics.m2l_operators;
        result["m2l_theoretical_maximum_classes"] =
            StaticPlanStatistics::theoretical_maximum_m2l_classes;
        result["m2l_operator_bytes"] = statistics.m2l_operator_bytes;
        result["m2l_interaction_bytes"] = statistics.m2l_interaction_bytes;
        result["l2l_operators"] = statistics.l2l_operators;
        result["l2l_theoretical_interactions"] = statistics.l2l_theoretical_interactions;
        result["l2l_operator_bytes"] = statistics.l2l_operator_bytes;
        result["l2p_operator_bytes"] = statistics.l2p_operator_bytes;
        result["near_field_operator_bytes"] = statistics.near_field_operator_bytes;
        result["p2p_interactions"] = statistics.p2p_interactions;
        result["p2p_value_bytes"] = statistics.p2p_value_bytes;
        result["p2p_index_bytes"] = statistics.p2p_index_bytes;
        result["p2p_canonical_total_bytes"] = statistics.p2p_canonical_total_bytes;
        result["p2p_unique_tensors"] = statistics.p2p_unique_tensors;
        result["p2p_dictionary_tokens"] = statistics.p2p_dictionary_tokens;
        result["p2p_dictionary_token_width_bytes"] = statistics.p2p_dictionary_token_width_bytes;
        result["p2p_dictionary_token_bytes"] = statistics.p2p_dictionary_token_bytes;
        result["p2p_dictionary_tensor_bytes"] = statistics.p2p_dictionary_tensor_bytes;
        result["p2p_dictionary_total_bytes"] = statistics.p2p_dictionary_total_bytes;
        result["tree_bytes"] = statistics.tree_bytes;
        result["topology_bytes"] = statistics.topology_bytes;
        result["topology_construction_seconds"] = statistics.topology_construction.total_seconds;
        result["translation_operator_bytes"] = statistics.translation_operator_bytes();
        result["dense"] = statistics.dense;
        result["sparse"] = statistics.sparse;
        result["numerically_pruned"] = statistics.numerically_pruned;
        result["symmetry_compressed"] = statistics.symmetry_compressed;
        result["total_bytes"] = statistics.total_bytes();
        result["total_persistent_bytes"] = statistics.total_persistent_bytes();
        result["setup_seconds"] = statistics.total.total_seconds;
        result["normalisation_seconds"] = statistics.normalisation.total_seconds;
        result["tree_construction_seconds"] = statistics.tree_construction.total_seconds;
        result["universal_cache_lookup_seconds"] = statistics.universal_cache_lookup.total_seconds;
        result["universal_cache_load_seconds"] = statistics.universal_cache_load.total_seconds;
        result["universal_operator_build_seconds"] = statistics.universal_operator_build.total_seconds;
        result["universal_cache_write_seconds"] = statistics.universal_cache_write.total_seconds;
        result["periodic_cache_lookup_seconds"] = statistics.periodic_cache_lookup.total_seconds;
        result["periodic_cache_load_seconds"] = statistics.periodic_cache_load.total_seconds;
        result["periodic_operator_build_seconds"] = statistics.periodic_operator_build.total_seconds;
        result["geometry_hash_seconds"] = statistics.geometry_hash.total_seconds;
        result["geometry_cache_lookup_seconds"] = statistics.geometry_cache_lookup.total_seconds;
        result["geometry_cache_load_seconds"] = statistics.geometry_cache_load.total_seconds;
        result["geometry_cache_write_seconds"] = statistics.geometry_cache_write.total_seconds;
        result["p2m_construction_seconds"] = statistics.p2m_plan.total_seconds;
        result["m2m_construction_seconds"] = statistics.m2m_plan.total_seconds;
        result["m2l_construction_seconds"] = statistics.m2l_plan.total_seconds;
        result["l2l_construction_seconds"] = statistics.l2l_plan.total_seconds;
        result["l2p_construction_seconds"] = statistics.l2p_plan.total_seconds;
        result["p2p_construction_seconds"] = statistics.p2p_tensor_plan.total_seconds;
        result["p2p_interaction_setup_seconds"] =
            statistics.p2p_interaction_setup.total_seconds;
        result["p2p_canonical_operator_seconds"] =
            statistics.p2p_canonical_operator.total_seconds;
        result["p2p_derived_packing_seconds"] =
            statistics.p2p_derived_packing.total_seconds;
        result["precision_conversion_seconds"] =
            statistics.precision_conversion.total_seconds;
        result["backend_packing_seconds"] = statistics.backend_packing.total_seconds;
        result["far_field_packing_seconds"] =
            statistics.far_field_packing.total_seconds;
        result["cuda_upload_seconds"] = statistics.cuda_upload.total_seconds;
        result["total_setup_seconds"] = statistics.total_setup.total_seconds;
        result["universal_cache_hit"] = statistics.universal_cache_hit;
        result["periodic_cache_hit"] = statistics.periodic_cache_hit;
        result["geometry_cache_hit"] = statistics.geometry_cache_hit;
        result["cache_bytes_read"] = statistics.cache_bytes_read;
        result["cache_bytes_written"] = statistics.cache_bytes_written;
        return result;
      })
      .def_property_readonly("universal_cache_key", &UniformFmm::universal_cache_key)
      .def_property_readonly("periodic_cache_key", &UniformFmm::periodic_cache_key)
      .def_property_readonly("geometry_cache_key", &UniformFmm::geometry_cache_key)
      .def_property_readonly("root_multipole", [](const UniformFmm& fmm) {
        if (fmm.precision() == StaticPrecision::Float32) {
          return py::object(coefficients_to_array(fmm.root_multipole_float32()));
        }
        const auto coefficients = fmm.root_multipole();
        return py::object(coefficients_to_array(
            CoeffVector(coefficients.begin(), coefficients.end())));
      })
      .def("multipole", [](const UniformFmm& fmm, const int node_index) {
        if (fmm.precision() == StaticPrecision::Float32) {
          return py::object(coefficients_to_array(fmm.multipole_float32(node_index)));
        }
        const auto coefficients = fmm.multipole(node_index);
        return py::object(coefficients_to_array(
            CoeffVector(coefficients.begin(), coefficients.end())));
      }, py::arg("node_index"))
      .def("local", [](const UniformFmm& fmm, const int node_index) {
        if (fmm.precision() == StaticPrecision::Float32) {
          return py::object(coefficients_to_array(fmm.local_float32(node_index)));
        }
        const auto coefficients = fmm.local(node_index);
        return py::object(coefficients_to_array(
            CoeffVector(coefficients.begin(), coefficients.end())));
      }, py::arg("node_index"));
}

} // namespace cdfmm::python_detail
