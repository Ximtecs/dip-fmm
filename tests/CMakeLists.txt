add_executable(cdfmm_tests
  test_multi_index.cpp
  test_p2p.cpp
  test_taylor_jet.cpp
  test_laplace_derivatives.cpp
  test_p2m_m2l_l2p.cpp
  test_multipole_accuracy.cpp
  test_output_flags.cpp
  test_operator_consistency.cpp
  test_uniform_tree.cpp
  test_uniform_fmm.cpp
  test_static_m2l.cpp
  test_cuda_backend.cpp
  test_cuda_m2l.cpp
  test_parameter_selection.cpp
)
target_link_libraries(cdfmm_tests PRIVATE cdfmm_core Catch2::Catch2WithMain)
target_include_directories(cdfmm_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
cdfmm_enable_ipo(cdfmm_tests)
include(Catch)
catch_discover_tests(cdfmm_tests)
