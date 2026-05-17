add_executable(cdfmm_tests
  test_multi_index.cpp
  test_p2p.cpp
  test_taylor_jet.cpp
  test_laplace_derivatives.cpp
  test_p2m_m2l_l2p.cpp
  test_multipole_accuracy.cpp
  test_output_flags.cpp
)
target_link_libraries(cdfmm_tests PRIVATE cdfmm_core Catch2::Catch2WithMain)
include(Catch)
catch_discover_tests(cdfmm_tests)
