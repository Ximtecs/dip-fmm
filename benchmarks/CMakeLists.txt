add_executable(benchmark_p2p benchmark_p2p.cpp)
target_link_libraries(benchmark_p2p PRIVATE cdfmm_core)
cdfmm_enable_ipo(benchmark_p2p)

add_executable(benchmark_uniform_fmm benchmark_uniform_fmm.cpp)
target_link_libraries(benchmark_uniform_fmm PRIVATE cdfmm_core)
target_include_directories(benchmark_uniform_fmm PRIVATE ${PROJECT_SOURCE_DIR}/src)
cdfmm_enable_ipo(benchmark_uniform_fmm)

add_executable(benchmark_cache_initialisation benchmark_cache_initialisation.cpp)
target_link_libraries(benchmark_cache_initialisation PRIVATE cdfmm_core)
cdfmm_enable_ipo(benchmark_cache_initialisation)

add_executable(benchmark_operator_representation
               benchmark_operator_representation.cpp)
target_link_libraries(benchmark_operator_representation PRIVATE cdfmm_core)
target_include_directories(benchmark_operator_representation
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(CDFMM_ENABLE_CUDA)
  # The procedural prism P2P device kernel of the representation study.
  target_sources(benchmark_operator_representation PRIVATE
                 benchmark_operator_representation_cuda.cu)
  set_target_properties(benchmark_operator_representation PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
endif()
cdfmm_enable_ipo(benchmark_operator_representation)
