cmake_minimum_required(VERSION 3.20)
project(cartesian_dipole_fmm LANGUAGES CXX)

option(CDFMM_BUILD_TESTS "Build C++ tests" ON)
option(CDFMM_BUILD_EXAMPLES "Build examples" ON)
option(CDFMM_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CDFMM_BUILD_PYTHON "Build Python bindings" ON)
option(CDFMM_ENABLE_OPENMP "Enable internally parallel CPU execution" ON)
option(CDFMM_ENABLE_IPO "Enable interprocedural optimisation for Release builds" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cdfmm_core
    src/taylor_jet.cpp
    src/laplace_derivatives.cpp
    src/operators.cpp
    src/validation.cpp
    src/uniform_tree.cpp
    src/uniform_fmm.cpp
)
target_include_directories(cdfmm_core PUBLIC include)
set_target_properties(cdfmm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CDFMM_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(cdfmm_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(cdfmm_core PUBLIC CDFMM_USE_OPENMP=1)
endif()

if(CDFMM_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT CDFMM_IPO_SUPPORTED OUTPUT CDFMM_IPO_ERROR)
  if(CDFMM_IPO_SUPPORTED)
    set_property(TARGET cdfmm_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  else()
    message(WARNING "IPO requested but unavailable: ${CDFMM_IPO_ERROR}")
  endif()
endif()

if(CDFMM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CDFMM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(CDFMM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(CDFMM_BUILD_PYTHON)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(cdfmm python/bindings.cpp)
  target_link_libraries(cdfmm PRIVATE cdfmm_core)
  target_include_directories(cdfmm PRIVATE include)
  install(TARGETS cdfmm LIBRARY DESTINATION .)
endif()
