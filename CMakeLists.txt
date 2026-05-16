cmake_minimum_required(VERSION 3.20)
project(cartesian_dipole_fmm LANGUAGES CXX)

option(CDFMM_BUILD_TESTS "Build C++ tests" ON)
option(CDFMM_BUILD_EXAMPLES "Build examples" ON)
option(CDFMM_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CDFMM_BUILD_PYTHON "Build Python bindings" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cdfmm_core
    src/taylor_jet.cpp
    src/laplace_derivatives.cpp
    src/operators.cpp
)
target_include_directories(cdfmm_core PUBLIC include)
set_target_properties(cdfmm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
endif()
