cmake_minimum_required(VERSION 3.20)

# oneMKL's CMake package uses the detected C compiler family to expose the
# matching OpenMP threading layer.  Enable C even though cdfmm itself is C++ so
# GNU builds can select MKL_THREADING=gnu_thread.
project(cartesian_dipole_fmm LANGUAGES C CXX)

option(CDFMM_BUILD_TESTS "Build C++ tests" ON)
option(CDFMM_BUILD_EXAMPLES "Build examples" ON)
option(CDFMM_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CDFMM_BUILD_TOOLS "Build cache precomputation tools" ON)
option(CDFMM_BUILD_PYTHON "Build Python bindings" ON)
option(CDFMM_BUILD_FORTRAN_INTERFACE
       "Build the ISO_C_BINDING module, smoke test, and Fortran example" OFF)
option(CDFMM_INSTALL_PYTHON_TO_ENV
       "Install Python bindings into the configured interpreter environment" OFF)
option(CDFMM_ENABLE_OPENMP "Enable internally parallel CPU execution" ON)
option(CDFMM_ENABLE_LTO "Enable interprocedural/link-time optimisation" ON)
option(CDFMM_ENABLE_IPO "Deprecated alias for CDFMM_ENABLE_LTO" OFF)
option(CDFMM_ENABLE_NATIVE_ARCH
       "Optimise CPU code for the architecture of the build machine" ON)
option(CDFMM_ENABLE_FAST_MATH
       "Enable non-IEEE fast floating-point transformations" OFF)
set(CDFMM_EXTRA_COMPILE_OPTIONS "" CACHE STRING
    "Additional semicolon-separated C++ compile options")
option(CDFMM_ENABLE_MKL
       "Include runtime-selectable oneMKL dense operator backends" OFF)
option(CDFMM_ENABLE_CUDA "Enable optional NVIDIA CUDA execution backends" OFF)
option(CDFMM_ENABLE_PROFILING
       "Enable NVTX ranges for perf and NVIDIA Nsight captures" OFF)
option(CDFMM_ENABLE_VECTORIZATION_REPORTS
       "Emit compiler vectorization diagnostics for static_operators.cpp" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cdfmm_core
    src/periodic.cpp
    src/spherical_harmonics.cpp
    src/taylor_jet.cpp
    src/laplace_derivatives.cpp
    src/operators.cpp
    src/cuboid.cpp
    src/static_operators.cpp
    src/validation.cpp
    src/uniform_tree.cpp
    src/uniform_fmm.cpp
    src/cache.cpp
    src/far_field.cpp
    src/near_field.cpp
    src/parameter_selection.cpp
)
if(CDFMM_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(cdfmm_core PRIVATE src/cuda_fmm.cu)
  target_link_libraries(cdfmm_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
  if(CDFMM_ENABLE_PROFILING)
    if(NOT TARGET CUDA::nvtx3)
      message(FATAL_ERROR
        "CDFMM_ENABLE_PROFILING requires CMake 3.25 or newer for CUDA::nvtx3")
    endif()
    # NVTX3 is header-only. CUDA 12.9 and newer no longer provide the legacy
    # nvToolsExt binary library or its corresponding imported CMake target.
    target_link_libraries(cdfmm_core PUBLIC CUDA::nvtx3)
    target_compile_definitions(cdfmm_core PUBLIC CDFMM_ENABLE_NVTX=1)
  endif()
  target_compile_definitions(cdfmm_core PUBLIC CDFMM_ENABLE_CUDA=1)
  set_target_properties(cdfmm_core PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
else()
  target_sources(cdfmm_core PRIVATE src/cuda_fmm_stub.cpp)
endif()
target_include_directories(cdfmm_core PUBLIC include)
target_compile_definitions(cdfmm_core PRIVATE
  CDFMM_DEFAULT_CACHE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/caches")
set_target_properties(cdfmm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Project optimisation options append to, rather than replace, flags selected by
# the standard CMake build type.  Generator expressions keep host-only options
# away from nvcc when CUDA is enabled.
if(CDFMM_ENABLE_NATIVE_ARCH)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    target_compile_options(cdfmm_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  elseif(MSVC)
    target_compile_options(cdfmm_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  else()
    message(FATAL_ERROR "CDFMM_ENABLE_NATIVE_ARCH is unsupported by ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

if(CDFMM_ENABLE_FAST_MATH)
  if(MSVC)
    target_compile_options(cdfmm_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>)
  else()
    target_compile_options(cdfmm_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
  endif()
endif()

if(CDFMM_EXTRA_COMPILE_OPTIONS)
  target_compile_options(cdfmm_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CDFMM_EXTRA_COMPILE_OPTIONS}>)
endif()

if(CDFMM_ENABLE_VECTORIZATION_REPORTS)
  set(CDFMM_VECTORIZATION_REPORT
      "${CMAKE_CURRENT_BINARY_DIR}/cdfmm-static-operators.vec")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_property(SOURCE src/static_operators.cpp APPEND PROPERTY COMPILE_OPTIONS
      "-fopt-info-vec-optimized-missed=${CDFMM_VECTORIZATION_REPORT}")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set_property(SOURCE src/static_operators.cpp APPEND PROPERTY COMPILE_OPTIONS
      "-qopt-report=3"
      "-qopt-report-phase=vec"
      "-qopt-report-file=${CDFMM_VECTORIZATION_REPORT}")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_property(SOURCE src/static_operators.cpp APPEND PROPERTY COMPILE_OPTIONS
      "-Rpass=loop-vectorize"
      "-Rpass-missed=loop-vectorize"
      "-Rpass-analysis=loop-vectorize")
  else()
    message(WARNING "Vectorization reports are unsupported by ${CMAKE_CXX_COMPILER_ID}")
  endif()
  message(STATUS "  Vector report:       ${CDFMM_VECTORIZATION_REPORT}")
endif()

if(CDFMM_ENABLE_MKL)
  find_package(MKL CONFIG REQUIRED)
  # oneMKL's Conda package exposes the environment-wide include directory.
  # Keeping it private prevents Intel compiler compatibility headers such as
  # <complex> from shadowing libstdc++ in pybind11 and downstream consumers.
  target_link_libraries(cdfmm_core PRIVATE MKL::MKL)
  if(UNIX AND NOT APPLE AND MKL_LINK STREQUAL "dynamic")
    # Dynamic oneMKL uses separate interface, threading, and core libraries.
    # The Conda GNU toolchain enables --as-needed, while oneMKL's CMake package
    # omits --no-as-needed unless GNU Fortran is enabled.  Preserve the full
    # component set in every final consumer of this static core library.
    target_link_options(cdfmm_core INTERFACE "LINKER:--no-as-needed")
  endif()
  target_compile_definitions(cdfmm_core PUBLIC CDFMM_USE_MKL=1)
endif()

if(CDFMM_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(cdfmm_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(cdfmm_core PUBLIC CDFMM_USE_OPENMP=1)
endif()

if(CDFMM_ENABLE_IPO)
  message(DEPRECATION "CDFMM_ENABLE_IPO is deprecated; use CDFMM_ENABLE_LTO")
  set(CDFMM_ENABLE_LTO ON)
endif()

if(CDFMM_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT CDFMM_IPO_SUPPORTED OUTPUT CDFMM_IPO_ERROR)
  if(NOT CDFMM_IPO_SUPPORTED)
    message(WARNING "IPO requested but unavailable: ${CDFMM_IPO_ERROR}")
  endif()
endif()

function(cdfmm_enable_ipo target_name)
  if(CDFMM_ENABLE_LTO AND CDFMM_IPO_SUPPORTED)
    set_property(
      TARGET ${target_name}
      PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
  endif()
endfunction()

cdfmm_enable_ipo(cdfmm_core)

# The shared integration boundary contains only plain C symbols.  Its C++
# implementation and all backend resources remain private behind an opaque plan.
add_library(cdfmm_c SHARED src/c_api.cpp)
target_link_libraries(cdfmm_c PRIVATE cdfmm_core)
target_include_directories(cdfmm_c PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(cdfmm_c PRIVATE CDFMM_C_EXPORTS=1)
cdfmm_enable_ipo(cdfmm_c)

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

if(CDFMM_BUILD_FORTRAN_INTERFACE)
  enable_language(Fortran)
  enable_testing()
  add_library(cdfmm_fortran STATIC fortran/cdfmm_fortran.f90)
  target_link_libraries(cdfmm_fortran PUBLIC cdfmm_c)
  set_target_properties(cdfmm_fortran PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fortran-modules")
  target_include_directories(cdfmm_fortran INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran-modules>)

  add_executable(cdfmm_fortran_smoke tests/test_fortran_api.f90)
  target_link_libraries(cdfmm_fortran_smoke PRIVATE cdfmm_fortran)
  add_test(NAME cdfmm_fortran_smoke COMMAND cdfmm_fortran_smoke)

  add_executable(magtense_style_demag
    examples/fortran/magtense_style_demag.f90)
  target_link_libraries(magtense_style_demag PRIVATE cdfmm_fortran)
endif()

if(CDFMM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(CDFMM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(CDFMM_BUILD_TOOLS)
  add_executable(cdfmm-precompute tools/cdfmm_precompute.cpp)
  target_link_libraries(cdfmm-precompute PRIVATE cdfmm_core)
  cdfmm_enable_ipo(cdfmm-precompute)
endif()

if(CDFMM_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
  endif()
  pybind11_add_module(cdfmm python/bindings.cpp)
  target_link_libraries(cdfmm PRIVATE cdfmm_core)
  target_include_directories(cdfmm PRIVATE include)
  cdfmm_enable_ipo(cdfmm)
  if(CDFMM_INSTALL_PYTHON_TO_ENV)
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c
              "import sysconfig; print(sysconfig.get_path('platlib'))"
      OUTPUT_VARIABLE CDFMM_PYTHON_INSTALL_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
  else()
    set(CDFMM_PYTHON_INSTALL_DIR .)
  endif()
  install(TARGETS cdfmm LIBRARY DESTINATION "${CDFMM_PYTHON_INSTALL_DIR}")
endif()

include(GNUInstallDirs)
install(TARGETS cdfmm_c
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(CDFMM_BUILD_TOOLS)
  install(TARGETS cdfmm-precompute RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(FILES include/cdfmm/c_api.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cdfmm)
install(FILES fortran/cdfmm_fortran.f90 DESTINATION ${CMAKE_INSTALL_DATADIR}/cdfmm/fortran)

# A normal rebuild must not require users to remove a previous installation by
# hand. Keep this target deliberately narrower than the install prefix: Conda,
# CUDA, oneMKL, Jupyter, unrelated headers, and persistent caches are untouched.
set(CDFMM_CLEAN_INSTALL_COMMANDS
  COMMAND ${CMAKE_COMMAND} -E rm -f
          "${CMAKE_INSTALL_FULL_LIBDIR}/$<TARGET_FILE_NAME:cdfmm_c>"
  COMMAND ${CMAKE_COMMAND} -E rm -rf
          "${CMAKE_INSTALL_FULL_INCLUDEDIR}/cdfmm"
          "${CMAKE_INSTALL_FULL_DATADIR}/cdfmm")
if(CDFMM_BUILD_PYTHON)
  list(APPEND CDFMM_CLEAN_INSTALL_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E rm -f
            "${CDFMM_PYTHON_INSTALL_DIR}/$<TARGET_FILE_NAME:cdfmm>")
endif()
if(CDFMM_BUILD_TOOLS)
  list(APPEND CDFMM_CLEAN_INSTALL_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E rm -f
            "${CMAKE_INSTALL_FULL_BINDIR}/$<TARGET_FILE_NAME:cdfmm-precompute>")
endif()
add_custom_target(cdfmm-clean-install
  ${CDFMM_CLEAN_INSTALL_COMMANDS}
  COMMENT "Removing installed dip-fmm artifacts from ${CMAKE_INSTALL_PREFIX}"
  VERBATIM)

message(STATUS "CDFMM optimisation configuration")
message(STATUS "  Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  Native architecture: ${CDFMM_ENABLE_NATIVE_ARCH}")
message(STATUS "  IPO/LTO:             ${CDFMM_ENABLE_LTO}")
message(STATUS "  Fast math:           ${CDFMM_ENABLE_FAST_MATH}")
message(STATUS "  OpenMP:              ${CDFMM_ENABLE_OPENMP}")
message(STATUS "  oneMKL:              ${CDFMM_ENABLE_MKL}")
message(STATUS "  CUDA:                ${CDFMM_ENABLE_CUDA}")
message(STATUS "  NVTX profiling:      ${CDFMM_ENABLE_PROFILING}")
if(CDFMM_ENABLE_CUDA)
  message(STATUS "  CUDA architecture:   ${CMAKE_CUDA_ARCHITECTURES}")
endif()
