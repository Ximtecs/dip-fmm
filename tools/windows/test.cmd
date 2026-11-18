@echo off
rem Build the C++ test suite and the Fortran smoke test of dip-fmm on Windows,
rem with the toolchain and options of build.cmd, in a build directory of its
rem own (build-wintest; nothing is installed), and run them.
rem
rem   tools\windows\test.cmd [fresh]
setlocal
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
@echo off

for %%i in ("%~dp0..\..") do set "SRC=%%~fi"
set "BLD=%SRC%\build-wintest"
set "FRESH="
if /i "%~1"=="fresh" set "FRESH=--fresh"

rem The tests write plan caches; keep them out of caches\ in the repository.
set "CDFMM_CACHE_DIR=%BLD%\test-caches"
rem cdfmm_tests.exe (in tests\) loads cdfmm_c.dll from the build root; without
rem this, Catch2's test discovery dies with 0xc0000135 (DLL not found).
set "PATH=%BLD%;%PATH%"

echo === configure
cmake %FRESH% -S "%SRC%" -B "%BLD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_Fortran_COMPILER=ifx ^
  -DCMAKE_CUDA_COMPILER="%CDFMM_ENV%\Library\bin\nvcc.exe" ^
  -DCMAKE_CUDA_ARCHITECTURES=native ^
  -DCUDAToolkit_ROOT="%CDFMM_ENV%\Library" ^
  -DCDFMM_ENABLE_CUDA=ON ^
  -DCDFMM_PROCEDURAL_MAX_ORDER=10 ^
  -DCDFMM_EXTRA_COMPILE_OPTIONS=-fp:precise ^
  -DCDFMM_ENABLE_MKL=ON ^
  -DMKL_DIR="%CDFMM_ENV%\Library\lib\cmake\mkl" ^
  -DMKL_INTERFACE=lp64 ^
  -DMKL_THREADING=intel_thread ^
  -DMKL_LINK=dynamic ^
  -DMKL_SYCL_LINK=none ^
  -DCDFMM_ENABLE_OPENMP=ON ^
  -DCDFMM_BUILD_PYTHON=OFF ^
  -DCDFMM_BUILD_FORTRAN_INTERFACE=ON ^
  -DCDFMM_BUILD_TESTS=ON ^
  -DCDFMM_BUILD_EXAMPLES=OFF ^
  -DCDFMM_BUILD_BENCHMARKS=OFF
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 1)

echo === build
rem -k 0: keep going, so one run reports every translation unit that fails.
cmake --build "%BLD%" --parallel -- -k 0
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)

echo === test
ctest --test-dir "%BLD%" --output-on-failure -j 4
if errorlevel 1 (echo TESTS_FAILED & exit /b 1)
echo DIPFMM_TESTS_OK
