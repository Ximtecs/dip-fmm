@echo off
rem Configure, build and install dip-fmm on Windows.
rem
rem   tools\windows\build.cmd [fresh] [static]
rem
rem   fresh   start from a new CMake configuration
rem   static  link oneMKL into cdfmm_c.dll instead of using the MKL DLLs. Needed
rem           when the library is loaded into a host process that brings its
rem           own oneMKL, such as MATLAB.
rem
rem   C/C++  : icx from the conda environment (MSVC ABI, Intel OpenMP runtime)
rem   CUDA   : the environment's nvcc with MSVC cl as host compiler
rem   oneMKL : the environment's MKL, lp64 / intel_thread
rem
rem Installs under local\ in the repository: bin\cdfmm_c.dll, lib\cdfmm_c.lib,
rem include\ and share\cdfmm\fortran\cdfmm_fortran.f90. The CUDA architectures
rem are those of the local GPU; set CDFMM_CUDA_ARCHITECTURES to a list such as
rem 75;80;86;89;90;120 for a library that has to run on other machines.
setlocal
call "%~dp0env.cmd"
if errorlevel 1 exit /b 1
@echo off

for %%i in ("%~dp0..\..") do set "SRC=%%~fi"
set "BLD=%SRC%\build-release"
set "PREFIX=%SRC%\local"
set "FRESH="
set "CDFMM_MKL_LINK=dynamic"
for %%a in (%*) do (
  if /i "%%a"=="fresh" set "FRESH=--fresh"
  if /i "%%a"=="static" set "CDFMM_MKL_LINK=static"
)
if not defined CDFMM_CUDA_ARCHITECTURES set "CDFMM_CUDA_ARCHITECTURES=native"

echo === configure (oneMKL %CDFMM_MKL_LINK%, CUDA architectures %CDFMM_CUDA_ARCHITECTURES%)
rem MKL_SYCL_LINK=none: MKLConfig.cmake sees icx and would otherwise ask for
rem the SYCL libraries of oneMKL, which a static link does not find.
cmake %FRESH% -S "%SRC%" -B "%BLD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_CUDA_COMPILER="%CDFMM_ENV%\Library\bin\nvcc.exe" ^
  -DCMAKE_CUDA_ARCHITECTURES="%CDFMM_CUDA_ARCHITECTURES%" ^
  -DCUDAToolkit_ROOT="%CDFMM_ENV%\Library" ^
  -DCDFMM_ENABLE_CUDA=ON ^
  -DCDFMM_PROCEDURAL_MAX_ORDER=10 ^
  -DCDFMM_EXTRA_COMPILE_OPTIONS=-fp:precise ^
  -DCDFMM_ENABLE_MKL=ON ^
  -DMKL_DIR="%CDFMM_ENV%\Library\lib\cmake\mkl" ^
  -DMKL_INTERFACE=lp64 ^
  -DMKL_THREADING=intel_thread ^
  -DMKL_LINK=%CDFMM_MKL_LINK% ^
  -DMKL_SYCL_LINK=none ^
  -DCDFMM_ENABLE_OPENMP=ON ^
  -DCDFMM_INSTALL_PYTHON_TO_ENV=OFF ^
  -DCDFMM_BUILD_PYTHON=OFF ^
  -DCDFMM_BUILD_FORTRAN_INTERFACE=OFF ^
  -DCDFMM_BUILD_TESTS=OFF ^
  -DCDFMM_BUILD_EXAMPLES=OFF ^
  -DCDFMM_BUILD_BENCHMARKS=OFF
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 1)

echo === build
cmake --build "%BLD%" --parallel
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)

echo === install
cmake --install "%BLD%"
if errorlevel 1 (echo INSTALL_FAILED & exit /b 1)
echo DIPFMM_BUILD_OK
