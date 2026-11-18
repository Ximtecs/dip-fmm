@echo off
rem Build environment for dip-fmm on Windows.
rem
rem   call tools\windows\env.cmd
rem
rem Needs a conda environment with the Intel compilers (icx, and ifx for the
rem Fortran wrapper), oneMKL, the CUDA toolkit (nvcc), CMake and Ninja, and
rem Visual Studio 2022 with the C++ workload: nvcc uses MSVC cl as its host
rem compiler and icx links with the MSVC linker.
rem
rem The active conda environment is used when it holds icx. Otherwise
rem %CDFMM_CONDA_ENV% (default: cdfmm) is activated, which needs conda on PATH.
rem CDFMM_ENV is set to the prefix of the environment. A project that builds
rem dip-fmm as part of its own build can call this script for the same
rem environment, after setting CDFMM_CONDA_ENV to the name of its own.
rem
rem conda is activated BEFORE vcvars64, so that MSVC's link.exe comes ahead of
rem the msys coreutils link.exe that a conda environment can carry.
if not defined CDFMM_CONDA_ENV set "CDFMM_CONDA_ENV=cdfmm"
if not exist "%CONDA_PREFIX%\Library\bin\icx.exe" call conda activate %CDFMM_CONDA_ENV%
if not exist "%CONDA_PREFIX%\Library\bin\icx.exe" (
  echo env.cmd: no icx.exe in the conda environment "%CONDA_PREFIX%"
  exit /b 1
)
@echo off
rem (the activation scripts of a conda environment turn command echo back on)
set "CDFMM_ENV=%CONDA_PREFIX%"

rem Visual Studio 2022 (version 17) by name: a newer Visual Studio next to it
rem brings an MSVC that the CUDA toolkit of the environment may not support.
set "CDFMM_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "CDFMM_VSROOT="
for /f "usebackq delims=" %%i in (`"%CDFMM_VSWHERE%" -latest -products * -version [17.0^,18.0^) -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "CDFMM_VSROOT=%%i"
if not defined CDFMM_VSROOT (
  echo env.cmd: no Visual Studio 2022 installation with the C++ tools was found
  exit /b 1
)
call "%CDFMM_VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul

rem A pip-installed ninja lives in Scripts. Do NOT prepend Library\bin here:
rem the msys shell of a conda make maps that directory onto /bin, which is
rem bound to Library\usr\bin, and the msys link.exe there would then shadow
rem MSVC's ("link: unknown option -- d"). The activation above already has
rem Library\bin on PATH behind the MSVC directories.
set "PATH=%CDFMM_ENV%\Scripts;%PATH%"

rem The conda nvcc activation leaves CUDA_PATH pointing at whichever standalone
rem toolkit was installed last; the build has to use the environment's own.
set "CUDA_PATH=%CDFMM_ENV%\Library"
set "NVCC_PREPEND_FLAGS="
rem conda's vs2022 activation exports CC=CXX=cl.exe and a Visual Studio generator.
set "CC="
set "CXX="
set "FC="
set "CMAKE_GENERATOR="
set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="
exit /b 0
