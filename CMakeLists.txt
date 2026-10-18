cmake_minimum_required(VERSION 3.18)
project(micromag LANGUAGES CXX)

option(MICROMAG_CUDA "Build the CUDA backend" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Must come before any target is created: targets copy CMAKE_CUDA_ARCHITECTURES at creation time.
# D25: V100 (70) where the toolkit supports it; CUDA 13 dropped Volta, so there the oldest it has, 75 (T4).
# The PTX embedded with either runs on any newer GPU. Override with -DCMAKE_CUDA_ARCHITECTURES=... or CUDAARCHS.
if(MICROMAG_CUDA)
  set(user_archs "${CMAKE_CUDA_ARCHITECTURES}$ENV{CUDAARCHS}")  # read before enable_language fills in its own default
  enable_language(CUDA)
  if(NOT user_archs)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
      set(CMAKE_CUDA_ARCHITECTURES 75)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70)
    endif()
  endif()
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# D15: header-only pocketfft for the CPU FFT, pinned by hash; single-threaded so the oracle is reproducible
include(FetchContent)
FetchContent_Declare(pocketfft
  URL https://github.com/mreineck/pocketfft/archive/c90e55b3d529f8efa40ed01a20de22405f45fc65.tar.gz
  URL_HASH SHA256=2d90016da31a8cc1dd0ac0c89c1f2fb55df8f4ccc3f0386dcfbad09ac6d1c3ba)
FetchContent_MakeAvailable(pocketfft)  # no CMakeLists.txt in the archive, so this only downloads

nanobind_add_module(_core src/bindings.cpp src/fields.cpp src/demag.cpp src/problem.cpp src/llg.cpp src/minimizer.cpp)
target_include_directories(_core PRIVATE ${pocketfft_SOURCE_DIR})
target_compile_definitions(_core PRIVATE POCKETFFT_NO_MULTITHREADING)

# D24/D25: the CUDA sources build as their own static library, so nanobind's host compiler flags never reach nvcc.
if(MICROMAG_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(micromag_cuda STATIC src/scale.cu src/cuda_backend.cu)
  set_target_properties(micromag_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(micromag_cuda PRIVATE MICROMAG_CUDA)
  target_link_libraries(micromag_cuda PRIVATE CUDA::cudart_static CUDA::cufft)
  target_compile_definitions(_core PRIVATE MICROMAG_CUDA)
  target_link_libraries(_core PRIVATE micromag_cuda)
endif()

install(TARGETS _core LIBRARY DESTINATION micromag)
