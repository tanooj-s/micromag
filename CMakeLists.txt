cmake_minimum_required(VERSION 3.18)
project(micromag LANGUAGES CXX)

option(MICROMAG_CUDA "Build the CUDA backend" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Must come before any target is created: targets copy CMAKE_CUDA_ARCHITECTURES at creation time.
if(MICROMAG_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)  # V100; override with -DCMAKE_CUDA_ARCHITECTURES=... on other GPUs
  endif()
  enable_language(CUDA)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# D15: header-only pocketfft for the CPU FFT, pinned by hash; single-threaded so the oracle is reproducible
include(FetchContent)
FetchContent_Declare(pocketfft
  URL https://github.com/mreineck/pocketfft/archive/c90e55b3d529f8efa40ed01a20de22405f45fc65.tar.gz
  URL_HASH SHA256=2d90016da31a8cc1dd0ac0c89c1f2fb55df8f4ccc3f0386dcfbad09ac6d1c3ba)
FetchContent_MakeAvailable(pocketfft)  # no CMakeLists.txt in the archive, so this only downloads

nanobind_add_module(_core src/bindings.cpp src/fields.cpp src/demag.cpp)
target_include_directories(_core PRIVATE ${pocketfft_SOURCE_DIR})
target_compile_definitions(_core PRIVATE POCKETFFT_NO_MULTITHREADING)

if(MICROMAG_CUDA)
  target_sources(_core PRIVATE src/scale.cu)
  target_compile_definitions(_core PRIVATE MICROMAG_CUDA)
endif()

install(TARGETS _core LIBRARY DESTINATION micromag)
