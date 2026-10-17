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

nanobind_add_module(_core src/bindings.cpp)

if(MICROMAG_CUDA)
  target_sources(_core PRIVATE src/scale.cu)
  target_compile_definitions(_core PRIVATE MICROMAG_CUDA)
endif()

install(TARGETS _core LIBRARY DESTINATION micromag)
