cmake_minimum_required(VERSION 3.12...3.18)
project(ehrlich_aberth_jax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories("lib/extern/cub-1.15.0")
include_directories("lib/extern/thrust-1.15.0")

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# CPU op library
pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc)
install(TARGETS cpu_ops DESTINATION ehrlich_aberth_jax)

if (EHRLICH_ABERTH_JAX_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  pybind11_add_module(
    gpu_ops
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cc.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc)
  install(TARGETS gpu_ops DESTINATION ehrlich_aberth_jax)
else()
  message(STATUS "Building without CUDA")
endif()
