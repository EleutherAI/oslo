cmake_minimum_required(VERSION 3.10)

project(LightseqProtoType LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_library(
  lsflow STATIC
  context.cpp
  node.cpp
  manager.cpp
  layer.cpp
  tensor.cpp
  allocator.cpp
  lsflow_util.cpp
  operator.cpp
  shape.cpp
  variable.cpp)

target_link_libraries(lsflow PUBLIC lightseq_kernels)
