cmake_minimum_required(VERSION 3.16)

# This file is here to serve as a minimal CMakeLists.txt at the root of the
# metatensor repository, allowing to easily include it inside other projects as
# a git submodule (or more generally as a CMake subdirectory).
#
# metatensor-core is always built, and additional options control building the
# other languages bindings:
#
# - `BUILD_METATENSOR_TORCH` to build the TorchScript bindings

project(metatensor-meta-project)

add_subdirectory(metatensor-core)

option(BUILD_METATENSOR_TORCH "Build the TorchScript bindings to metatensor" OFF)
if (BUILD_METATENSOR_TORCH)
    add_subdirectory(metatensor-torch)
endif()
