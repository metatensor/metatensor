# Torch seems to require features added in CMake 3.15, but does not
# declare it in its own `cmake_minimum_required`
cmake_minimum_required(VERSION 3.15)

if(NOT "${LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    set(LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
        message(STATUS "Running CMake version ${CMAKE_VERSION}")
    endif()
endif()

project(equistore_torch CXX)

option(EQUISTORE_TORCH_TESTS "Build equistore-torch C++ tests" OFF)
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install binaries/DLL")
set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

# Set a default build type if none was specified
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()

find_package(equistore 0.1 CONFIG REQUIRED)

# FindCUDNN.cmake distributed with PyTorch is a bit broken, so we have a
# fixed version in `cmake/FindCUDNN.cmake`
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(Torch 1.11 REQUIRED)

set(EQUISTORE_TORCH_HEADERS
    "include/equistore/torch/array.hpp"
    "include/equistore/torch/labels.hpp"
    "include/equistore/torch/block.hpp"
    "include/equistore/torch/tensor.hpp"
    "include/equistore/torch.hpp"
)

set(EQUISTORE_TORCH_SOURCE
    "src/array.cpp"
    "src/labels.cpp"
    "src/block.cpp"
    "src/tensor.cpp"
    "src/register.cpp"
)

add_library(equistore_torch SHARED
    ${EQUISTORE_TORCH_HEADERS}
    ${EQUISTORE_TORCH_SOURCE}
)


target_link_libraries(equistore_torch PUBLIC torch equistore::shared)
target_compile_features(equistore_torch PUBLIC cxx_std_14)
target_include_directories(equistore_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create a header defining EQUISTORE_TORCH_EXPORT for to export classes/functions
# in DLL on Windows.
set_target_properties(equistore_torch PROPERTIES
    # hide non-exported symbols by default, this mimics Windows behavior on Unix
    CXX_VISIBILITY_PRESET hidden
)

include(GenerateExportHeader)
generate_export_header(equistore_torch
    BASE_NAME EQUISTORE_TORCH
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/equistore/torch/exports.h
)
target_compile_definitions(equistore_torch PRIVATE equistore_torch_EXPORTS)



if (EQUISTORE_TORCH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#

install(TARGETS equistore_torch
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
)

install(DIRECTORY "include" DESTINATION ${INCLUDE_INSTALL_DIR})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include DESTINATION ${INCLUDE_INSTALL_DIR})
