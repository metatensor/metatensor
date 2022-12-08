# This file defines the CMake build system for the C and C++ API of equistore.
#
# This API is implemented in Rust, in the equistore-core crate, but Rust users
# of the API should use the equistore crate instead, wrapping equistore-core in
# an easier to use, idiomatic Rust API.

cmake_minimum_required(VERSION 3.10)

if (POLICY CMP0077)
    # use variables to set OPTIONS
    cmake_policy(SET CMP0077 NEW)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "" FORCE)

file(STRINGS "Cargo.toml" CARGO_TOML_CONTENT)
foreach(line ${CARGO_TOML_CONTENT})
    string(REGEX REPLACE "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1" EQUISTORE_VERSION ${line})
    if (NOT ${CMAKE_MATCH_COUNT} EQUAL 0)
        # stop on the first regex match, this should be the right version
        break()
    endif()
endforeach()

project(equistore
    VERSION ${EQUISTORE_VERSION}
    LANGUAGES C # we need to declare a language to access CMAKE_SIZEOF_VOID_P later
)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")
set(RUST_BUILD_TARGET "" CACHE STRING "Cross-compilation target for rust code. Leave empty to build for the host")

set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}")

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode")

# TODO: support multiple configuration generators (MSVC, ...)
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ("${BUILD_TYPE}" STREQUAL "debug")
    set(CARGO_BUILD_ARG "")
    set(CARGO_BUILD_TYPE "debug")
elseif("${BUILD_TYPE}" STREQUAL "release")
    set(CARGO_BUILD_ARG "--release")
    set(CARGO_BUILD_TYPE "release")
elseif("${BUILD_TYPE}" STREQUAL "relwithdebinfo")
    set(CARGO_BUILD_ARG "--release")
    set(CARGO_BUILD_TYPE "release")
else()
    message(FATAL_ERROR "unsuported build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/target)
set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target-dir=${CARGO_TARGET_DIR}")
# Handle cross compilation with RUST_BUILD_TARGET
if ("${RUST_BUILD_TARGET}" STREQUAL "")
    set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${CARGO_BUILD_TYPE}")
else()
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target=${RUST_BUILD_TARGET}")
    set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${RUST_BUILD_TARGET}/${CARGO_BUILD_TYPE}")
endif()

find_program(CARGO_EXE "cargo" DOC "path to cargo (Rust build system)")
if (NOT CARGO_EXE)
    message(FATAL_ERROR
        "could not find cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install) or set CARGO_EXE"
    )
endif()

execute_process(
    COMMAND ${CARGO_EXE} "--version"
    RESULT_VARIABLE CARGO_STATUS
    OUTPUT_VARIABLE CARGO_VERSION
)

if(CARGO_STATUS AND NOT CARGO_STATUS EQUAL 0)
    message(FATAL_ERROR
        "could not run cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install)"
    )
endif()

set(REQUIRED_RUST_VERSION "1.53.0")
string(REGEX REPLACE "cargo ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" CARGO_VERSION ${CARGO_VERSION})
if (${CARGO_VERSION} VERSION_LESS ${REQUIRED_RUST_VERSION})
    message(FATAL_ERROR
        "your Rust installation is too old (you have version ${CARGO_VERSION}), \
        at least ${REQUIRED_RUST_VERSION} is required"
    )
else()
    message(STATUS "Using cargo version ${CARGO_VERSION} at ${CARGO_EXE}")
endif()

file(GLOB_RECURSE ALL_RUST_SOURCES
    ${PROJECT_SOURCE_DIR}/Cargo.toml
    ${PROJECT_SOURCE_DIR}/src/**.rs
)


add_library(equistore SHARED IMPORTED GLOBAL)
set(EQUISTORE_SHARED_CORE_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}equistore_core${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(EQUISTORE_SHARED_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}equistore${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_library(equistore::static STATIC IMPORTED GLOBAL)
set(EQUISTORE_STATIC_CORE_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}equistore_core${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(EQUISTORE_STATIC_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}equistore${CMAKE_STATIC_LIBRARY_SUFFIX}")

get_filename_component(EQUISTORE_SHARED_DIRECTORY ${EQUISTORE_SHARED_CORE_LOCATION} DIRECTORY)
set(EQUISTORE_SHARED_LOCATION "${EQUISTORE_SHARED_DIRECTORY}/${EQUISTORE_SHARED_LIB_NAME}")

get_filename_component(EQUISTORE_STATIC_DIRECTORY ${EQUISTORE_STATIC_CORE_LOCATION} DIRECTORY)
set(EQUISTORE_STATIC_LOCATION "${EQUISTORE_STATIC_DIRECTORY}/${EQUISTORE_STATIC_LIB_NAME}")

# We need to add some metadata to the shared library to enable linking to it
# without using an absolute path.
if (UNIX)
    if (APPLE)
        # set the install name to `@rpath/libequistore.dylib`
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-install_name,@rpath/${EQUISTORE_SHARED_LIB_NAME}")
    else() # LINUX
        # set the SONAME to libequistore.so
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-soname,${EQUISTORE_SHARED_LIB_NAME}")
    endif()
else()
    set(CARGO_RUSTC_ARGS "")
endif()

if (EQUISTORE_BUILD_FOR_PYTHON AND ${CARGO_VERSION} VERSION_GREATER "1.59")
    # strip dynamic library for smaller wheels to download/install
    set(CARGO_RUSTC_ARGS "${CARGO_RUSTC_ARGS};-Cstrip=symbols")
endif()

add_custom_target(cargo-build-equistore ALL
    COMMAND
        ${CMAKE_COMMAND} -E env
        MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
        cargo rustc ${CARGO_BUILD_ARG} -- ${CARGO_RUSTC_ARGS}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${EQUISTORE_LIB_NAME} with cargo"
)

add_custom_command(TARGET cargo-build-equistore
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${EQUISTORE_SHARED_CORE_LOCATION} ${EQUISTORE_SHARED_LOCATION}
    COMMAND ${CMAKE_COMMAND} -E copy ${EQUISTORE_STATIC_CORE_LOCATION} ${EQUISTORE_STATIC_LOCATION}
    DEPENDS cargo-build-equistore
)

add_dependencies(equistore cargo-build-equistore)
add_dependencies(equistore::static cargo-build-equistore)
set(EQUISTORE_HEADERS
    "${PROJECT_SOURCE_DIR}/include/equistore.h"
    "${PROJECT_SOURCE_DIR}/include/equistore.hpp"
)
set(EQUISTORE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)

set_target_properties(equistore PROPERTIES
    IMPORTED_LOCATION ${EQUISTORE_SHARED_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${EQUISTORE_INCLUDE_DIR}
)

set_target_properties(equistore::static PROPERTIES
    IMPORTED_LOCATION ${EQUISTORE_STATIC_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${EQUISTORE_INCLUDE_DIR}
)

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/equistore-config.in.cmake"
    "${PROJECT_BINARY_DIR}/equistore-config.cmake"
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/equistore
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/equistore-config-version.in.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/equistore-config-version.cmake"
    @ONLY
)

install(FILES ${EQUISTORE_HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})
install(FILES ${EQUISTORE_SHARED_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
install(FILES ${EQUISTORE_STATIC_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
install(FILES
    ${PROJECT_BINARY_DIR}/equistore-config-version.cmake
    ${PROJECT_BINARY_DIR}/equistore-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/equistore
)
