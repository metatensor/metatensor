# This file defines the CMake build system for the C and C++ API of equistore.
#
# This API is implemented in Rust, in the equistore-core crate, but Rust users
# of the API should use the equistore crate instead, wrapping equistore-core in
# an easier to use, idiomatic Rust API.
cmake_minimum_required(VERSION 3.16)

# Is equistore the main project configured by the user? Or is this being used
# as a submodule/subdirectory?
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(EQUISTORE_MAIN_PROJECT ON)
else()
    set(EQUISTORE_MAIN_PROJECT OFF)
endif()

if(${EQUISTORE_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    # We use CACHED_LAST_CMAKE_VERSION to only print the cmake version
    # once in the configuration log
    set(CACHED_LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    message(STATUS "Running CMake version ${CMAKE_VERSION}")
endif()

if (POLICY CMP0077)
    # use variables to set OPTIONS
    cmake_policy(SET CMP0077 NEW)
endif()

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
    LANGUAGES C CXX # we need to declare a language to access CMAKE_SIZEOF_VOID_P later
)

# We follow the standard CMake convention of using BUILD_SHARED_LIBS to provide
# either a shared or static library as a default target. But since cargo always
# builds both versions by default, we also install both versions by default.
# `EQUISTORE_INSTALL_BOTH_STATIC_SHARED=OFF` allow to disable this behavior, and
# only install the file corresponding to `BUILD_SHARED_LIBS=ON/OFF`.
#
# BUILD_SHARED_LIBS controls the `equistore` cmake target, making it an alias of
# either `equistore::static` or `equistore::shared`. This is mainly relevant
# when using equistore from another cmake project, either as a submodule or from
# an installed library (see cmake/equistore-config.cmake)
option(BUILD_SHARED_LIBS "Use a shared library by default instead of a static one" ON)
option(EQUISTORE_INSTALL_BOTH_STATIC_SHARED "Install both shared and static libraries" ON)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")
set(RUST_BUILD_TARGET "" CACHE STRING "Cross-compilation target for rust code. Leave empty to build for the host")

set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}")

if (${EQUISTORE_MAIN_PROJECT})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

if(${EQUISTORE_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_BUILD_TYPE}" STREQUAL ${CMAKE_BUILD_TYPE})
    set(CACHED_LAST_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Last build type used in configuration")
    message(STATUS "Building equistore in ${CMAKE_BUILD_TYPE} mode")
endif()

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
    if(NOT "${CACHED_LAST_CARGO_VERSION}" STREQUAL ${CARGO_VERSION})
        set(CACHED_LAST_CARGO_VERSION ${CARGO_VERSION} CACHE INTERNAL "Last version of cargo used in configuration")
        message(STATUS "Using cargo version ${CARGO_VERSION} at ${CARGO_EXE}")
    endif()
endif()

file(GLOB_RECURSE ALL_RUST_SOURCES
    ${PROJECT_SOURCE_DIR}/Cargo.toml
    ${PROJECT_SOURCE_DIR}/src/**.rs
)


add_library(equistore::shared SHARED IMPORTED GLOBAL)
set(EQUISTORE_CORE_SHARED_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}equistore_core${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(EQUISTORE_SHARED_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}equistore${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_library(equistore::static STATIC IMPORTED GLOBAL)
set(EQUISTORE_CORE_STATIC_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}equistore_core${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(EQUISTORE_STATIC_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}equistore${CMAKE_STATIC_LIBRARY_SUFFIX}")

get_filename_component(EQUISTORE_SHARED_DIRECTORY ${EQUISTORE_CORE_SHARED_LOCATION} DIRECTORY)
set(EQUISTORE_SHARED_LOCATION "${EQUISTORE_SHARED_DIRECTORY}/${EQUISTORE_SHARED_LIB_NAME}")

get_filename_component(EQUISTORE_STATIC_DIRECTORY ${EQUISTORE_CORE_STATIC_LOCATION} DIRECTORY)
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

# TODO: introduce a generic RUST_FLAGS variable and use that instead
if (EQUISTORE_BUILD_FOR_PYTHON)
    # strip dynamic library for smaller wheels to download/install
    set(CARGO_RUSTC_ARGS "${CARGO_RUSTC_ARGS};-Cstrip=symbols")
endif()

add_custom_target(cargo-build-equistore ALL
    COMMAND
        ${CMAKE_COMMAND} -E env
        MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
        RUSTC_WRAPPER=$ENV{RUSTC_WRAPPER}
        cargo rustc ${CARGO_BUILD_ARG} -- ${CARGO_RUSTC_ARGS}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${EQUISTORE_SHARED_LIB_NAME} and ${EQUISTORE_STATIC_LIB_NAME} with cargo"
)

add_custom_command(TARGET cargo-build-equistore
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${EQUISTORE_CORE_SHARED_LOCATION} ${EQUISTORE_SHARED_LOCATION}
    COMMAND ${CMAKE_COMMAND} -E copy ${EQUISTORE_CORE_STATIC_LOCATION} ${EQUISTORE_STATIC_LOCATION}
    DEPENDS cargo-build-equistore
    BYPRODUCTS ${EQUISTORE_SHARED_LOCATION} ${EQUISTORE_STATIC_LOCATION}
)

add_dependencies(equistore::shared cargo-build-equistore)
add_dependencies(equistore::static cargo-build-equistore)
set(EQUISTORE_HEADERS
    "${PROJECT_SOURCE_DIR}/include/equistore.h"
    "${PROJECT_SOURCE_DIR}/include/equistore.hpp"
)
set(EQUISTORE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)

set_target_properties(equistore::shared PROPERTIES
    IMPORTED_LOCATION ${EQUISTORE_SHARED_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${EQUISTORE_INCLUDE_DIR}
)
target_compile_features(equistore::shared INTERFACE cxx_std_11)

set_target_properties(equistore::static PROPERTIES
    IMPORTED_LOCATION ${EQUISTORE_STATIC_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${EQUISTORE_INCLUDE_DIR}
)
target_compile_features(equistore::static INTERFACE cxx_std_11)

if (BUILD_SHARED_LIBS)
    add_library(equistore ALIAS equistore::shared)
else()
    add_library(equistore ALIAS equistore::static)
endif()

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

if (EQUISTORE_INSTALL_BOTH_STATIC_SHARED OR BUILD_SHARED_LIBS)
    install(FILES ${EQUISTORE_SHARED_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
endif()

if (EQUISTORE_INSTALL_BOTH_STATIC_SHARED OR NOT BUILD_SHARED_LIBS)
    install(FILES ${EQUISTORE_STATIC_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
endif()

install(FILES
    ${PROJECT_BINARY_DIR}/equistore-config-version.cmake
    ${PROJECT_BINARY_DIR}/equistore-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/equistore
)
