# Basic CMake integration for rascaline.
cmake_minimum_required(VERSION 3.10)

if (POLICY CMP0077)
    # use variables to set OPTIONS
    cmake_policy(SET CMP0077 NEW)
endif()

file(STRINGS "Cargo.toml" CARGO_TOML_CONTENT)
foreach(line ${CARGO_TOML_CONTENT})
    string(REGEX REPLACE "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1" AML_STORAGE_VERSION ${line})
    if (NOT ${CMAKE_MATCH_COUNT} EQUAL 0)
        # stop on the first regex match, this should be the right version
        break()
    endif()
endforeach()

project(aml_storage
    VERSION ${AML_STORAGE_VERSION}
    LANGUAGES C # we need to declare a language to access CMAKE_SIZEOF_VOID_P later
)

option(BUILD_SHARED_LIBS "Build a shared library instead of a static one" ON)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

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

set(CARGO_TARGET_DIR "${PROJECT_SOURCE_DIR}/target/${CARGO_BUILD_TYPE}")
if(${BUILD_SHARED_LIBS})
    add_library(aml_storage SHARED IMPORTED GLOBAL)
    set(AML_STORAGE_LOCATION "${CARGO_TARGET_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}aml_storage${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    add_library(aml_storage STATIC IMPORTED GLOBAL)
    set(AML_STORAGE_LOCATION "${CARGO_TARGET_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}aml_storage${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

get_filename_component(AML_STORAGE_LIB_NAME ${AML_STORAGE_LOCATION} NAME)
add_custom_target(cargo-build-aml-storage ALL
    COMMAND cargo build ${CARGO_BUILD_ARG}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${AML_STORAGE_LIB_NAME} with cargo"
)


if (APPLE)
    if (AML_STORAGE_BUILD_FOR_PYTHON)
        # set the build id of librascaline.dylib to `@rpath/librascaline.dylib`
        # instead of the full build path (${RASCALINE_LOCATION})
        add_custom_command(TARGET cargo-build-aml-storage POST_BUILD
           COMMAND install_name_tool -id "@rpath/${AML_STORAGE_LIB_NAME}" ${AML_STORAGE_LOCATION}
        )
    endif()
endif()

add_dependencies(aml_storage cargo-build-aml-storage)
set(AML_STORAGE_HEADERS
    "${PROJECT_SOURCE_DIR}/include/aml_storage.h"
)
set(AML_STORAGE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)

set_target_properties(aml_storage PROPERTIES
    IMPORTED_LOCATION ${AML_STORAGE_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${AML_STORAGE_INCLUDE_DIR}
)

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/aml-storage-config.in.cmake"
    "${PROJECT_BINARY_DIR}/aml-storage-config.cmake"
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/aml-storage
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/aml-storage-config-version.in.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/aml-storage-config-version.cmake"
    @ONLY
)

install(FILES ${AML_STORAGE_HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})
install(FILES ${AML_STORAGE_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
install(FILES
    ${PROJECT_BINARY_DIR}/aml-storage-config-version.cmake
    ${PROJECT_BINARY_DIR}/aml-storage-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/aml-storage
)
