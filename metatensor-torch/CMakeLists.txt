cmake_minimum_required(VERSION 3.16)

if (POLICY CMP0135)
    # Use download time as timestamp when extracting files from archives
    cmake_policy(SET CMP0135 NEW)
endif()


if(NOT "${LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    set(LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
        message(STATUS "Running CMake version ${CMAKE_VERSION}")
    endif()
endif()


file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION METATENSOR_TORCH_VERSION)
string(STRIP ${METATENSOR_TORCH_VERSION} METATENSOR_TORCH_VERSION)

include(cmake/dev-versions.cmake)
create_development_version("${METATENSOR_TORCH_VERSION}" METATENSOR_TORCH_FULL_VERSION "metatensor-torch-v")
message(STATUS "Building metatensor-torch v${METATENSOR_TORCH_FULL_VERSION}")

# strip any -dev/-rc suffix on the version since project(VERSION) does not support it
string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" METATENSOR_TORCH_VERSION ${METATENSOR_TORCH_FULL_VERSION})
project(metatensor_torch
    VERSION ${METATENSOR_TORCH_VERSION}
    LANGUAGES CXX
)
set(PROJECT_VERSION ${METATENSOR_TORCH_FULL_VERSION})

option(METATENSOR_TORCH_TESTS "Build metatensor-torch C++ tests" OFF)
include(GNUInstallDirs)

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

function(check_compatible_versions _actual_ _requested_)
    if(${_actual_} MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_actual_major_ "${CMAKE_MATCH_1}")
        set(_actual_minor_ "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "Failed to parse actual version: ${_actual_}")
    endif()

    if(${_requested_} MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_requested_major_ "${CMAKE_MATCH_1}")
        set(_requested_minor_ "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "Failed to parse requested version: ${_requested_}")
    endif()

    if (${_requested_major_} EQUAL 0 AND ${_actual_minor_} EQUAL ${_requested_minor_})
        # major version is 0 and same minor version, everything is fine
    elseif (${_actual_major_} EQUAL ${_requested_major_})
        # same major version, everything is fine
    else()
        # not compatible
        message(FATAL_ERROR "Incompatible versions: we need ${_requested_}, but we got ${_actual_}")
    endif()
endfunction()


set(REQUIRED_METATENSOR_VERSION "0.1.14")
if (NOT "$ENV{METATENSOR_NO_LOCAL_DEPS}" STREQUAL "1")
    # If building a dev version, we also need to update the
    # REQUIRED_METATENSOR_VERSION in the same way we update the metatensor-torch
    # version
    create_development_version("${REQUIRED_METATENSOR_VERSION}" METATENSOR_CORE_FULL_VERSION "metatensor-core-v")
else()
    set(METATENSOR_CORE_FULL_VERSION ${REQUIRED_METATENSOR_VERSION})
endif()
string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_METATENSOR_VERSION ${METATENSOR_CORE_FULL_VERSION})

# Either metatensor is built as part of the same CMake project, or we try to
# find the corresponding CMake package
if (TARGET metatensor::shared)
    get_target_property(METATENSOR_BUILD_VERSION metatensor::shared BUILD_VERSION)
    check_compatible_versions(${METATENSOR_BUILD_VERSION} ${REQUIRED_METATENSOR_VERSION})
else()
    find_package(metatensor ${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)
endif()

# FindCUDNN.cmake distributed with PyTorch is a bit broken, so we have a
# fixed version in `cmake/FindCUDNN.cmake`
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(Torch 2.1 REQUIRED)

set(METATENSOR_TORCH_HEADERS
    "include/metatensor/torch/array.hpp"
    "include/metatensor/torch/labels.hpp"
    "include/metatensor/torch/block.hpp"
    "include/metatensor/torch/tensor.hpp"
    "include/metatensor/torch/module.hpp"
    "include/metatensor/torch.hpp"
)

set(METATENSOR_TORCH_SOURCES
    "src/array.cpp"
    "src/labels.cpp"
    "src/block.cpp"
    "src/tensor.cpp"
    "src/misc.cpp"
    "src/module.cpp"
    "src/register.cpp"
)

add_library(metatensor_torch SHARED
    ${METATENSOR_TORCH_HEADERS}
    ${METATENSOR_TORCH_SOURCES}
)

set_target_properties(metatensor_torch PROPERTIES
    BUILD_VERSION ${METATENSOR_TORCH_FULL_VERSION}
)

target_link_libraries(metatensor_torch PUBLIC torch metatensor::shared ${CMAKE_DL_LIBS})
target_compile_features(metatensor_torch PUBLIC cxx_std_17)
target_include_directories(metatensor_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Create a header defining METATENSOR_TORCH_EXPORT for to export classes/functions
# in DLL on Windows.
set_target_properties(metatensor_torch PROPERTIES
    # hide non-exported symbols by default, this mimics Windows behavior on Unix
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

include(GenerateExportHeader)
generate_export_header(metatensor_torch
    BASE_NAME METATENSOR_TORCH
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/metatensor/torch/exports.h
)
target_compile_definitions(metatensor_torch PRIVATE metatensor_torch_EXPORTS)


set(_path_ "${CMAKE_CURRENT_BINARY_DIR}/generated-version.h")
file(WRITE ${_path_} "/** Full version of metatensor-torch as a string */\n")
file(APPEND ${_path_} "#define METATENSOR_TORCH_VERSION \"${METATENSOR_TORCH_FULL_VERSION}\"\n\n")
file(APPEND ${_path_} "/** Major version number of metatensor-torch as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_TORCH_VERSION_MAJOR ${PROJECT_VERSION_MAJOR}\n\n")
file(APPEND ${_path_} "/** Minor version number of metatensor-torch as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_TORCH_VERSION_MINOR ${PROJECT_VERSION_MINOR}\n\n")
file(APPEND ${_path_} "/** Patch version number of metatensor-torch as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_TORCH_VERSION_PATCH ${PROJECT_VERSION_PATCH}\n")

function(copy_if_different _src_ _dst_)
    # file(COPY_FILE ...) was added in cmake 3.21, this emulates the same
    # behavior on older versions
    get_filename_component(_dst_dir_ ${_dst_} DIRECTORY)
    get_filename_component(_src_name_ ${_src_} NAME)

    if (EXISTS ${_dst_})
        file(SHA1 ${_src_} new_hash)
        file(SHA1 ${_dst_} old_hash)

        if ("${new_hash}" STREQUAL "${old_hash}")
            set(_do_copy_ FALSE)
        else()
            set(_do_copy_ TRUE)
        endif()
    else()
        # The destination does not exist
        set(_do_copy_ TRUE)
    endif()

    if (${_do_copy_})
        file(COPY ${_src_} DESTINATION ${_dst_dir_})
        file(RENAME ${_dst_dir_}/${_src_name_} ${_dst_})
    endif()
endfunction()

set(_destination_ "${CMAKE_CURRENT_BINARY_DIR}/include/metatensor/torch/version.h")
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.21")
    file(COPY_FILE ${_path_} ${_destination_} ONLY_IF_DIFFERENT)
else()
    copy_if_different(${_path_} ${_destination_})
endif()

if (METATENSOR_TORCH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    metatensor_torch-config-version.cmake
    VERSION ${METATENSOR_TORCH_FULL_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(TARGETS metatensor_torch
    EXPORT metatensor_torch-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(EXPORT metatensor_torch-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metatensor_torch
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/metatensor_torch-config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_torch-config.cmake
    @ONLY
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_torch-config-version.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_torch-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metatensor_torch
)

install(DIRECTORY "include/metatensor" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/metatensor DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
