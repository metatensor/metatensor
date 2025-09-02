# This file allow the python module in metatensor-core to either use an
# externally-provided version of the shared metatensor library; or to build the
# code from source and bundle the shared library inside the wheel.
#
# The first case is used when distirbuting the code in conda (since we have a
# separate libmetatensor package), the second one is used everywhere else (for
# local development builds and for the PyPI distribution).

cmake_minimum_required(VERSION 3.16)
project(metatensor-python NONE)

option(METATENSOR_CORE_PYTHON_USE_EXTERNAL_LIB "Force the usage of an external version of metatensor-core" OFF)
set(METATENSOR_CORE_SOURCE_DIR "" CACHE PATH "Path to the sources of metatensor-core")

file(REMOVE ${CMAKE_INSTALL_PREFIX}/_external.py)

set(REQUIRED_METATENSOR_VERSION "0.1.17")
if(${METATENSOR_CORE_PYTHON_USE_EXTERNAL_LIB})
    # when building a source checkout, update version to include git information
    # this will not apply when building a sdist
    if (EXISTS ${CMAKE_SOURCE_DIR}/../../metatensor-core/cmake/dev-versions.cmake)
        include(${CMAKE_SOURCE_DIR}/../../metatensor-core/cmake/dev-versions.cmake)
        create_development_version("${REQUIRED_METATENSOR_VERSION}" REQUIRED_METATENSOR_VERSION "metatensor-core-v")
        # strip any -dev/-rc suffix on the version since find_package does not support it
        string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" REQUIRED_METATENSOR_VERSION ${REQUIRED_METATENSOR_VERSION})
    endif()

    find_package(metatensor ${REQUIRED_METATENSOR_VERSION} REQUIRED)

    get_target_property(METATENSOR_LOCATION metatensor::shared LOCATION)
    message(STATUS "Using external metatensor-core v${metatensor_VERSION} at ${METATENSOR_LOCATION}")

    # Get the prefix to use as cmake_prefix_path when trying to load this
    # version of the library again
    get_filename_component(METATENSOR_PREFIX "${METATENSOR_LOCATION}" DIRECTORY)
    get_filename_component(METATENSOR_PREFIX "${METATENSOR_PREFIX}" DIRECTORY)

    file(WRITE ${CMAKE_INSTALL_PREFIX}/_external.py
        "EXTERNAL_METATENSOR_PATH = \"${METATENSOR_LOCATION}\"\n\n"
    )
    file(APPEND ${CMAKE_INSTALL_PREFIX}/_external.py
        "EXTERNAL_METATENSOR_PREFIX = \"${METATENSOR_PREFIX}\"\n"
    )

    install(CODE "message(STATUS \"nothing to install\")")
else()
    if ("${METATENSOR_CORE_SOURCE_DIR}" STREQUAL "")
        message(FATAL_ERROR
            "Missing METATENSOR_CORE_SOURCE_DIR, please specify where to \
            find the source code for metatensor-core"
        )
    endif()

    message(STATUS "Using internal metatensor-core from ${METATENSOR_CORE_SOURCE_DIR}")

    set(BUILD_SHARED_LIBS ON)
    set(METATENSOR_INSTALL_BOTH_STATIC_SHARED OFF)
    # strip dynamic library for smaller wheels to download/install
    set(EXTRA_RUST_FLAGS "-Cstrip=symbols")

    add_subdirectory("${METATENSOR_CORE_SOURCE_DIR}" metatensor-core)
endif()
