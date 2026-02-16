# This file allow the python module in metatensor-torch to either use an
# externally-provided version of the shared metatensor_torch library; or to
# build the code from source and bundle the shared library inside the wheel.
#
# The first case is used when distirbuting the code in conda (since we have a
# separate libmetatensor package), the second one is used everywhere else (for
# local development builds and for the PyPI distribution).

cmake_minimum_required(VERSION 3.22)
project(metatensor-torch-python CXX)

option(METATENSOR_TORCH_PYTHON_USE_EXTERNAL_LIB "Force the usage of an external version of metatensor-torch" OFF)
set(METATENSOR_TORCH_SOURCE_DIR "" CACHE PATH "Path to the sources of metatensor-torch")

file(REMOVE ${CMAKE_INSTALL_PREFIX}/_external.py)

set(REQUIRED_METATENSOR_TORCH_VERSION "0.8.4")
if(${METATENSOR_TORCH_PYTHON_USE_EXTERNAL_LIB})
    # when building a source checkout, update version to include git information
    # this will not apply when building a sdist
    if (EXISTS ${CMAKE_SOURCE_DIR}/../../metatensor-torch/cmake/dev-versions.cmake)
        include(${CMAKE_SOURCE_DIR}/../../metatensor-torch/cmake/dev-versions.cmake)
        create_development_version(
            "${REQUIRED_METATENSOR_TORCH_VERSION}"
            REQUIRED_METATENSOR_TORCH_VERSION
            "metatensor-torch-v"
        )
        # strip any -dev/-rc suffix on the version since find_package does not support it
        string(
            REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3"
            REQUIRED_METATENSOR_TORCH_VERSION
            ${REQUIRED_METATENSOR_TORCH_VERSION}
        )
    endif()

    find_package(metatensor_torch ${REQUIRED_METATENSOR_TORCH_VERSION} REQUIRED)

    get_target_property(METATENSOR_TORCH_LOCATION metatensor_torch LOCATION)
    message(STATUS "Using external metatensor-torch v${metatensor_torch_VERSION} at ${METATENSOR_TORCH_LOCATION}")

    # Get the prefix to use as cmake_prefix_path when trying to load this
    # version of the library again
    get_filename_component(METATENSOR_TORCH_PREFIX "${METATENSOR_TORCH_LOCATION}" DIRECTORY)
    get_filename_component(METATENSOR_TORCH_PREFIX "${METATENSOR_TORCH_PREFIX}" DIRECTORY)

    file(WRITE ${CMAKE_INSTALL_PREFIX}/_external.py
        "EXTERNAL_METATENSOR_TORCH_PATH = \"${METATENSOR_TORCH_LOCATION}\"\n\n"
    )
    file(APPEND ${CMAKE_INSTALL_PREFIX}/_external.py
        "EXTERNAL_METATENSOR_TORCH_PREFIX = \"${METATENSOR_TORCH_PREFIX}\"\n"
    )

    install(CODE "message(STATUS \"nothing to install\")")
else()
    if ("${METATENSOR_TORCH_SOURCE_DIR}" STREQUAL "")
        message(FATAL_ERROR
            "Missing METATENSOR_TORCH_SOURCE_DIR, please specify where to \
            find the source code for metatensor-torch"
        )
    endif()

    message(STATUS "Using internal metatensor-torch from ${METATENSOR_TORCH_SOURCE_DIR}")

    add_subdirectory("${METATENSOR_TORCH_SOURCE_DIR}" metatensor-torch)

    if (LINUX)
        set_target_properties(
            # when loading the libraries from a Python installation,
            # $ORIGIN/../../../lib is where libmetatensor.so will be, and
            # $ORIGIN/../../../../torch/lib is where libtorch.so will be.
            metatensor_torch PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../lib;$ORIGIN/../../../../torch/lib"
        )
    elseif(APPLE)
        set_target_properties(
            # same as above, but `$ORIGIN` is `@loader_path` on macOS
            metatensor_torch PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH};@loader_path/../../../lib;@loader_path/../../../../torch/lib"
        )
    endif()
endif()
