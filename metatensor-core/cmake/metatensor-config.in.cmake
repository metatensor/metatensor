@PACKAGE_INIT@

cmake_minimum_required(VERSION 3.16)

include(FindPackageHandleStandardArgs)

if(metatensor_FOUND)
    return()
endif()

enable_language(CXX)

get_filename_component(METATENSOR_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/@PACKAGE_RELATIVE_PATH@" ABSOLUTE)

if (WIN32)
    set(METATENSOR_SHARED_LOCATION ${METATENSOR_PREFIX_DIR}/@CMAKE_INSTALL_BINDIR@/@METATENSOR_SHARED_LIB_NAME@)
    set(METATENSOR_IMPLIB_LOCATION ${METATENSOR_PREFIX_DIR}/@CMAKE_INSTALL_LIBDIR@/@METATENSOR_IMPLIB_NAME@)
else()
    set(METATENSOR_SHARED_LOCATION ${METATENSOR_PREFIX_DIR}/@CMAKE_INSTALL_LIBDIR@/@METATENSOR_SHARED_LIB_NAME@)
endif()

set(METATENSOR_STATIC_LOCATION ${METATENSOR_PREFIX_DIR}/@CMAKE_INSTALL_LIBDIR@/@METATENSOR_STATIC_LIB_NAME@)
set(METATENSOR_INCLUDE ${METATENSOR_PREFIX_DIR}/@CMAKE_INSTALL_INCLUDEDIR@/)

if (NOT EXISTS ${METATENSOR_INCLUDE}/metatensor.h OR NOT EXISTS ${METATENSOR_INCLUDE}/metatensor.hpp)
    message(FATAL_ERROR "could not find metatensor headers in '${METATENSOR_INCLUDE}', please re-install metatensor")
endif()


# Shared library target
if (@METATENSOR_INSTALL_BOTH_STATIC_SHARED@ OR @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${METATENSOR_SHARED_LOCATION})
        message(FATAL_ERROR "could not find metatensor library at '${METATENSOR_SHARED_LOCATION}', please re-install metatensor")
    endif()

    add_library(metatensor::shared SHARED IMPORTED)
    set_target_properties(metatensor::shared PROPERTIES
        IMPORTED_LOCATION ${METATENSOR_SHARED_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${METATENSOR_INCLUDE}
        BUILD_VERSION "@METATENSOR_FULL_VERSION@"
    )

    target_compile_features(metatensor::shared INTERFACE cxx_std_17)

    if (WIN32)
        if (NOT EXISTS ${METATENSOR_IMPLIB_LOCATION})
            message(FATAL_ERROR "could not find metatensor library at '${METATENSOR_IMPLIB_LOCATION}', please re-install metatensor")
        endif()

        set_target_properties(metatensor::shared PROPERTIES
            IMPORTED_IMPLIB ${METATENSOR_IMPLIB_LOCATION}
        )
    endif()
endif()


# Static library target
if (@METATENSOR_INSTALL_BOTH_STATIC_SHARED@ OR NOT @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${METATENSOR_STATIC_LOCATION})
        message(FATAL_ERROR "could not find metatensor library at '${METATENSOR_STATIC_LOCATION}', please re-install metatensor")
    endif()

    add_library(metatensor::static STATIC IMPORTED)
    set_target_properties(metatensor::static PROPERTIES
        IMPORTED_LOCATION ${METATENSOR_STATIC_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${METATENSOR_INCLUDE}
        INTERFACE_LINK_LIBRARIES "@CARGO_DEFAULT_LIBRARIES@"
        BUILD_VERSION "@METATENSOR_FULL_VERSION@"
    )

    target_compile_features(metatensor::static INTERFACE cxx_std_17)
endif()

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
    # Export either the shared or static library as the metatensor target
    # This requires cmake 3.18+ to use ALIAS for non-GLOBAL IMPORTED targets
    if (@BUILD_SHARED_LIBS@)
        add_library(metatensor ALIAS metatensor::shared)
    else()
        add_library(metatensor ALIAS metatensor::static)
    endif()
endif()


if (@BUILD_SHARED_LIBS@)
    find_package_handle_standard_args(metatensor DEFAULT_MSG METATENSOR_SHARED_LOCATION METATENSOR_INCLUDE)
else()
    find_package_handle_standard_args(metatensor DEFAULT_MSG METATENSOR_STATIC_LOCATION METATENSOR_INCLUDE)
endif()
