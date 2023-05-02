@PACKAGE_INIT@

cmake_minimum_required(VERSION 3.16)

if(equistore_FOUND)
    return()
endif()

enable_language(CXX)

if (WIN32)
    set(EQUISTORE_SHARED_LOCATION ${PACKAGE_PREFIX_DIR}/@BIN_INSTALL_DIR@/@EQUISTORE_SHARED_LIB_NAME@)
    set(EQUISTORE_IMPLIB_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@EQUISTORE_IMPLIB_NAME@)
else()
    set(EQUISTORE_SHARED_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@EQUISTORE_SHARED_LIB_NAME@)
endif()

set(EQUISTORE_STATIC_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@EQUISTORE_STATIC_LIB_NAME@)
set(EQUISTORE_INCLUDE ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/)

if (NOT EXISTS ${EQUISTORE_INCLUDE}/equistore.h OR NOT EXISTS ${EQUISTORE_INCLUDE}/equistore.hpp)
    message(FATAL_ERROR "could not find equistore headers in '${EQUISTORE_INCLUDE}', please re-install equistore")
endif()


# Shared library target
if (@EQUISTORE_INSTALL_BOTH_STATIC_SHARED@ OR @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${EQUISTORE_SHARED_LOCATION})
        message(FATAL_ERROR "could not find equistore library at '${EQUISTORE_SHARED_LOCATION}', please re-install equistore")
    endif()

    add_library(equistore::shared SHARED IMPORTED GLOBAL)
    set_target_properties(equistore::shared PROPERTIES
        IMPORTED_LOCATION ${EQUISTORE_SHARED_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${EQUISTORE_INCLUDE}
    )

    target_compile_features(equistore::shared INTERFACE cxx_std_11)

    if (WIN32)
        if (NOT EXISTS ${EQUISTORE_IMPLIB_LOCATION})
            message(FATAL_ERROR "could not find equistore library at '${EQUISTORE_IMPLIB_LOCATION}', please re-install equistore")
        endif()

        set_target_properties(equistore::shared PROPERTIES
            IMPORTED_IMPLIB ${EQUISTORE_IMPLIB_LOCATION}
        )
    endif()
endif()


# Static library target
if (@EQUISTORE_INSTALL_BOTH_STATIC_SHARED@ OR NOT @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${EQUISTORE_STATIC_LOCATION})
        message(FATAL_ERROR "could not find equistore library at '${EQUISTORE_STATIC_LOCATION}', please re-install equistore")
    endif()

    add_library(equistore::static STATIC IMPORTED GLOBAL)
    set_target_properties(equistore::static PROPERTIES
        IMPORTED_LOCATION ${EQUISTORE_STATIC_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${EQUISTORE_INCLUDE}
        INTERFACE_LINK_LIBRARIES "@CARGO_DEFAULT_LIBRARIES@"
    )

    target_compile_features(equistore::static INTERFACE cxx_std_11)
endif()


# Export either the shared or static library as the equistore target
if (@BUILD_SHARED_LIBS@)
    add_library(equistore ALIAS equistore::shared)
else()
    add_library(equistore ALIAS equistore::static)
endif()
