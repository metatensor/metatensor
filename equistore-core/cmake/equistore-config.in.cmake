@PACKAGE_INIT@

cmake_minimum_required(VERSION 3.10)

if(equistore_FOUND)
    return()
endif()

enable_language(CXX)

set(EQUISTORE_SHARED_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@EQUISTORE_SHARED_LIB_NAME@)
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

    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.11)
        # we can not set compile features for imported targets before cmake 3.11
        # users will have to manually request C++11
        target_compile_features(equistore::shared INTERFACE cxx_std_11)
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
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        # the rust standard lib uses pthread and libdl on linux
        target_link_libraries(equistore::static INTERFACE Threads::Threads dl)

        # num_bigint uses fmod
        target_link_libraries(equistore::static INTERFACE m)
    endif()

    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.11)
        # we can not set compile features for imported targets before cmake 3.11
        # users will have to manually request C++11
        target_compile_features(equistore::static INTERFACE cxx_std_11)
    endif()
endif()


# Export either the shared or static library as the equistore target
if (@BUILD_SHARED_LIBS@)
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.11)
        add_library(equistore ALIAS equistore::shared)
    else()
        # CMake 3.10 (default on Ubuntu 20.04) does not support ALIAS for IMPORTED
        # libraries
        add_library(equistore INTERFACE)
        target_link_libraries(equistore INTERFACE equistore::shared)
    endif()
else()
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.11)
        add_library(equistore ALIAS equistore::static)
    else()
        # CMake 3.10 (default on Ubuntu 20.04) does not support ALIAS for IMPORTED
        # libraries
        add_library(equistore INTERFACE)
        target_link_libraries(equistore INTERFACE equistore::static)
    endif()
endif()
