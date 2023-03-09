@PACKAGE_INIT@

cmake_minimum_required(VERSION 3.10)

if(equistore_FOUND)
    return()
endif()

enable_language(CXX)

add_library(equistore::shared SHARED IMPORTED GLOBAL)
add_library(equistore::static STATIC IMPORTED GLOBAL)

set_target_properties(equistore::shared PROPERTIES
    IMPORTED_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@EQUISTORE_SHARED_LIB_NAME@
    INTERFACE_INCLUDE_DIRECTORIES ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/
)

set_target_properties(equistore::static PROPERTIES
    IMPORTED_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@EQUISTORE_STATIC_LIB_NAME@
    INTERFACE_INCLUDE_DIRECTORIES ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/
)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.11)
    # we can not set compile features for imported targets before cmake 3.11
    # users will have to manually request C++11
    target_compile_features(equistore::shared INTERFACE cxx_std_11)
    target_compile_features(equistore::static INTERFACE cxx_std_11)
endif()


if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.11)
    # The default library for external users should be the shared one
    add_library(equistore ALIAS equistore::shared)
else()
    # CMake 3.10 (default on Ubuntu 20.04) does not support ALIAS for IMPORTED
    # libraries
    add_library(equistore INTERFACE)
    target_link_libraries(equistore INTERFACE equistore::shared)
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    # the rust standard lib uses pthread and libdl on linux
    target_link_libraries(equistore::static INTERFACE Threads::Threads dl)

    # num_bigint uses fmod
    target_link_libraries(equistore::static INTERFACE m)
endif()
