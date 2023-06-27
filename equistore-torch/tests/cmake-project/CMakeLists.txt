cmake_minimum_required(VERSION 3.16)

project(equistore-torch-test-cmake-project CXX)

find_package(equistore_torch 0.1 CONFIG REQUIRED)

add_executable(torch-main src/main.cpp)
target_link_libraries(torch-main equistore_torch)

enable_testing()
add_test(NAME torch-main COMMAND torch-main)

if(WIN32)
    # We need to set the path to allow access to equistore.dll
    STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
    set_tests_properties(torch-main PROPERTIES
        ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:torch>\;$<TARGET_FILE_DIR:equistore::shared>\;$<TARGET_FILE_DIR:equistore_torch>"
    )
endif()
