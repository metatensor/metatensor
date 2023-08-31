cmake_minimum_required(VERSION 3.16)

project(metatensor-torch-test-cmake-project CXX)

find_package(metatensor_torch 0.1 CONFIG REQUIRED)

add_executable(torch-main src/main.cpp)
target_link_libraries(torch-main metatensor_torch)

enable_testing()
add_test(NAME torch-main COMMAND torch-main)

if(WIN32)
    # We need to set the path to allow access to metatensor.dll
    STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
    set_tests_properties(torch-main PROPERTIES
        ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:torch>\;$<TARGET_FILE_DIR:metatensor::shared>\;$<TARGET_FILE_DIR:metatensor_torch>"
    )
endif()
