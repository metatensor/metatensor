cmake_minimum_required(VERSION 3.16)

project(equistore-test-cmake-project C CXX)

find_package(equistore 0.1 REQUIRED)

add_executable(c-main src/main.c)
target_link_libraries(c-main equistore)

add_executable(c-main-static src/main.c)
target_link_libraries(c-main-static equistore::static)

add_executable(cxx-main src/main.cpp)
target_link_libraries(cxx-main equistore)

add_executable(cxx-main-static src/main.cpp)
target_link_libraries(cxx-main-static equistore::static)

enable_testing()
add_test(NAME c-main COMMAND c-main)
add_test(NAME c-main-static COMMAND c-main-static)
add_test(NAME cxx-main COMMAND cxx-main)
add_test(NAME cxx-main-static COMMAND cxx-main-static)
