file(GLOB ALL_BENCHES *.cpp)

list(REMOVE_ITEM ALL_BENCHES ${CMAKE_CURRENT_SOURCE_DIR}/bench_utils.cpp)

add_library(torch_bench_utils STATIC ${CMAKE_CURRENT_SOURCE_DIR}/bench_utils.cpp)
target_link_libraries(torch_bench_utils metatensor)

foreach(_file_ ${ALL_BENCHES})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(bench-torch-${_name_} ${_file_})
    target_link_libraries(bench-torch-${_name_} metatensor_torch torch_bench_utils)
endforeach()
