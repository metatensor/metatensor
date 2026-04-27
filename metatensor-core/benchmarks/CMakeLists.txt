get_target_property(METATENSOR_IMPORTED_LOCATION metatensor::shared IMPORTED_LOCATION)
get_filename_component(METATENSOR_DIR ${METATENSOR_IMPORTED_LOCATION} DIRECTORY)

file(GLOB ALL_BENCHES *.cpp)

list(REMOVE_ITEM ALL_BENCHES ${CMAKE_CURRENT_SOURCE_DIR}/bench_utils.cpp)

add_library(bench_utils STATIC ${CMAKE_CURRENT_SOURCE_DIR}/bench_utils.cpp)
target_link_libraries(bench_utils metatensor)

foreach(_file_ ${ALL_BENCHES})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(bench-${_name_} ${_file_})
    target_link_libraries(bench-${_name_} metatensor bench_utils)

    set_target_properties(bench-${_name_} PROPERTIES
        # Ensure that the binaries find the right shared library.
        #
        # Without this, when configuring with cmake before the library is built,
        # cmake does not find the library on the filesystem and does not add the
        # RPATH to executables linking to it
        BUILD_RPATH ${METATENSOR_DIR}
        NO_SYSTEM_FROM_IMPORTED ON
    )

    if(WIN32)
        # We need to set the path to allow access to metatensor.dll
        # this does a similar job to the BUILD_RPATH above
        STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
        set_tests_properties(bench-${_name_} PROPERTIES
            ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:metatensor>"
        )
    endif()
endforeach()
