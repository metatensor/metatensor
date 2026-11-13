cmake_minimum_required(VERSION 3.22)
project(metatensor-tests)

if (POLICY CMP0135)
    # timestamps in FetchContent
    cmake_policy(SET CMP0135 NEW)
endif()

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

if (MINGW)
    # CI can't find libsdc++, so we statically link it
    set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++")
endif()

add_subdirectory(../../ metatensor)
get_target_property(METATENSOR_IMPORTED_LOCATION metatensor::shared IMPORTED_LOCATION)
get_filename_component(METATENSOR_DIR ${METATENSOR_IMPORTED_LOCATION} DIRECTORY)

add_subdirectory(external)

set(METATENSOR_TESTS_KOKKOS_VERSION "5.2.2" CACHE STRING "Version of Kokkos to download if it is not found")
set(METATENSOR_TESTS_KOKKOS_DEVICE "auto" CACHE STRING
    "Device backend to enable when downloading Kokkos: auto (detect available GPUs), none, CUDA, HIP or SYCL"
)
set_property(CACHE METATENSOR_TESTS_KOKKOS_DEVICE PROPERTY STRINGS auto none CUDA HIP SYCL)

# Try to find a usable GPU (compiler + device) for one of the Kokkos device
# backends, and store the corresponding backend name in `_output_`, or "none"
# if no GPU is available.
function(detect_kokkos_device _output_)
    include(CheckLanguage)

    # NVIDIA GPU with CUDA
    find_program(NVIDIA_SMI nvidia-smi)
    if (NVIDIA_SMI)
        execute_process(COMMAND ${NVIDIA_SMI} -L
            RESULT_VARIABLE _status_ OUTPUT_VARIABLE _devices_ ERROR_QUIET
        )
        if (_status_ EQUAL 0 AND "${_devices_}" MATCHES "GPU")
            check_language(CUDA)
            if (CMAKE_CUDA_COMPILER)
                set(${_output_} "CUDA" PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # AMD GPU with HIP. `rocm_agent_enumerator` also lists the CPU as `gfx000`
    find_program(ROCM_AGENT_ENUMERATOR rocm_agent_enumerator HINTS $ENV{ROCM_PATH}/bin /opt/rocm/bin)
    if (ROCM_AGENT_ENUMERATOR)
        execute_process(COMMAND ${ROCM_AGENT_ENUMERATOR}
            RESULT_VARIABLE _status_ OUTPUT_VARIABLE _devices_ ERROR_QUIET
        )
        if (_status_ EQUAL 0 AND "${_devices_}" MATCHES "gfx[1-9]")
            check_language(HIP)
            if (CMAKE_HIP_COMPILER)
                set(${_output_} "HIP" PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # Intel GPU with SYCL. This requires using the Intel compiler as the C++
    # compiler, and we only look for Intel GPUs (Level Zero backend), since
    # metatensor/kokkos.hpp does not support SYCL on NVIDIA/AMD GPUs.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        find_program(SYCL_LS sycl-ls)
        if (SYCL_LS)
            execute_process(COMMAND ${SYCL_LS}
                RESULT_VARIABLE _status_ OUTPUT_VARIABLE _devices_ ERROR_QUIET
            )
            if (_status_ EQUAL 0 AND "${_devices_}" MATCHES "level_zero:gpu")
                set(${_output_} "SYCL" PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    set(${_output_} "none" PARENT_SCOPE)
endfunction()

find_package(Kokkos QUIET)
if (Kokkos_FOUND)
    message(STATUS "Using pre-existing Kokkos v${Kokkos_VERSION} from ${Kokkos_DIR}")
    # the language used by Kokkos (CXX, CUDA or HIP), this is set by
    # Kokkos' CMake configuration
    set(MTS_KOKKOS_TESTS_LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
    if ("${MTS_KOKKOS_TESTS_LANGUAGE}" STREQUAL "")
        set(MTS_KOKKOS_TESTS_LANGUAGE CXX)
    endif()
else()
    if (METATENSOR_TESTS_KOKKOS_DEVICE STREQUAL "auto")
        detect_kokkos_device(KOKKOS_TESTS_DEVICE)
    else()
        set(KOKKOS_TESTS_DEVICE ${METATENSOR_TESTS_KOKKOS_DEVICE})
    endif()

    if (NOT KOKKOS_TESTS_DEVICE MATCHES "^(none|CUDA|HIP|SYCL)$")
        message(FATAL_ERROR
            "invalid METATENSOR_TESTS_KOKKOS_DEVICE '${METATENSOR_TESTS_KOKKOS_DEVICE}', "
            "expected one of none, CUDA, HIP or SYCL"
        )
    endif()

    message(STATUS "Kokkos not found, downloading Kokkos v${METATENSOR_TESTS_KOKKOS_VERSION} with device backend: ${KOKKOS_TESTS_DEVICE}")
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)

    set(MTS_KOKKOS_TESTS_LANGUAGE CXX)
    foreach(_device_ CUDA HIP SYCL)
        if (KOKKOS_TESTS_DEVICE STREQUAL ${_device_})
            set(Kokkos_ENABLE_${_device_} ON CACHE BOOL "" FORCE)
        else()
            set(Kokkos_ENABLE_${_device_} OFF CACHE BOOL "" FORCE)
        endif()
    endforeach()

    if (KOKKOS_TESTS_DEVICE STREQUAL "CUDA" OR KOKKOS_TESTS_DEVICE STREQUAL "HIP")
        # compile Kokkos and the tests using CMake's native CUDA/HIP
        # support, instead of replacing the C++ compiler with
        # nvcc_wrapper/hipcc. The GPU architecture is auto-detected by
        # Kokkos.
        set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL "" FORCE)
        set(MTS_KOKKOS_TESTS_LANGUAGE ${KOKKOS_TESTS_DEVICE})
        # the language must be enabled in this directory to be usable
        # for the tests below
        enable_language(${MTS_KOKKOS_TESTS_LANGUAGE})
    else()
        set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE OFF CACHE BOOL "" FORCE)
    endif()

    include(FetchContent)
    FetchContent_Declare(kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/${METATENSOR_TESTS_KOKKOS_VERSION}.tar.gz
    )

    # Always build Kokkos as a static library. Otherwise, it would inherit
    # `BUILD_SHARED_LIBS=ON` from metatensor, and on MinGW the DLL exports
    # libstdc++ symbols that conflict with `-static-libstdc++` in the tests.
    set(_mts_build_shared_libs_ ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(kokkos)
    set(BUILD_SHARED_LIBS ${_mts_build_shared_libs_})
endif()

set(KOKKOS_TESTS_HAS_GPU OFF)
foreach(_device_ CUDA HIP SYCL)
    if (Kokkos_ENABLE_${_device_})
        message(STATUS "Running Kokkos tests on ${_device_} devices")
        set(KOKKOS_TESTS_HAS_GPU ON)
    endif()
endforeach()

find_program(VALGRIND valgrind)
if (VALGRIND)
    if (NOT "$ENV{METATENSOR_DISABLE_VALGRIND}" EQUAL "1")
        message(STATUS "Running tests using valgrind")
        set(TEST_COMMAND
            "${VALGRIND}" "--tool=memcheck" "--dsymutil=yes" "--error-exitcode=125"
            "--leak-check=full" "--show-leak-kinds=definite,indirect,possible" "--track-origins=yes"
            "--gen-suppressions=all"
        )
    endif()
else()
    set(TEST_COMMAND "")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-equal")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow-uncaptured-local")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unsafe-buffer-usage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-poison-system-directories")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-allocator-wrappers")
endif()


enable_testing()

file(GLOB ALL_TESTS *.cpp)

foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} metatensor catch)

    set_target_properties(${_name_} PROPERTIES
        # Ensure that the binaries find the right shared library.
        #
        # Without this, when configuring with cmake before the library is built,
        # cmake does not find the library on the filesystem and does not add the
        # RPATH to executables linking to it
        BUILD_RPATH ${METATENSOR_DIR}
        NO_SYSTEM_FROM_IMPORTED ON
    )

    # Give path to tests data files to the tests through defines
    target_compile_definitions(${_name_} PRIVATE "-DTEST_DATA_MTS_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../data.mts\"")
    target_compile_definitions(${_name_} PRIVATE "-DTEST_BLOCK_MTS_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../block.mts\"")
    target_compile_definitions(${_name_} PRIVATE "-DTEST_KEYS_MTS_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../keys.mts\"")

    if (${_name_} STREQUAL "kokkos-arrays" AND KOKKOS_TESTS_HAS_GPU)
        # valgrind does not work with GPU runtimes
        add_test(NAME ${_name_} COMMAND $<TARGET_FILE:${_name_}>)
    else()
        add_test(
            NAME ${_name_}
            COMMAND ${TEST_COMMAND} $<TARGET_FILE:${_name_}>
        )
    endif()

    if(WIN32)
        # We need to set the path to allow access to metatensor.dll
        # this does a similar job to the BUILD_RPATH above
        STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
        set_tests_properties(${_name_} PROPERTIES
            ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:metatensor>"
        )
    endif()
endforeach()

target_link_libraries(kokkos-arrays Kokkos::kokkos)
# compile the Kokkos tests with the same language as Kokkos itself
# (CUDA or HIP when using these backends as CMake languages)
set_source_files_properties(kokkos-arrays.cpp PROPERTIES LANGUAGE ${MTS_KOKKOS_TESTS_LANGUAGE})

# Kokkos and the templated tests create too many sections for the default COFF
# object format on Windows
if (MSVC)
    target_compile_options(kokkos-arrays PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
elseif (MINGW)
    target_compile_options(kokkos-arrays PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wa,-mbig-obj>)
endif()
