# This cmake script adds `#define METATENSOR_VERSION_XXX` to the C API header
# for metatensor. This is done by modifying a copy of the header in cmake's
# build dir (instead of modifying the one in the source dir) to mimize git noise
# (since the version changes for every git commit).

cmake_minimum_required(VERSION 3.16)

function(copy_if_different _src_ _dst_)
    # file(COPY_FILE ...) was added in cmake 3.21, this emulates the same
    # behavior on older versions
    get_filename_component(_dst_dir_ ${_dst_} DIRECTORY)
    get_filename_component(_src_name_ ${_src_} NAME)

    if (EXISTS ${_dst_})
        file(SHA1 ${_src_} new_hash)
        file(SHA1 ${_dst_} old_hash)

        if ("${new_hash}" STREQUAL "${old_hash}")
            set(_do_copy_ FALSE)
        else()
            set(_do_copy_ TRUE)
        endif()
    else()
        # The destination does not exist
        set(_do_copy_ TRUE)
    endif()

    if (${_do_copy_})
        file(COPY ${_src_} DESTINATION ${_dst_dir_})
        file(RENAME ${_dst_dir_}/${_src_name_} ${_dst_})
    endif()
endfunction()


file(COPY ${MTS_SOURCE_DIR}/include/metatensor.hpp DESTINATION ${MTS_BINARY_DIR}/include)
file(READ ${MTS_SOURCE_DIR}/include/metatensor.h _header_content_)

set(_path_ ${MTS_BINARY_DIR}/generated-metatensor.h)
file(WRITE ${_path_} "#ifndef METATENSOR_H\n")

file(APPEND ${_path_} "/** Full version of metatensor as a string */\n")
file(APPEND ${_path_} "#define METATENSOR_VERSION \"${MTS_VERSION}\"\n\n")

file(APPEND ${_path_} "/** Major version of metatensor as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_VERSION_MAJOR ${MTS_VERSION_MAJOR}\n\n")

file(APPEND ${_path_} "/** Minor version of metatensor as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_VERSION_MINOR ${MTS_VERSION_MINOR}\n\n")

file(APPEND ${_path_} "/** Patch version of metatensor as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_VERSION_PATCH ${MTS_VERSION_PATCH}\n")

file(APPEND ${_path_} "#endif\n\n")

string(REPLACE ";" "\\;" _header_content_ "${_header_content_}")
file(APPEND ${_path_} ${_header_content_})

set(_destination_ "${MTS_BINARY_DIR}/include/metatensor.h")
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.21")
    file(COPY_FILE ${_path_} ${_destination_} ONLY_IF_DIFFERENT)
else()
    copy_if_different(${_path_} ${_destination_})
endif()
