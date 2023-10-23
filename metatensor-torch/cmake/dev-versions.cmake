# Parse a `_version_` number, and store its components in `_major_` `_minor_`
# `_patch_` and `_rc_`
function(parse_version _version_ _major_ _minor_ _patch_ _rc_)
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)(-rc)?([0-9]+)?" _ "${_version_}")

    if(${CMAKE_MATCH_COUNT} EQUAL 3)
        set(${_rc_} "" PARENT_SCOPE)
    elseif(${CMAKE_MATCH_COUNT} EQUAL 5)
        set(${_rc_} ${CMAKE_MATCH_5} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "invalid version string ${_version_}")
    endif()

    set(${_major_} ${CMAKE_MATCH_1} PARENT_SCOPE)
    set(${_minor_} ${CMAKE_MATCH_2} PARENT_SCOPE)
    set(${_patch_} ${CMAKE_MATCH_3} PARENT_SCOPE)
endfunction()


if (CMAKE_VERSION VERSION_LESS "3.17")
    # CMAKE_CURRENT_FUNCTION_LIST_DIR was added in CMake 3.17
    set(CMAKE_CURRENT_FUNCTION_LIST_DIR "${CMAKE_CURRENT_LIST_DIR}")
endif()

# Get the number of commits since the last tag matching `_tag_prefix_`
function(n_commits_since_last_tag _tag_prefix_ _output_)
    if (EXISTS "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/n_commits_since_last_tag")
        # When building from a tarball, the script is executed and the result
        # put in this file
        file(READ "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/n_commits_since_last_tag" _n_commits_)

    elseif (EXISTS "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../scripts/n-commits-since-last-tag.py")
        # When building from a checkout, we'll need to run the script
        find_package(Python COMPONENTS Interpreter REQUIRED)
        set(_script_ "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../scripts/n-commits-since-last-tag.py")
        execute_process(
            COMMAND "${Python_EXECUTABLE}" "${_script_}" "${_tag_prefix_}"
            RESULT_VARIABLE _status_
            OUTPUT_VARIABLE _n_commits_
            ERROR_VARIABLE _stderr_
            WORKING_DIRECTORY ${CMAKE_CURRENT_FUNCTION_LIST_DIR}
        )

        if (NOT ${_status_} EQUAL 0)
            set(${_output_} 0 PARENT_SCOPE)
            return()
        endif()

        if (NOT "${_stderr_}" STREQUAL "")
            message(WARNING "n-commits-since-last-tag.py gave some errors, version number might be wrong:\n${_stderr_}")
        endif()
    else()
        message(FATAL_ERROR "could not find the number of commits since the last tag")
    endif()

    string(STRIP ${_n_commits_} _n_commits_)
    set(${_output_} ${_n_commits_} PARENT_SCOPE)
endfunction()


# Take the version declared in the package, and increase the right number if we
# are actually installing a developement version from after the latest
# release/tag with the given prefix
function(create_development_version _version_ _tag_prefix_ _output_)
    n_commits_since_last_tag(${_tag_prefix_} _n_commits_)

    parse_version(${_version_} _major_ _minor_ _patch_ _rc_)
    if(${_n_commits_} STREQUAL "0")
        # we are building a release, leave the version number as-is
        if("${_rc_}" STREQUAL "")
            set(${_output_} "${_major_}.${_minor_}.${_patch_}" PARENT_SCOPE)
        else()
            set(${_output_} "${_major_}.${_minor_}.${_patch_}-rc${_rc_}" PARENT_SCOPE)
        endif()
    else()
        # we are building a development version, increase the right part of the version
        if("${_rc_}" STREQUAL "")
            math(EXPR _minor_ "${_minor_} + 1")
            set(${_output_} "${_major_}.${_minor_}.0-dev${_n_commits_}" PARENT_SCOPE)
        else()
            math(EXPR _rc_ "${_rc_} + 1")
            set(${_output_} "${_major_}.${_minor_}.${_patch_}-rc${_rc_}-dev${_n_commits_}" PARENT_SCOPE)
        endif()
    endif()
endfunction()
