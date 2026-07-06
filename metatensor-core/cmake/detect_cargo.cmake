# This module finds a suitable cargo binary. It tries plain "cargo" first, then
# searches for versioned cargo binaries (e.g. cargo-1.82) commonly installed on
# Ubuntu. If a binary is found but too old, it continues searching for a newer
# one.
#
# Sets:
#   CARGO_EXE                   - path to the chosen cargo binary
#   CARGO_VERSION               - parsed version string (e.g. 1.96.1)
#   RUST_HOST_TARGET            - host target triple (e.g. x86_64-unknown-linux-gnu)
#   RUST_HOST_ARCH              - host CPU architecture (e.g. x86_64)
#   CACHED_LAST_CARGO_VERSION   - cache variable for change detection
#   CARGO_VERSION_CHANGED       - true if the version differs from the last run

set(REQUIRED_RUST_VERSION "1.88.0")

# ---------------------------------------------------------------------------
# Helper: run cargo --version --verbose, extract version & host target
# ---------------------------------------------------------------------------
function(_try_cargo _exe _ok_var _version_var _host_target_var _host_arch_var)
    execute_process(
        COMMAND "${_exe}" "--version" "--verbose"
        RESULT_VARIABLE _status
        OUTPUT_VARIABLE _raw
        ERROR_QUIET
    )

    if (NOT _status EQUAL 0)
        set(${_ok_var} FALSE PARENT_SCOPE)
        return()
    endif()

    set(_ok TRUE)
    set(_version "")
    set(_host_target "")

    if (_raw MATCHES "cargo ([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(_version "${CMAKE_MATCH_1}")
    else()
        set(_ok FALSE)
    endif()

    if (_raw MATCHES "host: ([a-zA-Z0-9_\\-]*)\n")
        set(_host_target "${CMAKE_MATCH_1}")
    else()
        set(_ok FALSE)
    endif()

    set(${_ok_var} ${_ok} PARENT_SCOPE)
    set(${_version_var} "${_version}" PARENT_SCOPE)
    set(${_host_target_var} "${_host_target}" PARENT_SCOPE)

    if (_host_target MATCHES "([a-zA-Z0-9_]*)\\-")
        set(${_host_arch_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        set(${_host_arch_var} "" PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Step 1: try plain "cargo" (or respect a pre-defined CARGO_EXE)
# ---------------------------------------------------------------------------
set(_cargo_found FALSE)
if (DEFINED CARGO_EXE AND NOT CARGO_EXE STREQUAL "CARGO_EXE-NOTFOUND")
    _try_cargo("${CARGO_EXE}" _ok _ver _target _arch)
    if (_ok AND ${_ver} VERSION_GREATER_EQUAL ${REQUIRED_RUST_VERSION})
        set(_cargo_found TRUE)
        set(CARGO_VERSION "${_ver}")
        set(RUST_HOST_TARGET "${_target}")
        set(RUST_HOST_ARCH "${_arch}")
    else()
        # Cache is stale or binary changed; re-search below
        message(STATUS "cargo at ${CARGO_EXE} is not usable, searching for alternatives...")
        unset(CARGO_EXE)
        unset(CARGO_EXE CACHE)
    endif()
endif()

if (NOT _cargo_found)
    find_program(_cargo_vanilla "cargo")
    if (_cargo_vanilla)
        _try_cargo("${_cargo_vanilla}" _ok _ver _target _arch)
        if (_ok AND ${_ver} VERSION_GREATER_EQUAL ${REQUIRED_RUST_VERSION})
            set(_cargo_found TRUE)
            set(CARGO_EXE "${_cargo_vanilla}")
            set(CARGO_VERSION "${_ver}")
            set(RUST_HOST_TARGET "${_target}")
            set(RUST_HOST_ARCH "${_arch}")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Step 2: search for versioned cargo-* binaries across PATH
# ---------------------------------------------------------------------------
if (NOT _cargo_found)
    # Collect all directories to search
    set(_search_dirs ${CMAKE_PROGRAM_PATH})

    if (WIN32)
        foreach(_dir IN LISTS $ENV{PATH})
            list(APPEND _search_dirs "${_dir}")
        endforeach()
    else()
        string(REPLACE ":" ";" _sys_path "$ENV{PATH}")
        list(APPEND _search_dirs ${_sys_path})
    endif()

    if (NOT "$ENV{HOME}" STREQUAL "")
        list(APPEND _search_dirs "$ENV{HOME}/.cargo/bin")
    endif()

    set(_cargo_candidates "")
    foreach(_dir IN LISTS _search_dirs)
        if (IS_DIRECTORY "${_dir}")
            file(GLOB _bins "${_dir}/cargo-*")
            list(APPEND _cargo_candidates ${_bins})
        endif()
    endforeach()

    if (_cargo_candidates)
        list(REMOVE_DUPLICATES _cargo_candidates)
    endif()

    set(_best_exe "")
    set(_best_version "0.0.0")
    set(_best_target "")
    set(_best_arch "")

    foreach(_bin IN LISTS _cargo_candidates)
        _try_cargo("${_bin}" _ok _ver _target _arch)
        if (_ok AND ${_ver} VERSION_GREATER_EQUAL ${REQUIRED_RUST_VERSION}
              AND ${_ver} VERSION_GREATER ${_best_version})
            set(_best_exe "${_bin}")
            set(_best_version "${_ver}")
            set(_best_target "${_target}")
            set(_best_arch "${_arch}")
        endif()
    endforeach()

    if (_best_exe)
        set(_cargo_found TRUE)
        set(CARGO_EXE "${_best_exe}")
        set(CARGO_VERSION "${_best_version}")
        set(RUST_HOST_TARGET "${_best_target}")
        set(RUST_HOST_ARCH "${_best_arch}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------
if (NOT _cargo_found)
    message(FATAL_ERROR
        "could not find a suitable cargo binary (version >= ${REQUIRED_RUST_VERSION}).\n"
        "Please install Rust from https://www.rust-lang.org/tools/install\n"
        "or set CARGO_EXE to point to your cargo binary before calling CMake."
    )
endif()

if (NOT RUST_HOST_TARGET)
    message(FATAL_ERROR
        "failed to determine host target from cargo --version --verbose"
    )
endif()

if (NOT RUST_HOST_ARCH)
    message(FATAL_ERROR
        "failed to determine host CPU arch from target: ${RUST_HOST_TARGET}"
    )
endif()

# ---------------------------------------------------------------------------
# Cache for change detection across CMake re-configures
# ---------------------------------------------------------------------------
if (NOT "${CACHED_LAST_CARGO_VERSION}" STREQUAL "${CARGO_VERSION}")
    set(CACHED_LAST_CARGO_VERSION "${CARGO_VERSION}"
        CACHE INTERNAL "Last version of cargo used in configuration"
    )
    message(STATUS "Using cargo version ${CARGO_VERSION} at ${CARGO_EXE}")
    set(CARGO_VERSION_CHANGED TRUE)
endif()
