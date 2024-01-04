# Create a temporary directory using mktemp on *nix and powershell on windows
function(get_tempdir _outvar_)
    # special case for github actions, where $TEMP might
    # exist but point to nowhere/a non writable location
    # https://docs.github.com/en/actions/learn-github-actions/variables
    if (DEFINED ENV{RUNNER_TEMP})
        string(RANDOM LENGTH 12 _dirname_)
        set(_output_ $ENV{RUNNER_TEMP}/${_dirname_})
        file(TO_NATIVE_PATH "${_output_}" _output_)
        file(MAKE_DIRECTORY ${_output_})
        set(${_outvar_} ${_output_} PARENT_SCOPE)
        return()
    endif()

    find_program(MKTEMP_EXE NAMES mktemp)
    if(MKTEMP_EXE)
        execute_process(
            COMMAND ${MKTEMP_EXE} -d
            OUTPUT_VARIABLE _output_
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _status_
        )

        if(_status_ EQUAL 0)
            set(${_outvar_} ${_output_} PARENT_SCOPE)
            return()
        endif()
    endif()


    find_program(POWERSHELL_EXE NAMES pwsh)
    if(POWERSHELL_EXE)
        execute_process(
            COMMAND ${POWERSHELL_EXE} -c "[System.IO.Path]::GetTempPath()"
            OUTPUT_VARIABLE _output_
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _status_
        )

        if(_status_ EQUAL 0)
            string(RANDOM LENGTH 12 _dirname_)
            set(_output_ ${_output_}${_dirname_})
            file(MAKE_DIRECTORY ${_output_})
            set(${_outvar_} ${_output_} PARENT_SCOPE)
            return()
        endif()
    endif()

    message(FATAL_ERROR "Could not find mktemp or powsershell to make temporary directory")
endfunction()
