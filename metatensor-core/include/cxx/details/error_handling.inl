// -*- mode:cpp; -*-

/// Check if a return status from the C API indicates an error, and if it is
/// the case, throw an exception of type `metatensor::Error` with the last
/// error message from the library.
inline void check_status(mts_status_t status) {
    if (status == MTS_SUCCESS) {
        return;
    } else if (status > 0) {
        throw Error(mts_last_error());
    } else { // status < 0
        throw Error("error in C++ callback: " + LastCxxError::message());
    }
}

/// Call the given `function` with the given `args` (the function should
/// return an `mts_status_t`), catching any C++ exception, and translating
/// them to negative metatensor error code.
///
/// This is required to prevent callbacks unwinding through the C API.
template<typename Function, typename ...Args>
inline mts_status_t catch_exceptions(Function function, Args ...args) {
    try {
        return function(std::move(args)...);
    } catch (const std::exception& e) {
        details::LastCxxError::set_message(e.what());
        return -1;
    } catch (...) {
        details::LastCxxError::set_message("error was not an std::exception");
        return -128;
    }
}

/// Check if a pointer allocated by the C API is null, and if it is the
/// case, throw an exception of type `metatensor::Error` with the last error
/// message from the library.
inline void check_pointer(const void* pointer) {
    if (pointer == nullptr) {
        throw Error(mts_last_error());
    }
}
