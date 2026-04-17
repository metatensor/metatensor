#pragma once

#include <cstring>

#include <iostream>

#include <stdexcept>
#include <string>
#include <utility>

#include <metatensor.h>

namespace metatensor {
    /// Exception class used for all errors in metatensor
    class Error: public std::runtime_error {
    public:
        /// Create a new Error with the given `message`
        Error(const std::string& message): std::runtime_error(message) {}
    };

    namespace details {
        /// Check if a return status from the C API indicates an error, and if it is
        /// the case, throw an exception of type `metatensor::Error` with the last
        /// error message from the library.
        inline void check_status(mts_status_t status) {
            if (status == MTS_SUCCESS) {
                return;
            } else if (status == MTS_CALLBACK_ERROR) {
                const char* message = nullptr;
                const char* origin = nullptr;
                void* data = nullptr;
                mts_last_error(&message, &origin, &data);
                if (origin != nullptr &&std::strcmp(origin, "C++ exception") == 0 && data != nullptr) {
                    std::rethrow_exception(*static_cast<std::exception_ptr*>(data));
                } else {
                    throw Error(message == nullptr ? "unknown error" : message);
                }
            } else {
                const char* message = nullptr;
                mts_last_error(&message, nullptr, nullptr);
                throw Error(message == nullptr ? "unknown error" : message);
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
                function(std::move(args)...);
                return MTS_SUCCESS;
            } catch (...) {
                auto* exception_ptr = new std::exception_ptr(std::current_exception());

                const char* message = nullptr;
                try {
                    std::rethrow_exception(*exception_ptr);
                } catch (const std::exception& e) {
                    message = e.what();
                } catch (...) {
                    message = "C++ code threw an exception that was not an std::exception";
                }

                auto status = mts_set_last_error(
                    message,
                    "C++ exception",
                    exception_ptr,
                    [](void *ptr) { delete static_cast<std::exception_ptr*>(ptr); }
                );

                if (status != MTS_SUCCESS) {
                    // If we failed to set the error, we are in a very bad state,
                    // but we should still try to report the original error
                    // message if possible.
                    std::cerr << "INTERNAL ERROR: unable to set last error after C++ callback failure (status: " << status << ")" << std::endl;
                    if (message != nullptr) {
                        std::cerr << "C++ error was: " << message << std::endl;
                    } else {
                        std::cerr << "unknown C++ error" << std::endl;
                    }
                    delete exception_ptr;
                }

                return MTS_CALLBACK_ERROR;
            }
        }

        /// Check if a pointer allocated by the C API is null, and if it is the
        /// case, throw an exception of type `metatensor::Error` with the last
        /// error message from the library.
        inline void check_pointer(const void* pointer) {
            if (pointer == nullptr) {
                const char* message = nullptr;
                const char* origin = nullptr;
                void* data = nullptr;
                mts_last_error(&message, &origin, &data);
                if (std::strcmp(origin, "C++ exception") == 0 && data != nullptr) {
                    std::rethrow_exception(*static_cast<std::exception_ptr*>(data));
                } else {
                    throw Error(message);
                }
            }
        }
    } // namespace details
} // namespace metatensor
