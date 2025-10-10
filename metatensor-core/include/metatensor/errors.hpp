#pragma once

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
        /// Singleton class storing the last exception throw by a C++ callback.
        ///
        /// When passing callbacks from C++ to Rust, we need to convert exceptions
        /// into status code (see the `catch` blocks in this file). This class
        /// allows to save the message associated with an exception, and rethrow an
        /// exception with the same message later (the actual exception type is lost
        /// in the process).
        class LastCxxError {
        public:
            /// Set the last error message to `message`
            static void set_message(std::string message) {
                auto& stored_message = LastCxxError::get();
                stored_message = std::move(message);
            }

            /// Get the last error message
            static const std::string& message() {
                return LastCxxError::get();
            }

        private:
            static std::string& get() {
                #pragma clang diagnostic push
                #pragma clang diagnostic ignored "-Wexit-time-destructors"
                /// we are using a per-thread static value to store the last C++
                /// exception.
                static thread_local std::string STORED_MESSAGE;
                #pragma clang diagnostic pop

                return STORED_MESSAGE;
            }
        };

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

    } // namespace details
} // namespace metatensor
