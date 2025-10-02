#pragma once
#include <stdexcept>
#include <string>
#include <utility>

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

    #include "error_handling.inl"

} // namespace details
} // namespace metatensor

