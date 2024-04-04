#include <mutex>
#include <cstdlib>
#include <iostream>
#include <filesystem>

#include "shared_libraries.hpp"

static std::mutex MUTEX;

#if defined(__linux__)

#ifndef _GNU_SOURCE
#defined _GNU_SOURCE
#endif

#include <link.h>
#include <dlfcn.h>


static int phdr_callback(struct dl_phdr_info *info, size_t, void *data) {
    auto* result = reinterpret_cast<std::vector<std::string>*>(data);
    result->emplace_back(info->dlpi_name);
    return 0;
}

std::vector<std::string> metatensor_torch::details::get_loaded_libraries() {
    // prevent simultaneous calls to this function
    auto guard = std::lock_guard<std::mutex>(MUTEX);

    auto result = std::vector<std::string>();

    dl_iterate_phdr(phdr_callback, &result);

    return result;
}

static bool try_load(const std::string& path, bool debug) {
    auto* lib = dlopen(path.c_str(), RTLD_LAZY);

    if (debug && lib == nullptr) {
        std::cerr << dlerror() << std::endl;
    }

    return lib != nullptr;
}


#elif defined(__APPLE__)

#include <mach-o/dyld.h>
#include <dlfcn.h>

std::vector<std::string> metatensor_torch::details::get_loaded_libraries() {
    // prevent simultaneous calls to this function
    auto guard = std::lock_guard<std::mutex>(MUTEX);

    auto result = std::vector<std::string>();

    auto count = _dyld_image_count();
    for (uint32_t i=0; i<count; i++) {
        result.emplace_back(_dyld_get_image_name(i));
    }

    return result;
}

static bool try_load(const std::string& path, bool debug) {
    auto* lib = dlopen(path.c_str(), RTLD_LAZY);

    if (debug) {
        if (lib == nullptr) {
            std::cerr << "failed: " << dlerror() << std::endl;
        } else {
            std::cerr << " … success!" << std::endl;
        }
    }

    return lib != nullptr;
}

#elif defined(_WIN32)

#include <Windows.h>
#include <TlHelp32.h>

#include <cstring>

std::vector<std::string> metatensor_torch::details::get_loaded_libraries() {
    // prevent simultaneous calls to this function
    auto guard = std::lock_guard<std::mutex>(MUTEX);

    auto handle = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, 0);

    if (handle == INVALID_HANDLE_VALUE) {
        auto error = GetLastError();
        throw std::runtime_error(
            "failed to get a process snapshot, error code " + std::to_string(error)
        );
    }

    MODULEENTRY32 module;
    std::memset(&module, 0, sizeof(MODULEENTRY32));
    module.dwSize = sizeof(MODULEENTRY32);

    if (!Module32First(handle, &module)) {
        auto error = GetLastError();
        CloseHandle(handle);
        throw std::runtime_error(
            "failed to get the first module in process, error code " + std::to_string(error)
        );
    }

    auto result = std::vector<std::string>();
    do {
        result.emplace_back(module.szExePath);
    } while (Module32Next(handle, &module));

    CloseHandle(handle);
    return result;
}

static bool try_load(const std::string& path, bool debug) {
    // do not open a dialog window if the DLL fails to load
    auto previous_error_mode = GetErrorMode();
    SetErrorMode(previous_error_mode | SEM_FAILCRITICALERRORS);

    HMODULE library = nullptr;
    // convert file path to absolute path, but leave file name alone
    if (path.find('/') != std::string::npos || path.find('\\') != std::string::npos) {
        auto full_path = std::string(4096, '\0');
        auto size = GetFullPathNameA(path.c_str(), full_path.size(), &full_path[0], nullptr);
        full_path.resize(size);

        library = LoadLibraryA(full_path.c_str());
    } else {
        library = LoadLibraryA(path.c_str());
    }

    if (debug) {
        if (library == nullptr) {
            auto status = GetLastError();

            LPTSTR message = nullptr;
            FormatMessage(
                FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS,
                /* language */ nullptr,
                status,
                /* language */0,
                reinterpret_cast<LPTSTR>(&message),
                /* buffer size */0,
                /* format arguments */nullptr
            );

            if (message != nullptr) {
                std::cerr << "failed: " << message << std::endl;
                LocalFree(message);
            }
        } else {
            std::cerr << " … success!" << std::endl;
        }
    }

    // reset error mode
    SetErrorMode(previous_error_mode);

    return library != nullptr;
}

#else

#error "Unsupported OS, please add it to this file!"

#endif

bool metatensor_torch::details::load_library(
    const std::string& name,
    const std::vector<std::string>& candidates
) {
    auto debug = getenv("METATENSOR_DEBUG_EXTENSIONS_LOADING") != nullptr;

    for (const auto& path: candidates) {
        if (std::filesystem::exists(path)) {
            if (debug) {
                std::cerr << "trying to load '" << path << "' …" << std::endl;
            }
            auto loaded = try_load(path, debug);
            if (loaded) {
                return true;
            }
        } else {
            if (debug) {
                std::cerr << "skipping '" << path << "', the file does not exists" << std::endl;
            }
        }
    }

    if (debug) {
        std::cerr << "trying to load by name '" << name << "' …" << std::endl;
    }
    return try_load(name, debug);
}
