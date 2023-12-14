#include <mutex>

#include "shared_libraries.hpp"

static std::mutex MUTEX;

#if defined(__linux__)

#ifndef _GNU_SOURCE
#defined _GNU_SOURCE
#endif

#include <link.h>


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


#elif defined(__APPLE__)

#include <mach-o/dyld.h>

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

#elif defined(_WIN32)

#include <Windows.h>
#include <TlHelp32.h>

#include <cstring>

std::vector<std::string> metatensor_torch::details::get_loaded_libraries() {
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

#else

#error "Unsupported OS, please add it to this file!"

#endif
