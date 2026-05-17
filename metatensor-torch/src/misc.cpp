#include <cerrno>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

#include <torch/torch.h>

#include <metatensor.hpp>
#include <torch/types.h>

#include "metatensor/torch/version.h"
#include "metatensor/torch/array.hpp"
#include "metatensor/torch/misc.hpp"

using namespace metatensor_torch;

std::string metatensor_torch::version() {
    return METATENSOR_TORCH_VERSION;
}

static torch::ScalarType dlpack_dtype_to_torch(DLDataType dtype) {
    if (dtype.lanes != 1) {
        throw metatensor::Error(
            "unsupported DLDataType for torch: lanes=" +
            std::to_string(dtype.lanes) + " (expected 1)"
        );
    }
    if (dtype.code == kDLFloat && dtype.bits == 16) {
        return torch::kFloat16;
    } else if (dtype.code == kDLFloat && dtype.bits == 32) {
        return torch::kFloat32;
    } else if (dtype.code == kDLFloat && dtype.bits == 64) {
        return torch::kFloat64;
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
        return torch::kInt8;
    } else if (dtype.code == kDLInt && dtype.bits == 16) {
        return torch::kInt16;
    } else if (dtype.code == kDLInt && dtype.bits == 32) {
        return torch::kInt32;
    } else if (dtype.code == kDLInt && dtype.bits == 64) {
        return torch::kInt64;
    } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
        return torch::kBFloat16;
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
        return torch::kUInt8;
    } else if (dtype.code == kDLBool && dtype.bits == 8) {
        return torch::kBool;
    } else if (dtype.code == kDLComplex && dtype.bits == 64) {
        return torch::kComplexFloat;
    } else if (dtype.code == kDLComplex && dtype.bits == 128) {
        return torch::kComplexDouble;
    } else {
        throw metatensor::Error(
            "unsupported DLDataType for torch: code="
            + std::to_string(dtype.code) + " bits=" + std::to_string(dtype.bits)
        );
    }
}

mts_status_t metatensor_torch::details::create_torch_array(
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    DLDataType dtype,
    mts_array_t* array
) {
    return metatensor::details::catch_exceptions([](
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        DLDataType dtype,
        mts_array_t* array
    ) {
        auto sizes = std::vector<int64_t>();
        for (size_t i=0; i<shape_count; i++) {
            sizes.push_back(static_cast<int64_t>(shape_ptr[i]));
        }

        auto torch_dtype = dlpack_dtype_to_torch(dtype);
        auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch_dtype);
        auto tensor = torch::zeros(sizes, options);

        auto cxx_array = std::unique_ptr<metatensor::DataArrayBase>(new TorchDataArray(tensor));
        *array = metatensor::DataArrayBase::to_mts_array(std::move(cxx_array)).release();
    }, shape_ptr, shape_count, dtype, array);
}

/******************************************************************************/

namespace {

// Cross-platform read-only memory map. Owned by a shared_ptr that is captured
// by every torch::Tensor returned from the mmap loader, so the mapping stays
// alive for as long as any tensor refers into it.
struct MmapFile {
    const uint8_t* data = nullptr;
    size_t length = 0;

#if defined(_WIN32)
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = nullptr;
#else
    int fd = -1;
#endif

    MmapFile() = default;
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    ~MmapFile() {
#if defined(_WIN32)
        if (data != nullptr) { UnmapViewOfFile(data); }
        if (mapping_handle != nullptr) { CloseHandle(mapping_handle); }
        if (file_handle != INVALID_HANDLE_VALUE) { CloseHandle(file_handle); }
#else
        if (data != nullptr && length > 0) {
            munmap(const_cast<uint8_t*>(data), length);
        }
        if (fd >= 0) { close(fd); }
#endif
    }
};

static std::shared_ptr<MmapFile> mmap_open(const std::string& path) {
    auto handle = std::make_shared<MmapFile>();

#if defined(_WIN32)
    handle->file_handle = CreateFileA(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr
    );
    if (handle->file_handle == INVALID_HANDLE_VALUE) {
        throw metatensor::Error("could not open '" + path + "' for mmap loading");
    }
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(handle->file_handle, &fsize)) {
        throw metatensor::Error("could not stat '" + path + "' for mmap loading");
    }
    handle->length = static_cast<size_t>(fsize.QuadPart);
    handle->mapping_handle = CreateFileMappingA(
        handle->file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr
    );
    if (handle->mapping_handle == nullptr) {
        throw metatensor::Error("CreateFileMapping failed for '" + path + "'");
    }
    handle->data = reinterpret_cast<const uint8_t*>(
        MapViewOfFile(handle->mapping_handle, FILE_MAP_READ, 0, 0, 0)
    );
    if (handle->data == nullptr) {
        throw metatensor::Error("MapViewOfFile failed for '" + path + "'");
    }
#else
    handle->fd = open(path.c_str(), O_RDONLY);
    if (handle->fd < 0) {
        throw metatensor::Error(
            "could not open '" + path + "' for mmap loading: " + std::strerror(errno)
        );
    }
    struct stat st {};
    if (fstat(handle->fd, &st) != 0) {
        throw metatensor::Error(
            "could not stat '" + path + "': " + std::strerror(errno)
        );
    }
    handle->length = static_cast<size_t>(st.st_size);
    if (handle->length == 0) {
        throw metatensor::Error("file '" + path + "' is empty");
    }
    void* mapped = mmap(nullptr, handle->length, PROT_READ, MAP_PRIVATE, handle->fd, 0);
    if (mapped == MAP_FAILED) {
        throw metatensor::Error(
            "mmap of '" + path + "' failed: " + std::strerror(errno)
        );
    }
    // Audit #8: load_mmap is a single-pass read of the whole file, so
    // MADV_SEQUENTIAL lets the kernel read ahead aggressively and drop
    // pages early. Improves first-load throughput on cold cache.
    (void) madvise(mapped, handle->length, MADV_SEQUENTIAL);
    handle->data = reinterpret_cast<const uint8_t*>(mapped);
#endif

    return handle;
}

// `user_data` for the mmap callback. Owns a strong reference to the mmap
// handle and re-shares it into every materialised tensor via from_blob's
// deleter so the mapping lives as long as any referencing tensor.
struct MmapCallbackContext {
    std::shared_ptr<MmapFile> file;
};

static mts_status_t torch_mmap_create_array(
    void* user_data,
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    DLDataType dtype,
    uintptr_t file_offset,
    mts_array_t* array
) {
    return metatensor::details::catch_exceptions([](
        void* user_data,
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        DLDataType dtype,
        uintptr_t file_offset,
        mts_array_t* array
    ) {
        auto* ctx = static_cast<MmapCallbackContext*>(user_data);

        auto torch_dtype = dlpack_dtype_to_torch(dtype);
        auto sizes = std::vector<int64_t>();
        sizes.reserve(shape_count);
        int64_t num_elements = 1;
        for (size_t i = 0; i < shape_count; ++i) {
            sizes.push_back(static_cast<int64_t>(shape_ptr[i]));
            num_elements *= static_cast<int64_t>(shape_ptr[i]);
        }

        size_t element_bytes = (dtype.bits / 8) * dtype.lanes;
        size_t total_bytes = static_cast<size_t>(num_elements) * element_bytes;
        if (file_offset + total_bytes > ctx->file->length) {
            throw metatensor::Error(
                "mmap-backed array extends beyond the end of the file"
            );
        }

        auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch_dtype);
        // Capture a fresh shared_ptr to the mmap so this tensor keeps it
        // alive even if the load function and other tensors are dropped.
        auto file_for_capture = ctx->file;
        const void* raw = ctx->file->data + file_offset;
        auto tensor = torch::from_blob(
            const_cast<void*>(raw),
            sizes,
            // deleter: keeps the mmap alive until this tensor's storage is freed
            [file_for_capture](void*) mutable { file_for_capture.reset(); },
            options
        );

        auto cxx_array = std::unique_ptr<metatensor::DataArrayBase>(new TorchDataArray(tensor));
        *array = metatensor::DataArrayBase::to_mts_array(std::move(cxx_array)).release();
    }, user_data, shape_ptr, shape_count, dtype, file_offset, array);
}

}  // anonymous namespace

TensorMap metatensor_torch::load(const std::string& path) {
    return TensorMapHolder::load(path);
}

TensorMap TensorMapHolder::load_mmap(const std::string& path) {
    auto file = mmap_open(path);
    MmapCallbackContext ctx{file};
    return torch::make_intrusive<TensorMapHolder>(
        TensorMapHolder(metatensor::io::load_mmap(
            path, torch_mmap_create_array, &ctx
        ))
    );
}

TensorMap metatensor_torch::load_mmap(const std::string& path) {
    return TensorMapHolder::load_mmap(path);
}

TensorMap metatensor_torch::load_buffer(torch::Tensor buffer) {
    return TensorMapHolder::load_buffer(buffer);
}


void metatensor_torch::save(const std::string& path, TensorMap tensor) {
    tensor->save(path);
}

torch::Tensor metatensor_torch::save_buffer(TensorMap tensor) {
    return tensor->save_buffer();
}

/******************************************************************************/

TensorBlock metatensor_torch::load_block(const std::string& path) {
    return TensorBlockHolder::load(path);
}

TensorBlock TensorBlockHolder::load_mmap(const std::string& path) {
    auto file = mmap_open(path);
    MmapCallbackContext ctx{file};
    return torch::make_intrusive<TensorBlockHolder>(
        TensorBlockHolder(
            metatensor::io::load_block_mmap(path, torch_mmap_create_array, &ctx),
            /*parent=*/torch::IValue()
        )
    );
}

TensorBlock metatensor_torch::load_block_mmap(const std::string& path) {
    return TensorBlockHolder::load_mmap(path);
}

TensorBlock metatensor_torch::load_block_buffer(torch::Tensor buffer) {
    return TensorBlockHolder::load_buffer(buffer);
}


void metatensor_torch::save(const std::string& path, TensorBlock block) {
    block->save(path);
}

torch::Tensor metatensor_torch::save_buffer(TensorBlock block) {
    return block->save_buffer();
}

/******************************************************************************/

Labels metatensor_torch::load_labels(const std::string& path) {
    return LabelsHolder::load(path);
}

Labels metatensor_torch::load_labels_buffer(torch::Tensor buffer) {
    return LabelsHolder::load_buffer(buffer);
}

void metatensor_torch::save(const std::string& path, Labels labels) {
    labels->save(path);
}

torch::Tensor metatensor_torch::save_buffer(Labels labels) {
    return labels->save_buffer();
}
