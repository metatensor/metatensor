#include <cerrno>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
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

// torch has a `torch.load(path, mmap=True)` for its own `.pt` format (a zip
// of pickled torch tensors). It does not work for `.mts`: different
// container layout, different per-array metadata, and the torch loader
// expects pickled torch.Storage objects rather than raw NPY arrays. The
// `MmapFile` + custom `torch::from_blob` deleter machinery below is what
// bridges `.mts`'s STORED-NPY entries into torch tensors that share storage
// with the mapping; there is no torch-side API that does this for a third-
// party container.

// Cross-platform private memory map. Owned by a shared_ptr that is captured
// by every torch::Tensor returned from the mmap loader, so the mapping stays
// alive for as long as any tensor refers into it. The mapping is writable but
// private: in-place tensor writes fault private pages and never modify the file.
struct MmapFile {
    uint8_t* data = nullptr;
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
        handle->file_handle, nullptr, PAGE_WRITECOPY, 0, 0, nullptr
    );
    if (handle->mapping_handle == nullptr) {
        throw metatensor::Error("CreateFileMapping failed for '" + path + "'");
    }
    handle->data = reinterpret_cast<uint8_t*>(
        MapViewOfFile(handle->mapping_handle, FILE_MAP_COPY, 0, 0, 0)
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
    void* mapped = mmap(nullptr, handle->length, PROT_READ | PROT_WRITE, MAP_PRIVATE, handle->fd, 0);
    if (mapped == MAP_FAILED) {
        throw metatensor::Error(
            "mmap of '" + path + "' failed: " + std::strerror(errno)
        );
    }
    handle->data = reinterpret_cast<uint8_t*>(mapped);
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
        void* raw = ctx->file->data + file_offset;
        torch::Tensor tensor;
        if (num_elements == 0) {
            tensor = torch::empty(sizes, options);
        } else if (reinterpret_cast<uintptr_t>(raw) % element_bytes == 0) {
            // Capture a fresh shared_ptr to the mmap so this tensor keeps it
            // alive even if the load function and other tensors are dropped.
            auto file_for_capture = ctx->file;
            tensor = torch::from_blob(
                raw,
                sizes,
                // deleter: keeps the mmap alive until this tensor's storage is freed
                [file_for_capture](void*) mutable { file_for_capture.reset(); },
                options
            );
        } else {
            tensor = torch::empty(sizes, options);
            std::memcpy(tensor.data_ptr(), raw, total_bytes);
        }

        auto cxx_array = std::unique_ptr<metatensor::DataArrayBase>(new TorchDataArray(tensor));
        *array = metatensor::DataArrayBase::to_mts_array(std::move(cxx_array)).release();
    }, user_data, shape_ptr, shape_count, dtype, file_offset, array);
}

TensorMap metatensor_torch::load(const std::string& path, bool mmap) {
    if (mmap) {
        auto file = mmap_open(path);
        MmapCallbackContext ctx{file};
        return torch::make_intrusive<TensorMapHolder>(
            TensorMapHolder(metatensor::io::load(
                path, torch_mmap_create_array, &ctx
            ))
        );
    }
    return TensorMapHolder::load(path);
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

TensorBlock metatensor_torch::load_block(const std::string& path, bool mmap) {
    if (mmap) {
        auto file = mmap_open(path);
        MmapCallbackContext ctx{file};
        return torch::make_intrusive<TensorBlockHolder>(
            TensorBlockHolder(
                metatensor::io::load_block(path, torch_mmap_create_array, &ctx),
                /*parent=*/torch::IValue()
            )
        );
    }
    return TensorBlockHolder::load(path);
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

Labels metatensor_torch::load_labels(const std::string& path, bool mmap) {
    if (mmap) {
        auto file = mmap_open(path);
        MmapCallbackContext ctx{file};
        return torch::make_intrusive<LabelsHolder>(
            LabelsHolder(metatensor::io::load_labels(
                path, torch_mmap_create_array, &ctx
            ))
        );
    }
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
