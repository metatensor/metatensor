#include <torch/torch.h>

#include <metatensor.h>
#include <metatensor.hpp>
#include <torch/types.h>

#include "metatensor/torch/version.h"
#include "metatensor/torch/array.hpp"
#include "metatensor/torch/misc.hpp"

using namespace metatensor_torch;

std::string metatensor_torch::version() {
    return METATENSOR_TORCH_VERSION;
}

mts_status_t metatensor_torch::details::create_torch_array(
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    mts_array_t* array
) {
    return metatensor::details::catch_exceptions([](
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        mts_array_t* array
    ) {
        auto sizes = std::vector<int64_t>();
        for (size_t i=0; i<shape_count; i++) {
            sizes.push_back(static_cast<int64_t>(shape_ptr[i]));
        }

        auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kF64);
        auto tensor = torch::zeros(sizes, options);

        auto cxx_array = std::unique_ptr<metatensor::DataArrayBase>(new TorchDataArray(tensor));
        *array = metatensor::DataArrayBase::to_mts_array_t(std::move(cxx_array));
    }, shape_ptr, shape_count, array);
}

static torch::ScalarType dlpack_to_torch_type(DLDataType dtype) {
    switch (dtype.code) {
        case kDLInt:
            switch (dtype.bits) {
                case 8: return torch::kInt8;
                case 16: return torch::kInt16;
                case 32: return torch::kInt32;
                case 64: return torch::kInt64;
            }
            break;
        case kDLUInt:
            switch (dtype.bits) {
                case 8: return torch::kUInt8;
            }
            break;
        case kDLFloat:
            switch (dtype.bits) {
                case 16: return torch::kHalf;
                case 32: return torch::kFloat;
                case 64: return torch::kDouble;
            }
            break;
        case kDLBool:
            return torch::kBool;
        case kDLComplex:
            switch (dtype.bits) {
                case 64: return torch::kComplexFloat;
                case 128: return torch::kComplexDouble;
            }
            break;
    }
    throw metatensor::Error("unsupported DLPack dtype for torch: code=" + std::to_string(dtype.code) + " bits=" + std::to_string(dtype.bits));
}

mts_status_t metatensor_torch::details::create_mmap_torch_array(
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    DLDataType dtype,
    const void* data,
    uintptr_t data_len,
    void* mmap_ptr,
    mts_array_t* array
) {
    return metatensor::details::catch_exceptions([](
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        DLDataType dtype,
        const void* data,
        uintptr_t data_len,
        void* mmap_ptr,
        mts_array_t* array
    ) {
        auto sizes = std::vector<int64_t>();
        for (size_t i=0; i<shape_count; i++) {
            sizes.push_back(static_cast<int64_t>(shape_ptr[i]));
        }

        auto scalar_type = dlpack_to_torch_type(dtype);
        auto options = torch::TensorOptions().device(torch::kCPU).dtype(scalar_type);

        auto elem_bytes = (dtype.bits / 8) * dtype.lanes;
        if (elem_bytes > 1 && reinterpret_cast<uintptr_t>(data) % elem_bytes != 0) {
            // Not aligned, must copy
            auto tensor = torch::empty(sizes, options);
            std::memcpy(tensor.data_ptr(), data, data_len);

            // Free the memory mapping, we don't need it anymore
            ::mts_mmap_free(mmap_ptr);

            auto cxx_array = std::unique_ptr<metatensor::DataArrayBase>(new TorchDataArray(tensor));
            *array = metatensor::DataArrayBase::to_mts_array_t(std::move(cxx_array));
        } else {
            auto deleter = [mmap_ptr](void*) {
                ::mts_mmap_free(mmap_ptr);
            };

            // Create a tensor wrapping the mmap'ed data
            auto tensor = torch::from_blob(const_cast<void*>(data), sizes, deleter, options);

            auto cxx_array = std::unique_ptr<metatensor::DataArrayBase>(new TorchDataArray(tensor));
            *array = metatensor::DataArrayBase::to_mts_array_t(std::move(cxx_array));
        }
    }, shape_ptr, shape_count, dtype, data, data_len, mmap_ptr, array);
}

/******************************************************************************/

TensorMap metatensor_torch::load(const std::string& path) {
    return TensorMapHolder::load(path);
}

TensorMap metatensor_torch::load_buffer(torch::Tensor buffer) {
    return TensorMapHolder::load_buffer(buffer);
}


TensorMap metatensor_torch::load_mmap(const std::string& path) {
    return TensorMapHolder::load_mmap(path);
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

TensorBlock metatensor_torch::load_block_buffer(torch::Tensor buffer) {
    return TensorBlockHolder::load_buffer(buffer);
}


TensorBlock metatensor_torch::load_block_mmap(const std::string& path) {
    return TensorBlockHolder::load_mmap(path);
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
