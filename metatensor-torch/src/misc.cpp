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
    if (dtype.code == kDLFloat && dtype.bits == 16) return torch::kFloat16;
    if (dtype.code == kDLFloat && dtype.bits == 32) return torch::kFloat32;
    if (dtype.code == kDLFloat && dtype.bits == 64) return torch::kFloat64;
    if (dtype.code == kDLInt && dtype.bits == 8) return torch::kInt8;
    if (dtype.code == kDLInt && dtype.bits == 16) return torch::kInt16;
    if (dtype.code == kDLInt && dtype.bits == 32) return torch::kInt32;
    if (dtype.code == kDLInt && dtype.bits == 64) return torch::kInt64;
    if (dtype.code == kDLUInt && dtype.bits == 8) return torch::kUInt8;
    if (dtype.code == kDLBool && dtype.bits == 8) return torch::kBool;
    if (dtype.code == kDLComplex && dtype.bits == 64) return torch::kComplexFloat;
    if (dtype.code == kDLComplex && dtype.bits == 128) return torch::kComplexDouble;
    throw metatensor::Error(
        "unsupported DLDataType for torch: code="
        + std::to_string(dtype.code) + " bits=" + std::to_string(dtype.bits)
    );
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
        *array = metatensor::DataArrayBase::to_mts_array_t(std::move(cxx_array));
    }, shape_ptr, shape_count, dtype, array);
}

/******************************************************************************/

TensorMap metatensor_torch::load(const std::string& path) {
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

TensorBlock metatensor_torch::load_block(const std::string& path) {
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
