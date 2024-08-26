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

        return MTS_SUCCESS;
    }, shape_ptr, shape_count, array);
}

/******************************************************************************/

TorchTensorMap metatensor_torch::load(const std::string& path) {
    return TensorMapHolder::load(path);
}

TorchTensorMap metatensor_torch::load_buffer(torch::Tensor buffer) {
    return TensorMapHolder::load_buffer(buffer);
}


void metatensor_torch::save(const std::string& path, TorchTensorMap tensor) {
    tensor->save(path);
}

torch::Tensor metatensor_torch::save_buffer(TorchTensorMap tensor) {
    return tensor->save_buffer();
}

/******************************************************************************/

TorchTensorBlock metatensor_torch::load_block(const std::string& path) {
    return TensorBlockHolder::load(path);
}

TorchTensorBlock metatensor_torch::load_block_buffer(torch::Tensor buffer) {
    return TensorBlockHolder::load_buffer(buffer);
}


void metatensor_torch::save(const std::string& path, TorchTensorBlock block) {
    block->save(path);
}

torch::Tensor metatensor_torch::save_buffer(TorchTensorBlock block) {
    return block->save_buffer();
}

/******************************************************************************/

TorchLabels metatensor_torch::load_labels(const std::string& path) {
    return LabelsHolder::load(path);
}

TorchLabels metatensor_torch::load_labels_buffer(torch::Tensor buffer) {
    return LabelsHolder::load_buffer(buffer);
}

void metatensor_torch::save(const std::string& path, TorchLabels labels) {
    labels->save(path);
}

torch::Tensor metatensor_torch::save_buffer(TorchLabels labels) {
    return labels->save_buffer();
}
