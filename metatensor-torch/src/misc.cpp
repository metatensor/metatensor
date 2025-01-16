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
