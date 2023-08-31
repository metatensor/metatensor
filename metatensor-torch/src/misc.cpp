#include <torch/torch.h>

#include <metatensor.hpp>
#include <torch/types.h>

#include "metatensor/torch/array.hpp"
#include "metatensor/torch/misc.hpp"

using namespace metatensor_torch;


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


TorchTensorMap metatensor_torch::load(const std::string& path) {
    return torch::make_intrusive<TensorMapHolder>(
        metatensor::TensorMap::load(path, details::create_torch_array)
    );
}


void metatensor_torch::save(const std::string& path, TorchTensorMap tensor) {
    metatensor::TensorMap::save(path, tensor->as_metatensor());
}
