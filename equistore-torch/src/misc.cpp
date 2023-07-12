#include <torch/torch.h>

#include <equistore.hpp>
#include <torch/types.h>

#include "equistore/torch/array.hpp"
#include "equistore/torch/misc.hpp"

using namespace equistore_torch;


eqs_status_t equistore_torch::details::create_torch_array(
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    eqs_array_t* array
) {
    return equistore::details::catch_exceptions([](
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        eqs_array_t* array
    ) {
        auto sizes = std::vector<int64_t>();
        for (size_t i=0; i<shape_count; i++) {
            sizes.push_back(static_cast<int64_t>(shape_ptr[i]));
        }

        auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kF64);
        auto tensor = torch::zeros(sizes, options);

        auto cxx_array = std::unique_ptr<equistore::DataArrayBase>(new TorchDataArray(tensor));
        *array = equistore::DataArrayBase::to_eqs_array_t(std::move(cxx_array));

        return EQS_SUCCESS;
    }, shape_ptr, shape_count, array);
}


TorchTensorMap equistore_torch::load(const std::string& path) {
    return torch::make_intrusive<TensorMapHolder>(
        equistore::TensorMap::load(path, details::create_torch_array)
    );
}


void equistore_torch::save(const std::string& path, TorchTensorMap tensor) {
    equistore::TensorMap::save(path, tensor->as_equistore());
}
