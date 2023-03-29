#include <memory>

#include <equistore.hpp>

#include "equistore/torch/array.hpp"
#include "equistore/torch/block.hpp"

using namespace equistore_torch;


static std::vector<equistore::Labels> components_from_torch(std::vector<TorchLabels> components) {
    throw std::runtime_error("not implemented");
}

TensorBlockHolder::TensorBlockHolder(
    torch::Tensor data,
    TorchLabels samples,
    std::vector<TorchLabels> components,
    TorchLabels properties
): block_(std::make_unique<TorchDataArray>(TorchDataArray(std::move(data))), samples->as_equistore(), components_from_torch(std::move(components)), properties->as_equistore()) {
    throw std::runtime_error("not implemented");
}
