#include <equistore.hpp>

#include "equistore/torch/tensor.hpp"

using namespace equistore_torch;


static std::vector<equistore::TensorBlock> blocks_from_torch(std::vector<TorchTensorBlock> blocks) {
    throw std::runtime_error("not implemented");
}

TensorMapHolder::TensorMapHolder(
    TorchLabels keys,
    std::vector<TorchTensorBlock> blocks
):
    tensor_(keys->as_equistore(), blocks_from_torch(std::move(blocks)))
{
    throw std::runtime_error("not implemented");
}
