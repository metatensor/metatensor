#include <equistore.hpp>

#include "equistore/torch/labels.hpp"

using namespace equistore_torch;

static equistore::NDArray<int32_t> tensor_to_ndarray(torch::Tensor values) {
    throw std::runtime_error("not implemented");
}


LabelsHolder::LabelsHolder(std::vector<std::string> names, torch::Tensor values):
    labels_(equistore::details::labels_from_cxx(names, tensor_to_ndarray(std::move(values))))
{
    throw std::runtime_error("not implemented");
}
