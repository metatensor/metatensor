#include <vector>
#include <cstdint>

#include <torch/torch.h>

#include <equistore.hpp>

#include "equistore/torch/array.hpp"

using namespace equistore_torch;

TorchDataArray::TorchDataArray(torch::Tensor tensor): tensor_(std::move(tensor)) {
    throw std::runtime_error("not implemented");
}

eqs_data_origin_t TorchDataArray::origin() const {
    throw std::runtime_error("not implemented");
}

std::unique_ptr<equistore::DataArrayBase> TorchDataArray::copy() const {
    throw std::runtime_error("not implemented");
}

std::unique_ptr<equistore::DataArrayBase> TorchDataArray::create(std::vector<uintptr_t> shape) const {
    throw std::runtime_error("not implemented");
}

double* TorchDataArray::data() {
    throw std::runtime_error("not implemented");
}

const std::vector<uintptr_t>& TorchDataArray::shape() const {
    throw std::runtime_error("not implemented");
}

void TorchDataArray::reshape(std::vector<uintptr_t> shape) {
    throw std::runtime_error("not implemented");
}

void TorchDataArray::swap_axes(uintptr_t axis_1, uintptr_t axis_2) {
    throw std::runtime_error("not implemented");
}

void TorchDataArray::move_samples_from(
    const equistore::DataArrayBase& raw_input,
    std::vector<eqs_sample_mapping_t> samples,
    uintptr_t property_start,
    uintptr_t property_end
) {
    throw std::runtime_error("not implemented");
}
