#include <vector>
#include <cstdint>

#include <torch/script.h>

#include <equistore.hpp>

#include "equistore/torch/array.hpp"

using namespace equistore_torch;

// We need to register a data origin with equistore, that will be used for all
// eqs_array_t containing a C++ torch tensor. This is initialized to 0 (meaning
// "no origin"), and will be set by `EqsOriginRegistration` in
// `TorchDataArray::origin`.
eqs_data_origin_t equistore_torch::TORCH_DATA_ORIGIN = 0;

struct EqsOriginRegistration {
    EqsOriginRegistration(const char* name) {
        auto status = eqs_register_data_origin(name, &TORCH_DATA_ORIGIN);
        if (status != EQS_SUCCESS) {
            C10_THROW_ERROR(ValueError, "failed to register torch data origin");
        }
    }
};


TorchDataArray::TorchDataArray(torch::Tensor tensor): tensor_(std::move(tensor)) {
    this->update_shape();
}

eqs_data_origin_t TorchDataArray::origin() const {
    // eqs_data_origin registration in a thread-safe way through C++11 static
    // initialization of a class with a constructor.
    static EqsOriginRegistration REGISTRATION = EqsOriginRegistration("equistore_torch::TorchDataArray");
    return TORCH_DATA_ORIGIN;
}

std::unique_ptr<equistore::DataArrayBase> TorchDataArray::copy() const {
    return std::unique_ptr<DataArrayBase>(new TorchDataArray(this->tensor().clone()));
}

std::unique_ptr<equistore::DataArrayBase> TorchDataArray::create(std::vector<uintptr_t> shape) const {
    auto sizes = std::vector<int64_t>();
    for (auto size: shape) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    return std::unique_ptr<DataArrayBase>(new TorchDataArray(
        torch::zeros(
            sizes,
            torch::TensorOptions()
                .dtype(this->tensor().dtype())
                .device(this->tensor().device())
        )
    ));
}

double* TorchDataArray::data() {
    if (!this->tensor_.device().is_cpu()) {
        C10_THROW_ERROR(ValueError, "can not access the data of a torch::Tensor not on CPU");
    }

    if (this->tensor_.dtype() != torch::kF64) {
        C10_THROW_ERROR(ValueError,
            "can not access the data of this torch::Tensor: expected a dtype "
            "of float64, got " + std::string(this->tensor_.dtype().name())
        );
    }

    if (!this->tensor_.is_contiguous()) {
        C10_THROW_ERROR(ValueError, "can not access the data of a non contiguous torch::Tensor");
    }

    return static_cast<double*>(this->tensor_.data_ptr());
}

const std::vector<uintptr_t>& TorchDataArray::shape() const {
    return shape_;
}

void TorchDataArray::reshape(std::vector<uintptr_t> shape) {
    auto sizes = std::vector<int64_t>();
    for (auto size: shape) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    this->tensor_ = this->tensor().reshape(sizes).contiguous();

    this->update_shape();
}

void TorchDataArray::swap_axes(uintptr_t axis_1, uintptr_t axis_2) {
    this->tensor_ = this->tensor().swapaxes(axis_1, axis_2).contiguous();

    this->update_shape();
}

void TorchDataArray::move_samples_from(
    const equistore::DataArrayBase& raw_input,
    std::vector<eqs_sample_mapping_t> samples,
    uintptr_t property_start,
    uintptr_t property_end
) {
    const auto& input = dynamic_cast<const TorchDataArray&>(raw_input);
    auto input_tensor = input.tensor();

    auto input_samples = std::vector<int64_t>();
    input_samples.reserve(samples.size());
    auto output_samples = std::vector<int64_t>();
    output_samples.reserve(samples.size());

    for (const auto& sample: samples) {
        input_samples.push_back(static_cast<int64_t>(sample.input));
        output_samples.push_back(static_cast<int64_t>(sample.output));
    }

    using torch::indexing::Slice;
    using torch::indexing::Ellipsis;
    auto output_tensor = this->tensor();

    // output[output_samples, ..., properties] = input[input_samples, ..., :]
    output_tensor.index_put_(
        {torch::tensor(std::move(output_samples)), Ellipsis, Slice(property_start, property_end)},
        input_tensor.index({torch::tensor(std::move(input_samples)), Ellipsis, Slice()})
    );
}

void TorchDataArray::update_shape() {
    shape_.clear();
    for (auto size: this->tensor_.sizes()) {
        shape_.push_back(static_cast<uintptr_t>(size));
    }
}
