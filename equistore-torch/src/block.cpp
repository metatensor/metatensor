#include <memory>

#include <equistore.hpp>

#include "equistore/torch/array.hpp"
#include "equistore/torch/block.hpp"

using namespace equistore_torch;


static std::vector<equistore::Labels> components_from_torch(const std::vector<TorchLabels>& components) {
    auto result = std::vector<equistore::Labels>();
    for (const auto& component: components) {
        result.push_back(component->as_equistore());
    }
    return result;
}

TensorBlockHolder::TensorBlockHolder(
    torch::Tensor data,
    TorchLabels samples,
    std::vector<TorchLabels> components,
    TorchLabels properties
):
    block_(
        std::make_unique<TorchDataArray>(std::move(data)),
        samples->as_equistore(),
        components_from_torch(components),
        properties->as_equistore()
    )
{}


TensorBlockHolder::TensorBlockHolder(equistore::TensorBlock block):
    block_(std::move(block))
{}

torch::intrusive_ptr<TensorBlockHolder> TensorBlockHolder::copy() const {
    return torch::make_intrusive<TensorBlockHolder>(this->block_.clone());
}

torch::Tensor TensorBlockHolder::values() {
    auto array = block_.eqs_array();

    eqs_data_origin_t origin = 0;
    equistore::details::check_status(array.origin(array.ptr, &origin));
    if (origin != TORCH_DATA_ORIGIN) {
        C10_THROW_ERROR(ValueError,
            "this TensorBlock does not contain a C++ torch Tensor"
        );
    }

    auto ptr = static_cast<equistore::DataArrayBase*>(array.ptr);
    auto wrapper = dynamic_cast<TorchDataArray*>(ptr);
    if (wrapper == nullptr) {
        C10_THROW_ERROR(ValueError,
            "this TensorBlock does not contain a C++ torch Tensor"
        );
    }

    return wrapper->tensor();
}

TorchLabels TensorBlockHolder::labels(uintptr_t axis) const {
    return torch::make_intrusive<LabelsHolder>(block_.labels(axis));
}


void TensorBlockHolder::add_gradient(const std::string& parameter, TorchTensorBlock gradient) {
    // we need to move the tensor block in `add_gradient`, but we can not move
    // out of the `torch::intrusive_ptr` in `TorchTensorBlock`. So we create a
    // new temporary block, increasing the reference count to the values and
    // metadata of gradient.
    auto gradient_block = equistore::TensorBlock(
        std::make_unique<TorchDataArray>(gradient->values()),
        gradient->samples()->as_equistore(),
        components_from_torch(gradient->components()),
        gradient->properties()->as_equistore()
    );

    block_.add_gradient(parameter, std::move(gradient_block));
}

bool TensorBlockHolder::has_gradient(const std::string& parameter) const {
    auto list = this->block_.gradients_list();
    auto it = std::find(std::begin(list), std::end(list), parameter);
    return it != std::end(list);
}

TorchTensorBlock TensorBlockHolder::gradient(const std::string& parameter) const {
    return torch::make_intrusive<TensorBlockHolder>(block_.gradient(std::move(parameter)));
}

torch::IValue TensorBlockHolder::gradients() {
    auto result = c10::Dict<torch::IValue, torch::IValue>(
        c10::getTypePtr<std::string>(),
        c10::getTypePtr<TorchTensorBlock>()
    );
    for (const auto& parameter: this->gradients_list()) {
        result.insert(parameter, this->gradient(parameter));
    }
    return result;
}
