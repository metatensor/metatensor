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

TensorBlockHolder::TensorBlockHolder(equistore::TensorBlock block, std::string parameter):
    block_(std::move(block)),
    parameter_(std::move(parameter))
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
    // handle recursive gradients
    std::string gradient_parameter;
    if (!parameter_.empty()) {
        gradient_parameter = parameter_ + "/" + parameter;
    } else {
        gradient_parameter = parameter;
    }

    return torch::make_intrusive<TensorBlockHolder>(block_.gradient(parameter), gradient_parameter);
}

std::unordered_map<std::string, TorchTensorBlock> TensorBlockHolder::gradients() {
    auto result = std::unordered_map<std::string, TorchTensorBlock>();
    for (const auto& parameter: this->gradients_list()) {
        result.emplace(parameter, this->gradient(parameter));
    }
    return result;
}

static void print_labels(std::ostringstream& output, const equistore::Labels& labels, const char* labels_kind) {
    output << "    " << labels_kind << " (" << labels.count() << "): ";
    output << "[";
    auto first = true;
    for (const auto& name: labels.names()) {
        if (!first) {
            output << ", ";
        }
        output << '\'' << name << '\'';
        first = false;
    }
    output << "]\n";
}

std::string TensorBlockHolder::__repr__() const {
    auto output = std::ostringstream();

    if (parameter_.empty()) {
        output << "TensorBlock\n";
    } else {
        output << "Gradient TensorBlock ('" << parameter_ << "')\n";
    }

    print_labels(output, block_.samples(), "samples");

    auto components = block_.components();
    output << "    components (";
    auto first = true;
    for (const auto& component: components) {
        if (!first) {
            output << ", ";
        }
        output << component.count();
        first = false;
    }

    output << "): [";
    first = true;
    for (const auto& component: components) {
        if (!first) {
            output << ", ";
        }
        assert(component.size() == 1);
        output << '\'' << component.names()[0] << '\'';
        first = false;
    }
    output << "]\n";

    print_labels(output, block_.properties(), "properties");

    auto gradients = block_.gradients_list();
    output << "    gradients: ";
    if (gradients.empty()) {
        output << "None\n";
    } else {
        first = true;
        output << "[";
        for (const auto& parameter: gradients) {
            if (!first) {
                output << ", ";
            }
            output << '\'' << parameter << '\'';
            first = false;
        }
        output << "]\n";
    }

    return output.str();
}
