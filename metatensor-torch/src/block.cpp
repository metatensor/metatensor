#include <memory>

#include <metatensor.hpp>

#include "metatensor/torch/array.hpp"
#include "metatensor/torch/block.hpp"

using namespace metatensor_torch;


static std::vector<metatensor::Labels> components_from_torch(const std::vector<TorchLabels>& components) {
    auto result = std::vector<metatensor::Labels>();
    for (const auto& component: components) {
        result.push_back(component->as_metatensor());
    }
    return result;
}

TensorBlockHolder::TensorBlockHolder(
    torch::Tensor data,
    TorchLabels samples,
    std::vector<TorchLabels> components,
    TorchLabels properties
):
    TensorBlockHolder(
        metatensor::TensorBlock(
            std::make_unique<TorchDataArray>(std::move(data)),
            samples->as_metatensor(),
            components_from_torch(components),
            properties->as_metatensor()
        ),
        /* parameter */ "",
        /* parent */ torch::IValue()
    )
{}


TensorBlockHolder::TensorBlockHolder(metatensor::TensorBlock block, torch::IValue parent):
    TensorBlockHolder(std::move(block), "", std::move(parent))
{}

TensorBlockHolder::TensorBlockHolder(metatensor::TensorBlock block, std::string parameter, torch::IValue parent):
    block_(std::move(block)),
    parameter_(std::move(parameter)),
    parent_(std::move(parent))
{}

torch::intrusive_ptr<TensorBlockHolder> TensorBlockHolder::copy() const {
    return torch::make_intrusive<TensorBlockHolder>(this->block_.clone(), torch::IValue());
}

torch::intrusive_ptr<TensorBlockHolder> TensorBlockHolder::to(torch::Device device) {
    const auto values = this->values().to(device);
    const auto samples = this->samples()->to(device);
    auto components = std::vector<torch::intrusive_ptr<LabelsHolder>>();
    for (const auto& component : this->components()) {
        components.push_back(component->to(device));
    }
    const auto properties = this->properties()->to(device);
    auto block = torch::make_intrusive<TensorBlockHolder>(values, samples, components, properties);
    for (const auto& gradient_name : this->gradients_list()) {
        block->add_gradient(
            gradient_name,
            torch::make_intrusive<TensorBlockHolder>(this->block_.gradient(gradient_name), torch::IValue())->to(device)
        );
    }
    return block;
}

torch::Tensor TensorBlockHolder::values() {
    auto array = block_.mts_array();

    mts_data_origin_t origin = 0;
    metatensor::details::check_status(array.origin(array.ptr, &origin));
    if (origin != TORCH_DATA_ORIGIN) {
        C10_THROW_ERROR(ValueError,
            "this TensorBlock does not contain a C++ torch Tensor"
        );
    }

    auto* ptr = static_cast<metatensor::DataArrayBase*>(array.ptr);
    auto* wrapper = dynamic_cast<TorchDataArray*>(ptr);
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
    auto gradient_block = metatensor::TensorBlock(
        std::make_unique<TorchDataArray>(gradient->values()),
        gradient->samples()->as_metatensor(),
        components_from_torch(gradient->components()),
        gradient->properties()->as_metatensor()
    );

    block_.add_gradient(parameter, std::move(gradient_block));
}

bool TensorBlockHolder::has_gradient(const std::string& parameter) const {
    auto list = this->block_.gradients_list();
    auto it = std::find(std::begin(list), std::end(list), parameter);
    return it != std::end(list);
}

TorchTensorBlock TensorBlockHolder::gradient(TorchTensorBlock self, const std::string& parameter) {
    // handle recursive gradients
    std::string gradient_parameter;
    if (!self->parameter_.empty()) {
        gradient_parameter = self->parameter_ + "/" + parameter;
    } else {
        gradient_parameter = parameter;
    }

    return torch::make_intrusive<TensorBlockHolder>(self->block_.gradient(parameter), gradient_parameter, self);
}

std::vector<std::tuple<std::string, TorchTensorBlock>> TensorBlockHolder::gradients(TorchTensorBlock self) {
    auto result = std::vector<std::tuple<std::string, TorchTensorBlock>>();
    for (const auto& parameter: self->gradients_list()) {
        result.emplace_back(parameter, TensorBlockHolder::gradient(self, parameter));
    }
    return result;
}

static void print_labels(std::ostringstream& output, const metatensor::Labels& labels, const char* labels_kind) {
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

std::string TensorBlockHolder::repr() const {
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
