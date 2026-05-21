#include <memory>
#include <string>

#include <metatensor.hpp>

#include "metatensor/torch/array.hpp"
#include "metatensor/torch/block.hpp"
#include "metatensor/torch/misc.hpp"

#include "./utils.hpp"

using namespace metatensor_torch;


static std::vector<metatensor::Labels> components_from_torch(const std::vector<Labels>& components) {
    auto result = std::vector<metatensor::Labels>();
    result.reserve(components.size());
    for (const auto& component: components) {
        result.push_back(component->as_metatensor());
    }
    return result;
}

TensorBlockHolder::TensorBlockHolder(
    torch::Tensor data,
    Labels samples,
    std::vector<Labels> components,
    Labels properties
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

TensorBlock TensorBlockHolder::copy(bool deep) const {
    if (deep) {
        return torch::make_intrusive<TensorBlockHolder>(this->block_.clone(), torch::IValue());
    } else {
        auto new_block = torch::make_intrusive<TensorBlockHolder>(TensorBlockHolder(
            this->values(),
            this->samples(),
            this->components(),
            this->properties()
        ));

        for (const auto& parameter: this->gradients_list()) {
            auto gradient = TensorBlockHolder(
                this->block_.gradient(parameter),
                torch::IValue()
            );

            new_block->add_gradient(parameter, gradient.copy(/*deep=*/false));
        }

        return new_block;
    }
}

TensorBlock TensorBlockHolder::to(
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device,
    bool non_blocking
) const {
    auto values = this->values().to(
        dtype,
        /*layout*/ torch::nullopt,
        device,
        /*pin_memory*/ torch::nullopt,
        /*non_blocking*/ non_blocking,
        /*copy*/ false,
        /*memory_format*/ torch::MemoryFormat::Preserve
    );

    auto samples = this->samples()->to(device);
    auto components = std::vector<torch::intrusive_ptr<LabelsHolder>>();
    for (const auto& component : this->components()) {
        components.push_back(component->to(device));
    }
    auto properties = this->properties()->to(device);

    auto block = torch::make_intrusive<TensorBlockHolder>(values, samples, components, properties);
    for (const auto& parameter : this->gradients_list()) {
        auto gradient = TensorBlockHolder(
            this->block_.gradient(parameter),
            torch::IValue()
        );

        block->add_gradient(parameter, gradient.to(dtype, device, non_blocking));
    }
    return block;
}

TensorBlock TensorBlockHolder::to_positional(
    torch::IValue positional_1,
    torch::IValue positional_2,
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device,
    torch::optional<std::string> arrays,
    bool non_blocking
) const {
    if (arrays.value_or("torch") != "torch") {
        C10_THROW_ERROR(ValueError,
            "`arrays` must be None or 'torch', got '" + arrays.value() + "' instead"
        );
    }

    auto [parsed_dtype, parsed_device] = to_arguments_parse(
        positional_1,
        positional_2,
        dtype,
        device,
        "`TensorBlock.to`"
    );

    return this->to(parsed_dtype, parsed_device, non_blocking);
}

torch::Tensor TensorBlockHolder::values() const {
    auto array = block_.const_mts_array();

    if (array.origin() != TORCH_DATA_ORIGIN) {
        C10_THROW_ERROR(ValueError,
            "this TensorBlock does not contain a C++ torch Tensor"
        );
    }

    auto* ptr = static_cast<metatensor::DataArrayBase*>(array.as_mts_array_t().ptr);
    auto* wrapper = dynamic_cast<TorchDataArray*>(ptr);
    if (wrapper == nullptr) {
        C10_THROW_ERROR(ValueError,
            "this TensorBlock does not contain a C++ torch Tensor"
        );
    }

    return wrapper->tensor();
}

Labels TensorBlockHolder::labels(uintptr_t axis) const {
    return torch::make_intrusive<LabelsHolder>(block_.labels(axis));
}

static metatensor::TensorBlock torch_to_metatensor_block(TensorBlock block) {
    auto non_torch_block = metatensor::TensorBlock(
        std::make_unique<TorchDataArray>(block->values()),
        block->samples()->as_metatensor(),
        components_from_torch(block->components()),
        block->properties()->as_metatensor()
    );

    for (const auto& parameter : block->gradients_list()) {
        auto gradient = TensorBlockHolder::gradient(block, parameter);
        non_torch_block.add_gradient(parameter, torch_to_metatensor_block(gradient));
    }

    return non_torch_block;
}


void TensorBlockHolder::add_gradient(const std::string& parameter, TensorBlock gradient) {
    auto gradient_block = torch_to_metatensor_block(gradient);

    // device/dtype consistency is enforced by metatensor-core
    block_.add_gradient(parameter, std::move(gradient_block));
}

bool TensorBlockHolder::has_gradient(const std::string& parameter) const {
    auto list = this->block_.gradients_list();
    auto it = std::find(std::begin(list), std::end(list), parameter);
    return it != std::end(list);
}

TensorBlock TensorBlockHolder::gradient(TensorBlock self, const std::string& parameter) {
    // handle recursive gradients
    std::string gradient_parameter;
    if (!self->parameter_.empty()) {
        gradient_parameter = self->parameter_ + "/" + parameter;
    } else {
        gradient_parameter = parameter;
    }

    return torch::make_intrusive<TensorBlockHolder>(self->block_.gradient(parameter), gradient_parameter, self);
}

std::vector<std::tuple<std::string, TensorBlock>> TensorBlockHolder::gradients(TensorBlock self) {
    auto result = std::vector<std::tuple<std::string, TensorBlock>>();
    for (const auto& parameter: self->gradients_list()) {
        result.emplace_back(parameter, TensorBlockHolder::gradient(self, parameter));
    }
    return result;
}

static void print_labels(std::ostringstream& output, const metatensor::Labels& labels, const char* labels_kind) {
    output << "    " << labels_kind << ": [";
    auto first = true;
    for (const auto& name: labels.names()) {
        if (!first) {
            output << ", ";
        }
        output << name;
        first = false;
    }
    output << "]";
}

std::string TensorBlockHolder::repr() const {
    auto output = std::ostringstream();
    auto shape = this->values().sizes();

    output << "TensorBlock";
    if (!parameter_.empty()) {
        output << " gradient for '" << parameter_ << "',";
    }
    output << " with shape (";

    for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) {
            output << ", ";
        }
        output << shape[i];
    }
    output << ")\n";

    print_labels(output, block_.samples(), "samples");
    output << "\n";

    auto components = block_.components();
    if (!components.empty()) {
        output << "    components: [";
        auto first = true;
        for (const auto& component: components) {
            if (!first) {
                output << ", ";
            }
            assert(component.size() == 1);
            output << component.names()[0];
            first = false;
        }
        output << "]\n";
    }

    print_labels(output, block_.properties(), "properties");

    auto gradients = block_.gradients_list();
    if (!gradients.empty()) {
        output << "\n\n    gradients:";
        size_t max_len = 0;
        for (const auto& parameter: gradients) {
            max_len = std::max(max_len, parameter.size());
        }
        for (const auto& parameter: gradients) {
            auto grad_shape = block_.gradient(parameter).values_shape();
            output << "\n        " << parameter;
            output << std::string(max_len - parameter.size(), ' ');
            output << " => TensorBlock with shape (";
            for (size_t i = 0; i < grad_shape.size(); i++) {
                if (i != 0) {
                    output << ", ";
                }
                output << grad_shape[i];
            }
            output << ")";
        }
    }

    return output.str();
}

TensorBlock TensorBlockHolder::load(const std::string& path) {
    return torch::make_intrusive<TensorBlockHolder>(
        TensorBlockHolder(
            metatensor::io::load_block(path, details::create_torch_array),
            /*parent=*/torch::IValue()
        )
    );
}

TensorBlock TensorBlockHolder::load_buffer(torch::Tensor buffer) {
    if (buffer.scalar_type() != torch::kUInt8) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a tensor of uint8, not " +
            scalar_type_name(buffer.scalar_type())
        );
    }

    if (buffer.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a 1-dimensional tensor"
        );
    }

    auto block = metatensor::io::load_block_buffer(
        buffer.data_ptr<uint8_t>(),
        static_cast<size_t>(buffer.size(0)),
        details::create_torch_array
    );

    return torch::make_intrusive<TensorBlockHolder>(
        TensorBlockHolder(std::move(block), /*parent=*/torch::IValue())
    );
}


void TensorBlockHolder::save(const std::string& path) const {
    metatensor::io::save(path, this->block_);
}

torch::Tensor TensorBlockHolder::save_buffer() const {
    auto buffer = metatensor::io::save_buffer(this->block_);
    // move the buffer to the heap so it can escape this function
    // `torch::from_blob` does not take ownership of the data,
    // so we need to register a custom deleter to clean up when
    // the tensor is no longer used
    auto* buffer_data = new std::vector<uint8_t>(std::move(buffer));

    auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU);
    auto deleter = [=](void* data) {
        delete buffer_data;
    };

    // use a tensor of bytes to store the data
    return torch::from_blob(
        buffer_data->data(),
        {static_cast<int64_t>(buffer_data->size())},
        deleter,
        options
    );
}

metatensor::TensorBlock TensorBlockHolder::release() {
    if (!parent_.isNone()) {
        throw std::runtime_error(
            "can not release this TensorBlock, it is a view inside another"
            " TensorBlock or a TensorMap"
        );
    }

    return std::move(block_);
}

TensorBlock TensorBlockHolder::from_metatensor(metatensor::TensorBlock block) {
    return torch::make_intrusive<TensorBlockHolder>(std::move(block), torch::IValue());
}

TensorBlock TensorBlockHolder::view_from_metatensor(metatensor::TensorBlock block, torch::IValue parent) {
    if (parent.isNone()) {
        C10_THROW_ERROR(ValueError,
            "`parent` cannot be None when creating a TensorBlock view"
        );
    }

    if (!block.is_view()) {
        C10_THROW_ERROR(ValueError,
            "the provided metatensor::TensorBlock is not a view, "
            "cannot create a metatensor_torch::TensorBlock view from it"
        );
    }

    return torch::make_intrusive<TensorBlockHolder>(std::move(block), std::move(parent));
}
